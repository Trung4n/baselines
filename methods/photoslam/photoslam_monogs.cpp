/**
 * Photo-SLAM (RGB-D) on a MonoGS Replica-format dataset folder (baselines repo,
 * methods/photoslam). The SLAM part is Photo-SLAM's examples/replica_rgbd.cpp,
 * unchanged in what reaches the system: frames read with cv::imread and turned
 * to RGB, image scale, TrackRGBD with timestamp = frame index (as that example),
 * the GaussianMapper training thread, Shutdown, then the mapper's own final
 * training and saving (it runs until it stops by itself). Differences:
 * - frames are results/frame%06d.jpg and results/depth%06d.png by index;
 * - no viewer (Kaggle has no display);
 * - per frame: tracking state and tracking time to states.csv; progress lines;
 * - wall time of the tracking loop (frame reading included, as MonoGS times
 *   it), of start-up, and of the mapper's work after the loop, to timing.json;
 * - after the mapper has stopped: the camera-to-world pose of every frame of the
 *   map with the most keyframes, in that map's own world frame (the one the
 *   Gaussians live in; ORB-SLAM3's savers re-express poses relative to the first
 *   keyframe instead), its keyframe list, and renders of the final Gaussian map
 *   at those poses (GaussianMapper::renderFromPose, full resolution) for the
 *   frames the protocol scores: every `interval`-th frame below N-1 that is not
 *   a keyframe, plus the figure frames (render_rule.txt, written by the adapter);
 * - ORB-SLAM3's own trajectory files as the example writes them;
 * - ends with std::_Exit(0) once everything is written.
 *
 * Usage: photoslam_monogs vocabulary orb_settings mapper_settings dataset_folder
 *        n_frames render_rule.txt out_dir
 */
#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "ORB-SLAM3/include/System.h"
#include "include/gaussian_mapper.h"

using namespace std;

static double now_s()
{
    return chrono::duration<double>(chrono::steady_clock::now().time_since_epoch()).count();
}

static string frame_path(const string &dir, const char *prefix, int i, const char *ext)
{
    char name[64];
    snprintf(name, sizeof(name), "%s%06d.%s", prefix, i, ext);
    return dir + "/results/" + name;
}

// .npy (version 1.0), float32, C order, shape (h, w, c)
static void write_npy_f32(const string &path, const cv::Mat &m)
{
    CV_Assert(m.type() == CV_32FC3 && m.isContinuous());
    ostringstream hs;
    hs << "{'descr': '<f4', 'fortran_order': False, 'shape': (" << m.rows << ", " << m.cols << ", 3), }";
    string header = hs.str();
    size_t total = 10 + header.size() + 1;
    header.append((64 - total % 64) % 64, ' ');
    header.push_back('\n');
    ofstream f(path, ios::binary);
    const char magic[] = "\x93NUMPY\x01\x00";
    f.write(magic, 8);
    uint16_t len = header.size();
    f.write(reinterpret_cast<const char *>(&len), 2);
    f.write(header.data(), header.size());
    f.write(reinterpret_cast<const char *>(m.ptr<float>(0)), m.total() * 3 * sizeof(float));
}

int main(int argc, char **argv)
{
    if(argc != 8)
    {
        cerr << "Usage: photoslam_monogs vocabulary orb_settings mapper_settings dataset_folder "
                "n_frames render_rule.txt out_dir" << endl;
        return 1;
    }
    const string dataset = argv[4];
    const int nImages = atoi(argv[5]);
    const string out = argv[7];
    filesystem::path out_dir(out);
    filesystem::create_directories(out_dir / "renders");
    if(nImages <= 0)
    {
        cerr << "n_frames must be positive" << endl;
        return 1;
    }
    int interval = 0;
    vector<int> figure_frames;
    {
        ifstream f(argv[6]);
        string key;
        while(f >> key)
        {
            if(key == "interval")
                f >> interval;
            else if(key == "figures")
            {
                string line;
                getline(f, line);
                istringstream ls(line);
                int v;
                while(ls >> v)
                    figure_frames.push_back(v);
            }
        }
        if(interval <= 0)
        {
            cerr << argv[6] << ": no 'interval <n>' line" << endl;
            return 1;
        }
    }

    torch::DeviceType device_type = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    cout << (device_type == torch::kCUDA ? "CUDA available! Training on GPU." : "Training on CPU.") << endl;

    const double t_init0 = now_s();
    shared_ptr<ORB_SLAM3::System> pSLAM = make_shared<ORB_SLAM3::System>(argv[1], argv[2], ORB_SLAM3::System::RGBD);
    float imageScale = pSLAM->GetImageScale();
    filesystem::path gaussian_cfg_path(argv[3]);
    filesystem::path mapper_out = out_dir / "mapper";
    filesystem::create_directories(mapper_out);
    shared_ptr<GaussianMapper> pGausMapper =
        make_shared<GaussianMapper>(pSLAM, gaussian_cfg_path, mapper_out.string() + "/", 0, device_type);
    thread training_thd(&GaussianMapper::run, pGausMapper.get());
    const double init_s = now_s() - t_init0;

    vector<float> vTimesTrack(nImages, 0.f);
    vector<int> vState(nImages, -1);
    int width = 0, height = 0;
    cout << endl << "-------" << endl << "Start processing sequence ..." << endl
         << "Images in the sequence: " << nImages << endl << endl << flush;

    cv::Mat imRGB, imD;
    const double t_loop0 = now_s();
    int processed = 0;
    for(int ni = 0; ni < nImages; ni++)
    {
        if(pSLAM->isShutDown())
            break;
        imRGB = cv::imread(frame_path(dataset, "frame", ni, "jpg"), cv::IMREAD_UNCHANGED);
        imD = cv::imread(frame_path(dataset, "depth", ni, "png"), cv::IMREAD_UNCHANGED);
        if(imRGB.empty() || imD.empty())
        {
            cerr << endl << "Failed to load frame " << ni << " in " << dataset << endl;
            return 1;
        }
        cv::cvtColor(imRGB, imRGB, cv::COLOR_BGR2RGB);
        double tframe = ni;
        if(ni == 0)
        {
            width = imRGB.cols;
            height = imRGB.rows;
        }
        if(imageScale != 1.f)
        {
            int w = imRGB.cols * imageScale;
            int h = imRGB.rows * imageScale;
            cv::resize(imRGB, imRGB, cv::Size(w, h));
            cv::resize(imD, imD, cv::Size(w, h));
        }

        auto t1 = chrono::steady_clock::now();
        pSLAM->TrackRGBD(imRGB, imD, tframe, vector<ORB_SLAM3::IMU::Point>(), frame_path(dataset, "frame", ni, "jpg"));
        auto t2 = chrono::steady_clock::now();
        double ttrack = chrono::duration_cast<chrono::duration<double> >(t2 - t1).count();
        vTimesTrack[ni] = ttrack;
        vState[ni] = pSLAM->GetTrackingState();
        processed = ni + 1;
        if((ni + 1) % 100 == 0 || ni + 1 == nImages)
            cout << "frame " << ni + 1 << "/" << nImages << " state " << vState[ni] << " track " << fixed
                 << setprecision(1) << ttrack * 1000 << " ms, mapper iteration " << pGausMapper->getIteration()
                 << endl << flush;
    }
    const double loop_s = now_s() - t_loop0;
    if(processed != nImages)
    {
        cerr << "SLAM shut down after " << processed << " of " << nImages << " frames" << endl;
        return 1;
    }

    const double t_after0 = now_s();
    pSLAM->Shutdown();
    cout << "tracking done; waiting for the Gaussian mapper to finish its own final training and saving" << endl
         << flush;
    training_thd.join();
    const double mapper_after_s = now_s() - t_after0;
    cout << "mapper stopped after " << pGausMapper->getIteration() << " iterations" << endl << flush;

    pSLAM->SaveTrajectoryTUM((out_dir / "CameraTrajectory_TUM.txt").string());
    pSLAM->SaveKeyFrameTrajectoryTUM((out_dir / "KeyFrameTrajectory_TUM.txt").string());
    pSLAM->SaveTrajectoryEuRoC((out_dir / "CameraTrajectory_EuRoC.txt").string());

    // Frame poses in the world frame of the largest map (as SaveTrajectoryEuRoC,
    // without its re-expression relative to the map's first keyframe).
    ORB_SLAM3::Atlas *atlas = pSLAM->getAtlas();
    ORB_SLAM3::Tracking *tracker = pSLAM->getTracker();
    vector<ORB_SLAM3::Map *> maps = atlas->GetAllMaps();
    ORB_SLAM3::Map *big = nullptr;
    size_t most = 0;
    for(ORB_SLAM3::Map *m : maps)
        if(m && m->GetAllKeyFrames().size() > most)
        {
            most = m->GetAllKeyFrames().size();
            big = m;
        }
    if(!big)
    {
        cerr << "no map in the atlas" << endl;
        return 1;
    }
    map<int, Sophus::SE3f> Tcw_of;
    {
        auto lRit = tracker->mlpReferences.begin();
        auto lT = tracker->mlFrameTimes.begin();
        auto lbL = tracker->mlbLost.begin();
        for(auto lit = tracker->mlRelativeFramePoses.begin(); lit != tracker->mlRelativeFramePoses.end();
            lit++, lRit++, lT++, lbL++)
        {
            if(*lbL)
                continue;
            ORB_SLAM3::KeyFrame *pKF = *lRit;
            if(!pKF)
                continue;
            Sophus::SE3f Trw;
            while(pKF && pKF->isBad())
            {
                Trw = Trw * pKF->mTcp;
                pKF = pKF->GetParent();
            }
            if(!pKF || pKF->GetMap() != big)
                continue;
            Trw = Trw * pKF->GetPose();
            Tcw_of[(int)lround(*lT)] = (*lit) * Trw;
        }
    }
    set<int> keyframes;
    for(ORB_SLAM3::KeyFrame *pKF : big->GetAllKeyFrames())
        if(pKF && !pKF->isBad())
            keyframes.insert((int)lround(pKF->mTimeStamp));
    {
        ofstream f(out_dir / "trajectory_main_map_world.txt");
        f << fixed;
        for(auto &kv : Tcw_of)
        {
            Sophus::SE3f Twc = kv.second.inverse();
            Eigen::Vector3f t = Twc.translation();
            Eigen::Quaternionf q = Twc.unit_quaternion();
            f << kv.first << " " << setprecision(9) << t(0) << " " << t(1) << " " << t(2) << " " << q.x() << " "
              << q.y() << " " << q.z() << " " << q.w() << "\n";
        }
        ofstream k(out_dir / "keyframes_main_map.txt");
        for(int i : keyframes)
            k << i << "\n";
        ofstream s(out_dir / "states.csv");
        s << "frame,state,track_s\n";
        for(int ni = 0; ni < nImages; ni++)
            s << ni << "," << vState[ni] << "," << setprecision(9) << vTimesTrack[ni] << "\n";
    }
    cout << "atlas: " << maps.size() << " map(s); main map " << Tcw_of.size() << " frames, " << keyframes.size()
         << " keyframes" << endl << flush;

    // Renders of the final map at the estimated poses.
    set<int> want;
    for(int i = 0; i < nImages - 1; i += interval)
        if(!keyframes.count(i))
            want.insert(i);
    for(int i : figure_frames)
        if(i < nImages)
            want.insert(i);
    int done = 0, no_pose = 0;
    ofstream nopose(out_dir / "renders_without_pose.txt");
    const double t_render0 = now_s();
    for(int i : want)
    {
        auto it = Tcw_of.find(i);
        if(it == Tcw_of.end())
        {
            nopose << i << "\n";
            no_pose++;
            continue;
        }
        cv::Mat img = pGausMapper->renderFromPose(it->second, width, height, true);
        if(img.rows != height || img.cols != width || img.type() != CV_32FC3)
        {
            cerr << "render of frame " << i << ": " << img.cols << "x" << img.rows << " type " << img.type() << endl;
            return 1;
        }
        char name[32];
        snprintf(name, sizeof(name), "%06d.npy", i);
        write_npy_f32((out_dir / "renders" / name).string(), img.isContinuous() ? img : img.clone());
        if(++done % 50 == 0)
            cout << "rendered " << done << "/" << want.size() << endl << flush;
    }
    nopose.close();
    const double render_s = now_s() - t_render0;
    cout << "rendered " << done << " frames, " << no_pose << " without a pose" << endl << flush;

    {
        ofstream f(out_dir / "driver_timing.json");
        f << setprecision(9) << "{\n"
          << "  \"frames\": " << nImages << ",\n"
          << "  \"init_s\": " << init_s << ",\n"
          << "  \"loop_s\": " << loop_s << ",\n"
          << "  \"mapper_after_loop_s\": " << mapper_after_s << ",\n"
          << "  \"mapper_iterations\": " << pGausMapper->getIteration() << ",\n"
          << "  \"render_s\": " << render_s << ",\n"
          << "  \"maps\": " << maps.size() << ",\n"
          << "  \"width\": " << width << ",\n"
          << "  \"height\": " << height << "\n}\n";
        ofstream d(out_dir / "driver_done");
        d << "ok\n";
    }
    cout << "photoslam_monogs: done" << endl << flush;
    std::_Exit(0);
}
