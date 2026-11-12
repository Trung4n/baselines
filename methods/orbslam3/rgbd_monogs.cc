/**
 * ORB-SLAM3 RGB-D on a MonoGS Replica-format dataset folder (baselines repo,
 * methods/orbslam3). The loop is Examples/RGB-D/rgbd_tum.cc of ORB-SLAM3 v1.0,
 * unchanged in what reaches the SLAM system (cv::imread IMREAD_UNCHANGED, the
 * image scale, TrackRGBD with the frame timestamp, then a sleep until the next
 * timestamp, as a live sensor would deliver frames). Differences, all I/O:
 * - frames are results/frame%06d.jpg and results/depth%06d.png, frame i at
 *   timestamp i / fps (no association file; the dataset has no timestamps);
 * - the viewer is off (no display on Kaggle);
 * - per frame: tracking state and tracking time to states.csv; progress lines;
 * - wall time of the loop (frame reading included, as MonoGS times it) and of
 *   the system start-up (vocabulary loading) to timing.json;
 * - trajectories: SaveTrajectoryTUM / SaveKeyFrameTrajectoryTUM as rgbd_tum
 *   (every map of the atlas, relative to the first keyframe), and
 *   SaveTrajectoryEuRoC / SaveKeyFrameTrajectoryEuRoC (only the map with the
 *   most keyframes: frames of other maps are in other coordinate frames);
 * - ends with std::_Exit(0) once the files are written: Shutdown() does not
 *   join the threads (upstream), so normal exit can crash in destructors.
 *
 * Usage: rgbd_monogs vocabulary settings dataset_folder n_frames fps out_dir
 */
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <System.h>

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

int main(int argc, char **argv)
{
    if(argc != 7)
    {
        cerr << "Usage: rgbd_monogs vocabulary settings dataset_folder n_frames fps out_dir" << endl;
        return 1;
    }
    const string dataset = argv[3];
    const int nImages = atoi(argv[4]);
    const double fps = atof(argv[5]);
    const string out = argv[6];
    if(nImages <= 0 || fps <= 0)
    {
        cerr << "n_frames and fps must be positive" << endl;
        return 1;
    }

    const double t_init0 = now_s();
    ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::RGBD, false);
    float imageScale = SLAM.GetImageScale();
    const double init_s = now_s() - t_init0;

    vector<float> vTimesTrack(nImages);
    vector<int> vState(nImages);
    cout << endl << "-------" << endl << "Start processing sequence ..." << endl
         << "Images in the sequence: " << nImages << endl << endl << flush;

    cv::Mat imRGB, imD;
    const double t_loop0 = now_s();
    for(int ni = 0; ni < nImages; ni++)
    {
        imRGB = cv::imread(frame_path(dataset, "frame", ni, "jpg"), cv::IMREAD_UNCHANGED);
        imD = cv::imread(frame_path(dataset, "depth", ni, "png"), cv::IMREAD_UNCHANGED);
        double tframe = ni / fps;

        if(imRGB.empty() || imD.empty())
        {
            cerr << endl << "Failed to load frame " << ni << " in " << dataset << endl;
            return 1;
        }

        if(imageScale != 1.f)
        {
            int width = imRGB.cols * imageScale;
            int height = imRGB.rows * imageScale;
            cv::resize(imRGB, imRGB, cv::Size(width, height));
            cv::resize(imD, imD, cv::Size(width, height));
        }

        auto t1 = chrono::steady_clock::now();
        SLAM.TrackRGBD(imRGB, imD, tframe);
        auto t2 = chrono::steady_clock::now();
        double ttrack = chrono::duration_cast<chrono::duration<double> >(t2 - t1).count();
        vTimesTrack[ni] = ttrack;
        vState[ni] = SLAM.GetTrackingState();

        if((ni + 1) % 100 == 0 || ni + 1 == nImages)
            cout << "frame " << ni + 1 << "/" << nImages << " state " << vState[ni]
                 << " track " << fixed << setprecision(1) << ttrack * 1000 << " ms" << endl << flush;

        // Wait to load the next frame (as rgbd_tum)
        double T = 0;
        if(ni < nImages - 1)
            T = 1.0 / fps;
        else if(ni > 0)
            T = 1.0 / fps;
        if(ttrack < T)
            usleep((T - ttrack) * 1e6);
    }
    const double loop_s = now_s() - t_loop0;

    SLAM.Shutdown();

    {
        ofstream f(out + "/states.csv");
        f << "frame,state,track_s\n";
        for(int ni = 0; ni < nImages; ni++)
            f << ni << "," << vState[ni] << "," << setprecision(9) << vTimesTrack[ni] << "\n";
    }

    vector<float> sorted = vTimesTrack;
    sort(sorted.begin(), sorted.end());
    double total = 0;
    for(float t : sorted)
        total += t;
    {
        ofstream f(out + "/driver_timing.json");
        f << setprecision(9) << "{\n"
          << "  \"frames\": " << nImages << ",\n"
          << "  \"fps_timestamps\": " << fps << ",\n"
          << "  \"init_s\": " << init_s << ",\n"
          << "  \"loop_s\": " << loop_s << ",\n"
          << "  \"track_median_s\": " << sorted[nImages / 2] << ",\n"
          << "  \"track_mean_s\": " << total / nImages << "\n}\n";
    }
    cout << "-------" << endl << endl
         << "median tracking time: " << sorted[nImages / 2] << endl
         << "mean tracking time: " << total / nImages << endl
         << "loop wall time: " << loop_s << " s for " << nImages << " frames" << endl << flush;

    SLAM.SaveTrajectoryTUM(out + "/CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryTUM(out + "/KeyFrameTrajectory.txt");
    SLAM.SaveTrajectoryEuRoC(out + "/CameraTrajectory_main_map.txt");
    SLAM.SaveKeyFrameTrajectoryEuRoC(out + "/KeyFrameTrajectory_main_map.txt");

    {
        ofstream f(out + "/driver_done");
        f << "ok\n";
    }
    cout << "rgbd_monogs: done" << endl << flush;
    std::_Exit(0);
}
