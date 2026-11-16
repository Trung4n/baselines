# SEGS-SLAM (RGB-D)

- Bài báo: SEGS-SLAM: Structure-enhanced 3D Gaussian Splatting SLAM with Appearance Embedding (Wen, Liu, Fang, ICCV 2025, arXiv 2501.05242). Thêm vào `refs.bib` khi dùng.
- Mã nguồn: chép sẵn trong repo, `third_party/SEGS-SLAM` (leaner-forever/SEGS-SLAM @ `2110994a4acb6c1fe3d3e082f21899bc20d36d03`, 2025-10-18; README của repo ghi "Official code for ICCV 2025"), kèm bản ORB-SLAM3 đã sửa của nó và vocabulary. OpenCV có CUDA, LibTorch và TorchScatter **không** chép vào repo; `setup_kaggle.sh` tải chúng từ nguồn chính thức, phiên bản ghim trong script.
- Vai trò trong so sánh: **đối thủ trực tiếp nhất cho câu 3** (`CLAUDE.md`): hệ 3DGS-SLAM có cơ chế chống thay đổi ngoại hình. SEGS-SLAM xây trên Photo-SLAM (cùng tracking ORB-SLAM3), thêm bản đồ dạng anchor (Scaffold-GS: mỗi anchor sinh 10 Gaussian qua MLP) và **nhúng ngoại hình sinh từ pose camera**: một MLP nhận vị trí và quaternion của camera, ra vectơ ngoại hình đưa vào MLP màu (`src/gaussian_renderer.cpp`). Về nguyên tắc, nó có thể học ánh sáng thay đổi theo pose, đúng kiểu đèn đồng hành, nhưng bằng mạng học chứ không bằng mô hình vật lý. Đặt cạnh Photo-SLAM (cùng tracking, không có nhúng) cho biết riêng phần học ngoại hình giúp được bao nhiêu.

## File

| File | Việc |
|---|---|
| `segs_monogs.cpp` | Driver C++, sửa từ `methods/photoslam/photoslam_monogs.cpp`: vòng lặp của `examples/replica_rgbd.cpp` của SEGS-SLAM theo đúng cách `scripts/replica_rgbd.sh` gọi (không viewer, không `undistorted_image`). Sau frame cuối, như ví dụ: lưu `CameraTrajectory_TUM_bf.txt`, nạp vào mapper (`pose_`, `poseSaved`; mapper đặt lại pose keyframe từ đó khi SLAM dừng), `Shutdown`, chờ mapper train tiếp đến `max_num_iterations` và tự lưu. Trong lúc chờ, cứ 60 s in số vòng (để runner không coi là treo). Sau đó: pose mọi frame của bản đồ lớn nhất trong hệ world của nó, keyframe, render bản đồ cuối tại các pose đó (`renderFromPose`) |
| `adapter.py` | Sửa từ adapter Photo-SLAM: settings ORB-SLAM3 (camera của dataset), chép `replica_rgbd.yaml` của mapper, chạy driver, xuất `<work>/export/`, đóng bản đồ cuối (anchor `.ply` + thư mục `mlp/`, thiếu một trong hai thì không render lại được) thành một `.tar`, đóng zip |
| `setup_kaggle.sh` | Sửa từ setup Photo-SLAM: CUDA 12 cài cạnh CUDA 13 khi cần, LibTorch, OpenCV có CUDA (cùng tên cache với Photo-SLAM), thêm `libpcl-dev`, TorchScatter 2.1.2 build có CUDA cài vào `/usr/local`, build DBoW2, g2o, ORB-SLAM3 bản SEGS, SEGS-SLAM và driver |
| `kaggle.json` | `"checkpoints": false`, `"setup_timeout_min": 240` |

## Cách chạy

1. **Kaggle**, GPU T4, bật Internet: notebook chung `run_baseline.ipynb`, đặt `METHOD = "segsslam"`. Runner tự chạy với `--checkpoint_every 0` và nới thời gian cài đặt lên 240 phút.
2. **OpenCV có CUDA:** tên file cache giống hệt Photo-SLAM (`opencv-4.10.0-cuda<ver>-sm<arch>.tar.gz`), và setup tìm nó trong mọi thư mục của `/kaggle/input`. Nếu đã tạo Kaggle dataset từ cache của phiên Photo-SLAM thì **gắn dataset đó vào**, khỏi build lại khoảng một giờ. Không có thì setup build và lưu cache vào `/kaggle/working/segsslam_cache/`.
3. Mỗi run cho `/kaggle/working/segsslam__<scene>_<variant>__<thời điểm>.zip`, có render, ảnh 5 frame chung và bản đồ cuối (`run/map_iteration_<n>.tar`; `NO_MAP = True` bỏ nó ra).
4. Tải zip về, chấm cục bộ:

```
python baselines/common/score_zip.py --zip segsslam__office0_lit__<...>.zip \
    --dataset /mnt/d/CodeForYou/Replica/04_datasets/monogs/office0/lit --no_wandb
```

## Phụ thuộc và phiên bản

| Thứ | Phiên bản | Lý do |
|---|---|---|
| OpenCV + opencv_contrib | 4.10.0, có CUDA, cùng danh sách module như Photo-SLAM (SEGS-SLAM include đúng các module đó) | Như Photo-SLAM. Tác giả thử 4.7/4.8 |
| LibTorch (C++, cxx11 ABI) | CUDA 12.x → 2.1.2+cu121; CUDA 11.x → 2.0.1+cu118 (cấu hình của tác giả) | SEGS-SLAM thừa hưởng giới hạn ≤ 2.1.2 của Photo-SLAM |
| Bộ CUDA dùng để build | nvcc hệ thống nếu 11.x/12.x; nếu 13 trở lên (Kaggle 2026-11: 13.0), cài thêm CUDA 12 qua apt của NVIDIA vào `/usr/local/cuda-12.x` | Như Photo-SLAM |
| TorchScatter | pytorch_scatter 2.1.2 (bản tác giả ghi), thư viện C++, `WITH_CUDA=ON`, cài vào `/usr/local` như `docker/Dockerfile` của tác giả | SEGS-SLAM gọi `scatter_max` trên tensor CUDA. CMakeLists của nó link cứng `/usr/local/lib/libtorchscatter.so` và tìm header trong đường mặc định |
| PCL | `libpcl-dev` của Ubuntu, cộng `mpi-default-dev` | `gaussian_model.h`, `gaussian_mapper.h` include PCL, cả `pcl/visualization` (kéo theo VTK 9). Gói VTK của Ubuntu có target `VTK::mpi` link tới `MPI::MPI_C`, target chỉ có khi đã tìm MPI. Thiếu nó thì bước generate của CMake hỏng (Kaggle 2026-11-16). Setup cài MPI và chèn `find_package(MPI REQUIRED COMPONENTS C)` trước `find_package(PCL REQUIRED)` trên bản chép CMakeLists. FindMPI cần ngôn ngữ C, mà `project()` của SEGS-SLAM chỉ khai `CXX CUDA` (lần 2 hỏng vì "language C is not enabled"), nên bản chép thêm `C` vào `project()`. Đầu setup có **bước dò PCL** (`pcl_probe`): một project CMake nhỏ gọi đúng các `find_package` từ gói apt (MPI, PCL, Eigen, jsoncpp, OpenGL, glm, glfw, OpenMP) với cùng ngôn ngữ, build một chương trình dùng các header PCL của SEGS-SLAM (buộc nạp `pcl_visualization` và VTK) rồi chạy nó. Bước này chạy trước CUDA 12, OpenCV, LibTorch, nên lỗi loại này hiện sau vài phút |

## Khác biệt so với mặc định của SEGS-SLAM cho Replica

| Đổi | Lý do |
|---|---|
| `cfg/ORB_SLAM3/RGB-D/Replica/<scene>.yaml`: `Camera1.fx/fy/cx/cy`, `Camera.width/height` từ `protocol.calibration()` (600×340) | Bắt buộc |
| Bỏ `Camera1.k1, k2, k3, p1, p2` (nếu có) | File Replica có hệ số méo nhỏ khác 0, ảnh không méo (như Photo-SLAM) |
| Giữ nguyên | Mọi khóa ORB khác, kể cả **7200 đặc trưng** (`office0`; `room0` 9600), đặt cho ảnh 1200×680. Ở 600×340 ORB có thể không tìm đủ chừng đó điểm; giữ theo quy tắc không chỉnh tham số. Mapper: `cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml` nguyên văn, đúng file `scripts/replica_rgbd.sh` dùng cho mọi cảnh Replica (không phải `<scene>.yaml`) |
| Timestamp = chỉ số frame, không giữ nhịp | Như `replica_rgbd.cpp` |
| Viewer tắt, không truyền `undistorted_image` | Như `scripts/replica_rgbd.sh` |
| Build: như Photo-SLAM (`-include cstdint` cho C++, `<cstdint>`/`<cfloat>` chèn vào file CUDA, chính sách CMake ≥ 3.5, `-rpath-link`, `LD_LIBRARY_PATH` riêng lúc chạy); thêm `set(Torch_DIR ...)` trỏ vào LibTorch đã tải (CMakeLists ghi đường dẫn máy tác giả), TorchScatter build với `CMAKE_CXX_STANDARD 17` (gốc 14; header LibTorch ≥ 2.0 cần C++17) và `WITH_PYTHON=OFF` | Chỉ sửa lỗi build; làm trên bản chép |

## Điểm cần nói khi diễn giải

- **Tracking là ORB-SLAM3 bản SEGS** (sửa `Tracking`, `Frame`, `KeyFrame`), có loop closure. So ATE trực tiếp với Photo-SLAM và ORB-SLAM3; khác biệt về tracking còn đến từ số đặc trưng (7200 so với 1600 và 1000).
- **Bản đồ được chấm là bản đồ sau khi mapper train tới `max_num_iterations` (30000)**: khác Photo-SLAM (dừng ngay sau frame cuối), SEGS-SLAM mặc định train tiếp sau frame cuối, dùng pose keyframe lấy từ quỹ đạo đã lưu. Đây là cấu hình chuẩn của tác giả cho Replica, nên chấm như vậy; `method.json` ghi số vòng lúc frame cuối (`mapper_iterations_at_loop_end`) và tổng, `timing.json` ghi thời gian train thêm (`mapper_after_loop_s`). Khi so với MonoGS: phần train thêm này gần với color refinement của MonoGS (cũng ngoại tuyến, sau tracking), nên đặt SEGS-SLAM cạnh cột **sau refine** của đồ án, và ghi chú.
- **FPS** là vòng tracking (gồm đọc ảnh từ bản chép cục bộ), không gồm phần train thêm. Phải báo kèm thời gian train thêm để không làm SEGS-SLAM trông nhanh hơn thực tế.
- **Render:** nhúng ngoại hình là hàm của pose, nên frame chấm (không phải keyframe) nhận nhúng của chính pose ước lượng của nó; không có chọn nhúng theo ảnh đích như kiểu NeRF-W. Đánh giá riêng của tác giả (`renderAndRecordKeyframe`) nhân ảnh render với mặt nạ pixel khác 0 của ảnh quan sát; `score.py` không làm vậy (mặt nạ chỉ áp lên PSNR, như MonoGS).
- **Keyframe:** như Photo-SLAM, mapper train trên keyframe, frame chấm ảnh là frame chưa dùng để tối ưu.

## Kỹ thuật

- Đã kiểm cục bộ (không build C++, 2026-11-16): settings ORB (giá trị mới, hệ số méo bỏ, 7200 đặc trưng giữ; cảnh không có cấu hình thì báo lỗi), file mapper tồn tại, export đầu ra giả của driver, `.tar` bản đồ lấy đúng lần lưu cuối (có `mlp/`), đóng zip, `score_zip.py` chấm hết (PSNR 40.17 dB với nhiễu σ = 0.01), runner đọc `kaggle.json`.
- API ORB-SLAM3 và mapper mà driver dùng đã đối chiếu với mã SEGS-SLAM: `TrackRGBD` chỉ thêm một tham số tùy chọn; `getAtlas`, `getTracker`, `mlRelativeFramePoses`, `renderFromPose`, `pose_`, `poseSaved` đều public.

  **Build trên Kaggle (2026-11-16), lần 1:** CUDA 12, OpenCV, TorchScatter (cài vào `/usr/local`, CMake tìm thấy), ORB-SLAM3 bản SEGS đều build xong; CMake của SEGS-SLAM hỏng ở `find_package(PCL)` vì VTK cần `MPI::MPI_C` (đã sửa, xem bảng phụ thuộc). CMake còn cảnh báo `libtbb.so.12` trong `/usr/local/lib` (của ảnh Kaggle) có thể che bản hệ thống khi chạy, vì `/usr/local/lib` vào rpath cùng TorchScatter; bước `ldd` cuối setup sẽ cho biết. Từ lần 3, bước dò PCL chạy một chương trình có nạp VTK. Nếu chương trình hỏng với thứ tự thư viện mặc định nhưng chạy được khi đặt thư mục hệ thống lên trước, setup ghi `prefer_system_libs` và thêm thư mục đó vào `runtime_ld_library_path` của driver (sau CUDA 12).

  **Lần 2 (2026-11-16):** `find_package(MPI REQUIRED COMPONENTS C)` hỏng: "MPI component 'C' was requested, but language C is not enabled" (`project(photo-slam LANGUAGES CXX CUDA)`). Lần 1 nhiều khả năng cùng gốc: VTK tự gọi FindMPI, lặng lẽ thất bại vì thiếu C, rồi bước generate báo thiếu `MPI::MPI_C`. Đã sửa: thêm `C` vào `project()` trên bản chép, cộng bước dò PCL. Không có Ubuntu 24.04 cục bộ để chạy thử CMake (Docker không khởi động được, WSL là 22.04 không có PCL), nên chỉ mới kiểm cú pháp script và phép sửa CMakeLists trên bản chép.

  **Lần 3 (2026-11-16):** bước dò PCL đã **cấu hình và biên dịch được** (MPI với C, PCL, VTK, OpenMP, các header PCL của SEGS-SLAM), nên bản sửa MPI đã đúng. Bước dò hỏng lúc liên kết (`libOpenGL.so: undefined reference to '_glapi_tls_Current'`, lỗi ORB-SLAM3 từng gặp ngày 2026-11-12), vì nó thiếu `-rpath-link` vào thư mục hệ thống mà lệnh build SEGS-SLAM vốn có. Đã thêm cùng các cờ liên kết đó vào bước dò. Lỗi nằm ở bước dò, không ở SEGS-SLAM. **Chưa chạy.** Rủi ro còn lại: build TorchScatter 2.1.2 với LibTorch 2.1.2 và CUDA 12; `find_package(PCL)` trên Ubuntu 24.04; các sửa đổi ORB-SLAM3 riêng của SEGS với GCC 13; thời gian train thêm 30000 vòng trên T4 (ước lượng vài chục phút, chưa đo).

## Đã chạy

(chưa có)
