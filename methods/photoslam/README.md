# Photo-SLAM (RGB-D)

- Bài báo: Photo-SLAM: Real-time Simultaneous Localization and Photorealistic Mapping (Huang et al., CVPR 2024). Thêm vào `refs.bib` khi dùng.
- Mã nguồn: chép sẵn trong repo, `third_party/Photo-SLAM` (HuajianUP/Photo-SLAM @ `f8bfb2f0809c003ccc3fd577dc43c576fcafa4ac`, 2024-06-12), kèm bản ORB-SLAM3 đã sửa của nó và vocabulary. OpenCV có CUDA và LibTorch **không** chép vào repo; `setup_kaggle.sh` tải chúng từ nguồn chính thức, phiên bản ghim trong script.
- Vai trò trong so sánh: tracking bằng đặc trưng (ORB-SLAM3) cộng bản đồ Gaussian học song song. Phần tracking gần như ORB-SLAM3 đã đo; phần mới là PSNR của một bản đồ Gaussian dựng trên tracking đặc trưng. Đây là phương pháp GS-SLAM thời gian thực được trích dẫn nhiều, hội đồng có thể hỏi tới.

## File

| File | Việc |
|---|---|
| `photoslam_monogs.cpp` | Driver C++: vòng lặp của `examples/replica_rgbd.cpp` (đọc ảnh, chuyển RGB, `TrackRGBD` với timestamp = chỉ số frame, luồng `GaussianMapper::run`, `Shutdown`, chờ mapper tự train nốt và lưu). Không viewer. Sau đó: pose mọi frame của bản đồ lớn nhất **trong hệ world của bản đồ đó** (hệ chứa Gaussian), danh sách keyframe, và render bản đồ cuối tại các pose đó cho các frame giao thức chấm (`renderFromPose`, độ phân giải gốc), ghi `.npy` |
| `adapter.py` | Ghi settings ORB-SLAM3 (camera của dataset), chép settings Gaussian mapper của Photo-SLAM, chạy driver, xuất `<work>/export/` (render chuyển sang float16), đóng zip |
| `setup_kaggle.sh` | `apt`; dò phiên bản CUDA; tải LibTorch; build OpenCV 4.10 + contrib có CUDA (hoặc giải nén bản build đã lưu); build DBoW2, g2o, ORB-SLAM3, Photo-SLAM và driver |
| `kaggle.json` | `"checkpoints": false`, `"setup_timeout_min": 180` (runner tự dùng giá trị này nếu notebook đặt nhỏ hơn) |

## Cách chạy

1. **Kaggle**, GPU T4, bật Internet: notebook chung `run_baseline.ipynb`, đặt `METHOD = "photoslam"`. Runner tự chạy với `--checkpoint_every 0` và nới thời gian cài đặt lên 180 phút.
2. **Lần cài đặt đầu** build OpenCV có CUDA từ mã nguồn, khoảng một giờ. Bản build được nén vào `/kaggle/working/photoslam_cache/opencv-4.10.0-cuda<ver>-sm<arch>.tar.gz`. **Tạo một Kaggle dataset từ file đó và gắn làm input** cho các phiên sau: setup tự tìm nó trong `/kaggle/input` và giải nén thay vì build lại. Tên file ghi phiên bản CUDA và GPU; nếu Kaggle đổi CUDA thì build lại.
3. Mỗi run cho `/kaggle/working/photoslam__<scene>_<variant>__<thời điểm>.zip`, có render, ảnh 5 frame chung và bản đồ cuối (`run/point_cloud.ply`; `NO_MAP = True` bỏ nó ra).
4. Tải zip về, chấm cục bộ:

```
python baselines/common/score_zip.py --zip photoslam__office0_lit__<...>.zip \
    --dataset /mnt/d/CodeForYou/Replica/04_datasets/monogs/office0/lit --no_wandb
```

## Phụ thuộc và phiên bản

| Thứ | Phiên bản | Lý do |
|---|---|---|
| OpenCV + opencv_contrib | 4.10.0, có CUDA, chỉ các module `core, imgproc, imgcodecs, highgui, features2d, calib3d, flann, cudev, cudaarithm, cudafilters, cudaimgproc, cudawarping, cudastereo`, chỉ cho compute capability của GPU đang có | Photo-SLAM gọi `cv::cuda` trên đường chạy RGB-D. Tác giả thử 4.7/4.8; 4.10 là bản đầu có sửa lỗi cho CUDA 12.4+ và GCC 13 |
| LibTorch (C++, cxx11 ABI) | CUDA 11.x → 2.0.1+cu118 (cấu hình của tác giả); CUDA 12.x → 2.1.2+cu121 | Photo-SLAM chỉ hỗ trợ tới 2.1.2 |
| Bộ CUDA dùng để build | nvcc của hệ thống nếu là 11.x hoặc 12.x. Nếu là 13 trở lên (Kaggle 2026-11: nvcc 13.0), cài thêm CUDA 12 qua apt từ repo của NVIDIA (ưu tiên 12.6, rồi 12.5, 12.8) vào `/usr/local/cuda-12.x` và build OpenCV, Photo-SLAM bằng bộ đó; CUDA 13 của hệ thống giữ nguyên | LibTorch ≤ 2.1.2 không có bản cho CUDA 13. Driver của Kaggle chạy được chương trình CUDA 12 |
| `LD_LIBRARY_PATH` lúc chạy | `<build>/runtime_ld_library_path` do setup ghi: `lib64` của bộ CUDA đã build, cộng symlink tới riêng `libcuda.so*` của driver NVIDIA nếu bộ đệm của loader không có | Không dùng `LD_LIBRARY_PATH` của notebook: thư viện GL của NVIDIA trong đó làm hỏng GL hệ thống (`methods/orbslam3/README.md`), và `cuda/lib64` của nó là CUDA 13 |
| Trình biên dịch chủ cho CUDA | g++-12 nếu nvcc 12.0–12.3, g++-11 nếu CUDA 11.x; ngược lại GCC hệ thống | nvcc cũ không nhận GCC 13 |

## Khác biệt so với mặc định của Photo-SLAM cho Replica

| Đổi | Lý do |
|---|---|
| `cfg/ORB_SLAM3/RGB-D/Replica/<scene>.yaml`: `Camera1.fx/fy/cx/cy`, `Camera.width/height` từ `protocol.calibration()` (dataset là Replica nửa độ phân giải, 600×340) | Bắt buộc |
| Bỏ `Camera1.k1, k2, k3, p1, p2` | File Replica của Photo-SLAM có hệ số méo nhỏ khác 0, trong khi ảnh Replica (và dataset của đồ án) không méo |
| Giữ nguyên | Mọi khóa ORB khác (1600 đặc trưng, `Camera.RGB 1` vì ví dụ chuyển ảnh sang RGB, `Stereo.b`, `ThDepth`, `DepthMapFactor 6553.5` vốn đã đúng) và toàn bộ `cfg/gaussian_mapper/RGB-D/Replica/<scene>.yaml` (SH bậc 3, 30100 vòng, ...) |
| Timestamp = chỉ số frame, không giữ nhịp | Giữ như `replica_rgbd.cpp`. Với ORB-SLAM3 khác ví dụ TUM (30 Hz), vì đây là lựa chọn của chính Photo-SLAM cho Replica |
| Viewer tắt | Kaggle không có màn hình |
| Build: `-include cstdint`, chính sách CMake ≥ 3.5, `-rpath-link`, `LD_LIBRARY_PATH` riêng lúc chạy, CUDA 12 cài bên cạnh khi hệ thống là CUDA 13, trình biên dịch chủ CUDA cũ hơn khi cần | Chỉ sửa lỗi build với bộ công cụ mới của Kaggle (xem `methods/orbslam3/README.md`); làm trên bản chép |

## Điểm cần nói khi diễn giải

- **Tracking là ORB-SLAM3**, có loop closure; MonoGS không có. ATE nên gần với ORB-SLAM3 đã đo (`lit` 3.55 cm); khác biệt đến từ cấu hình Replica của Photo-SLAM (1600 so với 1000 đặc trưng, timestamp) và từ tính ngẫu nhiên.
- **Quỹ đạo được chấm là bản đồ lớn nhất**, giống ORB-SLAM3: frame thuộc bản đồ khác tính là thiếu. Phải báo ATE kèm `frames_missing`.
- **Render của frame không có pose**: driver ghi chúng vào `renders_without_pose.txt` và `method.json`. `score.py` khi đó dừng vì thiếu ảnh. Cách xử lý (bỏ frame, hay tính là ảnh đen) chưa có trong `PROTOCOL.md`; quyết khi gặp.
- **FPS** là vòng tracking, gồm đọc ảnh (từ bản chép cục bộ, `PROTOCOL.md`), không gồm phần mapper tự train thêm sau frame cuối (ghi riêng `mapper_after_loop_s`). Bản đồ chấm là bản đồ **sau** phần train thêm đó, đúng như Photo-SLAM tự đánh giá.
- **Keyframe:** mapper chỉ train trên keyframe của ORB-SLAM3, nên các frame chấm ảnh là frame chưa dùng để tối ưu (giống MonoGS và GS-ICP, khác SplaTAM).

## Kỹ thuật

- Đã kiểm cục bộ (không build C++):
  - file settings (giá trị mới, khóa méo bị bỏ, khóa ORB giữ nguyên; cảnh không có cấu hình thì báo lỗi);
  - export đầu ra giả: render float32 → float16, một frame không có pose được ghi nhận và bỏ khỏi ảnh hình, bản đồ `.ply` đóng vào zip, số Gaussian đọc từ header `.ply`;
  - `score_zip.py` chạy được, và dừng đúng khi thiếu ảnh render của một frame;
  - runner dùng `setup_timeout_min` của `kaggle.json`.

  **Chưa build xong trên Kaggle.** Lần setup đầu (2026-11-13) dừng vì Kaggle có nvcc 13.0 (GCC 13.3, T4 sm 7.5); setup giờ cài CUDA 12 bên cạnh. Lần thứ hai cài được CUDA 12 nhưng build OpenCV hỏng: cờ `-include cstdint` (dành cho ORB-SLAM3) bị FindCUDA của OpenCV chuyển sang nvcc, nvcc chèn nó hai lần (lỗi "redefinition"); giờ OpenCV build không có cờ này. Rủi ro còn lại: build OpenCV 4.10 với CUDA 12.x, LibTorch 2.1.2 (cu121) với nvcc 12.x mới hơn, và CMake của LibTorch tìm đủ thư viện trong bộ CUDA 12 vừa cài.

## Đã chạy

(chưa có)
