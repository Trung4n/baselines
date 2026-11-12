# ORB-SLAM3 RGB-D

- Bài báo: ORB-SLAM3 (Campos et al., T-RO 2021). Trong `refs.bib`: `campos2021orbslam3`.
- Mã nguồn: chép sẵn trong repo, `third_party/ORB_SLAM3` (UZ-SLAMLab/ORB_SLAM3 @ `4452a3c4ab75b1cde34e5505a36ec3f9edcdc4c4`, v1.0, 2022-02-10, kèm vocabulary ORB) và `third_party/Pangolin` (v0.8 @ `aff6883c`). ORB-SLAM3 bắt buộc link Pangolin dù không mở viewer, và Ubuntu 24.04 không có gói Pangolin. Những gì khác so với commit nằm trong `.vendor.json` của từng thư mục.
- Vai trò trong so sánh: tracking dựa trên đặc trưng ORB, không so cường độ pixel. Theo SpotlessGS, đèn đồng hành vẫn có thể gây hại cho đặc trưng: vùng tối không có đặc trưng, vùng bão hòa mất kết cấu, vệt sáng di động tạo đặc trưng giả. Thí nghiệm này kiểm điều đó trên dữ liệu của đồ án.

## File

| File | Việc |
|---|---|
| `rgbd_monogs.cc` | Driver C++: vòng lặp của `Examples/RGB-D/rgbd_tum.cc`, chỉ đổi phần vào/ra: đọc thẳng `results/frame%06d.jpg` và `depth%06d.png`, frame i có timestamp i / 30; tắt viewer; ghi trạng thái tracking và thời gian từng frame; lưu quỹ đạo của mọi bản đồ và của bản đồ lớn nhất; kết thúc bằng `std::_Exit(0)` sau khi ghi xong |
| `adapter.py` | Ghi file settings từ `TUM1.yaml`, chạy driver, xuất `<work>/export/`, đóng zip (`common/runzip.py`) |
| `setup_kaggle.sh` | `apt` cài OpenCV, Eigen, Boost, OpenSSL, GLEW; build Pangolin, DBoW2, g2o, `libORB_SLAM3` và driver trên bản chép |
| `kaggle.json` | `"checkpoints": false` |

## Cách chạy

1. **Kaggle** (chạy trên CPU; GPU không cần nhưng cũng không hại): notebook chung `run_baseline.ipynb`, đặt `METHOD = "orbslam3"`. Runner tự chạy với `--checkpoint_every 0`. Cài đặt lần đầu mất khoảng 15–25 phút (apt và build C++); một run mất vài phút.
2. Mỗi run cho `/kaggle/working/orbslam3__<scene>_<variant>__<thời điểm bắt đầu>.zip`. Zip không có render hay bản đồ; có hai quỹ đạo thô, trạng thái tracking từng frame và file settings.
3. Tải zip về, chấm cục bộ. `score_zip.py` tự dùng `--no_render`:

```
python baselines/common/score_zip.py --zip orbslam3__office0_lit__<...>.zip \
    --dataset /mnt/d/CodeForYou/Replica/04_datasets/monogs/office0/lit --no_wandb
```

## Khác biệt so với mặc định của ORB-SLAM3 (ví dụ RGB-D TUM)

Gốc: `Examples/RGB-D/rgbd_tum.cc` với `Examples/RGB-D/TUM1.yaml`. Mọi thay đổi ghi trong `overrides.json` và `method.json` (`settings_changes`).

| Đổi | Lý do |
|---|---|
| `Camera1.fx/fy/cx/cy`, `Camera.width/height` từ `protocol.calibration()`; bỏ `Camera1.k1..k3, p1, p2` | Bắt buộc: camera của dataset, không méo |
| `RGBD.DepthMapFactor` 5000 → 6553.5 | Bắt buộc: thang độ sâu PNG của dataset |
| `Camera.RGB` 1 → 0 | Ảnh đọc bằng `cv::imread` là BGR. TUM1.yaml ghi 1 dù `rgbd_tum` cũng đọc BGR; giá trị này chỉ ảnh hưởng bước chuyển sang ảnh xám |
| Giữ nguyên | 1000 đặc trưng ORB, 8 tầng, hệ số 1.2, ngưỡng FAST 20/7, `Camera.fps` 30, `Stereo.ThDepth` 40, `Stereo.b` 0.07732 m (ngưỡng gần/xa 3.1 m, tính bằng mét nên không phụ thuộc tiêu cự) |
| Timestamp frame i = i / 30, chờ tới timestamp kế tiếp như `rgbd_tum` | Dataset không có timestamp. ORB-SLAM3 dùng timestamp cho logic mất tracking (ví dụ thời gian "vừa mất"), nên nhịp 30 Hz giữ hành vi như dữ liệu TUM |
| Viewer tắt | Kaggle không có màn hình |
| Đọc ảnh thẳng theo chỉ số thay cho file association | Định dạng dataset |
| Build: C++17, `-include cstdint`, chính sách CMake ≥ 3.5; chỉ build thư viện và driver | Chỉ sửa lỗi build với Pangolin 0.8, GCC 13, CMake mới; không đổi thuật toán. Làm trên bản chép trong `/tmp` |
| `LoopClosing.cc`: `mnFullBAIdx++` → `mnFullBAIdx = true` (3 chỗ) | `mnFullBAIdx` khai báo là `bool`; C++17 cấm `++` trên `bool`. Với `bool`, `++` luôn gán `true`, nên hành vi giữ nguyên. Gặp ở lần build đầu trên Kaggle (2026-11-12) |

Loop closure và atlas nhiều bản đồ bật theo mặc định. MonoGS không có loop closure; báo cáo phải nêu điều này.

## Điểm cần nói khi diễn giải

- **Quỹ đạo được chấm là bản đồ lớn nhất** (`SaveTrajectoryEuRoC`). Khi mất tracking, ORB-SLAM3 mở bản đồ mới với hệ tọa độ riêng; frame thuộc bản đồ khác tính là thiếu (`frames_missing` trong `metrics.json`). ATE chỉ tính trên các frame có pose, nên **phải báo ATE cùng số frame thiếu**: ATE thấp trên một phần nhỏ của chuỗi không có nghĩa là chạy tốt. Quỹ đạo của mọi bản đồ (`SaveTrajectoryTUM`, cách `rgbd_tum` lưu) nằm ở `run/orbslam3_out/CameraTrajectory.txt`.
- `method.json` ghi số bản đồ trong atlas và số frame theo từng trạng thái tracking (OK, RECENTLY_LOST, LOST...).
- **FPS** là FPS của vòng lặp, gồm đọc ảnh, tracking và thời gian chờ tới timestamp kế tiếp. Vì vậy FPS bị chặn ở 30. Không tính thời gian khởi động (nạp vocabulary). `timing.json` ghi thêm thời gian tracking trung bình và trung vị mỗi frame. Chạy hoàn toàn trên CPU.
- Không có bản đồ kết xuất được: không có PSNR, SSIM, LPIPS hay ảnh 5 frame chung.
- Depth Blender không có nhiễu (`CLAUDE.md` mục 3).

## Kỹ thuật

- Đã kiểm cục bộ (WSL, môi trường MonoGS, không build C++):
  - file settings (giá trị mới, khóa méo bị bỏ, các khóa ORB giữ nguyên);
  - đọc quỹ đạo với timestamp tính bằng giây và bằng nano giây;
  - export → `runzip.package` → `score_zip.py` (`--no_render`) trên đầu ra giả có hai bản đồ và 50 frame mất tracking: 550 frame thiếu được đếm đúng, ATE đúng như tính tay.

  **Chưa build và chưa chạy ORB-SLAM3 thật**: phần C++ (Pangolin, ORB-SLAM3, driver) chỉ kiểm được trên Kaggle. Lỗi build dễ gặp nhất: `find_package(OpenCV 4.4)` với OpenCV 4.6 của Ubuntu, và lỗi biên dịch C++17/GCC 13 chưa lường trước.

## Đã chạy

(chưa có)
