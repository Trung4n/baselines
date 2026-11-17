# DarkGS

- Bài báo: DarkGS: Learning Neural Illumination and 3D Gaussians Relighting for Robotic Exploration in the Dark (Zhang, Huang, Zhi, Johnson-Roberson, IROS 2024, arXiv 2403.10814). Thêm vào `refs.bib` khi dùng.
- Mã nguồn: chép sẵn trong repo, `third_party/darkgs` = **bản fork của người dùng** <https://github.com/Trung4n/darkgs> @ `32945fcd` (2026-09-21), từ bản gốc tyz1030/darkgs @ `279bfa10`, kèm 2 extension CUDA của Inria ở đúng commit bản gốc ghim. Bản fork thêm `--linearize`, `--scaling_factor`, `--light_params`, chặn NaN khi `z = 0` và `sqrt(0)`, cảnh báo pháp tuyến 0 (`.vendor.json`).
- Vai trò: **nhóm C** (`CLAUDE.md`, `PROTOCOL.md` mục "Nhóm C"): 3DGS ngoại tuyến, **cùng bài toán** (đèn gắn camera, robot trong bóng tối), đèn **hiệu chuẩn trước** bằng bia, giống hướng hiệu chuẩn của đồ án (D106). Không phải SLAM: cần pose cho trước, nên chỉ so bản đồ.

## Giao thức (người dùng quyết 2026-11-17)

| | |
|---|---|
| Pose | **Pose tham chiếu** (`traj.txt`). `kaggle.json` khai báo `"reference_poses": true`, runner chép thêm `traj.txt` |
| Ảnh train | Lưới 0: frame 0, 5, …, 1995 (400 ảnh) |
| Đèn | **Đèn hiệu chuẩn của đồ án** (`calib/light_calibrated.json`, D106), **không** dùng tham số Blender |
| Chấm ảnh | Lưới dịch 2, 7, 12, … (không frame nào là ảnh train), render tại pose tham chiếu; không ATE, không FPS |

## File

| File | Việc |
|---|---|
| `adapter.py` | Dựng gói COLMAP, chạy `train.py` của DarkGS **không sửa** trong tiến trình con, nạp `chkpnt<it>.pth` và `shader<it>.pth`, render lưới dịch và 5 frame chung tại pose tham chiếu, đổi về sRGB, đóng zip (`given_poses`, `heldout_offset` trong `method.json`) |
| `colmap_pkg.py` | Gói COLMAP từ `results/` và `traj.txt`, chuyển từ `scripts/darkgs/build_dataset.py` của repo dữ liệu (cùng cách dựng), nội tại và `depth_scale` từ `protocol.calibration()` |
| `fit_light.py` | Chạy **cục bộ**, một lần mỗi cảnh, trong môi trường MonoGS: `light/<scene>_<variant>.pth` (khuôn `model_parameters.pth` của DarkGS) từ đèn hiệu chuẩn |
| `light/office0_lit.pth`, `.json` | Đèn đã fit cho `office0 lit`, kèm sai số |
| `lietorch_shim/lietorch.py` | Thay `lietorch` (SO3 bằng torch), không phải build |
| `test_lietorch_shim.py` | So shim với `scipy.spatial.transform.Rotation` |
| `probe.py` | Bước dò của setup: mã DarkGS qua shim, trên một mặt phẳng tổng hợp, với file đèn đã fit |
| `setup_kaggle.sh`, `kaggle.json` | Cài phụ thuộc, build 2 extension, chạy bước dò; `"reference_poses": true` |

## Khác biệt so với cách chạy gốc của DarkGS

| Đổi | Lý do |
|---|---|
| Pose tham chiếu thay cho COLMAP, nội tại của dataset | Giao thức nhóm C. Pose theo mét nên `--scaling_factor 1.0` (gốc: 0.1, đoán cho pose SfM tùy thang) |
| Đám mây điểm khởi tạo từ **độ sâu cảm biến** của các frame train tại pose tham chiếu (gộp voxel 0.04 m, mỗi 4 pixel, bỏ điểm xa hơn 8 m), kèm pháp tuyến từ độ sâu hướng **ra xa** camera | Thay cho điểm SfM của COLMAP: thiết lập RGB-D của đồ án. Pháp tuyến phải khác 0, vì DarkGS tính `cos = relu(n · (điểm − đèn))`, pháp tuyến 0 cho màu 0 và không có gradient |
| `--linearize` | Ảnh là sRGB 8 bit; DarkGS mô hình hóa cường độ tuyến tính (gốc dùng ảnh RAW). Render được đổi lại sang sRGB (`linear_to_srgb` của bản fork) trước khi chấm, cùng mã hóa với ảnh quan sát |
| Đèn: `light/<scene>_<variant>.pth` do `fit_light.py` fit từ `calib/light_calibrated.json` | Gốc dùng file của bộ hiệu chuẩn `neuralight` cho phần cứng của tác giả. Ở đây: MLP 1D của DarkGS fit theo K·Φ(atan r) của đèn hiệu chuẩn (Φ tính bằng đúng `_linear_in_cos` của mô hình chính), `t_l2c = t_CL`, `so3 = log(R_CL)`, suy giảm 1/d² (γ = 1, τ = 1e-4 m²). Ánh sáng nền bắt đầu ở 0.05, giá trị đoán của mô hình chính (`rgbd_lambert_calib_ambgrid.yaml`); DarkGS tự học tiếp (và học cả `scaling_factor`), như cách gốc |
| `lietorch` thay bằng `lietorch_shim` | Tránh build `lietorch` (CUDA) trên torch mới. DarkGS chỉ dùng `SO3.exp`, `inv`, `act`, `log` và `LieGroupParameter`; xoay của đèn **không** nằm trong bộ tối ưu (`ShadingModel.set_optimizer`), chỉ đặt lại từ số trong lúc warm-up. Khớp scipy tới sai số làm tròn (`test_lietorch_shim.py`) |
| Chỉ chạy biến thể `lit` | Dữ liệu `even` không có đèn gắn camera, nên mô hình đèn của DarkGS không áp dụng; adapter từ chối khi thiếu file đèn của biến thể |

Còn lại là mặc định của DarkGS: 30000 vòng, warm-up tư thế đèn từ vòng 1000 tới 2000, học shader tới vòng 9000, densification của 3DGS, seed 0 (`safe_state`).

**Kiểm chứng `fit_light.py` (office0 lit):** MLP khớp K·Φ với sai số RMS 0.2% và lớn nhất 2.2% của K (ở các khúc gấp của bảng Φ tuyến tính từng đoạn tại rìa nón). Gọi chính `ShadingModel` của DarkGS (qua shim) tại 50000 điểm ngẫu nhiên trong tầm nhìn, so với K·Φ(θ)/d² tính trực tiếp trong hệ camera, có cả độ lệch `t_CL`: lớn nhất 2.2%, phân vị 99 là 1.9%. Toàn bộ sai số đến từ việc fit MLP, nên phép đổi hệ tọa độ đúng.

## Điểm cần nói khi diễn giải

- **Mô hình ánh sáng nền khác đồ án:** DarkGS tính `albedo · (incident + ambient) · cos(n, l)`, tức ánh sáng nền cũng nhân `cos` với hướng **đèn**; dữ liệu (và đồ án) có ánh sáng nền nhân albedo, không phụ thuộc đèn. Ngoài nón, nơi chỉ có ánh sáng nền, DarkGS vì vậy lệch.
- **Không có bóng đổ** trong DarkGS (như đồ án).
- **Được lợi so với đồ án:** pose đúng, train ngoại tuyến 30000 vòng trên mọi ảnh train. Tập frame chấm (lưới dịch) khác của đồ án (lưới 0 trừ keyframe).
- Tâm quang: DarkGS (3DGS) đặt tâm quang ở giữa ảnh, chỉ đọc `fx`, `fy`; dataset có `cx = 299.5`, `cy = 169.5` (giữa ảnh theo quy ước OpenCV), khác nửa pixel so với quy ước tâm pixel của 3DGS.

## Kỹ thuật

- Đã kiểm cục bộ (WSL, GTX 1650; extension build vào thư mục `--target` riêng, không đụng môi trường MonoGS):
  - `test_lietorch_shim.py`: `exp` 8e-16, `log` 5e-10, `act` 3e-7 (float32), ngoài `state_dict`.
  - `probe.py`: qua (tâm nón sáng 9.0, góc ngoài nón 0.026, chỉ ánh sáng nền; gradient tới albedo và vị trí).
  - Adapter trên 120 frame đầu `office0 lit` (24 ảnh train, `--iterations` chỉ dùng cho thử cục bộ): dựng gói, `train.py`, nạp checkpoint, render, đóng zip đều chạy. Với 3000 vòng: 36.8k Gaussian, ánh sáng nền học từ 0.05 lên 0.06, PSNR lưới dịch 21.1. Ảnh frame 100 khớp hướng, vị trí đĩa sáng và độ sáng sRGB (còn nhòe vì ít vòng và ít ảnh). Với 500 vòng PSNR chỉ 16.8, vì trước vòng 1000 đèn còn ở xoay mặc định của lớp `LightBaseLie` (khoảng 20°): đúng cách DarkGS warm-up.
  - `score.py --given_poses`: ATE và FPS ghi `null`, chấm 24 frame lưới dịch; offset khác bị từ chối.
- Chưa chạy trên Kaggle.

## Đã chạy

| Ngày | Biến thể | PSNR / SSIM / LPIPS (lưới dịch, pose tham chiếu) | Thời gian train | Gaussian | Học được | Thư mục |
|---|---|---|---|---|---|---|
| 2026-11-17 | office0 / lit | 32.41 / 0.942 / 0.091 | 596 s (30000 vòng) | 234k | ánh sáng nền 0.059, `scaling_factor` 1.61 | `results/darkgs__office0_lit/2026-11-17-12-15-07` |

Chạy hết trên Kaggle ngay lần đầu (runner chép cả `traj.txt`, bước dò, train, render, đóng zip). PSNR train (tuyến tính, DarkGS tự báo) 33.2. Phân tích ở `RESULTS.md`.
