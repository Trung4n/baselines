# SpotlessGS

- Bài báo: SpotlessGS: Relightable 3D Gaussian Splatting under Dynamic Illumination for Robotic Perception (arXiv 2608.14713). Thêm vào `refs.bib` khi dùng.
- Mã nguồn: repo chính chủ <https://github.com/Liianne/SpotlessGS> @ `db380dcf`, chép phần Python vào `third_party/SpotlessGS` (`.vendor.json`). Hai extension CUDA của Inria giống hệt bản trong `third_party/darkgs/submodules` (đã so), nên setup build bản đó.
- Vai trò: **nhóm C** (`PROTOCOL.md` mục "Nhóm C"): 3DGS ngoại tuyến, cùng bài toán (đèn gắn camera), xây trên DarkGS nhưng **không cần hiệu chuẩn**: vị trí, hướng, độ suy giảm, MLP phát sáng của đèn, một BRDF dạng MLP và ánh sáng nền dạng SH đều học cùng bản đồ. Đặt cạnh DarkGS (đèn hiệu chuẩn) như chế độ tự hiệu chuẩn của đồ án đặt cạnh mô hình chính.

## File

| File | Việc |
|---|---|
| `adapter.py` | Chép `third_party/SpotlessGS` ra thư mục làm việc và vá `N_TEST`; dựng gói COLMAP (`methods/darkgs/colmap_pkg.py`, cùng gói với DarkGS); chạy `train.py` của SpotlessGS trong tiến trình con, lọc output; nạp `chkpnt<it>.pth`, `shader<it>.pth`; render lưới dịch và 5 frame chung tại pose tham chiếu; đóng zip |
| `lietorch_shim/lietorch.py` | Thay `lietorch`: `LieGroupParameter` là `nn.Parameter` giữ góc quay trục (học được), dùng toán SO3 của `methods/darkgs/lietorch_shim` |
| `test_lietorch_shim.py` | Gradient qua `act`, Adam khớp được một phép quay, `state_dict` khứ hồi |
| `probe.py` | Bước dò của setup: mã SpotlessGS trên một mặt phẳng tổng hợp; gradient tới Gaussian và tới đèn (vị trí, hướng, MLP), một bước tối ưu shader |
| `setup_kaggle.sh`, `kaggle.json` | Cài phụ thuộc, build 2 extension, chạy bước dò; `"reference_poses": true` |

## Giữ như bản công bố

- **Không hiệu chuẩn:** đèn bắt đầu từ mặc định của SpotlessGS (`LightBaseLie`: `t = (0.145, 0.03, 0.08)` m của phần cứng tác giả, xoay 0, MLP khởi tạo ngẫu nhiên) và được học; `train.py` chỉ đọc `./model_parameters.pth` để lấy kích thước `so3`. Đây là điểm chính của phương pháp, nên không đưa đèn hiệu chuẩn của đồ án vào (khác DarkGS).
- **Ảnh sRGB như đầu vào**, không tuyến tính hóa (SpotlessGS không có bước này); render cũng ở mã hóa đó.
- `SCALE_FACTOR = 1.0` trong `train.py` (pose theo mét), 30000 vòng, loss L1 + SSIM, mặc định 3DGS còn lại.

## Khác biệt (ghi trong `overrides.json`)

| Đổi | Lý do |
|---|---|
| Pose tham chiếu, 400 ảnh lưới 0, đám mây điểm từ độ sâu cảm biến kèm pháp tuyến (cùng gói với DarkGS; SpotlessGS đọc `points3D_with_normals.ply`, adapter ghi file này) | Giao thức nhóm C. Pháp tuyến hướng ra xa camera như DarkGS: shader của SpotlessGS dùng cùng `light_dir` (từ đèn tới điểm) |
| `scene/dataset_readers.py`: `N_TEST = 29` → 0, vá trên **bản chép** trong thư mục làm việc | Bộ đọc COLMAP của SpotlessGS **luôn** giữ lại 29 ảnh cuối để test, bất kể `--eval`: ở đây là các frame 1855–1995 của tập train. Giao thức train trên cả lưới 0 và chấm trên lưới dịch |
| `lietorch` thay bằng `lietorch_shim` (SO3 học được) | Tránh build `lietorch`. SpotlessGS đưa xoay đèn vào bộ tối ưu, nên shim là `nn.Parameter`; Adam bước trên góc quay trục thay vì trên nhóm (khớp bậc một quanh 0). Trên thực tế xoay **không đổi**: `train.py` gọi `set_r_vec` sau khi dựng bộ tối ưu, nên bộ tối ưu giữ tham số cũ (đặc điểm của mã gốc, giống hệt với `lietorch`) |
| Lọc output của `train.py` (không đổi phép tính) | `ShadingModel.forward` in cả tensor pháp tuyến ở mỗi lần render (3 lần mỗi vòng) và `train.py` in tư thế đèn mỗi vòng: hàng trăm nghìn dòng, làm ngập notebook. Adapter bỏ các bản in tensor, giữ một cặp `r is`/`t is` mỗi 1000, giữ mọi dòng khác; số dòng đã bỏ in ở cuối |

## Điểm cần nói khi diễn giải

- Mỗi vòng, SpotlessGS còn dựng và render một shader "relit" không đi vào loss (mã gốc): tốn thời gian train, không ảnh hưởng kết quả.
- Đèn học được không bị ràng buộc về dạng vật lý: thử cục bộ 3000 vòng, `gamma` (số mũ suy giảm) học về 0.21 thay vì 1 (1/d²), vị trí đèn khác vị trí thật vài cm. MLP BRDF và ánh sáng nền SH đủ linh hoạt để bù.
- Như DarkGS: pose đúng, train ngoại tuyến, tập frame chấm khác tập của đồ án.

## Kỹ thuật

- Đã kiểm cục bộ (WSL, GTX 1650; extension là bản đã build cho DarkGS; `tensorboard` cài vào thư mục tạm riêng):
  - `test_lietorch_shim.py`: qua (gradient, Adam khớp một phép quay, `state_dict` khứ hồi).
  - `probe.py`: qua.
  - Adapter trên 120 frame đầu `office0 lit` (24 ảnh train, `--iterations 3000` chỉ cho thử cục bộ): dựng gói, train (311 s), render, đóng zip, chấm `--given_poses`: PSNR lưới dịch 26.6 (DarkGS cùng điều kiện 21.1). Log 1076 dòng sau lọc.
- Chưa chạy trên Kaggle.

## Đã chạy

(chưa có)
