# Giao thức chấm điểm cho các phương pháp so sánh

Mọi phương pháp, kể cả MonoGS của đồ án, được chấm bằng **cùng một đoạn code**: `baselines/common/protocol.py` gọi lại đúng các hàm mà `utils/eval_utils.py` của MonoGS dùng. Không phương pháp nào được chấm bằng script đánh giá riêng của nó. Số tự báo của phương pháp chỉ ghi để đối chiếu, không đưa vào bảng.

## Dữ liệu vào

- Chỉ dùng đầu vào của một cảm biến thật, như ràng buộc ở `CLAUDE.md` mục 3 của repo: ảnh màu `results/frame*.jpg` (ảnh `rgb_lit` hoặc `rgb_even` của biến thể) và ảnh độ sâu `results/depth*.png` của thư mục dataset MonoGS. Không dùng EXR, pass albedo, pháp tuyến, hay quỹ đạo tham chiếu. `traj.txt` chỉ dùng để chấm.
- Nội tại camera và thang độ sâu lấy từ `configs/rgbd/replica/base_config.yaml`, khối `Dataset.Calibration`: kích thước 600×340, `fx`, `fy`, `cx`, `cy`, `depth_scale`. Đọc từ file đó, không chép số vào code của adapter.
- Đủ 2000 frame, không bỏ frame, cùng thứ tự.

## ATE

- Pose ước lượng ghi ra file TUM: `idx tx ty tz qx qy qz qw`, trong đó **cột thời gian là chỉ số frame** (0, 1, 2, ...), và pose là camera-to-world.
- Pose tham chiếu: các ma trận 4×4 trong `traj.txt` của dataset (camera-to-world). Đây cũng chính là ma trận evo nhận trong MonoGS.
- Gọi `utils.eval_utils.evaluate_evo`: căn chỉnh SE(3) bằng Umeyama cho RGB-D (Sim(3) chỉ cho camera đơn), lấy RMSE của phần tịnh tiến, đơn vị mét.
- Báo **hai số**:
  - `ate_all`: trên mọi frame có pose. Đây là số chính để so giữa các phương pháp, vì mỗi hệ chọn keyframe khác nhau.
  - `ate_kf`: chỉ trên keyframe của phương pháp, tức đúng cách MonoGS báo. Không có khái niệm keyframe thì để trống.
- Frame mất tracking (không có pose) được đếm vào `frames_missing` và báo kèm. ATE tính trên các frame còn lại; không được nội suy.
- MonoGS của đồ án cũng được tính `ate_all` bằng `common/from_monogs_record.py`, từ `record/frames_final.csv`: pose cuối, keyframe đã qua BA, các frame khác lấy pose lúc tracking. Kiểm chứng ngày 2026-11-12: `ate_kf` tính lại khớp số MonoGS log tới 7 chữ số.

## PSNR, SSIM, LPIPS

- Frame được chấm: chỉ số 0, 5, 10, ... nhỏ hơn N − 1, bỏ keyframe của phương pháp, đúng như `eval_rendering`. Với 2000 frame và khoảng 250 keyframe, còn khoảng 350–360 frame.
- Ảnh kết xuất: render bản đồ **cuối** của phương pháp tại **pose ước lượng** của frame đó, ở độ phân giải dataset, giá trị trong [0, 1]. Lưu `<idx:06d>.npy` dạng float16 hoặc float32, kích thước (H, W, 3) hoặc (3, H, W). PNG 8 bit vẫn được nhận, nhưng sẽ bị đánh dấu `quantized_8bit` trong kết quả.
- Ảnh quan sát được nạp như `utils/dataset.py`: đọc JPEG, chia cho 255.
- PSNR tính bằng `gaussian_splatting.utils.image_utils.psnr` trên các pixel có ảnh quan sát > 0, xét từng kênh. SSIM tính bằng `gaussian_splatting.utils.loss_utils.ssim` trên toàn ảnh. LPIPS dùng torchmetrics với mạng AlexNet, `normalize=True`. Ảnh kết xuất được kẹp về [0, 1] trước khi tính.
- MonoGS báo PSNR trước và sau color refinement (26000 vòng ngoại tuyến). Phương pháp khác chỉ báo bản đồ cuối, không thêm bước tinh chỉnh nào. Trong bảng so sánh, cột chính của MonoGS là **trước refine**; số sau refine ghi riêng.
- "Keyframe" là danh sách keyframe của chính phương pháp. Phương pháp tối ưu bản đồ trên **mọi** frame (SplaTAM, `map_every` 1) vẫn chấm cùng tập frame. Khi đó các frame được chấm đã từng là đích tối ưu, chỉ số ảnh thiên về phía có lợi, và bảng phải ghi chú điều này.
- Phương pháp không có bản đồ kết xuất được (như ORB-SLAM3) dùng `--no_render`, và để trống ba cột này.

## Ảnh so sánh trực quan

Mọi phương pháp render đúng **5 frame chung** `protocol.FIGURE_FRAMES` = **100, 500, 825, 1125, 1750**, dù frame đó có là keyframe của nó hay không.
- Rải đều trên 2000 frame; frame 100 lấy lúc bản đồ còn non.
- 825, 1125, 1750 trùng các frame đã dùng làm hình ánh sáng nền ở D127.
- Chỉ dùng làm hình, không tính vào chỉ số.

Adapter render tập `protocol.required_renders(N, keyframes)`, gồm các frame chấm chỉ số và 5 frame này. `score.py` ghi `figures/<idx:06d>_render.png` (8 bit, ảnh kết xuất đã kẹp về [0, 1]) và `figures/<idx:06d>_obs.jpg` (ảnh quan sát). Với MonoGS của đồ án, `from_monogs_record.py --figures --dataset <...>` render bản đồ trong record tại pose cuối, qua đúng mô hình đèn của run (`--map final` là trước refine, `final_refined` là sau refine).

Ghép các phương pháp thành bảng ảnh (hàng là frame, cột đầu là ảnh quan sát):

```
python baselines/common/compare_figures.py --out <png> --column "Đồ án=<dir>/figures" --column "SplaTAM=<dir>/figures" ...
```

## FPS

- FPS = số frame / thời gian thực của vòng SLAM (tracking và mapping) trên toàn chuỗi. Giống `Total FPS` của MonoGS: tính từ lúc bắt đầu tới lúc frame cuối được xử lý, **không tính** phần đánh giá cuối và refine.
- **Thời gian đọc ảnh đầu vào được tính** (quyết định 2026-11-13). MonoGS đọc từng frame bên trong khoảng được đo: `dataset[idx]`, và `FramePrefetcher` đọc trước một frame trên luồng phụ, từ `start.record()` tới `end.record()` trong `slam.py`. Một phương pháp đọc hết ảnh vào bộ nhớ **trước** vòng lặp (GS-ICP) vì vậy phải cộng thời gian đó vào. Để minh bạch, `timing.json` ghi thêm thời gian vòng lặp riêng (`loop_s`, `fps_loop_only`), và bảng kết quả nêu cả hai số. Thời gian khởi tạo một lần (dựng môi trường, nạp model LPIPS) không tính.
- Ghi vào `timing.json`:
  - `frames`, `seconds`;
  - `gpu`: tên GPU, có dùng một hay hai GPU;
  - `what`: đo từ đâu tới đâu.
- Chạy trên cùng loại GPU với MonoGS (Kaggle T4). MonoGS dùng hai GPU (mapping trên GPU 1). Nếu phương pháp khác chỉ chạy một GPU thì ghi rõ.

## Gói kết quả của một lần chạy

Phương pháp khác chạy trên Kaggle (không có repo MonoGS) và mỗi lần chạy cho đúng **một zip** `<method>__<scene>_<variant>__<thời điểm bắt đầu>.zip`. Trong zip có một thư mục cùng tên, chứa:

| File | Nội dung | Bắt buộc |
|---|---|---|
| `traj_est.tum` | pose camera-to-world, cột thời gian là chỉ số frame | có |
| `keyframes.txt` | keyframe của phương pháp, mỗi dòng một chỉ số | nếu có keyframe |
| `renders/<idx:06d>.npy` | render float tại pose ước lượng, mọi frame của `protocol.required_renders` | nếu có bản đồ kết xuất |
| `figures/<idx>_render.png`, `_obs.jpg` | 5 frame chung, để xem nhanh trước khi chấm | nếu có bản đồ kết xuất |
| `timing.json` | `frames`, `seconds`, `gpu`, `what` (mục FPS) | có |
| `calibration.json` | nội tại và `depth_scale` đã dùng, kiểm với `base_config.yaml` khi chấm | có |
| `method.json` | `method`, `scene`, `variant`, `started_at`, `commit`, `frames`, `notes`, phiên bản, config đã đổi | có |
| `notes.txt` | khác biệt so với mặc định, đi vào `metrics.json` | có |
| `summary.json` | số frame, keyframe, FPS, danh sách file | có |
| `run/` | config đầy đủ, log, thời gian từng phiên, bản đồ cuối | có (bản đồ tùy chọn) |

Chấm cục bộ trong repo và môi trường MonoGS:

```
python baselines/common/score_zip.py --zip <zip> --dataset <thư mục dataset cục bộ> --no_wandb
```

Script giải nén vào `baselines/runs/` (git bỏ qua), rồi kiểm nội tại, cảnh, biến thể và số frame. Sau đó nó chạy `score.py` vào `baselines/results/<method>__<scene>_<variant>/<started_at>/` và chép bản ghi của run (trừ render và bản đồ) vào `run/`, kèm `source.json` (tên, cỡ, sha256 của zip). Thư mục kết quả đã có thì không bao giờ ghi đè.

## Ghi lại cho mỗi lần chạy

`score.py` ghi các file sau vào `baselines/results/<method>__<scene>_<variant>/<thời gian>/`:

| File | Nội dung |
|---|---|
| `plot/stats_final.json`, `plot/evo_2dplot_final.png` | ATE trên keyframe, cùng tên file với MonoGS |
| `plot/stats_final_all.json`, `plot/evo_2dplot_final_all.png` | ATE trên mọi frame |
| `psnr/final/final_result.json` | `mean_psnr`, `mean_ssim`, `mean_lpips`, cùng khóa với MonoGS |
| `psnr/final/per_frame.csv` | các chỉ số theo từng frame |
| `traj_est.tum`, `keyframes.txt` | đầu vào đã dùng để chấm |
| `figures/` | ảnh kết xuất và ảnh quan sát của 5 frame chung |
| `metrics.json` | toàn bộ chỉ số, kèm commit của phương pháp và ghi chú |

Mặc định không đẩy W&B (`--no_wandb`, `CLAUDE.md`). Nếu dùng `--wandb`, kết quả được đẩy lên project `MonoGS`: tên run `<method>__<scene>_<variant>_<thời gian>`, tag `baseline`, kèm bảng `Metrics` cùng cột với MonoGS và thêm cột `ATE all frames`.
