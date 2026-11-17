# NFL-BA

- Bài báo: NFL-BA: Near-Field Light Bundle Adjustment for SLAM in Dynamic Lighting (Dunn Beltran, Rho, Niethammer, Sengupta, NeurIPS 2025, arXiv 2412.13176). Thêm vào `refs.bib` khi dùng.
- Mã nguồn: <https://github.com/asdunnbe/NFL-BA> @ `a01ae39d1aa914817d77248544259102b758d2b2` (2026-11-10). **Không chép vào repo:** repo NFL-BA không có giấy phép, còn repo `baselines` công khai. `setup_kaggle.sh` tải đúng commit này bằng git về `$TMPDIR/nflba_src` lúc setup (chỉ file Python, khoảng 0.3 MB) và từ chối commit khác; `adapter.py` kiểm lại commit trước khi chạy. Hai extension CUDA là của MonoGS, repo đã có sẵn đúng commit MonoGS ghim (`third_party/LoopSplat/_deps`: `simple-knn` giống hệt Inria `44f76429`, `diff-gaussian-rasterization-w-pose` `43e21bff`, đã so 2026-11-17).
- Vai trò trong so sánh: **công trình gần đồ án nhất**, cùng bài toán (đèn gắn cùng camera, đổ bóng phụ thuộc pose) và cùng nền MonoGS. NFL-BA thêm mô hình đèn vật lý vào renderer và loss của MonoGS; đồ án cũng vậy, nhưng mô hình khác. Đặt cạnh MonoGS gốc và mô hình chính của đồ án, nó cho biết phần nào của mô hình đèn quyết định kết quả. **Báo cáo phải trích dẫn và nêu khác biệt** (đồ án chưa trích dẫn bài này, kiểm 2026-11-17).

## NFL-BA thay đổi gì so với MonoGS (đã so từng file với MonoGS `6c9254c3`)

| Chỗ | Thay đổi | Bật khi |
|---|---|---|
| `gaussian_renderer/__init__.py` | Màu mỗi Gaussian nhân với `shading^(1/2.2)`, `shading = cos(n, l) / d²`: đèn điểm **đặt đúng tâm camera**, suy giảm nghịch đảo bình phương, Lambert với pháp tuyến là **trục ngắn nhất** của Gaussian (đổi dấu về phía camera). Không giản đồ nón (`attenuation = 0.0`, tức đẳng hướng), không ánh sáng nền, không tham số cường độ (gộp vào màu). Tính trên từng Gaussian trước khi rasterize | `pipeline_params.nfl` |
| `slam_utils.py` | Loss tracking và mapping bỏ pixel có độ xám ≥ 0.9 (`spec_thresh=0.9`, hằng mặc định, coi là vùng phản xạ gương); trọng số độ sâu `beta` (mặc định `1 − alpha`, như MonoGS) | luôn |
| `slam_frontend.py` | Khoảng cách keyframe đo giữa hai tâm camera; `znear` 0.01 → 0.0001 | luôn |
| `slam.py`, `eval_utils.py` | Đánh giá ATE trên mọi frame, ghi ảnh render, tham số dòng lệnh (`--nfl`, `--alpha`, `--beta`, …) | (adapter không dùng) |

**Vẫn giữ bù affine** `exposure_a/b` của MonoGS trong tracking và mapping (đồ án bỏ, vì nó nuốt tín hiệu chuyển động dọc trục quang).

## File

| File | Việc |
|---|---|
| `adapter.py` | Dựng config, dàn dữ liệu (`traj.txt` đơn vị, dùng lại `stage` của adapter LoopSplat), chạy đúng phần `SLAM.__init__` của `slam.py` NFL-BA (frontend trong tiến trình này, backend trong tiến trình con), không chạy đánh giá của nó; xuất pose mọi frame, keyframe, render bản đồ cuối, bản đồ `.ply`, đóng zip |
| `configs/replica/` | Config Replica RGB-D của MonoGS (`base_config.yaml`, `office0.yaml`, `room0.yaml`) chép nguyên văn từ MonoGS `6c9254c3` (`SOURCE.md`): NFL-BA không có config Replica |
| `probe.py` | Bước dò sau khi build: mã NFL-BA trên một khung RGB-D tổng hợp (tạo Gaussian từ độ sâu, render tắt và bật mô hình đèn, một vòng tracking đưa camera lệch 2 cm về, một bước mapping) |
| `setup_kaggle.sh` | Tải NFL-BA ở commit ghim, `uv` cài phụ thuộc trên đường chạy SLAM của MonoGS (không cài phần GUI), build 2 extension trên bản chép, chạy `probe.py` |
| `kaggle.json` | `"checkpoints": false` (MonoGS không chạy tiếp được) |

## Cách chạy

1. Kaggle, GPU T4, bật Internet: `run_baseline.ipynb`, `METHOD = "nflba"`.
2. Mỗi run cho `/kaggle/working/nflba__<scene>_<variant>__<thời điểm>.zip` (render, ảnh 5 frame, bản đồ `run/point_cloud_final.ply`).
3. Tải về, chấm: `python baselines/common/score_zip.py --zip <zip> --dataset <dataset cục bộ> --no_wandb`. Chấm ảnh theo tập thường của `PROTOCOL.md` (frame 0, 5, 10, … trừ keyframe), như MonoGS.

## Khác biệt so với mặc định

NFL-BA không có cấu hình Replica, nên gốc là **cấu hình Replica RGB-D của MonoGS** (`configs/rgbd/replica/<scene>.yaml`) cộng cờ `--nfl` của NFL-BA. Mọi tham số tracking, mapping, keyframe, cửa sổ giữ nguyên MonoGS (cùng cấu hình đồ án dùng cho MonoGS gốc, chỉ khác các khóa riêng của đồ án). Mọi thay đổi ghi trong `overrides.json` và `config_resolved.json`.

| Đổi | Lý do |
|---|---|
| `Dataset.Calibration` (`width`, `height`, `fx`, `fy`, `cx`, `cy`, `depth_scale`) từ `protocol.calibration()` | Bắt buộc: Replica nửa độ phân giải. Adapter từ chối cấu hình có méo ảnh |
| `Dataset.dataset_path` trỏ thư mục dàn dữ liệu có `traj.txt` là ma trận **đơn vị** | MonoGS đặt frame 0 tại pose tham chiếu đọc từ `traj.txt` (và chấm ATE với nó). Với ma trận đơn vị, không pose tham chiếu nào lọt vào run; căn chỉnh SE(3) khi chấm làm việc chọn frame 0 không ảnh hưởng ATE |
| `pipeline_params.nfl: True` | Đây là phương pháp (`slam.py --nfl`) |
| `Results.save_results`, `use_gui`, `eval_rendering`, `use_wandb`: tắt; không chạy `eval_ate`, `eval_rendering`, color refinement | Việc chấm là của `score.py`; đánh giá của MonoGS dùng pose tham chiếu. Bản đồ chấm là bản đồ trực tuyến, trước refine (như cột chính của đồ án) |
| Một GPU (`CUDA_VISIBLE_DEVICES=0`) | Như MonoGS gốc |

**Xuất kết quả:**
- **Pose mọi frame:** từ camera của frontend lúc kết thúc: keyframe là pose đã được backend tinh chỉnh, frame khác là pose tracking. Đây là đúng định nghĩa `from_monogs_record.py` dùng cho MonoGS của đồ án, nên ATE all so được trực tiếp.
- **Render:** bản đồ của frontend (như `eval_rendering`) qua renderer NFL-BA có mô hình đèn, kẹp về [0, 1], không áp bù affine của từng frame (như `eval_rendering`).
- Xuất và lưu bản đồ làm **trước khi dừng backend**: bản đồ của frontend đến từ tiến trình backend qua hàng đợi CUDA IPC, backend dừng thì vùng nhớ đó mất hiệu lực (`slam.py` cũng đánh giá trước lệnh `stop`).

## Điểm cần nói khi diễn giải

- **Mô hình đèn của NFL-BA không khớp đèn của dữ liệu đồ án**, và đó chính là điểm cần so: dữ liệu có đèn spot lệch khỏi camera (`t_CL`), giản đồ nón tắt ở rìa, ánh sáng nền nhân albedo. NFL-BA giả định đèn điểm đẳng hướng tại tâm camera. Trong nón, hình dạng suy giảm gần đúng; ngoài nón, NFL-BA dự đoán vẫn sáng, còn ảnh thật gần như tối. Bù affine có thể bù một phần theo toàn ảnh.
- **Pháp tuyến:** NFL-BA lấy trục ngắn nhất của từng Gaussian; Gaussian mới tạo là đẳng hướng nên pháp tuyến ban đầu tùy ý cho tới khi mapping kéo dẹt chúng. Đồ án dùng pháp tuyến từ độ sâu (D49, D94).
- **Mặt nạ vùng sáng:** NFL-BA bỏ pixel có độ xám ≥ 0.9 trong cả tracking lẫn mapping. Trên `lit`, tâm đĩa sáng hay bão hòa, nên phần này cũng giúp NFL-BA (tương tự trọng số pixel của đồ án).
- **FPS:** "Total time" của MonoGS (`slam.py`): sự kiện CUDA từ đầu phần dựng SLAM tới hết frontend, gồm đọc ảnh, tracking, mapping keyframe ở backend, và việc MonoGS tự giới hạn 3 FPS sau mỗi keyframe. Cùng cách đồ án và MonoGS gốc báo.

## Kỹ thuật

- Đã kiểm cục bộ (WSL, GTX 1650; 2 extension build vào thư mục `--target` riêng, không đụng môi trường MonoGS; NFL-BA tải ở commit ghim bằng git như setup):
  - `probe.py`: qua. Tạo Gaussian từ độ sâu, render bật mô hình đèn tối hơn đúng trong giới hạn lý thuyết (trung vị 0.27, lớn nhất 0.34, giới hạn 0.53), tracking đưa camera lệch 2 cm về 0.11 mm, mapping có gradient.
  - Toàn bộ vòng SLAM: **không chạy được trên WSL**. Backend của MonoGS là tiến trình con, nhận bản đồ qua bộ nhớ CUDA chia sẻ (CUDA IPC); WSL không hỗ trợ, nên backend chết khi khởi động ("CUDA error: invalid resource handle") và frontend chờ mãi (lần thử 2026-11-17 treo khoảng 50 phút). Trên Kaggle (Linux) đây là cách MonoGS vẫn chạy. Từ đó adapter có luồng giám sát: in tiến độ mỗi 60 s, và dừng run ngay nếu backend chết.
- Chưa chạy trên Kaggle.

## Đã chạy

| Ngày | Biến thể | ATE all (cm) | PSNR / SSIM / LPIPS | FPS | Keyframe | Thư mục |
|---|---|---|---|---|---|---|
| 2026-11-17 | office0 / lit | 109.19 | 12.91 / 0.529 / 0.548 | 0.29 | 389 | `results/nflba__office0_lit/2026-11-17-08-16-48` |
| 2026-11-17 | office0 / even | 6.29 | 22.24 / 0.803 / 0.338 | 0.46 | 391 | `results/nflba__office0_even/2026-11-17-10-17-07` |

Cả hai run chạy hết trên Kaggle ngay lần đầu (setup, bước dò, vòng SLAM, xuất). Cảnh báo `Producer process has been terminated before all shared CUDA tensors released` ở cuối log xuất hiện khi dừng backend, **sau** khi đã render và lưu bản đồ: vô hại, đúng lý do phần xuất làm trước lệnh dừng. Kết quả kém (gãy ở frame 82–90 trên `lit`, 6.3 cm trên `even`); **chưa có run đối chứng tắt `nfl`** để tách mô hình đèn khỏi các khác biệt khác. Phân tích ở `RESULTS.md`.
