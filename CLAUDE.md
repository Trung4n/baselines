# baselines/: so sánh với các phương pháp SLAM khác

Ngữ cảnh cho mọi phiên làm việc trong thư mục này. Đọc cùng `../CLAUDE.md` (dự án chính) và `PROTOCOL.md` (cách chấm điểm).

## Mục đích

Báo cáo đồ án cần một khối "phương pháp khác" trong bảng kết quả chính, để trả lời hai câu hội đồng dễ hỏi:

1. MonoGS hỏng dưới đèn đồng hành (ATE 0.84 m trên `lit`, so với 1.0 cm trên `even`) là lỗi riêng của MonoGS, hay của cả họ SLAM quang trắc?
2. Tracking dựa trên đặc trưng hoặc hình học, vốn ít phụ thuộc độ sáng, có tránh được không?

Thư mục này **chỉ chạy và chấm** các phương pháp khác trên đúng dữ liệu của đồ án, rồi ghi kết quả vào `RESULTS.md` cho phiên viết báo cáo. Không sửa code MonoGS của đồ án từ đây.

## Phương pháp dự kiến

| Thư mục | Phương pháp | Trả lời câu | Ghi chú |
|---|---|---|---|
| `methods/splatam/` | SplaTAM (CVPR 2024), 3DGS-SLAM RGB-D, tracking quang trắc kèm độ sâu | 1 | Python, đọc định dạng Replica; chậm |
| `methods/orbslam3/` | ORB-SLAM3 RGB-D | 2 (đặc trưng) | C++, không có bản đồ kết xuất: chỉ ATE và FPS |
| `methods/gsicp/` | GS-ICP SLAM, tracking G-ICP hình học, bản đồ Gaussian | 2 (hình học) | nhanh; depth không nhiễu là lợi thế của nó |

Không so ATE với DarkGS, SpotlessGS, SunFaded: chúng cần pose cho trước, không phải hệ SLAM. Lý do này được ghi trong báo cáo.

## Quy tắc (giữ như dự án chính)

- **Không rò rỉ ground truth.** Đầu vào chỉ gồm `results/frame*.jpg` và `results/depth*.png` của thư mục dataset MonoGS. `traj.txt` chỉ dùng để chấm. Không đọc EXR, pass albedo hay pháp tuyến. Không khởi tạo bằng pose ground truth. Riêng pose frame đầu: MonoGS đặt nó bằng pose tham chiếu, và việc căn chỉnh SE(3) làm điều này không ảnh hưởng ATE. Phương pháp khác cứ khởi tạo theo cách của chúng.
- **Chấm bằng `common/score.py`**, không bằng script đánh giá riêng của phương pháp. Số phương pháp tự báo chỉ ghi vào cột ghi chú để đối chiếu.
- **Không gán cứng** nội tại camera hay thang độ sâu trong adapter. Đọc từ `configs/rgbd/replica/base_config.yaml` (`Dataset.Calibration`).
- **Giữ nguyên tham số mặc định** của từng phương pháp cho Replica, chỉ đổi những gì bắt buộc: độ phân giải 600×340, nội tại, thang độ sâu, đường dẫn. Mọi thay đổi khác ghi vào README của phương pháp, kèm lý do.
- Cùng loại GPU (Kaggle T4). FPS đo đúng như `PROTOCOL.md`.
- Mỗi cấu hình chạy ít nhất 2 lần nếu quota cho phép. Dao động của đồ án là 0.37 cm ATE và 0.8 dB PSNR (D139).
- Phụ thuộc của các phương pháp **không** vào `environment.yml` của MonoGS. Dùng môi trường hoặc notebook riêng.
- W&B: project `MonoGS`, tag `baseline` (`score.py --wandb`). API key lấy từ Kaggle Secrets, **không** ghi key vào notebook hay vào file.

## Dữ liệu

| Biến trong notebook `run-monogs.ipynb` | Đường dẫn Kaggle | Ý nghĩa |
|---|---|---|
| `OFFICE0_LIT` | `/kaggle/input/replica-lunogs/office0/lit` | đèn đồng hành, cảnh chính |
| `OFFICE0_EVEN` | `/kaggle/input/replica-lunogs/office0/even` | chiếu sáng đều (mốc trên) |
| `ROOM0_LIT`, `ROOM0_EVEN` | `/kaggle/input/datasets/nguyntrungan/replica-lunogs/room0/{lit,even}` | cảnh thứ hai |
| `OFFICE0_AMB` | `/kaggle/input/datasets/nguyntrungan/office0` | ánh sáng nền không đều: `<OFFICE0_AMB>/ambvar_hard/ambvar_hard` |

Định dạng mỗi thư mục: `results/frame%06d.jpg` (8 bit sRGB), `results/depth%06d.png` (16 bit, chia cho `depth_scale` ra mét), `traj.txt` (2000 dòng, mỗi dòng 16 số là ma trận camera-to-world 4×4). Trên WSL cục bộ, dataset ở `/mnt/d/CodeForYou/Replica/04_datasets/monogs/<scene>/<variant>`.

**Thứ tự ưu tiên chạy:** `office0` `lit` và `even` trước, sau đó `room0` `lit` và `even`, cuối cùng `ambvar_hard` nếu còn quota.

## Quy trình cho một phương pháp

1. Dựng môi trường trên Kaggle theo `methods/<name>/README.md`. Ghi commit của repo phương pháp.
2. Viết adapter `methods/<name>/adapter.py` (hoặc script). Adapter đọc dataset MonoGS, chạy phương pháp và ghi ra:
   - quỹ đạo TUM với cột thời gian là chỉ số frame (`common/protocol.write_tum`);
   - `keyframes.txt`, nếu phương pháp có khái niệm keyframe;
   - render `<idx:06d>.npy` tại pose ước lượng, cho các frame `protocol.required_renders(N, keyframes)`. Tập này gồm các frame chấm chỉ số và 5 frame chung để so ảnh: 100, 500, 825, 1125, 1750, render kể cả khi là keyframe;
   - `timing.json`.
3. Chạy `common/score.py --wandb`, kết quả vào `results/<method>__<scene>_<variant>/<thời gian>/`.
4. Thêm một dòng vào bảng chi tiết của `RESULTS.md`, cập nhật bảng tổng, ghi mọi khác biệt so với mặc định của phương pháp.

Với MonoGS của đồ án: `common/from_monogs_record.py --record record/runs/<run> --figures --dataset <...>` ghi ra quỹ đạo, keyframe và ảnh 5 frame chung từ bản đồ trong record (`--map final` hoặc `final_refined`), sau đó chạy `score.py --no_render`. PSNR, SSIM, LPIPS lấy từ log của MonoGS. Cách này cho ATE trên mọi frame theo cùng giao thức.

## Đã kiểm chứng (2026-11-12)

- `score.py` tính lại ATE trên keyframe của một run MonoGS (`cell0p5`, `lit`) ra 0.010947566 m, khớp số MonoGS log (0.010947593) tới 7 chữ số, đúng độ chính xác của `frames_final.csv`.
- Ảnh 5 frame chung: `from_monogs_record.py --figures` đã render được từ record (trước và sau refine), `compare_figures.py` ghép đúng, nhãn tiếng Việt có dấu hiển thị đúng.
- Phần chấm ảnh: render giả bằng ảnh quan sát cộng nhiễu σ = 0.01 cho PSNR 40.17 dB, đúng mức lý thuyết (khoảng 40 dB). Chấm 357 frame, cùng tập frame `eval_rendering` dùng.

## Đưa thư mục này lên Kaggle

`baselines/` nằm trong `.gitignore` của repo chính, nên `git clone` repo trên Kaggle sẽ **không** có nó. Hai cách:

- `git init` riêng trong `baselines/` và push lên một repo GitHub riêng, rồi clone trên Kaggle vào `LunoGS/baselines`. Cách này giữ được lịch sử.
- Hoặc nén và upload thành một Kaggle dataset, rồi copy vào `LunoGS/baselines`.

Adapter chạy trong môi trường của phương pháp khác, nên `common/protocol.py` không đưa repo MonoGS vào `sys.path` lúc import (gói `utils` của MonoGS trùng tên với gói của SplaTAM). Chỉ `ate()` và `RenderScorer` gọi `monogs_on_path()`. Adapter đọc nội tại camera bằng `protocol.calibration()`.

`common/` import code đánh giá của MonoGS (`utils/eval_utils.py`, `gaussian_splatting/utils/*`), nên `baselines/` phải nằm **bên trong** repo MonoGS đã clone, và chạy trong môi trường có `evo` và `torchmetrics` (môi trường MonoGS). Phương pháp khác chạy trong môi trường riêng; chỉ bước chấm điểm chạy trong môi trường MonoGS.

## Tình trạng

- [x] Giao thức, `common/`, `RESULTS.md` cùng số của đồ án
- [ ] SplaTAM: adapter và notebook Kaggle đã viết (2026-11-12, `methods/splatam/`), phần không cần GPU đã kiểm cục bộ; **chưa chạy** `office0` `lit` và `even`
- [ ] ORB-SLAM3: như trên
- [ ] GS-ICP SLAM: như trên
- [ ] MonoGS gốc chạy lại bằng code hiện tại (`lit`, `even`, có `--record`), chấm bằng `from_monogs_record.py`
