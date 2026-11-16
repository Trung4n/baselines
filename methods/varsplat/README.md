# VarSplat

- Bài báo: VarSplat: Uncertainty-aware 3D Gaussian Splatting for Robust RGB-D SLAM (Tran, Košecká, CVPR 2026, arXiv 2603.09673). Thêm vào `refs.bib` khi dùng.
- Mã nguồn: chép sẵn trong repo, `third_party/VarSplat` (anhthuan1999/varsplat @ `4a21a0aaade40b7c42d47181108e49a8a0e695f2`, 2026-06-14, repo chính chủ, MIT), kèm gói `hloc` và 3 extension CUDA ở đúng commit submodule ghim, trong `thirdparty/`. Xem `.vendor.json`.
- Vai trò trong so sánh: **nhóm B, câu 3** (`CLAUDE.md`): 3DGS-SLAM RGB-D có cơ chế chống pixel không tin cậy. Mỗi Gaussian học thêm một **phương sai màu** σ²; rasterizer render ra bản đồ phương sai theo pixel (luật tổng phương sai qua alpha compositing). Bản đồ này làm **trọng số pixel** ở ba chỗ: tracking, đăng ký submap, và phát hiện vòng (độ tin cậy mỗi submap). Về nguyên tắc, vùng đĩa sáng của đèn đồng hành, vốn đổi màu theo pose, sẽ có phương sai cao và bị giảm trọng số.
- **Đối chứng trực tiếp với LoopSplat:** VarSplat xây trên đúng mã LoopSplat (cùng `GaussianSLAM`, submap, `gs_reg`, NetVLAD, cùng `configs/Replica/*.yaml` trừ khối `var`). LoopSplat trên `lit` hỏng nặng (ATE 105 cm, gãy ở frame 257, `RESULTS.md`). Đặt hai hệ cạnh nhau cho biết riêng phần trọng số bất định cứu được bao nhiêu.

## File

| File | Việc |
|---|---|
| `adapter.py` | Adapter LoopSplat (`methods/loopsplat/adapter.py`, **import**, không chép) chạy trên mã VarSplat: dàn dữ liệu, dựng config, chạy `GaussianSLAM(config).run()`, xuất quỹ đạo, keyframe, render submap cuối, đóng zip. Thêm kiểm khối `var` và một bản vá riêng (`varsplat_patches`, dưới đây) |
| `probe.py` | Bước dò sau khi build: render một cảnh tổng hợp nhỏ bằng mã VarSplat, lan truyền ngược như mapping (NLL, gradient tới phương sai), tracking (gradient tới pose) và đăng ký (`viewpoint_localizer`, rasterizer có gradient pose). Lỗi build hay lệch ABI với torch hiện trong setup, không phải sau một giờ chạy |
| `setup_kaggle.sh` | Như LoopSplat: `uv` cài phụ thuộc; faiss GPU nếu có wheel, không thì faiss CPU cộng `methods/loopsplat/faiss_shim.py`; build 3 extension trên bản chép; chạy `probe.py` |
| `kaggle.json` | `"checkpoints": false` (không chạy tiếp được, như LoopSplat) |

## Cách chạy

1. Kaggle, GPU T4, bật Internet: `run_baseline.ipynb`, `METHOD = "varsplat"`. Lần chạy đầu tải trọng số NetVLAD (Pitts30K, 554 MB) từ máy chủ của ETH.
2. Mỗi run cho `/kaggle/working/varsplat__<scene>_<variant>__<thời điểm>.zip` (render, ảnh 5 frame, các submap trong `run/submaps.tar`).
3. Tải về, chấm: `python baselines/common/score_zip.py --zip <zip> --dataset <dataset cục bộ> --no_wandb`. Chấm trên **tập dịch lưới** như LoopSplat (`method.json` khai báo `heldout_offset`).

## Khác biệt so với mặc định của VarSplat cho Replica

Gốc: `run_slam.py` với `configs/Replica/<scene>.yaml` (kế thừa `replica.yaml`). Mọi tham số tracking, mapping, submap, loop closure và khối `var` giữ nguyên: `use_nlldepth: True`, `tau_track: 100`, `tau_registration: 100`, `tau_lc: 30`, `w_limit: 3` (đây là khối đang dùng; `replica.yaml` còn một khối `var` khác bị comment, τ = 10). Mọi thay đổi ghi trong `overrides.json` và `config_resolved.json`.

| Đổi | Lý do |
|---|---|
| **Mọi thay đổi của adapter LoopSplat**, cùng lý do (`methods/loopsplat/README.md`): nội tại và thang độ sâu từ `protocol.calibration()`; `traj.txt` là ma trận đơn vị; số pixel lấy mẫu khi mở submap nhân 0.25; W&B tắt; không chạy `Evaluator`; torch của Kaggle và `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD`; `torch_compat`; một GPU; bỏ `analyse_pgo`; lưu submap cuối; faiss shim; `#include` khi build | Mã VarSplat ở các chỗ này giống hệt LoopSplat (đã so từng file). VarSplat đã tự bỏ `verbose=False` ở cả hai chỗ gọi `ReduceLROnPlateau`, nên lớp bọc của `torch_compat` (vẫn được cài trên torch mới, ghi trong `overrides.json`) không đổi lời gọi nào |
| `Loop_closure` giữ, cho mỗi submap đã xong, **chỉ phương sai màu và opacity** (vá trong adapter, `varsplat_patches`) thay vì cả `GaussianModel` | VarSplat lưu nguyên mô hình của mọi submap đã xong trong `submap_gaussians`, chỉ để `submap_reliability` đọc `get_color_variance()` và `get_opacity()` khi phát hiện vòng. Mô hình nguyên còn gradient và trạng thái Adam (amsgrad), khoảng 5 lần số tham số, nên với 30–40 submap là vài GB trên T4 16 GB (tác giả chạy A100 80 GB). Mô hình không đổi sau lời gọi này (bị thay bằng mô hình mới ở `start_new_submap`; hiệu chỉnh đóng vòng ghi lại file submap trên đĩa, và là phép cứng, không đổi phương sai hay opacity). Đã kiểm cục bộ: `submap_reliability` trên bản chụp **bằng đúng** trên mô hình gốc, và mô hình gốc được giải phóng. Thiếu mô hình (đóng vòng cuối, `lc.final`) thì báo lỗi; cấu hình Replica có `lc.final: False`, và adapter từ chối cấu hình bật nó |

## Điểm cần nói khi diễn giải

- **Như LoopSplat:** render là bản đồ cuối dạng submap, mỗi frame bằng submap của nó, tại pose ước lượng; chấm ảnh trên tập dịch lưới (frame cách frame mapping 2 frame). PSNR vì vậy thiên về phía có lợi so với phương pháp render một bản đồ chung (`RESULTS.md`, mục LoopSplat). VarSplat và LoopSplat chấm cùng cách, nên so với nhau thì công bằng.
- **Loop closure:** VarSplat có, MonoGS không.
- **Một chỗ lệch trong mã của tác giả, giữ nguyên:** `gaussian_registration` gọi `viewpoint_localizer(viewpoint, gs, base_lr, tau_registration, w_limit)` theo vị trí, nhưng hàm nhận `(viewpoint, gaussians, base_lr, use_variance, tau_registration, w_limit)`. Kết quả: `use_variance` nhận 100 (đúng, vẫn dùng phương sai), còn τ của đăng ký thực tế là `w_limit` = 3 chứ không phải 100 như cấu hình. Không sửa, vì đây là hành vi của mã công bố (và có thể là của các số trong bài báo).
- `VALUE_INIT = -7` (log phương sai ban đầu, comment ghi "Scannet++") là hằng trong `gaussian_model.py`, dùng cho mọi dataset.
- Ảnh phương sai `var_vis/` (chỉ để xem) dùng `matplotlib.cm.get_cmap`, đã bị gỡ ở matplotlib mới; lỗi bị `try/except` của tác giả nuốt, nên không có ảnh, không ảnh hưởng ước lượng.
- **FPS:** `GaussianSLAM.run()` từ frame đầu tới frame cuối, như LoopSplat. VarSplat ghi log ra file ở mọi dòng in (mở file mỗi lần), phần này nằm trong FPS vì là mã của tác giả. `timing.json` ghi thêm bộ nhớ GPU đỉnh.

## Kỹ thuật

- **So mã với LoopSplat** (2026-11-16, từng file trong `src/`): khác ở `gaussian_model.py` (tham số `color_variance`, khởi tạo `VALUE_INIT`, lưu vào checkpoint submap), `utils.py` (`render_gaussian_model` luôn trả bản đồ phương sai), `mapper.py` (thêm NLL × 1e-4, có cả phần độ sâu khi `use_nlldepth`), `tracker.py` (trọng số pixel `exp(−(log v − trung vị)/τ_track)`, kẹp trong [1/`w_limit`, `w_limit`]), `gsr/` (đăng ký có trọng số phương sai), `lc.py` (độ tin cậy submap nhân vào độ tương tự NetVLAD), cộng log ra file. `datasets.py`, `run_slam.py`, cách mở submap, đóng vòng, lưu và hiệu chỉnh submap: như LoopSplat. Hàm `hloc/extractors/netvlad.py` giống hệt bản LoopSplat đã chạy trên Kaggle.
- **Chạy thử cục bộ** (WSL, GTX 1650 4 GB, torch 2.4.1 + CUDA 12.1 của môi trường MonoGS; extension build vào một thư mục `--target` riêng, không đụng môi trường MonoGS). NetVLAD được thay bằng mô tả ảnh thu nhỏ trong một script thử ở thư mục tạm, để khỏi tải 554 MB; mã trong repo không đổi.
  - `probe.py`: qua hết (render, gradient tới phương sai, gradient pose khi tracking, đăng ký đưa camera lệch 2 cm về 0.55 cm).
  - Bản vá `varsplat_patches`: `submap_reliability` trên bản chụp bằng đúng trên mô hình gốc (3 cỡ mô hình), mô hình gốc được giải phóng, thiếu mô hình thì báo lỗi.
  - 120 frame đầu `office0 lit`: chạy hết, xuất render, đóng zip. PSNR mapping 25 frame mapping đầu gần như trùng LoopSplat trên Kaggle (58.1, 16.9, 16.7, 17.5, …, 14.7). 0.19 FPS trên GTX 1650, bộ nhớ GPU đỉnh 0.46 GB.
  - 599 frame (frame 0→299 rồi quay ngược về 0, để có vòng lặp): 6 submap (frame 144, 211, 257, 286, 338, 361; LoopSplat trên Kaggle cũng mở submap ở frame 257, nơi nó gãy), phát hiện vòng có trọng số tin cậy, đăng ký `gs_reg` có phương sai (300 s trên GTX 1650), tối ưu pose graph (1 cạnh vòng), hiệu chỉnh submap và pose, rồi tracking và mapping tiếp. Lần đầu dừng ở frame 368 với `CUDA error: unknown error` trong tracking, sau 7 frame chạy bình thường kể từ lúc hiệu chỉnh; máy có chuyển trạng thái nguồn và màn hình trước đó. Lần chạy lại qua frame 368 bình thường, nên lỗi đó do môi trường WSL. Lần chạy lại: submap ở frame 144, 210, 258, 362, 405, 463; tối ưu pose graph ở submap frame 405 (1 cạnh vòng); rồi hết bộ nhớ GPU (4 GB) ở lần phát hiện vòng kế tiếp, trong `faiss_shim` (khối khoảng cách 1024 × khoảng 400 nghìn điểm = 1.54 GiB). Đây là giới hạn của bài thử cục bộ: cục bộ chỉ có faiss CPU nên dùng shim, còn trên Kaggle LoopSplat đã dùng faiss GPU thật (`method.json`: `faiss-gpu`), và T4 có 16 GB. Phần xuất (lưu submap cuối, render từng submap, đóng zip) đã qua ở bài 120 frame và giống hệt LoopSplat đã chạy trên Kaggle.
  - Tín hiệu sớm (một run cục bộ, không phải kết quả): trên `lit`, VarSplat **cũng nhảy ở đoạn bảng trắng**, 26.6 cm trong một frame ở frame 258 (LoopSplat trên Kaggle: 20.6 cm ở frame 257). ATE frame 0–299: 15.9 cm (LoopSplat 14.2 cm).

## Đã chạy

(chưa có)
