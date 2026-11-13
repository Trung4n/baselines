# LoopSplat

- Bài báo: LoopSplat: Loop Closure by Registering 3D Gaussian Splats (Zhu et al., 3DV 2025). Thêm vào `refs.bib` khi dùng.
- Mã nguồn: chép sẵn trong repo, `third_party/LoopSplat` (GradientSpaces/LoopSplat @ `676e18f950b2de6be39525a613120712b67768bb`, 2025-02-03), kèm gói `hloc` (Hierarchical-Localization @ `41860cf4`, chỉ dùng NetVLAD) và 3 extension CUDA ở đúng commit `requirements.txt` ghim, trong `_deps/`. Xem `.vendor.json`.
- Vai trò trong so sánh: 3DGS-SLAM RGB-D dựa trên Gaussian-SLAM (tracking quang trắc cộng độ sâu, bản đồ chia submap), có **loop closure** bằng đăng ký Gaussian giữa các submap. Câu hỏi: thêm loop closure có cứu được tracking quang trắc dưới đèn đồng hành không (SplaTAM thì không)?

## File

| File | Việc |
|---|---|
| `adapter.py` | Dàn dữ liệu (`traj.txt` giả), dựng config từ `configs/Replica/<scene>.yaml`, chạy `GaussianSLAM(config).run()` ngay trong tiến trình, xuất quỹ đạo, keyframe, render submap cuối, đóng zip |
| `faiss_shim.py` | Thay chỉ số faiss GPU bằng tìm láng giềng chính xác bằng torch trên GPU khi Kaggle không có faiss GPU cho Python của nó; cho kết quả như tìm vét cạn (đã kiểm với numpy) |
| `setup_kaggle.sh` | `uv` cài phụ thuộc trên đường chạy SLAM; faiss GPU nếu có wheel, không thì faiss CPU cộng shim; build 3 extension CUDA trên bản chép |
| `kaggle.json` | `"checkpoints": false` (`checkpoint_path` của LoopSplat chỉ nạp lại run đã xong) |

## Cách chạy

1. Kaggle, GPU T4, bật Internet: `run_baseline.ipynb`, `METHOD = "loopsplat"`. Lần chạy đầu tải trọng số NetVLAD (Pitts30K) từ máy chủ của ETH.
2. Mỗi run cho `/kaggle/working/loopsplat__<scene>_<variant>__<thời điểm>.zip` (render, ảnh 5 frame, các submap trong `run/submaps.tar`).
3. Tải về, chấm: `python baselines/common/score_zip.py --zip <zip> --dataset <dataset cục bộ> --no_wandb`. **Xem mục chấm ảnh bên dưới trước.**

## Khác biệt so với mặc định của LoopSplat cho Replica

Gốc: `run_slam.py` với `configs/Replica/<scene>.yaml` (kế thừa `replica.yaml`). Mọi tham số tracking, mapping, submap, loop closure giữ nguyên (60 vòng tracking, mapping mỗi 5 frame, 100 vòng mapping, submap theo chuyển động, đăng ký `gs_reg`, seed 1). Mọi thay đổi ghi trong `overrides.json` và `config_resolved.json`.

| Đổi | Lý do |
|---|---|
| `cam.H/W/fx/fy/cx/cy/depth_scale` từ `protocol.calibration()` | Bắt buộc: Replica nửa độ phân giải |
| `data.input_path` trỏ thư mục dàn dữ liệu có `traj.txt` là ma trận **đơn vị**; `output_path`, `scene_name` | LoopSplat lấy **pose thật cho frame 0 và frame 1** từ `traj.txt` (rồi mới tracking từ frame 2), và in ATE với nó trong lúc đóng vòng. Với ma trận đơn vị, không pose tham chiếu nào lọt vào run. Cái giá: frame 1 bắt đầu ở pose của frame 0, mất chuyển động thật giữa hai frame đầu (vài mm). Adapter từ chối config dùng pose thật (`gt_camera`, `odometry_type: gt`, đăng ký `gt`) |
| `use_wandb: false` | Chỉ là log |
| Không chạy `Evaluator` (đánh giá của LoopSplat, có tinh chỉnh bản đồ toàn cục 30000 vòng) | Việc chấm là của `score.py`; tinh chỉnh ngoại tuyến tương tự color refinement của MonoGS, không thuộc bản đồ cuối |
| Torch của Kaggle thay vì 2.1.2; `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` | Torch ≥ 2.6 mặc định `weights_only=True`, còn LoopSplat tự `torch.load` lại các checkpoint submap nó vừa ghi |
| faiss GPU → `faiss_shim.py` khi không có wheel GPU | Chỉ thay nơi tính láng giềng gần nhất; cùng kết quả. `method.json` ghi `faiss` đã dùng |
| Build: `#include <cstdint>`, `#include <cfloat>` trên bản chép | Chỉ sửa lỗi biên dịch với GCC/CUDA mới |

## Chấm ảnh: tập dịch lưới (quyết 2026-11-13)

LoopSplat mapping ở frame 0, 5, 10, … (`map_every: 5`), cộng frame bắt đầu mỗi submap. Tập chấm ảnh thường của `PROTOCOL.md` (frame 0, 5, 10, … trừ keyframe) vì vậy **rỗng**: mọi frame trong lưới đều là frame đã dùng để tối ưu.

Người dùng chọn chấm trên **tập dịch lưới** (`PROTOCOL.md`, mục PSNR): frame 2, 7, 12, … trừ keyframe, khoảng 400 frame chưa dùng để tối ưu. Adapter render tập này (cùng phần còn lại của lưới 0, nếu có, và 5 frame chung) rồi ghi `"heldout_offset": 2` vào `method.json`. `score_zip.py` đọc khóa đó và chấm bằng `score.py --render_offset 2`, `metrics.json` ghi `render.grid_offset: 2`. Kiểm cục bộ bằng zip giả: 400 frame, PSNR 40.17 dB với nhiễu σ = 0.01; zip không khai báo bị từ chối và không để lại thư mục kết quả dở. Trong bảng, dòng LoopSplat ghi chú "chấm trên lưới dịch".

## Điểm cần nói khi diễn giải

- **Render là bản đồ cuối dạng submap**, sau khi loop closure hiệu chỉnh, trước tinh chỉnh toàn cục: frame i render bằng submap chứa nó, đúng cách `Evaluator` của LoopSplat render keyframe.
- **Loop closure:** LoopSplat có, MonoGS không. Báo cáo phải nêu.
- **FPS:** `GaussianSLAM.run()` từ frame đầu tới frame cuối, gồm đọc ảnh, tracking, mapping, đóng vòng; không gồm khởi tạo (tải NetVLAD) và bước xuất.

## Kỹ thuật

- Đã kiểm cục bộ: `faiss_shim` khớp tìm vét cạn bằng numpy trên GPU (khoảng cách, chỉ số, cách đệm khi ít điểm hơn k, `reset`), config dựng đúng (thay đổi như bảng trên, cảnh không có config thì báo lỗi), dàn dữ liệu, tập dịch lưới.

  **Chưa build, chưa chạy trên Kaggle**: 3 extension CUDA, faiss, NetVLAD, vòng SLAM và phần render submap (`render_submaps`) chỉ kiểm được ở đó.

## Đã chạy

(chưa có)
