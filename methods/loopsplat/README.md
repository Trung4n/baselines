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
| `mapping.new_submap_points_num` 600000 → 150000, `new_submap_gradient_points_num` 50000 → 12500 (nhân tỉ lệ số pixel, 600×340 / 1200×680 = 0.25) | Bắt buộc do độ phân giải: khi mở submap, LoopSplat lấy mẫu không lặp số pixel này trong ảnh; 600000 lớn hơn số pixel của ảnh 600×340, nên run đầu trên Kaggle (2026-11-15) dừng ở frame 0. Giữ đúng tỉ lệ pixel được lấy như cấu hình gốc (khoảng 74% và 6%) |
| Không chạy `Evaluator` (đánh giá của LoopSplat, có tinh chỉnh bản đồ toàn cục 30000 vòng) | Việc chấm là của `score.py`; tinh chỉnh ngoại tuyến tương tự color refinement của MonoGS, không thuộc bản đồ cuối |
| Torch của Kaggle thay vì 2.1.2; `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` | Torch ≥ 2.6 mặc định `weights_only=True`, còn LoopSplat tự `torch.load` lại các checkpoint submap nó vừa ghi |
| `ReduceLROnPlateau(verbose=False)`: bỏ tham số `verbose` (vá trong adapter, `torch_compat`, chỉ khi torch đang chạy không còn tham số này; `verbose=True` thì báo lỗi) | Torch 2.11 của Kaggle đã gỡ tham số này, run thứ hai (2026-11-15) dừng ở frame 1 với `TypeError`. LoopSplat truyền `False` (tracking trong `gaussian_model.py`, đăng ký submap trong `gsr/solver.py`), tức chỉ tắt một dòng log, nên bỏ đi không đổi phép tính. Ghi vào `overrides.json`. Đã quét LoopSplat tìm các API torch/numpy khác đã bị gỡ (`torch.lstsq`, `symeig`, `solve`, `np.float`, `np.int`, …): không còn chỗ nào |
| Chạy trên **một GPU**: adapter đặt `CUDA_VISIBLE_DEVICES=0` trước khi nạp torch và faiss (nếu biến đã đặt nhiều GPU thì từ chối) | Kaggle có 2 GPU (T4 x2). Khi có nhiều GPU, `faiss.index_cpu_to_all_gpus` (`gsr/overlap.py`, bước đóng vòng) trả về lớp bọc `IndexReplicas` chứ không phải chỉ mục GPU, và `faiss.contrib.torch_utils` từ chối tensor CUDA trên nó. Run thứ ba (2026-11-15) dừng ở lần đóng vòng đầu, frame 701. LoopSplat vốn chỉ dùng GPU 0 (mapping dùng `index_cpu_to_gpu(res, 0, ...)`), tác giả chạy trên một GPU, nên không đổi thuật toán. Ghi vào `overrides.json` (`gpus`) |
| Bỏ qua `Loop_closure.analyse_pgo` (vá trong adapter, `loopsplat_patches`) | Hàm này chỉ **đánh giá theo pose tham chiếu**: ATE trước và sau pose graph, đồ thị sai số cạnh trong `<output>/pgo/`. Nó chạy sau khi hiệu chỉnh đã tính xong và chỉ trên bản chép của camera, nên bỏ đi không đổi kết quả. Ở đây pose tham chiếu là `traj.txt` đơn vị, nên con số vô nghĩa, và hàm không chạy được: run thứ tư (2026-11-16, `even`, lần đóng vòng đầu ở frame 701, pose graph đã tối ưu xong) dừng vì evo mới không còn `trajectory.align_trajectory`. Ghim evo 1.11 như tác giả cũng không cứu được: đã thử cục bộ, căn Umeyama với các pose tham chiếu trùng nhau báo `Degenerate covariance rank`. Số lần tối ưu pose graph ghi vào `method.json` (`pose_graph_optimizations`) |
| Lưu **submap cuối** sau khi chạy xong (`save_current_submap` của LoopSplat) | `GaussianSLAM.run()` chỉ lưu một submap khi submap kế tiếp bắt đầu, nên submap đang mở ở frame cuối không bao giờ được ghi; `Evaluator` của tác giả chỉ render các submap đã lưu, tức bỏ qua đoạn cuối. Không lưu thì các frame của đoạn cuối bị render bằng submap trước đó, và keyframe của đoạn cuối thiếu trong `keyframes.txt`. Adapter giữ mô hình ở lần gọi `Mapper.map` cuối (frame N−1 luôn là frame mapping) rồi lưu; nếu lần gọi cuối không phải frame N−1, hoặc file đã có, thì báo lỗi. Ngoài thời gian đo FPS |
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

  **Trên Kaggle** (tới 2026-11-16): 3 extension CUDA, faiss, NetVLAD, tracking, mapping, mở submap, phát hiện vòng, đăng ký `gs_reg` và tối ưu pose graph đã chạy (run `even` tới frame 701). Sau bản sửa, cả hai run `office0` chạy hết (2026-11-16): `lit` 1 lần tối ưu pose graph, `even` 4 lần, rồi lưu submap cuối, render và đóng zip. Kết quả ở `RESULTS.md`.

  Rà soát 2026-11-16, sau lỗi evo, toàn bộ phần chưa chạy (`gaussian_slam.py` từ đóng vòng tới hết `run()`, `lc.py`, `Evaluator` để đối chiếu cách render, và adapter). Kết quả: hai lỗi trên đã sửa. Ngoài ra, ở cấu hình Replica, `lc.final` là `False`, nên lần đóng vòng cuối (vốn đọc cả submap cuối chưa lưu) không chạy. evo chỉ được dùng trong `analyse_pgo`. Các checkpoint `torch.load` lại đều nằm trên đường đã chạy, hoặc có `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD`. Phép biến đổi submap chạy trên CPU, cùng thiết bị.

## Đã chạy

| Ngày | Biến thể | ATE all (cm) | PSNR / SSIM / LPIPS (tập dịch lưới) | FPS | Submap | Tối ưu pose graph | Thư mục |
|---|---|---|---|---|---|---|---|
| 2026-11-16 | office0 / lit | 105.02 | 20.71 / 0.869 / 0.144 | 0.53 | 40 | 1 (frame 1102) | `results/loopsplat__office0_lit/2026-11-16-07-47-36` |
| 2026-11-16 | office0 / even | 0.72 | 45.09 / 0.987 / 0.010 | 0.50 | 28 | 4 | `results/loopsplat__office0_even/2026-11-16-08-55-23` |
