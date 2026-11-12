# SplaTAM

- Bài báo: SplaTAM: Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM (Keetha et al., CVPR 2024). Trong `refs.bib` của báo cáo: `keetha2024splatam`.
- Repo: <https://github.com/spla-tam/SplaTAM>, commit đã dùng `da6bbcd24c248dc884ac7f49d62e91b841b26ccc` (2024-06-19). Rasterizer: `JonathonLuiten/diff-gaussian-rasterization-w-depth` @ `cb65e4b86bc3bd8ed42174b72a62e8d3a3a71110` (đúng commit trong `requirements.txt` của SplaTAM).
- Vai trò trong so sánh: một hệ 3DGS-SLAM RGB-D khác cũng tracking bằng sai khác ảnh, có cộng thêm độ sâu. Nếu nó cũng hỏng trên `lit` thì lỗi là của cả họ quang trắc, không riêng MonoGS.

## File

| File | Việc |
|---|---|
| `adapter.py` | `run`: dàn dữ liệu, dựng config, chạy SLAM có đo giờ, rồi `export`. `export`: `params.npz` → `export/` (quỹ đạo TUM, keyframe, render `.npy`, `timing.json`, `method.json`, `notes.txt`) |
| `run_splatam.ipynb` | Notebook Kaggle: dựng môi trường SplaTAM, chạy `office0` `lit` rồi `even`, dọn output |

## Cách chạy

1. Kaggle, GPU T4, bật Internet: chạy `run_splatam.ipynb` (điền `BASELINES_SRC`). Mỗi run ghi vào `/kaggle/working/splatam__<scene>_<variant>/`, log ở file `.log` cùng tên.
2. Chấm trong môi trường MonoGS (notebook `run-monogs` hoặc WSL), sau khi lấy thư mục `export/` về. Cuối lệnh `run`, adapter in sẵn câu lệnh:

```
python baselines/common/score.py --method splatam --dataset <dataset> \
    --traj <export>/traj_est.tum --keyframes <export>/keyframes.txt --renders <export>/renders \
    --timing <export>/timing.json --out baselines/results/splatam__office0_lit/<thời gian> \
    --method_commit <commit> --notes "$(cat <export>/notes.txt)" --wandb
```

Phiên Kaggle bị ngắt: chép thư mục run của phiên trước về `/kaggle/working`, chạy lại cùng lệnh với `--resume`. Adapter tiếp tục từ checkpoint đầy đủ mới nhất (`params<i>.npz` cùng `keyframe_time_indices<i>.npy`), dùng cơ chế checkpoint có sẵn của SplaTAM. `timing.json` cộng thời gian các phiên tới đúng checkpoint mà phiên sau bắt đầu, và ghi rõ là đã resume.

## Khác biệt so với mặc định của SplaTAM cho Replica

Gốc: `configs/replica/splatam.py` và `configs/data/replica.yaml` của SplaTAM. Mọi tham số tracking, mapping, keyframe, pruning và seed giữ nguyên: 40 vòng tracking, 60 vòng mapping, `keyframe_every` 5, `map_every` 1, cửa sổ 24, `seed` 0. Mọi giá trị bị đổi ghi trong `<work>/overrides.json`, config đầy đủ trong `<work>/config_resolved.json`.

| Đổi | Lý do |
|---|---|
| Kích thước 600×340, `fx`, `fy`, `cx`, `cy`, `png_depth_scale`, đọc bằng `protocol.calibration()` từ `configs/rgbd/replica/base_config.yaml` | Bắt buộc: dataset của đồ án là Replica giảm nửa độ phân giải. SplaTAM tự đổi tỉ lệ nội tại theo `desired_image_*`, nên khai báo thẳng kích thước gốc của dataset (tỉ lệ 1). `crop_edge` giữ 0 như gốc |
| Đường dẫn, `run_name` | Bắt buộc |
| `use_wandb: false` | Chỉ là log. Run được chấm sẽ log lên W&B qua `score.py --wandb` |
| `traj.txt` của thư mục dàn dữ liệu là ma trận **đơn vị** | Loader Replica của SplaTAM đọc `traj.txt` để in ATE trong lúc chạy. Thay bằng ma trận đơn vị thì không pose tham chiếu nào lọt vào run. Tracking không đọc nó vì `tracking.use_gt_poses` là `false` (adapter kiểm). Hệ quả: ATE SplaTAM tự in trong log là vô nghĩa. `traj.txt` thật chỉ do `score.py` đọc |
| Bỏ bước `eval()` cuối của SplaTAM | Việc chấm là của `score.py`. Lời gọi đó được dùng làm mốc kết thúc vòng SLAM để đo FPS |
| Checkpoint mỗi `--checkpoint_every` frame (notebook: 100), chỉ giữ checkpoint mới nhất | Phiên Kaggle tối đa 12 giờ. Thời gian lưu checkpoint nằm trong FPS (khoảng 20 lần lưu trên 2000 frame) |
| Torch của Kaggle thay vì torch 1.12.1 + CUDA 11.6 | README của SplaTAM ghi đã thử với torch 2.3 + CUDA 12.1. Phiên bản thật ghi trong `method.json` |
| Build rasterizer: thêm `#include <cstdint>` vào `rasterizer_impl.h` nếu thiếu | Chỉ sửa lỗi biên dịch với GCC mới, không đổi thuật toán |

## Điểm cần nói khi diễn giải

- **SplaTAM mapping ở mọi frame** (`map_every` 1): frame hiện tại luôn nằm trong cửa sổ mapping. Keyframe (`keyframe_time_indices`: frame 0, 4, 9, 14, …) chỉ là bộ nhớ để chọn cửa sổ. Giao thức bỏ keyframe khỏi tập chấm ảnh, nên các frame 5, 10, 15, … vẫn được chấm, nhưng với SplaTAM chúng **đã từng là đích tối ưu khi đang là frame hiện tại**. MonoGS chỉ tối ưu bản đồ trên keyframe. Vì vậy PSNR, SSIM, LPIPS của SplaTAM thiên về phía có lợi, và bảng phải ghi rõ điều này.
- Chỉ dùng một GPU (`cuda:0`). MonoGS của đồ án dùng hai GPU.
- Depth Blender không có nhiễu (`../../../CLAUDE.md` mục 3).

## Kỹ thuật

- Các thư mục `datasets/`, `utils/`, `scripts/` của SplaTAM là namespace package (file khởi tạo tên `_init_.py`). Vì vậy gói cài sẵn trùng tên (ví dụ `datasets` của Hugging Face trên Kaggle) sẽ che mất chúng. `import_splatam()` đăng ký thẳng ba gói này vào `sys.modules`. `protocol.py` không còn tự đưa repo MonoGS vào `sys.path` lúc import, vì gói `utils` của MonoGS cũng trùng tên.
- Pose: SplaTAM render frame t bằng cách dời Gaussian theo `rel_w2c(t)` (`transform_to_frame`) rồi nhìn qua camera frame đầu `w2c`. Pose camera-to-world xuất ra là `inv(w2c @ rel_w2c(t))`, trong hệ của frame 0. Căn chỉnh SE(3) trong `score.py` xử lý phần còn lại.
- Render: đúng các bước tô màu trong `eval()` của SplaTAM (`setup_camera`, `transform_to_frame`, `transformed_params2rendervar`, nền đen). Lưu `float16` (H, W, 3) chưa kẹp; `score.py` kẹp về [0, 1].
- Đã kiểm cục bộ (WSL, môi trường MonoGS, không GPU SplaTAM): dàn dữ liệu, file data config, config sau khi đổi (mặc định giữ nguyên, khóa sai thì báo lỗi), tính thời gian khi resume, công thức quaternion `(w, x, y, z)` của `build_rotation` khớp scipy, quỹ đạo TUM ghi rồi đọc lại khớp, `score.py` vẫn gọi được `evaluate_evo` của MonoGS. **Chưa chạy SplaTAM thật**: phần `run_slam` và `export` (import SplaTAM, render) chỉ kiểm được trên Kaggle.

## Đã chạy

(chưa có)
