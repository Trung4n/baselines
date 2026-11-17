# Luminance-GS

- Phương pháp: Cui, Chu, Harada, "Luminance-GS: Adapting 3D Gaussian Splatting to Challenging Lighting Conditions with View-Adaptive Curve Adjustment", CVPR 2025 (arXiv 2504.01503). Học **một đường cong độ sáng cho mỗi ảnh** (curve toàn cục cộng phần bù sinh từ ảnh và pose camera, cộng phép chiếu màu cho mỗi ảnh), đồng thời với 3DGS. Gaussian có hai nhánh màu: `enh` (màu SH, khớp với ảnh đã chỉnh đường cong, tức ảnh "chiếu sáng chuẩn") và `low` (`adjust_k · màu + adjust_b`, hệ số affine **riêng cho mỗi Gaussian, chung mọi ảnh**, khớp với ảnh quan sát). Thêm vào `refs.bib` khi dùng.
- Mã nguồn: <https://github.com/cuiziteng/Luminance-GS> commit `f9b4b866` (2026-05-13, có sửa lỗi train 2026-05-10), Apache-2.0. Các file cần để train chép vào `third_party/Luminance-GS` (`.vendor.json`). Repo mang theo bản gsplat 1.0.0 **đã sửa** (thêm số kênh 5, 9, 17, 33 trong kernel, thêm `rendering_double.py`), biên dịch JIT từ mã đó.
- Vai trò: **nhóm C** (`PROTOCOL.md` mục "Nhóm C"). Bù chiếu sáng theo từng ảnh bằng đường cong học được, không có mô hình đèn. Đánh giá gốc của tác giả là ảnh thiếu sáng, dư sáng, phơi sáng thay đổi giữa các ảnh (LOM, MipNeRF360-varying), không phải chiếu sáng thay đổi **trong** ảnh như đèn đồng hành.

## Cách chạy và chấm

- `simple_trainer_ours.py --data_dir <gói> --exp_name low --result_dir <thư mục>`, đúng lệnh ảnh thiếu sáng trong README của tác giả. Mặc định còn lại: 10000 bước, SH bậc 3, chuẩn hóa hệ thế giới. `exp_name` chỉ chọn bộ đọc dữ liệu và trọng số của loss histogram sau bước 3000; `low`, `variance` và để trống cho cùng trọng số 0.5 (chỉ `over_exp` khác).
- Gói dữ liệu theo bố cục LOM mà bộ đọc mặc định (`datasets/colmap.py`) cần, dựng từ gói COLMAP chung (`methods/darkgs/colmap_pkg.py`, file nhị phân cho bản `pycolmap` mà repo dùng):
  - `colmap_sparse/0/`: camera, pose tham chiếu của ảnh train và ảnh chấm, đám mây điểm từ độ sâu cảm biến của **ảnh train** (tác giả khởi tạo từ điểm SfM);
  - `low/`: mọi ảnh có trong COLMAP (bộ đọc ghép danh sách tên COLMAP đã sắp với danh sách file đã sắp của thư mục này);
  - `high/`: ảnh kiểm định của trainer, là lưới dịch. Với LOM đây là ảnh chiếu sáng chuẩn; ở đây là **ảnh quan sát** (run không thấy gì khác), nên số `eval()` của trainer không dùng;
  - `transforms_train.json` (lưới 0), `transforms_test.json` (lưới dịch). Train chỉ đọc `lentrain` ảnh đầu.
- Render cả hai nhánh tại pose tham chiếu, từ checkpoint cuối, đưa vào hệ chuẩn hóa bằng chính `transform_cameras` của repo (adapter kiểm: mọi pose trong trainer trùng pose tham chiếu sau phép đổi):
  - `renders/`: nhánh `low`, tức **dự đoán ảnh quan sát**. `score_zip.py` chấm bộ này với `lit`, như mọi phương pháp khác.
  - `renders_enh/`: nhánh `enh`, ảnh đã chuẩn hóa chiếu sáng (cái tác giả chấm với ảnh chuẩn). Chấm riêng với biến thể `even` (dưới đây). `even` chỉ dùng để chấm, không phải đầu vào.

Chấm bộ thứ hai sau `score_zip.py` (đã giải nén vào `runs/<tên zip>/`), trong repo MonoGS:

```
E=baselines/runs/<tên zip>
python baselines/common/score.py --method luminancegs_enh --dataset <dataset office0/even> --given_poses \
    --keyframes $E/keyframes.txt --renders $E/renders_enh --timing $E/timing.json \
    --out baselines/results/luminancegs_enh__office0_even/<thời điểm> --render_offset 2 --no_wandb
```

## Khác biệt (ghi trong `overrides.json`)

Vá trên bản chép ở đường dẫn cố định `$TMPDIR/luminancegs_src`. Đường dẫn cố định để run dùng lại bản build JIT của probe lúc setup.

| Đổi | Lý do |
|---|---|
| Bỏ `from turtle import color` trong `simple_trainer_ours.py` | Không dùng; `turtle` cần tkinter, Python của Kaggle có thể không có |
| Thêm `examples/datasets/__init__.py` rỗng | `datasets/` là namespace package; gói `datasets` của Hugging Face trên Kaggle sẽ che mất nó |
| Chép glm vào `gsplat/cuda/csrc/third_party/glm` | Mã CUDA include glm nhưng repo mất submodule. Dùng đúng commit gsplat 1.0.0 ghim (`g-truc/glm` `45008b22`, glm 1.0.2), setup tải bằng git |
| Không cài gsplat bằng pip (README tác giả: gsplat 1.0.0 từ git) | `_backend.py` nhập kernel của gói `gsplat` đã cài nếu có, mà kernel gốc thiếu các số kênh repo thêm. Setup gỡ gsplat nếu có; adapter và probe từ chối khi thấy gói này |
| Không ép `numpy==1.24.3` | Kaggle (2026-11-18) dùng Python 3.13, không có numpy 1.x. Parser của repo đã kiểm trong env Python 3.13 + numpy 2.5.5: ra pose, điểm, phép chuẩn hóa trùng numpy 1 (lệch ≤ 2e-15) |
| `pycolmap` (bản fork rmbrualla) vá cho numpy 2 lúc cài: `np.uint64(-1)` thành `np.uint64(2**64 - 1)`, cùng giá trị (`common/install_pycolmap_fork.sh`, dùng chung với bilagrid) | Dòng gốc báo `OverflowError` khi import dưới numpy 2 |
| Lọc output: bỏ dòng `the image train is 0 ...` | Bộ đọc in cả danh sách ảnh train một lần cho mỗi ảnh (bậc hai theo số ảnh) |

## Kỹ thuật

- **Biên dịch JIT** backend đi kèm: probe trong `setup_kaggle.sh` biên dịch ngay lúc setup, tại bản chép `$TMPDIR/luminancegs_src`. Run làm lại bản chép ở **cùng đường dẫn**, `copytree` giữ thời gian file, nên torch dùng lại bản build. Adapter đặt `TORCH_CUDA_ARCH_LIST` bằng compute capability của GPU, đúng dạng setup export cho probe (`cuda_arch_env`). Không có nó thì torch ghi cờ `-gencode` khác, và ninja biên dịch lại ngay trong run.
- **Bộ nhớ khi biên dịch:** mỗi job `nvcc` (`rasterization.cu`) dùng tới khoảng 2.9 GB RAM. Trên WSL cục bộ (giới hạn 4 GB), `MAX_JOBS=4` làm máy ảo hết bộ nhớ và treo. Phải dùng `MAX_JOBS=1` (9 phút, GTX 1650). Kaggle có khoảng 30 GB nên giữ `MAX_JOBS=4` của setup.
- **Không chạy song song với gsplat 1.5.3 (bilagrid) trong một môi trường:** cả hai biên dịch JIT dưới cùng tên `gsplat_cuda`, và gói `gsplat` đã cài sẽ được nhập thay cho kernel đi kèm. Trên Kaggle mỗi phiên chỉ chạy một phương pháp.
- **`TMPDIR` dài làm hỏng DataLoader** (lỗi `AF_UNIX path too long` ở worker, rồi treo). Chỉ gặp khi chạy thử cục bộ với `TMPDIR` trong scratch; Kaggle dùng `/tmp`.
- Trainer tự đánh giá ở các bước 5000, 7000, 10000 (số so `enh` với ảnh quan sát, không dùng) và render video quỹ đạo (cần `imageio[ffmpeg]`). Thời gian đó nằm trong `timing.json`.
- Checkpoint cuối (Gaussian, gồm cả `adjust_k`, `adjust_b`) vào `run/` của zip. Đường cong và hai mạng sinh phần bù đường cong không được trainer lưu. Hai nhánh render không cần chúng.

## Đã chạy

**Chạy thử cục bộ** (GTX 1650, 2026-11-18): 120 frame đầu của `office0 lit`, 24 ảnh train, 3000 bước (thay 10000), 442 s, 342 nghìn Gaussian. Chấm trên 24 frame lưới dịch:

| Render | So với | PSNR | SSIM | LPIPS |
|---|---|---|---|---|
| `low` (`renders/`) | `lit` | 18.04 | 0.824 | 0.192 |
| `enh` (`renders_enh/`) | `even` | 11.50 | 0.728 | 0.233 |
| `low` | `even` (đối chiếu) | 14.63 | 0.604 | 0.311 |

Nhìn ảnh:
- **`low`** giữ vệt đèn cố định trên bản đồ, vì hệ số affine là của từng Gaussian và dùng chung cho mọi ảnh. Ở frame chấm, quầng sáng không theo camera. Cùng hiện tượng với bilagrid khi không dùng grid (19.3 dB trong cùng phép thử).
- **`enh`** đều hơn nhiều, chỉ còn dấu vết nón mờ, nhưng sáng gấp gần đôi `even` (trung bình 0.42 so với 0.24). Đường cong kéo về histogram cân bằng, không về mức phơi sáng của `even`, nên PSNR với `even` thấp chủ yếu vì lệch độ sáng tổng thể.
- Đánh giá riêng của trainer (so `enh` với ảnh quan sát): 13.05.

Chưa chạy Kaggle.
