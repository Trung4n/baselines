# 3DGS + bilateral grid (gsplat)

- Phương pháp: 3DGS với một **bilateral grid cho mỗi ảnh train** (Wang et al., "Bilateral Guided Radiance Field Processing", SIGGRAPH 2024), cài trong gsplat. Mỗi grid là một phép biến đổi màu affine thay đổi theo vị trí trong ảnh và theo độ sáng, học cùng bản đồ để hấp thụ khác biệt ngoại hình giữa các ảnh (phơi sáng, cân bằng trắng, vùng sáng tối). Thêm vào `refs.bib` khi dùng.
- Mã nguồn: gsplat <https://github.com/nerfstudio-project/gsplat> **v1.5.3** (Apache-2.0). Thư viện cài bằng `pip gsplat==1.5.3`; script train mẫu `examples/simple_trainer.py` cùng các file nó cần chép vào `third_party/gsplat_examples` (`.vendor.json`).
- Vai trò: **nhóm C** (`PROTOCOL.md` mục "Nhóm C"). Trả lời câu "sao không dùng bù ngoại hình chung có sẵn trong 3DGS": bù ngoại hình học được cho mỗi ảnh đã thấy, nhưng không có mô hình đèn nên **không dự đoán được** ngoại hình ở pose mới.

## Cách chạy và chấm

- `simple_trainer.py default --data_dir <gói> --data_factor 1 --result_dir <thư mục> --use_bilateral_grid --disable_viewer`, đúng như dòng lệnh của gsplat. Mặc định còn lại: 30000 bước, SH bậc 3, chuẩn hóa hệ thế giới, grid 16 × 16 × 8, chiến lược densify mặc định.
- Gói COLMAP chung với DarkGS và SpotlessGS (`methods/darkgs/colmap_pkg.py`, thêm mô hình COLMAP **nhị phân** (`cameras.bin`, `images.bin`, `points3D.bin`), vì bộ đọc `pycolmap` mà gsplat dùng ưu tiên file nhị phân và phần đọc file text của nó còn mã Python 2, không chạy được): 400 ảnh lưới 0 tại pose tham chiếu, đám mây điểm từ độ sâu cảm biến (gsplat vốn khởi tạo từ điểm SfM).
- Frame chấm (lưới dịch 2, 7, 12, …) render tại pose tham chiếu, đưa vào hệ chuẩn hóa bằng chính `transform_cameras` của gsplat (adapter kiểm: pose ảnh train sau phép đổi trùng pose trong trainer). Adapter dựng lại `Runner` của gsplat với cùng cấu hình và nạp checkpoint cuối. **Hai bộ render**, cùng một lần train:
  - `renders/`: **chỉ Gaussian, không grid**, như `eval()` của gsplat (grid thuộc về ảnh train, frame chưa thấy không có grid riêng). `score_zip.py` chấm bộ này.
  - `renders_nngrid/`: Gaussian **cộng grid của ảnh train gần nhất** (với lưới dịch 2: frame `i − 2`), áp đúng như lúc train (`slice` với chỉ số ảnh đó). Đây là cách dùng bù ngoại hình có lợi nhất cho phương pháp: ngoại hình ở frame lân cận đã thấy. 5 frame ảnh chung (100, 500, …) là ảnh train nên dùng grid của chính nó. Chấm riêng (dưới đây).
- **Không tính** `cc_psnr` của gsplat: số đó hiệu chỉnh màu ảnh render theo ảnh thật (affine fit trên ảnh thật), tức dùng đáp án lúc chấm.

Chấm bộ thứ hai sau `score_zip.py` (đã giải nén vào `runs/<tên zip>/`), trong repo MonoGS:

```
E=baselines/runs/<tên zip>
python baselines/common/score.py --method bilagrid_nngrid --dataset <dataset> --given_poses --keyframes $E/keyframes.txt \
    --renders $E/renders_nngrid --timing $E/timing.json --out baselines/results/bilagrid_nngrid__office0_lit/<thời điểm> \
    --render_offset 2 --no_wandb
```

## Khác biệt (ghi trong `overrides.json`), vá trên bản chép trong thư mục làm việc

| Đổi | Lý do |
|---|---|
| `datasets/colmap.py`: train trên **mọi** ảnh | Gốc giữ lại mỗi ảnh thứ `test_every` (mặc định 8), **luôn gồm ảnh 0**, để tự đánh giá. Giao thức train trên đủ lưới 0 và chấm trên lưới dịch |
| Thêm `datasets/__init__.py` rỗng | `examples/datasets/` của gsplat là namespace package; trên Kaggle có gói `datasets` của Hugging Face, sẽ che mất nó khi import |
| `pycolmap` (bản fork rmbrualla, đúng commit gsplat ghim) vá cho numpy 2 lúc cài (`common/install_pycolmap_fork.sh`): `INVALID_POINT3D = np.uint64(-1)` thành `np.uint64(2**64 - 1)`, cùng giá trị | Kaggle (2026-11-18) dùng Python 3.13 với numpy 2: dòng gốc báo `OverflowError` ngay khi import. Ghim `numpy<2` như `examples/requirements.txt` của gsplat là không thể, vì numpy 1.x không có bản cho Python 3.13. Đã kiểm trong env Python 3.13 + numpy 2.5.5: gói nhị phân và parser của gsplat đọc ra pose, điểm, phép chuẩn hóa trùng numpy 1 (lệch ≤ 2e-15) |
| Một GPU: adapter đặt `CUDA_VISIBLE_DEVICES=0` nếu chưa đặt, và từ chối nếu biến này chỉ nhiều hơn một GPU | `cli()` của gsplat mở một process cho mỗi GPU nhìn thấy. Trên Kaggle T4 ×2, lần chạy 2026-11-18 vào chế độ phân tán và hỏng (`NameError: BilateralGrid`: process con tạo bằng `spawn` không chạy khối `__main__` của `simple_trainer.py`, nơi tên này được gán). Hướng dẫn của gsplat cũng chạy một GPU; nhiều GPU tương đương batch lớn hơn và phải đổi số bước |
| `simple_trainer.py`: checkpoint lưu thêm `bil_grids` | Bản gốc chỉ lưu Gaussian, nên không render được `renders_nngrid/` từ checkpoint. Không đổi việc train |

## Kỹ thuật

- gsplat 1.5.3 biên dịch CUDA lúc dùng lần đầu (JIT, cần `nvcc` và `ninja`). `probe.py` trong `setup_kaggle.sh` kích hoạt việc này ngay lúc setup; trên GTX 1650 mất khoảng 4 phút.
- Trainer gốc tự đánh giá ở các bước `eval_steps` (7000 và 30000) trên tập `val` của nó (mỗi ảnh thứ 8, sau bản vá đều là ảnh train, render không grid), rồi render một video quỹ đạo (cần `imageio[ffmpeg]`). Thời gian đó nằm trong `timing.json`; số của nó không dùng.
- Checkpoint cuối (Gaussian và grid) vào `run/` của zip làm bản đồ.

## Đã chạy

Chạy thử cục bộ (GTX 1650, 2026-11-17), 120 frame đầu của `office0 lit`, 24 ảnh train, 3000 bước (thay 30000), 127 s: 92 nghìn Gaussian; 24 frame lưới dịch: **không grid PSNR 19.30** / SSIM 0.852 / LPIPS 0.114; **grid ảnh train gần nhất 27.52** / 0.956 / 0.054. `eval()` của gsplat (ảnh train, không grid) ra 20.1, khớp bộ không grid: grid nuốt phần ánh sáng đèn, Gaussian một mình giữ màu tối. Chưa chạy Kaggle.
