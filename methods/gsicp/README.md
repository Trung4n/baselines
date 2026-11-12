# GS-ICP SLAM

- Bài báo: RGBD GS-ICP SLAM (Ha et al., ECCV 2024). Thêm vào `refs.bib` khi dùng.
- Mã nguồn: chép sẵn trong repo, `third_party/GS_ICP_SLAM` (Lab-of-AI-and-Robotics/GS_ICP_SLAM @ `5f996a872a979406b270fe0ee3b0a8f25c5e9ae3`, 2026-06-20), kèm 3 submodule ở commit repo ghim: rasterizer `95dbb69d`, `fast_gicp` (nhánh `gs_icp_slam`) `e2954b1a`, `simple-knn` `44f76429`. Những gì khác so với commit nằm trong `third_party/GS_ICP_SLAM/.vendor.json`.
- Vai trò trong so sánh: tracking thuần hình học, bằng Generalized-ICP trên đám mây điểm lấy từ depth. Màu không đi vào tracking; bản đồ Gaussian chỉ để kết xuất. Đây là đối chứng bên ngoài cho câu hỏi "chỉ dùng depth có đủ không".
- Lưu ý khi diễn giải: depth Blender không có nhiễu, nên dữ liệu mô phỏng có lợi cho tracking hình học. Cần nói rõ điều này khi so (`CLAUDE.md` mục 3).

## File

| File | Việc |
|---|---|
| `adapter.py` | `run`: dàn dữ liệu theo cấu trúc GS-ICP cần, ghi `caminfo.txt`, đọc tham số Replica từ `replica.sh`, chạy `gs_icp_slam.py` với hai lớp con ở `hooks.py`, xuất `<work>/export/` rồi đóng zip (`common/runzip.py`) |
| `hooks.py` | `ExportTracker`: đo thời gian (nạp ảnh và vòng tracking tách riêng), bỏ ATE tự tính của GS-ICP. `ExportMapper`: thay bước đánh giá cuối của GS-ICP bằng việc render bản đồ cuối tại pose cuối cho các frame `protocol.required_renders`, cùng cách `calc_2d_metric` render, và ghi danh sách keyframe |
| `setup_kaggle.sh` | `apt` cài PCL và Eigen; `uv` cài phụ thuộc; build rasterizer, simple-knn và `fast_gicp` (pygicp) trên bản chép, cho đúng GPU đang có |

GS-ICP chạy tracking và mapping thành hai tiến trình con theo kiểu `spawn`. Vì vậy không vá lúc chạy được như với SplaTAM. Adapter thay hai lớp `Tracker` và `Mapper` trong module `gs_icp_slam` bằng lớp con ở `hooks.py`; các tiến trình con import lại module đó qua `sys.path` mà `spawn` truyền xuống.

## Cách chạy

1. **Kaggle**, GPU T4, bật Internet: notebook chung `run_baseline.ipynb`, đặt `METHOD = "gsicp"`. GS-ICP không có checkpoint (`kaggle.json`: `"checkpoints": false`), nên runner tự chạy với `--checkpoint_every 0` dù notebook đặt `CKPT_EVERY` bao nhiêu, có in thông báo. Bản thân adapter vẫn từ chối mọi giá trị khác 0 và từ chối `--resume`. Run lỗi thì xóa thư mục làm việc rồi chạy lại. Một run chỉ vài phút: tracking bị giới hạn 30 FPS, cộng thời gian nạp ảnh và xuất kết quả.
2. Mỗi run cho `/kaggle/working/gsicp__<scene>_<variant>__<thời điểm bắt đầu>.zip`. Bản đồ cuối (`run/scene.ply`) có trong zip; `NO_MAP = True` bỏ nó ra.
3. Tải zip về, chấm cục bộ:

```
python baselines/common/score_zip.py --zip gsicp__office0_lit__<...>.zip \
    --dataset /mnt/d/CodeForYou/Replica/04_datasets/monogs/office0/lit --no_wandb
```

## Khác biệt so với mặc định của GS-ICP cho Replica

Gốc: `gs_icp_slam.py`, tức chế độ "Limited to 30 FPS", chế độ đầu tiên trong mục Run của README và là chế độ `replica.sh` dùng. Tham số lấy từ `replica.sh`, đọc thẳng từ file đó: `keyframe_th` 0.7, `knn_maxd` 99999, `overlapped_th` 5e-4, `max_correspondence_distance` 0.02, `trackable_opacity_th` 0.05, `overlapped_th2` 5e-5, `--save_results`. Các tham số còn lại giữ mặc định trong `arguments/SLAMParameters`. Mọi thay đổi ghi trong `overrides.json` và `params.json` (thư mục `run/` của zip).

| Đổi | Lý do |
|---|---|
| Camera: 600×340, `fx`, `fy`, `cx`, `cy`, `depth_scale` từ `protocol.calibration()`, ghi vào bản chép `configs/Replica/caminfo.txt`; `depth_trunc` 12.0 và loại dataset giữ nguyên | Bắt buộc: dataset của đồ án là Replica giảm nửa độ phân giải |
| `downsample_rate` 10 → **5** | Đây là bước nhảy pixel khi lấy điểm cho GICP và cho Gaussian mới. Giá trị 10 được chọn cho ảnh rộng 1200 px; giữ 10 ở 600 px thì số điểm mỗi frame giảm 4 lần (2100 so với 8280). Adapter đổi theo tỉ lệ chiều rộng: 10 × 600 / 1200 = 5, cho đúng 8280 điểm như cấu hình gốc (đã kiểm). Đây là lựa chọn của adapter; cần nêu trong báo cáo |
| Liên kết từng file `images/frame*.jpg`, `depth_images/depth*.png` → `results/` | GS-ICP đòi cấu trúc thư mục này (README của nó); ảnh không đổi |
| `traj.txt` là ma trận **đơn vị** | GS-ICP lấy pose frame đầu từ `traj.txt` và in một ATE với nó. Với ma trận đơn vị, không pose tham chiếu nào lọt vào run, và pose đầu thành ma trận đơn vị, được căn chỉnh SE(3) khi chấm hấp thụ. ATE GS-ICP tự in ra là `nan` |
| Bỏ đánh giá cuối của GS-ICP (PSNR, SSIM, LPIPS trên mọi frame) | Việc chấm là của `score.py`. Thay bằng bước xuất ở `hooks.ExportMapper`, render đúng cách GS-ICP render |
| Bỏ `opencv-contrib-python` | Chỉ dùng `cv2.imread`, `cvtColor`, `resize`, đều có trong `opencv-python` của Kaggle |
| Build: `#include <cstdint>` cho rasterizer, `#include <cfloat>` cho simple-knn; `fast_gicp` dùng pybind11 mới (`find_package`) thay bản 2.6 đi kèm | Chỉ sửa lỗi build với GCC, CUDA, Python ≥ 3.11 và CMake mới; không đổi thuật toán. Làm trên bản chép trong `/tmp`, mã trong `third_party/` giữ đúng commit |

## Điểm cần nói khi diễn giải

- **Tracking không dùng màu.** Nếu GS-ICP chạy tốt trên `lit`, điều đó cho thấy hình học đủ để bám pose trên dữ liệu này. Khi đó cần nói ba điều:
  - depth mô phỏng không có nhiễu;
  - bản đồ màu của GS-ICP vẫn nung ánh sáng của đèn, nên PSNR so với ảnh `lit` không đo việc tách albedo;
  - mô hình của đồ án nhắm tới tracking quang trắc, và lợi ích của nó nằm ở bản đồ albedo và ở thiết lập camera đơn.
- **Mapping chạy song song với tracking và dừng khi tracking xong.** Chất lượng bản đồ vì vậy phụ thuộc vào số vòng mapping kịp chạy, tức là phụ thuộc tốc độ máy. Số vòng này ghi trong `method.json` (`mapping_iterations`).
- **Keyframe:** gồm keyframe tracking (khi tỉ lệ điểm trùng < `keyframe_th`) và keyframe mapping (mỗi 10 frame kể từ keyframe tracking gần nhất). Mapping chỉ tối ưu trên các keyframe này, nên các frame chấm ảnh là frame chưa dùng để tối ưu, giống MonoGS và khác SplaTAM.
- **FPS** tính cả thời gian đọc ảnh, cùng cách MonoGS tính `Total FPS` (`PROTOCOL.md`): GS-ICP đọc hết ảnh trước vòng lặp, MonoGS đọc từng frame bên trong khoảng được đo. `timing.json` ghi thêm `loop_s` và `fps_loop_only` ("System FPS" của GS-ICP, bị giới hạn ở 30; GICP chạy trên CPU, 4 nhân trên Kaggle).
- Chỉ dùng một GPU (`cuda:0`) cho cả hai tiến trình.

## Kỹ thuật

- Đã kiểm cục bộ (WSL, môi trường MonoGS, không build GS-ICP):
  - dàn dữ liệu (2000 cặp liên kết, đúng cách GS-ICP suy tên file depth từ tên ảnh), `caminfo.txt`, tham số đọc từ `replica.sh`;
  - `downsample_rate` 5 cho đúng 8280 điểm mỗi frame như cấu hình gốc;
  - adapter từ chối checkpoint;
  - `export` → `runzip.package` → `score_zip.py` trên một đầu ra giả của hooks: PSNR 40.17 dB với nhiễu σ = 0.01, ATE đúng như tính tay.

  **Chưa chạy GS-ICP thật**: phần build (PCL, pygicp) và `hooks.py` chỉ kiểm được trên Kaggle.

## Đã chạy

| Ngày | Biến thể | ATE all (cm) | PSNR / SSIM / LPIPS | FPS | Kết quả |
|---|---|---|---|---|---|
| 2026-11-12 | office0 / lit | 0.032 | 16.71 / 0.758 / 0.289 | 5.14 (vòng tracking 10.77) | `results/gsicp__office0_lit/2026-11-12-14-04-31` |
| 2026-11-12 | office0 / even | 0.041 | 44.25 / 0.987 / 0.013 | 3.13 (vòng tracking 10.57) | `results/gsicp__office0_even/2026-11-12-14-05-04` |

Python 3.13, torch 2.11.0+cu130, PCL 1.14 trên Kaggle. Build và `hooks.py` chạy đúng ngay lần đầu, mỗi run một phiên. Phân tích ở `RESULTS.md`, mục "GS-ICP SLAM trên `office0`".
