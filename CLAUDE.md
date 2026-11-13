# baselines/: so sánh với các phương pháp SLAM khác

Ngữ cảnh cho mọi phiên làm việc trong thư mục này. Đọc cùng `../CLAUDE.md` (dự án chính) và `PROTOCOL.md` (cách chấm điểm).

## Mục đích

Báo cáo đồ án cần một khối "phương pháp khác" trong bảng kết quả chính, để trả lời ba câu hội đồng dễ hỏi:

1. MonoGS hỏng dưới đèn đồng hành (ATE 0.84 m trên `lit`, so với 1.0 cm trên `even`) là lỗi riêng của MonoGS, hay của cả họ SLAM quang trắc?
2. Tracking dựa trên đặc trưng hoặc hình học, vốn ít phụ thuộc độ sáng, có tránh được không? Và bản đồ của chúng có đúng màu không?
3. Các hệ 3DGS-SLAM đã có cơ chế chống thay đổi ánh sáng hoặc phơi sáng (học ngoại hình, tách albedo theo từng ảnh, trọng số bất định) có xử lý được đèn đồng hành không? Hay cần mô hình đèn theo vật lý như đồ án?

Thư mục này **chỉ chạy và chấm** các phương pháp khác trên đúng dữ liệu của đồ án, rồi ghi kết quả vào `RESULTS.md` cho phiên viết báo cáo. Không sửa code MonoGS của đồ án từ đây.

## Danh sách phương pháp (chốt với người dùng 2026-11-13)

Nguyên tắc chọn: ưu tiên **3DGS-SLAM và 3DGS** chạy được trên dữ liệu đồ án để đo số, nhất là phương pháp có cơ chế chống thay đổi ánh sáng. Phần tổng quan (chương 2) chỉ giữ chi tiết cho các bài được so hoặc định nghĩa bài toán, và thu gọn các bài xa (xem `../reports/CLAUDE.md`, mục 9).

**A. 3DGS-SLAM tổng quát** (đối chứng, không có cơ chế ánh sáng):

| Thư mục | Phương pháp | Câu | Tình trạng / ghi chú |
|---|---|---|---|
| (repo đồ án) | MonoGS gốc | 1 | cần chạy lại bằng code hiện tại, có `--record` |
| `methods/splatam/` | SplaTAM (CVPR 2024), quang trắc kèm độ sâu | 1 | ✅ `lit`/`even`, 1 run mỗi biến thể |
| `methods/gsicp/` | GS-ICP SLAM (ECCV 2024), tracking G-ICP hình học | 2 | ✅ `lit`/`even`, 1 run. **Cần thêm điều kiện độ sâu có nhiễu** (dưới đây) |
| `methods/photoslam/` | Photo-SLAM (CVPR 2024), tracking ORB-SLAM3 + bản đồ Gaussian | 2 | đang làm. **Thay ORB-SLAM3 trong bảng chính**: cùng tracking, có thêm ảnh |
| `methods/loopsplat/` | LoopSplat (3DV 2025), có đóng vòng | 1, và "đóng vòng có cứu được trôi không" | chưa làm |

**B. 3DGS-SLAM có cơ chế chống thay đổi ánh sáng hoặc phơi sáng** (đối thủ trực tiếp, câu 3). Trước hết kiểm code có công bố không và có chạy RGB-D không; không có code thì chỉ trích dẫn.

| Phương pháp | Cơ chế | Ghi chú |
|---|---|---|
| SEGS-SLAM (ICCV 2025, arXiv 2501.05242) | nhúng ngoại hình sinh từ pose camera (AfME), tracking ORB-SLAM3 | có RGB-D; repo có thể là `leaner-forever/SEGS-SLAM`, chưa xác nhận chính chủ. Ưu tiên cao nhất nhóm B |
| Taming the Light (arXiv 2511.22968) | tách màu thành albedo nhân hệ số chiếu sáng theo từng ảnh | gần ý tưởng đồ án nhất (nhưng học, không vật lý). Code chưa xác nhận; kiểm có RGB-D không |
| RoGER-SLAM (arXiv 2510.22600) | kết hợp ảnh, độ sâu, cạnh; tăng cường ảnh khi suy giảm nặng | đánh giá trên Replica; code chưa xác nhận |
| VarSplat (CVPR 2026, arXiv 2603.09673) | phương sai ngoại hình cho mỗi Gaussian làm trọng số tracking | RGB-D; có project page, repo chưa xác nhận. Tùy chọn |
| AERGS-SLAM (CVPR 2026), RawSLAM (arXiv 2609.20589) | mạng phơi sáng / ảnh HDR tuyến tính | **không chạy**: AERGS chỉ stereo, RawSLAM cần ảnh RAW. Chỉ trích dẫn |

**C. 3DGS ngoại tuyến xử lý ánh sáng** (cần pose cho trước, không có ATE; so PSNR trên `lit` và albedo với pass albedo như `tools/verify/eval_map.py`). Pose cấp vào: pose ước lượng của đồ án hoặc pose tham chiếu, ghi rõ trong kết quả. Cần thêm một mục giao thức riêng trong `PROTOCOL.md` trước khi chạy.

| Phương pháp | Ghi chú |
|---|---|
| 3DGS + bilateral grid / affine theo từng ảnh (gsplat) | bù ngoại hình phổ biến nhất; bilateral grid sửa được độ sáng thay đổi theo vùng trong ảnh. Trả lời "sao không dùng bù ngoại hình chung" |
| SpotlessGS | cùng bài toán, học đèn; code có ở `D:\CodeForYou\SpotlessGS` |
| DarkGS | cùng bài toán, hiệu chuẩn bằng bia; code có ở `D:\CodeForYou\darkgs` |

**Kết quả phụ (không vào bảng chính):** ORB-SLAM3 RGB-D (`methods/orbslam3/`, ✅ `lit` 3.55 cm, `even` 1.36 cm). Giữ trong `RESULTS.md` làm phương án dự phòng nếu Photo-SLAM không build được, và để đối chiếu ATE của Photo-SLAM (cùng tracking).

**Thí nghiệm GS-ICP với độ sâu có nhiễu** (ưu tiên cao): GS-ICP bỏ qua màu, nên trên depth Blender không nhiễu nó có ATE 0.032 cm, tốt hơn đồ án khoảng 40 lần. Cần chạy lại với depth có nhiễu kiểu Kinect v1, dùng đúng `utils/depth_noise.py` của repo đồ án (khối `Dataset.depth_noise`, cùng tham số và seed của thí nghiệm D72 và D75). Cách làm: một script trong repo đồ án xuất bộ dataset `lit` có `depth*.png` nhiễu (ảnh màu giữ nguyên), upload thành Kaggle dataset, chạy GS-ICP trên đó như một biến thể mới. Đồ án đã có các run nhiễu riêng để đặt cạnh (D75).

Không so ATE với DarkGS, SpotlessGS, SunFaded: chúng cần pose cho trước, không phải hệ SLAM (nhóm C chỉ so bản đồ).

## Quy tắc (giữ như dự án chính)

- **Không rò rỉ ground truth.** Đầu vào chỉ gồm `results/frame*.jpg` và `results/depth*.png` của thư mục dataset MonoGS. `traj.txt` chỉ dùng để chấm. Không đọc EXR, pass albedo hay pháp tuyến. Không khởi tạo bằng pose ground truth. Riêng pose frame đầu: MonoGS đặt nó bằng pose tham chiếu, và việc căn chỉnh SE(3) làm điều này không ảnh hưởng ATE. Phương pháp khác cứ khởi tạo theo cách của chúng.
- **Chấm bằng `common/score.py`**, không bằng script đánh giá riêng của phương pháp. Số phương pháp tự báo chỉ ghi vào cột ghi chú để đối chiếu.
- **Không gán cứng** nội tại camera hay thang độ sâu trong adapter. Nguồn là `configs/rgbd/replica/base_config.yaml` (`Dataset.Calibration`). Adapter chạy trên Kaggle không có repo MonoGS, nên đọc bằng `protocol.calibration()` từ `common/calibration.yaml`, là bản chép nguyên văn khối đó. `score_zip.py` từ chối run nếu `calibration.json` của run khác `base_config.yaml`. Đổi nội tại trong MonoGS thì chép lại file này, không sửa tay.
- **Giữ nguyên tham số mặc định** của từng phương pháp cho Replica, chỉ đổi những gì bắt buộc: độ phân giải 600×340, nội tại, thang độ sâu, đường dẫn. Mọi thay đổi khác ghi vào README của phương pháp, kèm lý do.
- Cùng loại GPU (Kaggle T4). FPS đo đúng như `PROTOCOL.md`.
- Mỗi cấu hình chạy ít nhất 2 lần nếu quota cho phép. Dao động của đồ án là 0.37 cm ATE và 0.8 dB PSNR (D139).
- Phụ thuộc của các phương pháp **không** vào `environment.yml` của MonoGS. Dùng môi trường hoặc notebook riêng.
- W&B: **không đẩy** kết quả baseline lên W&B (người dùng quyết định 2026-11-12). Chấm bằng `--no_wandb`; số liệu nằm ở `results/` và `RESULTS.md`. Cờ `--wandb` vẫn còn trong `score.py`; nếu dùng thì lấy API key từ Kaggle Secrets hay biến môi trường, **không** ghi key vào notebook hay vào file.

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

Phân công: **người dùng chạy trên Kaggle, tải về một file zip cho mỗi lần chạy, rồi đưa cho Claude chấm và phân tích cục bộ.** Notebook Kaggle chỉ clone repo `baselines`, không clone repo đồ án. Mã nguồn các phương pháp đã chép sẵn vào `third_party/` của repo, ghim commit trong `.vendor.json` của từng thư mục, theo quy tắc ở `third_party/README.md`: không có `.git` lồng, kiểm `.gitignore` của repo được chép.

1. **Một notebook chung cho mọi phương pháp:** `run_baseline.ipynb` ở gốc repo. Người dùng đặt `METHOD = "<name>"` và danh sách `RUNS`. Notebook gọi `common/kaggle_runner.py`: `setup(METHOD)`, rồi `run(...)` cho từng dataset (tuần tự, log vào `<work>/run.log`, có timeout), cuối cùng `collect(METHOD, OUT)` để kiểm zip và dọn thư mục làm việc. Mỗi phiên notebook chỉ chạy một phương pháp, vì gói của hai phương pháp có thể trùng tên.

   Thêm một phương pháp là thêm thư mục `methods/<name>/` (tên viết thường), không sửa notebook. Thư mục cần đúng ba file theo giao diện chung:
   - `kaggle.json`: khả năng của phương pháp, bắt buộc có khóa `"checkpoints": true|false`. Khóa tùy chọn `"setup_timeout_min"`: runner dùng giá trị này nếu nó lớn hơn `SETUP_TIMEOUT_MIN` của notebook (Photo-SLAM build OpenCV có CUDA, khoảng một giờ). Runner đọc file này. Nếu phương pháp không có checkpoint, runner tự chạy với `--checkpoint_every 0` dù notebook đặt `CKPT_EVERY` bao nhiêu (có in thông báo), và từ chối `resume`. Thông tin này nằm trong repo chứ không trong notebook, vì notebook trên Kaggle là bản chép, không tự cập nhật theo repo: một notebook cũ vẫn chạy được phương pháp thêm sau (bài học 2026-11-12, GS-ICP gặp `CKPT_EVERY = 100` của notebook cũ);
   - `setup_kaggle.sh`: cài phụ thuộc vào Python của notebook (biến `PYTHON`, do runner đặt) bằng `uv`, build extension cho GPU đang có (đọc compute capability, không gán cứng), gọi hai lần trong một phiên vẫn an toàn;
   - `adapter.py run --dataset <thư mục> --work <thư mục> --checkpoint_every <n> [--resume] [--no_map]`: mọi tham số khác của phương pháp có mặc định trỏ vào `third_party/`. Adapter phải báo lỗi với tùy chọn nó không làm được (ví dụ checkpoint), không được lặng lẽ bỏ qua.
   - Runner cho output hiện ngay trong notebook (dòng tiến trình lặp, như tqdm, được rút bớt theo `PROGRESS_S`; lỗi luôn hiện) và ghi đầy đủ vào log. Runner dừng cả nhóm tiến trình khi quá giờ (`SETUP_TIMEOUT_MIN`, `TIMEOUT_H`) hoặc khi **không in gì** trong `STALL_MIN` phút. Vì vậy adapter và `setup_kaggle.sh` phải in tiến trình thường xuyên hơn `STALL_MIN`, kể cả ở các bước im lặng như render hàng loạt hay nén zip.
   - Setup lỗi thì runner in lại 40 dòng cuối của log setup ngay trên thông báo lỗi, và `run()` từ chối chạy (trước khi chép dataset) cho tới khi `setup()` của phương pháp đó thành công trong cùng kernel (2026-11-13, sau lần Photo-SLAM chép hai dataset mất 10 phút rồi mới báo thiếu file build).
   - Trước mỗi run, runner chép `results/` của dataset ra `/tmp/baselines_data/<scene>/<variant>/` (không tính giờ) và đưa đường dẫn đó cho adapter (`stage_dataset`, quyết định 2026-11-13, `PROTOCOL.md` mục FPS). Adapter chỉ thấy `results/`: không có `traj.txt` hay pass nào khác. Vì vậy adapter không được đọc gì ngoài `results/` của `--dataset`.

   Mã của phương pháp lấy từ `third_party/<repo>`. Windows không phân biệt hoa thường, nên `methods/SplaTAM` và `methods/splatam` là **cùng một thư mục**: không clone repo phương pháp vào `methods/`, và kiểm `git status` rằng đường dẫn mới có chữ thường.
2. Adapter `methods/<name>/adapter.py` đọc dataset MonoGS, chạy phương pháp, rồi đóng **một zip** `<method>__<scene>_<variant>__<thời điểm bắt đầu>.zip` theo bố cục ở `PROTOCOL.md` (mục "Gói kết quả của một lần chạy"). Zip gồm:
   - quỹ đạo TUM với cột thời gian là chỉ số frame (`common/protocol.write_tum`);
   - `keyframes.txt`, nếu phương pháp có khái niệm keyframe;
   - render `<idx:06d>.npy` tại pose ước lượng, cho các frame `protocol.required_renders(N, keyframes)`. Tập này gồm các frame chấm chỉ số và 5 frame chung để so ảnh: 100, 500, 825, 1125, 1750, render kể cả khi là keyframe;
   - `figures/`, `timing.json`, `calibration.json`, `method.json`, `summary.json`;
   - `run/`: config, log, bản đồ cuối.
3. Chấm cục bộ trong repo và môi trường MonoGS (WSL: `~/bin/micromamba run -n MonoGS`). Lệnh `common/score_zip.py --zip <zip> --dataset <dataset cục bộ> --no_wandb` giải nén vào `runs/` (bị git bỏ qua), kiểm nội tại, cảnh và số frame, rồi chạy `score.py` vào `results/<method>__<scene>_<variant>/<thời điểm bắt đầu>/`. Bản ghi của run nằm ở `run/` trong thư mục đó, gồm cả sha256 của zip.
4. Thêm một dòng vào bảng chi tiết của `RESULTS.md`, cập nhật bảng tổng, ghi mọi khác biệt so với mặc định của phương pháp.

Với MonoGS của đồ án: `common/from_monogs_record.py --record record/runs/<run> --figures --dataset <...>` ghi ra quỹ đạo, keyframe và ảnh 5 frame chung từ bản đồ trong record (`--map final` hoặc `final_refined`), sau đó chạy `score.py --no_render`. PSNR, SSIM, LPIPS lấy từ log của MonoGS. Cách này cho ATE trên mọi frame theo cùng giao thức.

## Đã kiểm chứng (2026-11-12)

- `score.py` tính lại ATE trên keyframe của một run MonoGS (`cell0p5`, `lit`) ra 0.010947566 m, khớp số MonoGS log (0.010947593) tới 7 chữ số, đúng độ chính xác của `frames_final.csv`.
- Ảnh 5 frame chung: `from_monogs_record.py --figures` đã render được từ record (trước và sau refine), `compare_figures.py` ghép đúng, nhãn tiếng Việt có dấu hiển thị đúng.
- Phần chấm ảnh: render giả bằng ảnh quan sát cộng nhiễu σ = 0.01 cho PSNR 40.17 dB, đúng mức lý thuyết (khoảng 40 dB). Chấm 357 frame, cùng tập frame `eval_rendering` dùng.

## Đưa thư mục này lên Kaggle

`baselines/` nằm trong `.gitignore` của repo chính và là repo git riêng, <https://github.com/Trung4n/baselines>. Trên Kaggle chỉ clone repo này, đặt ngoài `/kaggle/working`, để output chỉ còn các zip:

```
!git clone --depth 1 https://github.com/Trung4n/baselines /tmp/code/baselines
```

Cài phụ thuộc bằng `uv` như `run-monogs` (`pip install uv`, rồi `uv pip install --system --python {sys.executable} --break-system-packages ...`).

Phần chạy trên Kaggle (adapter, cùng `protocol.calibration`, `required_renders`, `write_tum`, `to_png8`, `load_render`) không cần repo MonoGS. Vì vậy `common/protocol.py` không đưa repo MonoGS vào `sys.path` lúc import; gói `utils` của MonoGS cũng trùng tên với gói của SplaTAM. Chỉ `ate()` và `RenderScorer` gọi `monogs_on_path()`.

Phần chấm (`score.py`, `score_zip.py`, `from_monogs_record.py`) import code đánh giá của MonoGS (`utils/eval_utils.py`, `gaussian_splatting/utils/*`). Do đó chỉ chạy được khi `baselines/` nằm **bên trong** repo MonoGS, trong môi trường MonoGS (có `evo` và `torchmetrics`): đó là máy cục bộ.

## Tình trạng

- [x] Giao thức, `common/`, `RESULTS.md` cùng số của đồ án
- [x] Đóng gói zip và `score_zip.py` (2026-11-13). Đã thử với một run giả: quỹ đạo tham chiếu cộng lệch đã biết, render bằng ảnh quan sát cộng nhiễu σ = 0.01. Kết quả PSNR 40.17 dB, ATE đúng như tính tay. Zip giả 402 MiB, phần lớn là render; render thật nén tốt hơn ảnh nhiễu.
- [x] Notebook chung `run_baseline.ipynb` và `common/kaggle_runner.py` (2026-11-13). Đã thử với một phương pháp giả: run xong ra zip, run lỗi, run quá giờ, chạy tiếp thì log ghi nối, `collect` giữ thư mục của run khác.
- [x] SplaTAM `office0` `lit` và `even`, mỗi biến thể một run (2026-11-12). Đã chấm, kết quả và phân tích ở `RESULTS.md`: `lit` ATE 62.6 cm, `even` 0.80 cm. Hai run chạy ở hai phiên Kaggle riêng, nên FPS dùng được. Chạy song song nhiều phiên riêng là cách hợp lệ để tiết kiệm thời gian. Còn mở: run thứ hai cho mỗi biến thể.
- [x] ORB-SLAM3 RGB-D `office0` `lit` và `even`, mỗi biến thể một run (2026-11-12): ATE 3.55 / 1.36 cm, 1 bản đồ, không mất frame; FPS bị chi phối bởi việc đọc ảnh từ `/kaggle/input` (riêng tracking hơn 50 FPS). Phân tích ở `RESULTS.md`. Còn mở: run thứ hai.
- [ ] Photo-SLAM RGB-D (người dùng thêm, 2026-11-13): mã chép vào `third_party/Photo-SLAM`; `photoslam_monogs.cpp`, `adapter.py`, `setup_kaggle.sh` (OpenCV 4.10 có CUDA, build và lưu cache; LibTorch ≤ 2.1.2), `kaggle.json` đã viết. Phần Python đã kiểm cục bộ; **chưa build, chưa chạy**. Còn mở: cách chấm khi một frame chấm ảnh không có pose (`methods/photoslam/README.md`).
- [ ] LoopSplat (người dùng thêm, 2026-11-13): mã chép vào `third_party/LoopSplat` (kèm `hloc` và 3 extension trong `_deps/`); `adapter.py`, `faiss_shim.py`, `setup_kaggle.sh`, `kaggle.json` đã viết; phần không cần build đã kiểm cục bộ; **chưa chạy**. LoopSplat mapping mọi frame chia hết cho 5, nên tập chấm ảnh thường rỗng. Người dùng quyết (2026-11-13) chấm trên **tập dịch lưới** (frame 2, 7, 12, …): `method.json` khai báo `heldout_offset`, `score_zip.py` chuyển thành `score.py --render_offset`, quy tắc ở `PROTOCOL.md`; đã thử bằng zip giả.
- [x] GS-ICP SLAM `office0` `lit` và `even`, mỗi biến thể một run (2026-11-12): ATE 0.032 / 0.041 cm (depth không nhiễu), PSNR `lit` 16.7 so với `even` 44.3 (ánh sáng nung vào bản đồ). Phân tích ở `RESULTS.md`. Còn mở: run thứ hai; thí nghiệm depth có nhiễu. Đóng gói zip dùng `common/runzip.py`, phần chung cho các adapter mới (SplaTAM còn bản riêng, cùng bố cục).
- [ ] MonoGS gốc chạy lại bằng code hiện tại (`lit`, `even`, có `--record`), chấm bằng `from_monogs_record.py`
- [ ] GS-ICP với depth có nhiễu (script xuất dataset nhiễu: chưa viết, làm ở repo đồ án)
- [ ] LoopSplat
- [ ] Nhóm B: kiểm code SEGS-SLAM, Taming the Light, RoGER-SLAM, VarSplat; chạy những cái có code và hỗ trợ RGB-D
- [ ] Nhóm C: thêm mục giao thức cho phương pháp cần pose cho trước; 3DGS + bilateral grid, SpotlessGS, DarkGS
- [ ] Bảng chính: ORB-SLAM3 rời bảng khi Photo-SLAM có số (giữ trong `RESULTS.md` làm kết quả phụ)
