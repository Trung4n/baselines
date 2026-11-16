# Kết quả so sánh với các phương pháp khác

File này lưu số liệu cho chương thực nghiệm của báo cáo. Mỗi lần chấm xong một run, thêm một dòng vào bảng chi tiết và cập nhật bảng tổng. Mọi số đều chấm theo `PROTOCOL.md`.

## Bảng tổng (điền dần)

ATE tính bằng cm. `all` là ATE trên mọi frame, số chính để so giữa các phương pháp. **FPS của các run trước 2026-11-13 (SplaTAM, GS-ICP, ORB-SLAM3) đo khi đọc ảnh thẳng từ `/kaggle/input`; Photo-SLAM đọc từ ổ cục bộ.** Với GS-ICP và ORB-SLAM3, con số đó chủ yếu phản ánh tốc độ ổ đĩa. Từ 2026-11-13, runner đọc từ bản chép trên ổ cục bộ (`PROTOCOL.md`, mục FPS), và các run chạy lại sẽ thay số FPS ở bảng này. `kf` là ATE trên keyframe theo cách MonoGS báo. PSNR, SSIM, LPIPS lấy trên bản đồ cuối, không refine. FPS đo trên Kaggle T4.

| Phương pháp | Loại tracking | `office0` `even`: ATE all | `office0` `lit`: ATE all / kf | `lit`: PSNR / SSIM / LPIPS | FPS | Số run |
|---|---|---|---|---|---|---|
| **Đồ án** (mô hình chính, lưới 0.5 m, `_parallel_fused`) | quang trắc, mô hình đèn | chưa chạy | 1.45, 1.10 / 1.46, 1.09 | 35.84 / 0.958 / 0.043, 36.28 / 0.963 / 0.041 (trước refine) | 3.18, 3.05 (T4 x2) | 2 |
| MonoGS gốc (`configs/rgbd/replica/office0.yaml`) | quang trắc | kf 1.13 (D133) | kf 84 (CLAUDE.md, code cũ) | – | 2.28 (`even`) | 1 |
| SplaTAM | quang trắc + độ sâu | 0.80 | 62.6 / 63.0 | 13.59 / 0.512 / 0.591 (`even`: 38.92 / 0.969 / 0.055) | 0.24 `lit`, 0.20 `even` (T4, 1 GPU) | 1 |
| Photo-SLAM | đặc trưng (ORB-SLAM3) + bản đồ Gaussian | 1.66 | 7.93 / 6.09 | 17.44 / 0.696 / 0.401 (`even`: 35.51 / 0.939 / 0.089) | 25.6 `lit`, 26.9 `even` (gồm đọc ảnh từ ổ cục bộ; riêng tracking 29.7, 31.4; T4, 1 GPU, tracking trên CPU) | 1 |
| GS-ICP SLAM | hình học (G-ICP) | 0.041 | 0.032 / 0.032 | 16.71 / 0.758 / 0.289 (`even`: 44.25 / 0.987 / 0.013) | 5.14 `lit`, 3.13 `even` (tính cả đọc ảnh; riêng vòng tracking 10.8, 10.6; T4, 1 GPU, GICP trên CPU) | 1 |
| LoopSplat | quang trắc + độ sâu, submap, có đóng vòng | 0.72 | 105.0 / 106.1 | 20.71 / 0.869 / 0.144, trên tập dịch lưới (`even`: 45.09 / 0.987 / 0.010) | 0.53 `lit`, 0.50 `even` (T4, 1 GPU) | 1 |
| SEGS-SLAM | đặc trưng (ORB-SLAM3) + bản đồ anchor, nhúng ngoại hình theo pose | 2.78 | 3.06 / 3.30 (có 1 loop closure) | 21.57 / 0.815 / 0.220, mapper 30000 vòng (`even`: 38.68 / 0.964 / 0.041) | 25.2 `lit`, 24.0 `even` (gồm đọc ảnh từ ổ cục bộ; riêng tracking 28.8, 27.1; T4, 1 GPU, tracking trên CPU) | 1 |

**Kết quả phụ** (không vào bảng chính; Photo-SLAM thay chỗ, cùng tracking):

| Phương pháp | Loại tracking | `office0` `even`: ATE all | `office0` `lit`: ATE all / kf | FPS | Số run |
|---|---|---|---|---|---|
| ORB-SLAM3 RGB-D | đặc trưng | 1.36 | 3.55 / 3.94 (có 1 loop closure) | 5.51 `lit`, 5.56 `even` (gồm đọc ảnh từ `/kaggle/input`; riêng tracking 54, 51; CPU 4 nhân) | 1 |

**Cần chạy lại cho đồng nhất:**
- MonoGS gốc trên `lit` và `even` bằng code hiện tại, có `--record`, để có ATE all, PSNR và FPS cùng giao thức.
- Mô hình chính trên `even`.
- Mô hình chính đo lại bằng `from_monogs_record.py` để lấy ATE all; hai run `lit` ở trên đã làm.

## Bảng chi tiết từng run

| Ngày | Phương pháp | Cảnh / biến thể | ATE all (cm) | ATE kf (cm) | PSNR | SSIM | LPIPS | FPS | GPU | Thư mục kết quả / ghi chú |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-11-11 | đồ án, 0.5 m | office0 / lit | 1.445 | 1.463 | 35.84 (sau refine 36.28) | 0.9579 (0.9617) | 0.0435 (0.0434) | 3.18 | T4 x2 | record `..._cell0p5_parallel_fused__office0_lit__2026-11-11-13-06-50` |
| 2026-11-11 | đồ án, 0.5 m | office0 / lit | 1.104 | 1.095 | 36.28 (sau refine 37.14) | 0.9630 (0.9678) | 0.0410 (0.0377) | 3.05 | T4 x2 | record `...__2026-11-11-16-55-52` |
| 2026-11-11 | đồ án, 0.5 m | office0 / ambvar_hard | 1.635 | 1.627 | 34.51 (sau refine 34.83) | 0.9442 (0.9494) | 0.0493 (0.0489) | 2.86 | T4 x2 | record `...__ambvar_hard_ambvar_hard__2026-11-11-12-43-01` |
| 2026-11-11 | đồ án, 0.5 m | office0 / ambvar_hard | 1.354 | 1.369 | 34.81 (sau refine 35.26) | 0.9509 (0.9561) | 0.0477 (0.0443) | 3.14 | T4 x2 | record `...__ambvar_hard_ambvar_hard__2026-11-11-16-49-09` |
| 2026-11-12 | SplaTAM `da6bbcd2` | office0 / lit | 62.60 | 62.96 | 13.59 | 0.5124 | 0.5906 | 0.237 | T4, 1 GPU | `results/splatam__office0_lit/2026-11-12-09-12-50`; 877k Gaussian; gãy ở frame 280–300 (xem ghi chú) |
| 2026-11-12 | SplaTAM `da6bbcd2` | office0 / even | 0.798 | 0.796 | 38.92 | 0.9690 | 0.0547 | 0.198 | T4, 1 GPU | `results/splatam__office0_even/2026-11-12-09-12-29`; 1.48M Gaussian |
| 2026-11-12 | GS-ICP `5f996a87` | office0 / lit | 0.0325 | 0.0316 | 16.71 | 0.7580 | 0.2892 | 5.14 | T4, 1 GPU | `results/gsicp__office0_lit/2026-11-12-14-04-31`; 1.55M Gaussian, 210 keyframe, 7979 vòng mapping; FPS = 2000 / (đọc ảnh 203.5 s + vòng tracking 185.8 s); riêng vòng tracking 10.77; `metrics.json` của run này còn ghi 10.77 (cách tính cũ) |
| 2026-11-12 | GS-ICP `5f996a87` | office0 / even | 0.0413 | 0.0413 | 44.25 | 0.9872 | 0.0133 | 3.13 | T4, 1 GPU | `results/gsicp__office0_even/2026-11-12-14-05-04`; 1.48M Gaussian, 209 keyframe, 8626 vòng mapping; FPS = 2000 / (đọc ảnh 450.7 s + vòng tracking 189.2 s); riêng vòng tracking 10.57; `metrics.json` của run này còn ghi 10.57 (cách tính cũ) |
| 2026-11-12 | ORB-SLAM3 `4452a3c4` | office0 / lit | 3.548 | 3.937 | – | – | – | 5.51 | CPU 4 nhân | `results/orbslam3__office0_lit/2026-11-12-17-53-07`; 1 bản đồ, 2000/2000 frame OK, 162 keyframe, 1 loop closure (~frame 1700); tracking trung bình 18.4 ms/frame |
| 2026-11-12 | ORB-SLAM3 `4452a3c4` | office0 / even | 1.365 | 1.275 | – | – | – | 5.56 | CPU 4 nhân | `results/orbslam3__office0_even/2026-11-12-17-53-29`; 1 bản đồ, 2000/2000 frame OK, 91 keyframe, không loop closure; tracking trung bình 19.4 ms/frame |
| 2026-11-15 | Photo-SLAM `f8bfb2f0` | office0 / lit | 7.932 | 6.094 | 17.44 | 0.6959 | 0.4008 | 25.6 | T4, 1 GPU | `results/photoslam__office0_lit/2026-11-15-14-36-51`; 1 bản đồ, 2000/2000 frame OK, 169 keyframe, **không** loop closure; 140.8k Gaussian, 5883 vòng mapper; 366 frame chấm ảnh; riêng tracking 29.7 FPS; LibTorch 2.1.2+cu121, CUDA 12 cài cạnh CUDA 13 |
| 2026-11-15 | Photo-SLAM `f8bfb2f0` | office0 / even | 1.662 | 1.473 | 35.51 | 0.9392 | 0.0887 | 26.9 | T4, 1 GPU | `results/photoslam__office0_even/2026-11-15-14-43-30`; 1 bản đồ, 2000/2000 frame OK, 105 keyframe, không loop closure; 54.9k Gaussian, 7281 vòng mapper; 376 frame chấm ảnh; riêng tracking 31.4 FPS |
| 2026-11-16 | LoopSplat `676e18f9` | office0 / lit | 105.02 | 106.14 | 20.71 | 0.8692 | 0.1442 | 0.526 | T4, 1 GPU | `results/loopsplat__office0_lit/2026-11-16-07-47-36`; 40 submap, 429 keyframe (frame mapping); gãy ở frame 257 (bước 20.7 cm trong một frame); 1 lần tối ưu pose graph (frame 1102, 2 cạnh vòng); 391 frame chấm ảnh trên tập dịch lưới (2 mod 5), mỗi frame render bằng submap của nó |
| 2026-11-16 | LoopSplat `676e18f9` | office0 / even | 0.724 | 0.692 | 45.09 | 0.9869 | 0.0105 | 0.504 | T4, 1 GPU | `results/loopsplat__office0_even/2026-11-16-08-55-23`; 28 submap, 421 keyframe; 4 lần tối ưu pose graph (frame 701, 1103, 1769, 1838); 397 frame chấm ảnh trên tập dịch lưới |
| 2026-11-16 | SEGS-SLAM `2110994a` | office0 / lit | 3.058 | 3.297 | 21.57 | 0.8153 | 0.2201 | 25.2 | T4, 1 GPU | `results/segsslam__office0_lit/2026-11-16-09-58-39`; 1 bản đồ, 2000/2000 frame OK, 158 keyframe, 1 loop closure (frame 1100–1200); 93.5k anchor, mapper 2549 vòng ở frame cuối rồi train tiếp tới 30000 (902 s, không tính vào FPS); 366 frame chấm ảnh; riêng tracking 28.8 FPS |
| 2026-11-16 | SEGS-SLAM `2110994a` | office0 / even | 2.776 | 2.681 | 38.68 | 0.9636 | 0.0405 | 24.0 | T4, 1 GPU | `results/segsslam__office0_even/2026-11-16-10-19-28`; 1 bản đồ, 2000/2000 frame OK, 100 keyframe, không loop closure; 57.1k anchor, mapper 3008 vòng ở frame cuối rồi tới 30000 (729 s); 385 frame chấm ảnh; riêng tracking 27.1 FPS |

Trong ngoặc là số sau color refinement. MonoGS log cả hai dòng `mean psnr` trước và sau refine.

## Ảnh so sánh

Năm frame chung: 100, 500, 825, 1125, 1750 (`PROTOCOL.md`). Ảnh của từng run nằm ở `results/<run>/figures/`. Bảng ghép cho báo cáo ghi ở đây khi đã có đủ phương pháp:

| Cảnh / biến thể | File ghép | Cột |
|---|---|---|
| office0 / lit | (chưa có) | Quan sát, Đồ án, MonoGS gốc, SplaTAM, GS-ICP |
| office0 / lit, chỉ SplaTAM | `results/figures_splatam_office0_lit.png` | Quan sát, SplaTAM |
| office0 / even, chỉ SplaTAM | `results/figures_splatam_office0_even.png` | Quan sát, SplaTAM |
| office0 / lit | `results/figures_office0_lit.png` | Quan sát, SplaTAM, GS-ICP, Photo-SLAM, SEGS-SLAM, LoopSplat (2026-11-16) |
| office0 / even | `results/figures_office0_even.png` | Quan sát, SplaTAM, GS-ICP, Photo-SLAM, SEGS-SLAM, LoopSplat (2026-11-16) |

## Ghi chú cho báo cáo

- Mỗi hệ chọn keyframe khác nhau, nên so ATE trên mọi frame. Trên các run của đồ án, ATE trên mọi frame chênh dưới 0.02 cm so với ATE trên keyframe.
- Dao động giữa hai lần chạy của đồ án: tới khoảng 0.37 cm ATE và 0.8 dB PSNR (D139). Muốn kết luận một khác biệt thì nó phải lớn hơn mức này.
- SplaTAM tối ưu bản đồ ở mọi frame (`map_every` 1), nên các frame chấm ảnh của nó đã từng là đích tối ưu. PSNR, SSIM, LPIPS của SplaTAM vì vậy thiên về phía có lợi so với MonoGS, vốn chỉ tối ưu trên keyframe (`methods/splatam/README.md`).
- Depth Blender không có nhiễu (`CLAUDE.md` mục 3). Điều này có lợi cho các phương pháp tracking hình học như GS-ICP, cần nói rõ khi so.

### SplaTAM trên `office0` (2026-11-12, mỗi biến thể một run)

- **Trả lời câu 1 của `CLAUDE.md`:** lỗi thuộc về cả họ SLAM quang trắc, không riêng MonoGS. SplaTAM tracking bằng ảnh cộng độ sâu, cũng trên Gaussian. Trên `even` nó chạy tốt (ATE 0.80 cm, PSNR 38.9), nhưng trên `lit` hỏng (ATE 62.6 cm, PSNR 13.6). Mức hỏng cùng cỡ với MonoGS gốc (84 cm, code cũ). Cùng dữ liệu, cùng depth cảm biến, mô hình chính của đồ án ở mức 1.10–1.45 cm.
- **Run `even` là phép kiểm tra cả quy trình:** adapter, pose, render và cách chấm đều đúng. ATE 0.80 cm, thấp hơn MonoGS gốc trên `even` (kf 1.13 cm, D133). Ảnh 5 frame chung gần như trùng ảnh quan sát.
- **Kiểu hỏng trên `lit`: gãy đột ngột, không trôi dần.** Căn chỉnh trên 100 frame đầu thì sai số còn 1.3 cm ở frame 200, vượt 2 cm ở frame 237, 10 cm ở frame 293, 26 cm ở frame 300. Từ đó quỹ đạo không quay lại: RPE trung vị mỗi 10 frame là 4.4 cm (trên `even` là 0.12 cm), và quãng đường ước lượng 28.7 m so với 18.2 m thật.
- **Chỗ gãy:** từ frame 260 đến 300, camera tiến lại tấm bảng trắng lớn (độ sâu trung vị giảm từ 2.0 xuống 1.65 m). Đĩa sáng của đèn phủ gần hết bảng: pixel bão hòa tăng từ 4% (frame 200) lên 24% (frame 290), pixel tối dưới 0.05 khoảng 35%. Mặt bảng không có hoa văn, nên vật nổi bật nhất trong ảnh là chính vệt sáng, mà vệt sáng này đi theo camera. Trên `even`, cùng đoạn này không có vấn đề. Đây là giải thích hợp lý nhất từ dữ liệu, chưa kiểm bằng thí nghiệm riêng.
- **Bản đồ `lit` hỏng toàn cục.** Ảnh frame 100 render từ bản đồ cuối cũng vỡ, dù pose frame 100 còn đúng (0.4 cm): Gaussian đặt sai sau frame 300 đè lên vùng cũ. Vệt sáng của đèn bị nung vào màu bản đồ. PSNR 13.6 vì vậy đo cả lỗi pose lẫn lỗi bản đồ, không chỉ việc thiếu mô hình đèn.
- **FPS:** 0.20–0.24 trên một T4, chậm hơn mô hình chính khoảng 13–16 lần (3.05–3.18 trên T4 x2). `lit` nhanh hơn `even` vì bản đồ nhỏ hơn (877k so với 1.48M Gaussian). Hai run bắt đầu cách nhau 21 giây nhưng chạy trên **hai phiên Kaggle riêng**, mỗi phiên một máy (người dùng xác nhận, 2026-11-12). Không có tranh chấp tài nguyên, nên FPS dùng được.
- SplaTAM tự in "ATE RMSE" trong log (1.17 m) là so với quỹ đạo đơn vị giả mà adapter đưa vào, không có nghĩa. Điều này cũng xác nhận run không thấy pose tham chiếu.

### GS-ICP SLAM trên `office0` (2026-11-12, mỗi biến thể một run)

- **Trả lời câu 2 của `CLAUDE.md`, phía hình học:** tracking chỉ bằng depth hoàn toàn không bị đèn đồng hành ảnh hưởng. ATE trên `lit` là 0.032 cm, trên `even` là 0.041 cm. Hai con số ngang nhau, và thấp hơn khoảng 35–45 lần so với mô hình chính của đồ án trên `lit` (1.10–1.45 cm).
- **Lý do: depth Blender không nhiễu.** Bước lượng tử của PNG 16 bit chỉ 0.15 mm. GICP khớp đám mây điểm gần như chính xác: RPE trung vị mỗi 10 frame là 0.08 mm; so thẳng với quỹ đạo thật trong hệ frame 0, chưa căn chỉnh, sai số trung bình 0.4–0.5 mm, lớn nhất 3 mm. Con số này tốt hơn cả mức bài báo GS-ICP báo trên Replica (khoảng 0.16 cm). **Đây không phải kết luận về cảm biến thật**: depth thật có nhiễu, lỗ hổng và sai số theo khoảng cách, và tracking ICP nhạy với các yếu tố đó hơn tracking quang trắc. Báo cáo phải nói rõ (`CLAUDE.md` mục 3).
- **Đã kiểm không rò rỉ:** pose ước lượng nằm trong hệ riêng của GS-ICP (frame 0 là ma trận đơn vị từ `traj.txt` giả; frame 0 thật thì không). GS-ICP tự in `ATE RMSE: nan`. Chỉ `score.py` đọc quỹ đạo thật.
- **Bản đồ: đúng hình học, sai ánh sáng.** PSNR trên `even` là 44.25: frame chấm là frame chưa dùng để tối ưu, và pose gần như đúng. Trên `lit` PSNR chỉ còn 16.71 (SSIM 0.758, LPIPS 0.289), dù pose tốt như trên `even`. Lý do: mỗi Gaussian chỉ có một màu (SH bậc 0), nung ánh sáng của các keyframe đã thấy nó. Ảnh 5 frame chung (`results/figures_office0_lit.png`) cho thấy đĩa sáng bị in lên tường ở chỗ cũ (frame 500), và vùng đang được chiếu lại render tối (frame 1750). Với pose gần như đúng, phần lớn mức chênh 16.7 so với 44.3 dB đến từ việc thiếu mô hình chiếu sáng. Đây là phép đo sạch hơn SplaTAM, vì ở SplaTAM lỗi pose và lỗi bản đồ lẫn vào nhau.
- **So với đồ án:** trên `lit`, đồ án có PSNR 35.84–36.28 (trước refine) so với 16.71 của GS-ICP, nhờ mô hình đèn. Nhưng ATE của đồ án kém hơn hẳn trên depth không nhiễu. Cách trình bày trung thực: với depth sạch, tracking hình học là cận dưới về ATE; đóng góp của đồ án là tracking quang trắc đúng dưới đèn di động (thứ SplaTAM và MonoGS gốc không làm được), và một bản đồ tái tạo được ảnh dưới đèn đó.
- **FPS: 5.14 (`lit`) và 3.13 (`even`), tính cả việc đọc ảnh, cùng cách MonoGS tính** (`PROTOCOL.md`). GS-ICP đọc hết 2000 ảnh vào bộ nhớ trước vòng lặp, mất 204 s và 451 s; MonoGS đọc từng frame bên trong khoảng được đo. Riêng vòng tracking đạt 10.6–10.8 FPS ("System FPS" của GS-ICP), dưới trần 30 vì GICP chạy trên CPU (4 nhân). Thời gian đọc ảnh chênh hơn 2 lần giữa hai phiên do tốc độ đọc `/kaggle/input`, nên FPS của GS-ICP dao động theo I/O nhiều hơn theo thuật toán. So với mô hình chính của đồ án (3.05–3.18 trên T4 x2) thì GS-ICP ngang hoặc nhanh hơn vừa phải, không vượt trội. Mapping chạy song song và dừng khi tracking xong: 8000–8600 vòng.
- **Thí nghiệm nên làm thêm:** chạy GS-ICP (và mô hình chính) với depth có nhiễu kiểu cảm biến thật, để biết lợi thế của tracking hình học còn lại bao nhiêu. Thí nghiệm này cần sửa dữ liệu đầu vào, nên phải thống nhất cách tạo nhiễu trước.

### ORB-SLAM3 RGB-D trên `office0` (2026-11-12, mỗi biến thể một run)

- **Trả lời câu 2, phía đặc trưng:** đặc trưng ORB **bị** đèn đồng hành làm hại, nhưng không đến mức hỏng. Trên `lit`, ATE là 3.55 cm, so với 1.36 cm trên `even`, tức kém 2.6 lần. Cả hai run giữ đúng 1 bản đồ, 2000/2000 frame ở trạng thái OK, không mất tracking lần nào. Mô hình chính của đồ án trên `lit` (1.10–1.45 cm) tốt hơn ORB-SLAM3 khoảng 2.5–3 lần, dù ORB-SLAM3 có loop closure còn đồ án thì không.
- **Kiểu lỗi: rung từ frame sang frame, không gãy.** Sai số chuyển động mỗi 10 frame (RPE) trên `lit` có trung vị 8–21 mm tùy đoạn, so với 5–8 mm trên `even`. Quãng đường ước lượng là 27.96 m so với 18.20 m thật (trên `even`: 20.79 m). Pose rung quanh quỹ đạo đúng nên quãng đường bị cộng dồn. Căn chỉnh trên 150 frame đầu thì sai số trôi tới 18 cm (frame 1400) trên `lit`, so với khoảng 6 cm trên `even`.
- **Keyframe dày lên ở chỗ khó:** 162 keyframe trên `lit` so với 91 trên `even`. Đoạn bảng trắng (frame 250–350, nơi SplaTAM gãy, pixel bão hòa 18%) có 26 keyframe so với 6 trên `even`. ORB-SLAM3 chèn keyframe khi số điểm bám được giảm, nên đây là dấu hiệu tracking đặc trưng suy yếu dưới vệt sáng, đúng như SpotlessGS mô tả: vùng tối không có đặc trưng, vùng bão hòa mất kết cấu.
- **Loop closure đã giúp:** trên `lit`, ORB-SLAM3 phát hiện một vòng lặp quanh frame 1700 và hiệu chỉnh quỹ đạo; trên `even` thì không có. Sai số lớn nhất trên `lit` vẫn là 16 cm (frame 1733, ngay trước khi vòng lặp được đóng). MonoGS và đồ án không có loop closure, nên so ATE thì ORB-SLAM3 đang được lợi.
- **FPS bị chi phối bởi việc đọc ảnh:** vòng lặp mất 360 s cho 2000 frame (5.5 FPS), nhưng riêng `TrackRGBD` chỉ 37–39 s (51–54 FPS). Phần còn lại khoảng 160 ms/frame là đọc ảnh từ `/kaggle/input` (`cv::imread` trong vòng lặp, như `rgbd_tum`) cộng thời gian chờ để giữ nhịp 30 Hz (ít nhất 28 s). Cùng hiện tượng với GS-ICP: đọc 2000 ảnh mất 204–451 s. MonoGS cũng đọc từ `/kaggle/input`, nhưng có luồng đọc trước một frame nên che được độ trễ này. **FPS của các phương pháp đọc ảnh tuần tự vì vậy phản ánh tốc độ ổ đĩa Kaggle nhiều hơn tốc độ thuật toán.** Bảng cần ghi cả FPS riêng phần tracking. Nếu muốn so FPS chặt chẽ, nên chép dataset ra ổ cục bộ trước khi đo, cho mọi phương pháp kể cả MonoGS; việc này cần thống nhất trước.
- ATE kf (3.94 cm) cao hơn ATE all (3.55 cm) vì keyframe dồn vào các đoạn khó. Chính vì lý do này mà giao thức chọn ATE all làm số chính.

### Photo-SLAM trên `office0` (2026-11-15, mỗi biến thể một run)

- **Trả lời câu 2, đủ cả hai phía (tracking đặc trưng và bản đồ):** đèn đồng hành làm hại cả hai. ATE trên `lit` là 7.93 cm, so với 1.66 cm trên `even` (kém 4.8 lần). PSNR trên `lit` là 17.44, so với 35.51 trên `even`. Mô hình chính của đồ án trên `lit`: ATE 1.10–1.45 cm (tốt hơn 5–7 lần), PSNR 35.8–36.3 trước refine. Cả hai run giữ 1 bản đồ, 2000/2000 frame OK, không frame chấm ảnh nào thiếu pose.
- **Vì sao kém ORB-SLAM3 (3.55 cm) dù cùng tracking: chủ yếu do loop closure.** Sai số theo đoạn 200 frame (căn chỉnh SE(3) trên mọi frame):

  | Frame | 0–599 | 600–1599 | 1600–1799 | 1800–1999 |
  |---|---|---|---|---|
  | Photo-SLAM `lit` (cm) | 5.2–7.5 | 3.5–5.6 | 9.2 | 17.6 |
  | ORB-SLAM3 `lit` (cm) | 3.5–5.7 | 2.3–3.4 | 4.2 | 1.7 |

  Đến frame 1600, Photo-SLAM chỉ kém ORB-SLAM3 khoảng 1.2–1.7 lần. Khác biệt có thể do cấu hình Replica riêng của Photo-SLAM (1600 đặc trưng thay vì 1000, timestamp = chỉ số frame) và do tính ngẫu nhiên. Sau đó ORB-SLAM3 đóng được vòng lặp quanh frame 1700 và kéo đoạn cuối về 1.7 cm. Photo-SLAM không đóng được, nên sai số tăng tới 24.5 cm ở frame 1996; đoạn cuối chỉ có 3 keyframe, so với 13 của ORB-SLAM3. Vậy với tracking ORB trên `lit`, ATE nằm trong khoảng 3.5–8 cm tùy loop closure có xảy ra hay không: **run thứ hai rất cần** để biết khoảng này. Trên `even` không run nào có loop closure, và hai hệ ngang nhau (1.66 so với 1.36 cm).
- **Keyframe dày lên ở chỗ khó, như ORB-SLAM3:** 169 keyframe trên `lit` so với 105 trên `even`. Đoạn 200–599 (bảng trắng) có 53 keyframe, so với 27 trên `even`.
- **Bản đồ: ánh sáng bị nung vào màu.** Photo-SLAM có SH bậc 3, nhưng SH chỉ mô tả màu thay đổi theo hướng nhìn. Ánh sáng của đèn đồng hành thay đổi theo vị trí camera và khoảng cách tới bề mặt, nên SH không biểu diễn được. Ánh sáng nung vào bản đồ chỉ khớp ảnh khi góc nhìn gần một keyframe đã thấy cùng chỗ. Frame 500 là trường hợp đó: render khá giống ảnh quan sát, kể cả đĩa sáng trên bảng trắng. Ở frame 825 và 1125, vùng được chiếu trong render loang ra ngoài nón và nhòe, vì nhiều keyframe chiếu vào sàn từ các vị trí khác nhau. Ở frame 1750, ảnh vừa nhòe vừa lệch, vì pose sai 9–17 cm ở đoạn này (`results/figures_office0_lit.png`). PSNR theo đoạn thấp nhất ở 800–999 (13.0, chỉ 5 keyframe) và 1800–1999 (11.7, pose sai nhất). Mức 17.4 dB ngang GS-ICP trên `lit` (16.7). Ở GS-ICP pose gần như đúng, nên mức PSNR thấp ở đó chủ yếu đến từ việc thiếu mô hình chiếu sáng; ở Photo-SLAM thì lỗi pose cộng thêm vào.
- **Bản đồ `even` thua SplaTAM và GS-ICP** (35.5 so với 38.9 và 44.3), vì mapper được train ít: 5883 vòng (`lit`) và 7281 vòng (`even`) trên tối đa 30100 của cấu hình. Mapper chạy song song với tracking và dừng ngay sau frame cuối (chỉ train thêm 2–3 s). Tracking chạy nhanh, không giữ nhịp (timestamp = chỉ số frame, như ví dụ `replica_rgbd` của Photo-SLAM), nên cả chuỗi chỉ mất 78 s. Đây là thiết kế thời gian thực của Photo-SLAM. Trên GPU nhanh hơn T4, mapper được nhiều vòng hơn. PSNR theo đoạn trên `even` cũng giảm ở 800–999 và 1800–1999, hai đoạn ít keyframe.
- **FPS: 25.6 (`lit`) và 26.9 (`even`), gồm đọc ảnh** từ bản chép trên ổ cục bộ. Đây là các run đầu tiên đo theo cách mới (`PROTOCOL.md`, mục FPS). Riêng tracking đạt 29.7 và 31.4 FPS (CPU 4 nhân). So với mô hình chính của đồ án (3.05–3.18 trên T4 x2), Photo-SLAM nhanh hơn khoảng 8 lần, nhưng bản đồ của nó được train ít hơn nhiều.

### SEGS-SLAM trên `office0` (2026-11-16, mỗi biến thể một run)

- **Trả lời câu 3, phía nhúng ngoại hình học được: có học được đèn, nhưng chỉ bù được khoảng 1 dB.** Nhúng ngoại hình sinh từ pose camera có học ra một đĩa sáng đi theo camera: ở frame 500, 825, 1125 (`results/figures_office0_lit.png`), render của SEGS-SLAM có đĩa sáng gần đúng chỗ. Nhưng đĩa nhòe, to hơn nón thật, và có quầng sáng ngoài nón (frame 100). Ở frame 1750 đĩa vẫn có, nhưng nội dung cảnh (tranh trên tường) mất hẳn. PSNR trên `lit` là 21.57, hơn Photo-SLAM (17.44) 4.1 dB. Tuy vậy trên `even` SEGS-SLAM cũng hơn Photo-SLAM 3.2 dB (38.68 so với 35.51), vì bản đồ anchor kiểu Scaffold-GS và vì mapper train tới 30000 vòng (Photo-SLAM chỉ khoảng 6000–7000). Mức tụt từ `even` xuống `lit` là 17.1 dB ở SEGS-SLAM so với 18.1 dB ở Photo-SLAM. Vậy riêng phần nhúng ngoại hình chỉ lấy lại được khoảng 1 dB của mức tụt do đèn đồng hành.
- **So với đồ án:** SEGS-SLAM train tiếp bản đồ tới 30000 vòng sau frame cuối, nên so với cột sau refine của đồ án (`methods/segsslam/README.md`): 36.28 và 37.14 dB trên `lit`, so với 21.57. Đồ án hơn khoảng 15 dB. Một mạng ánh xạ pose sang ngoại hình phải học lại hình dạng nón, suy giảm theo khoảng cách và góc tới từ dữ liệu, cho từng vị trí camera. Mô hình vật lý có sẵn các quan hệ đó.
- **ATE:** `lit` 3.06 cm, có 1 loop closure (ORB-SLAM3 phát hiện ở frame 1100–1200). Sai số lớn nhất 11.7 cm ở frame 278, đoạn bảng trắng. `even` 2.78 cm, không loop closure, kém Photo-SLAM (1.66) và ORB-SLAM3 (1.36); sai số theo đoạn cao nhất ở frame 400–799 (3.5–4.2 cm). Cấu hình ORB-SLAM3 cho `office0` của SEGS-SLAM dùng 7200 đặc trưng (Photo-SLAM 1600, ORB-SLAM3 1000), giữ nguyên theo quy tắc. Với một run, chưa tách được ảnh hưởng của số đặc trưng khỏi tính ngẫu nhiên. Ba hệ dùng tracking ORB trên `lit` cho 3.06–7.93 cm; hai run có loop closure là 3.06 và 3.55 cm. Mô hình chính của đồ án (1.10–1.45 cm, không có loop closure) vẫn tốt hơn 2–3 lần.
- **PSNR theo đoạn thấp nhất ở frame 800–999** (13.0 dB, chỉ 3 keyframe), như Photo-SLAM: phần cảnh ít keyframe nhìn thấy thì bản đồ kém, có nhúng hay không.
- **FPS:** 25.2 (`lit`) và 24.0 (`even`), gồm đọc ảnh từ ổ cục bộ; riêng tracking 28.8 và 27.1. Phần train sau frame cuối (902 s và 729 s) không tính vào FPS, theo cùng quy tắc với phần refine của đồ án (`PROTOCOL.md`).

### LoopSplat trên `office0` (2026-11-16, mỗi biến thể một run)

- **Trả lời câu 1 và câu "đóng vòng có cứu được trôi không": không.** Trên `even`, LoopSplat tốt nhất trong mọi phương pháp quang trắc: ATE 0.72 cm, PSNR 45.09, 4 lần tối ưu pose graph (frame 701, 1103, 1769, 1838). Trên `lit` nó hỏng nặng nhất: ATE 105 cm, kém cả SplaTAM (62.6 cm). Cùng họ với SplaTAM (tracking bằng ảnh cộng độ sâu trên Gaussian), cùng kết luận: lỗi do đèn đồng hành thuộc cả họ SLAM quang trắc, và đóng vòng không sửa được.
- **Kiểu hỏng: trôi sớm, rồi gãy ở bảng trắng.** Căn chỉnh trên 30 frame đầu thì sai số đã 2–3 cm ở frame 40–170, rồi 8 cm ở frame 200. Ở frame 257 tracking nhảy 20.7 cm trong một frame (chuyển động thật 0.9 cm), sau đó 8.7 và 3.9 cm. Đây cùng đoạn camera tiến lại bảng trắng nơi SplaTAM gãy (frame 260–300). Bước nhảy này vượt ngưỡng chuyển động nên LoopSplat mở submap mới liên tục: 5 submap trong 90 frame (257, 286, 302, 326, 346). Đoạn hỏng thứ hai ở frame 600–799 (RPE trung vị mỗi 10 frame 10 cm, so với 0.3–0.8 cm ở các đoạn tốt). Quãng đường ước lượng 31.8 m so với 18.2 m thật. Run có 40 submap, so với 28 trên `even`.
- **Đóng vòng gần như không xảy ra trên `lit`:** chỉ 1 lần tối ưu pose graph (frame 1102, 2 cạnh vòng). Ở frame 1843 và 1890, LoopSplat tìm được ứng viên vòng nhưng loại hết ("No valid loop edges"). Run này chưa cho biết lý do (nhận dạng nơi chốn trên ảnh có đĩa sáng, hay đăng ký giữa các submap lệch nhau quá xa).
- **PSNR `lit` 20.71 không phản ánh bản đồ toàn cục.** Mỗi frame được render bằng submap của chính nó, tại pose ước lượng (cách `Evaluator` của LoopSplat làm). Submap được dựng từ cùng các pose ước lượng đó, nên nhất quán cục bộ dù pose sai toàn cục. Frame chấm lại chỉ cách frame mapping 2 frame (tập dịch lưới, `PROTOCOL.md`), nên ánh sáng nung từ frame gần đó còn khớp: ở frame 500, render có đĩa sáng gần đúng chỗ. Vì vậy PSNR của LoopSplat (cả `lit` lẫn `even`) thiên về phía có lợi so với phương pháp render một bản đồ chung, như ghi chú về SplaTAM ở trên. Ở frame 1750, pose sai nên ảnh lệch hẳn.
- **FPS:** 0.53 (`lit`) và 0.50 (`even`) trên một T4, chậm hơn mô hình chính của đồ án khoảng 6 lần. LoopSplat mapping mọi frame chia hết cho 5, mỗi lần khoảng 1.5 s.
