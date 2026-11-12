# Kết quả so sánh với các phương pháp khác

File này lưu số liệu cho chương thực nghiệm của báo cáo. Mỗi lần chấm xong một run, thêm một dòng vào bảng chi tiết và cập nhật bảng tổng. Mọi số đều chấm theo `PROTOCOL.md`.

## Bảng tổng (điền dần)

ATE tính bằng cm. `all` là ATE trên mọi frame, số chính để so giữa các phương pháp. `kf` là ATE trên keyframe theo cách MonoGS báo. PSNR, SSIM, LPIPS lấy trên bản đồ cuối, không refine. FPS đo trên Kaggle T4.

| Phương pháp | Loại tracking | `office0` `even`: ATE all | `office0` `lit`: ATE all / kf | `lit`: PSNR / SSIM / LPIPS | FPS | Số run |
|---|---|---|---|---|---|---|
| **Đồ án** (mô hình chính, lưới 0.5 m, `_parallel_fused`) | quang trắc, mô hình đèn | chưa chạy | 1.45, 1.10 / 1.46, 1.09 | 35.84 / 0.958 / 0.043, 36.28 / 0.963 / 0.041 (trước refine) | 3.18, 3.05 (T4 x2) | 2 |
| MonoGS gốc (`configs/rgbd/replica/office0.yaml`) | quang trắc | kf 1.13 (D133) | kf 84 (CLAUDE.md, code cũ) | – | 2.28 (`even`) | 1 |
| SplaTAM | quang trắc + độ sâu | 0.80 | 62.6 / 63.0 | 13.59 / 0.512 / 0.591 (`even`: 38.92 / 0.969 / 0.055) | 0.24 `lit`, 0.20 `even` (T4, 1 GPU) | 1 |
| ORB-SLAM3 RGB-D | đặc trưng | | | không có bản đồ | | |
| GS-ICP SLAM | hình học (G-ICP) | | | | | |

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

Trong ngoặc là số sau color refinement. MonoGS log cả hai dòng `mean psnr` trước và sau refine.

## Ảnh so sánh

Năm frame chung: 100, 500, 825, 1125, 1750 (`PROTOCOL.md`). Ảnh của từng run nằm ở `results/<run>/figures/`. Bảng ghép cho báo cáo ghi ở đây khi đã có đủ phương pháp:

| Cảnh / biến thể | File ghép | Cột |
|---|---|---|
| office0 / lit | (chưa có) | Quan sát, Đồ án, MonoGS gốc, SplaTAM, GS-ICP |
| office0 / lit, chỉ SplaTAM | `results/figures_splatam_office0_lit.png` | Quan sát, SplaTAM |
| office0 / even, chỉ SplaTAM | `results/figures_splatam_office0_even.png` | Quan sát, SplaTAM |

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
