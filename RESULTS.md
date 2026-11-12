# Kết quả so sánh với các phương pháp khác

File này lưu số liệu cho chương thực nghiệm của báo cáo. Mỗi lần chấm xong một run, thêm một dòng vào bảng chi tiết và cập nhật bảng tổng. Mọi số đều chấm theo `PROTOCOL.md`.

## Bảng tổng (điền dần)

ATE tính bằng cm. `all` là ATE trên mọi frame, số chính để so giữa các phương pháp. `kf` là ATE trên keyframe theo cách MonoGS báo. PSNR, SSIM, LPIPS lấy trên bản đồ cuối, không refine. FPS đo trên Kaggle T4.

| Phương pháp | Loại tracking | `office0` `even`: ATE all | `office0` `lit`: ATE all / kf | `lit`: PSNR / SSIM / LPIPS | FPS | Số run |
|---|---|---|---|---|---|---|
| **Đồ án** (mô hình chính, lưới 0.5 m, `_parallel_fused`) | quang trắc, mô hình đèn | chưa chạy | 1.45, 1.10 / 1.46, 1.09 | 35.84 / 0.958 / 0.043, 36.28 / 0.963 / 0.041 (trước refine) | 3.18, 3.05 (T4 x2) | 2 |
| MonoGS gốc (`configs/rgbd/replica/office0.yaml`) | quang trắc | kf 1.13 (D133) | kf 84 (CLAUDE.md, code cũ) | – | 2.28 (`even`) | 1 |
| SplaTAM | quang trắc + độ sâu | | | | | |
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

Trong ngoặc là số sau color refinement. MonoGS log cả hai dòng `mean psnr` trước và sau refine.

## Ảnh so sánh

Năm frame chung: 100, 500, 825, 1125, 1750 (`PROTOCOL.md`). Ảnh của từng run nằm ở `results/<run>/figures/`. Bảng ghép cho báo cáo ghi ở đây khi đã có đủ phương pháp:

| Cảnh / biến thể | File ghép | Cột |
|---|---|---|
| office0 / lit | (chưa có) | Quan sát, Đồ án, MonoGS gốc, SplaTAM, GS-ICP |

## Ghi chú cho báo cáo

- Mỗi hệ chọn keyframe khác nhau, nên so ATE trên mọi frame. Trên các run của đồ án, ATE trên mọi frame chênh dưới 0.02 cm so với ATE trên keyframe.
- Dao động giữa hai lần chạy của đồ án: tới khoảng 0.37 cm ATE và 0.8 dB PSNR (D139). Muốn kết luận một khác biệt thì nó phải lớn hơn mức này.
- SplaTAM tối ưu bản đồ ở mọi frame (`map_every` 1), nên các frame chấm ảnh của nó đã từng là đích tối ưu. PSNR, SSIM, LPIPS của SplaTAM vì vậy thiên về phía có lợi so với MonoGS, vốn chỉ tối ưu trên keyframe (`methods/splatam/README.md`).
- Depth Blender không có nhiễu (`CLAUDE.md` mục 3). Điều này có lợi cho các phương pháp tracking hình học như GS-ICP, cần nói rõ khi so.
