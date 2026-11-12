# GS-ICP SLAM

- Bài báo: RGBD GS-ICP SLAM (Ha et al., ECCV 2024). Thêm vào `refs.bib` khi dùng.
- Repo: <https://github.com/Lab-of-AI-and-Robotics/GS_ICP_SLAM>.
- Vai trò trong so sánh: tracking thuần hình học (Generalized-ICP trên đám mây điểm từ độ sâu), bản đồ Gaussian chỉ để kết xuất. Đây là đối chứng bên ngoài cho mục "chỉ dùng độ sâu" của báo cáo.
- Lưu ý khi diễn giải: depth Blender không có nhiễu, nên dữ liệu mô phỏng có lợi cho tracking hình học. Cần nói rõ điều này khi so (`CLAUDE.md` mục 3).

## Cần làm và cần kiểm (chưa làm)

1. Môi trường theo README của repo. Kiểm các submodule (fast_gicp, rasterizer) build được trên T4.
2. Dataset: repo có loader Replica. Kiểm mẫu tên file, nội tại 600×340 và thang độ sâu, đọc từ `configs/rgbd/replica/base_config.yaml`.
3. Keyframe: lấy từ output hoặc log của phương pháp.
4. Render: render bản đồ Gaussian cuối tại pose ước lượng cho các frame `protocol.required_renders(N, keyframes)` (có 5 frame chung để so ảnh, render cả khi là keyframe), lưu `.npy`.
5. FPS: phương pháp nhanh (tracking và mapping chạy song song, có thể giới hạn tốc độ mapping theo tham số). Giữ mặc định của repo và ghi rõ.

## Đã chạy

(chưa có)
