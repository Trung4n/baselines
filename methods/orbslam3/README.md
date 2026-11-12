# ORB-SLAM3 RGB-D

- Bài báo: ORB-SLAM3 (Campos et al., T-RO 2021). Trong `refs.bib`: `campos2021orbslam3`.
- Repo: <https://github.com/UZ-SLAMLab/ORB_SLAM3>.
- Vai trò trong so sánh: tracking dựa trên đặc trưng ORB, không so cường độ pixel. Theo SpotlessGS, đèn đồng hành vẫn gây hại cho đặc trưng: vùng tối không có đặc trưng, vùng bão hòa mất kết cấu, đỉnh sáng di động tạo đặc trưng giả. Thí nghiệm này kiểm điều đó trên dữ liệu của đồ án.

## Cần làm và cần kiểm (chưa làm)

1. Build trên Kaggle: cần Pangolin, OpenCV, Eigen. Có thể build không có viewer. Ghi commit.
2. File settings RGB-D: dựa trên file mẫu cho TUM hoặc Replica của repo, chỉ đổi nội tại, kích thước 600×340 và `DepthMapFactor`, tất cả đọc từ `configs/rgbd/replica/base_config.yaml`. Kiểm `DepthMapFactor` khớp `depth_scale` (độ sâu tính bằng mét = giá trị PNG / 6553.5). Giữ mặc định số đặc trưng và các tham số ORB.
3. Đầu vào: chương trình mẫu `rgbd_tum` cần file associations theo thời gian. Tạo timestamp giả bằng chỉ số frame (ví dụ `i / 30`) và ghi lại quy ước này. Khi xuất quỹ đạo, đổi timestamp về chỉ số frame trước khi đưa vào `score.py`.
4. Đầu ra: `CameraTrajectory.txt` (TUM, camera-to-world). Kiểm frame mất tracking, đếm vào `frames_missing`. Bật loop closure theo mặc định, và ghi rõ trong báo cáo là MonoGS không có loop closure.
5. Không có bản đồ kết xuất, nên chạy `score.py --no_render`. Keyframe lấy từ `KeyFrameTrajectory.txt`.
6. FPS: đo thời gian xử lý 2000 frame của tracking (vòng gọi `TrackRGBD`), chạy trên CPU. Ghi rõ CPU/GPU trong `timing.json`.

## Đã chạy

(chưa có)
