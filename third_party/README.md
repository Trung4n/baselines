# third_party/: mã nguồn phương pháp khác, ghim commit

Chép thẳng mã nguồn vào repo, không dùng submodule, để notebook Kaggle chỉ clone một repo `baselines`. Mỗi thư mục có `.vendor.json` ghi URL, commit, và mọi chỗ khác so với commit đó. Adapter đọc commit từ file này để ghi vào `method.json`.

| Thư mục | Repo | Commit | Dùng cho |
|---|---|---|---|
| `SplaTAM/` | spla-tam/SplaTAM | `da6bbcd2` (2024-06-19) | `methods/splatam/` |
| `diff-gaussian-rasterization-w-depth/` | JonathonLuiten/diff-gaussian-rasterization-w-depth | `cb65e4b8` (2023-09-19) | rasterizer của SplaTAM, build trên Kaggle |
| `ORB_SLAM3/` | UZ-SLAMLab/ORB_SLAM3 | `4452a3c4` (v1.0, 2022-02-10), kèm vocabulary | `methods/orbslam3/` |
| `Pangolin/` | stevenlovegrove/Pangolin | `aff6883c` (v0.8, 2022-05-02) | thư viện ORB-SLAM3 bắt buộc link; Ubuntu 24.04 không có gói |
| `GS_ICP_SLAM/` | Lab-of-AI-and-Robotics/GS_ICP_SLAM | `5f996a87` (2026-06-20), kèm 3 submodule ở commit nó ghim (rasterizer, fast_gicp, simple-knn) | `methods/gsicp/` |

Quy tắc:

- Không sửa code trong đây. Chỗ nào cần khác thì làm trong adapter, hoặc vá bản chép lúc build (`methods/<name>/setup_kaggle.sh`), rồi ghi vào `.vendor.json`.
- Không đặt repo git lồng (`.git`) trong đây. Repo `baselines` sẽ ghi nó thành gitlink, và Kaggle clone về sẽ chỉ thấy thư mục rỗng.
- Kiểm `.gitignore` của repo được chép. `data/` trong `.gitignore` của SplaTAM từng làm git bỏ qua `configs/data/*.yaml`, nên file đó đã đổi tên thành `.gitignore.upstream`. Sau khi chép, chạy `find third_party -type f | git check-ignore --stdin` và phải ra rỗng.
- Đổi commit: chép lại toàn bộ, so với bản clone bằng `git --git-dir=<clone>/.git --work-tree=<thư mục> status`, rồi cập nhật `.vendor.json` và bảng này.
