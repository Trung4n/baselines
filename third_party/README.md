# third_party/: mã nguồn phương pháp khác, ghim commit

Chép thẳng mã nguồn vào repo, không dùng submodule, để notebook Kaggle chỉ clone một repo `baselines`. Mỗi thư mục có `.vendor.json` ghi URL, commit, và mọi chỗ khác so với commit đó. Adapter đọc commit từ file này để ghi vào `method.json`.

| Thư mục | Repo | Commit | Dùng cho |
|---|---|---|---|
| `SplaTAM/` | spla-tam/SplaTAM | `da6bbcd2` (2024-06-19) | `methods/splatam/` |
| `diff-gaussian-rasterization-w-depth/` | JonathonLuiten/diff-gaussian-rasterization-w-depth | `cb65e4b8` (2023-09-19) | rasterizer của SplaTAM, build trên Kaggle |
| `ORB_SLAM3/` | UZ-SLAMLab/ORB_SLAM3 | `4452a3c4` (v1.0, 2022-02-10), kèm vocabulary | `methods/orbslam3/` |
| `Pangolin/` | stevenlovegrove/Pangolin | `aff6883c` (v0.8, 2022-05-02) | thư viện ORB-SLAM3 bắt buộc link; Ubuntu 24.04 không có gói |
| `Photo-SLAM/` | HuajianUP/Photo-SLAM | `f8bfb2f0` (2024-06-12), kèm bản ORB-SLAM3 đã sửa của nó và vocabulary | `methods/photoslam/` (OpenCV có CUDA và LibTorch tải lúc setup, không chép vào đây) |
| `SEGS-SLAM/` | leaner-forever/SEGS-SLAM (chính chủ, ICCV 2025) | `2110994a` (2025-10-18), kèm bản ORB-SLAM3 đã sửa của nó và vocabulary | `methods/segsslam/` (OpenCV có CUDA, LibTorch và TorchScatter tải lúc setup) |
| `LoopSplat/` | GradientSpaces/LoopSplat | `676e18f9` (2025-02-03), kèm gói `hloc` và 3 extension CUDA ở commit `requirements.txt` ghim (`_deps/`) | `methods/loopsplat/` |
| `VarSplat/` | anhthuan1999/varsplat (chính chủ, CVPR 2026) | `4a21a0aa` (2026-06-14), kèm gói `hloc` và 3 submodule ở commit nó ghim (`simple-knn`, `gaussian_rasterizer_var`, `diff-gaussian-rasterization-w-pose-var`) trong `thirdparty/` | `methods/varsplat/` |
| `darkgs/` | Trung4n/darkgs (bản fork của người dùng, từ tyz1030/darkgs `279bfa10`) | `32945fcd` (2026-09-21), kèm 2 extension Inria ở commit bản gốc ghim | `methods/darkgs/` (lietorch thay bằng `methods/darkgs/lietorch_shim`) |
| `SpotlessGS/` | Liianne/SpotlessGS (chính chủ) | `db380dcf`, chỉ phần Python (extension CUDA dùng bản của `darkgs/`, giống hệt) | `methods/spotlessgs/` (`N_TEST` vá trên bản chép lúc chạy; lietorch thay bằng shim học được) |
| `gsplat_examples/` | nerfstudio-project/gsplat, thư mục `examples/` | tag `v1.5.3`, chỉ các file `simple_trainer.py` cần; thư viện `gsplat==1.5.3` cài bằng pip | `methods/bilagrid/` (vá trên bản chép lúc chạy: train mọi ảnh, `datasets/__init__.py`, checkpoint lưu grid) |
| `Luminance-GS/` | cuiziteng/Luminance-GS (chính chủ, Apache-2.0) | `f9b4b866` (2026-05-13), chỉ các file trainer cần, kèm backend gsplat 1.0.0 đã sửa của repo (JIT); glm tải lúc setup (`g-truc/glm` `45008b22`, pin của gsplat 1.0.0) | `methods/luminancegs/` (vá trên bản chép lúc chạy: bỏ `import turtle`, `datasets/__init__.py`, glm) |
| `GS_ICP_SLAM/` | Lab-of-AI-and-Robotics/GS_ICP_SLAM | `5f996a87` (2026-06-20), kèm 3 submodule ở commit nó ghim (rasterizer, fast_gicp, simple-knn) | `methods/gsicp/` |

Quy tắc:

- Không sửa code trong đây. Chỗ nào cần khác thì làm trong adapter, hoặc vá bản chép lúc build (`methods/<name>/setup_kaggle.sh`), rồi ghi vào `.vendor.json`.
- Không đặt repo git lồng (`.git`) trong đây. Repo `baselines` sẽ ghi nó thành gitlink, và Kaggle clone về sẽ chỉ thấy thư mục rỗng.
- Kiểm `.gitignore` của repo được chép. `data/` trong `.gitignore` của SplaTAM từng làm git bỏ qua `configs/data/*.yaml`, nên file đó đã đổi tên thành `.gitignore.upstream`. Sau khi chép, chạy `find third_party -type f | git check-ignore --stdin` và phải ra rỗng.
- Đổi commit: chép lại toàn bộ, so với bản clone bằng `git --git-dir=<clone>/.git --work-tree=<thư mục> status`, rồi cập nhật `.vendor.json` và bảng này.
