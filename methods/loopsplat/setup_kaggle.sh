#!/usr/bin/env bash
# Kaggle setup of LoopSplat, called by common/kaggle_runner.setup("loopsplat")
# from run_baseline.ipynb. Installs into the notebook's Python ($PYTHON) with
# uv; Kaggle's torch is kept (LoopSplat pins torch 2.1.2 + CUDA 12.1 in conda;
# its code runs on newer torch, see methods/loopsplat/README.md). Second call
# in a session: skips what is built.
#
# - Python packages of LoopSplat's SLAM path (its requirements.txt without the
#   evaluation-only ones: evaluate_3d_reconstruction_lib, PyQt5; versions not
#   pinned where the pins have no wheel for this Python);
# - faiss: the GPU build (faiss-gpu-cu12) when a wheel exists for this Python,
#   else faiss-cpu, and the adapter swaps in faiss_shim.py (exact torch GPU
#   search) for the GPU index;
# - hloc: not installed; the adapter puts third_party/LoopSplat/thirdparty/
#   Hierarchical-Localization on sys.path (only hloc.extractors.netvlad is used;
#   its weights are downloaded at the first run);
# - the three CUDA extensions LoopSplat pins (third_party/LoopSplat/_deps/),
#   built for this GPU on COPIES, with '#include <cstdint>' (rasterizers) and
#   '#include <cfloat>' (simple-knn) added when missing.
set -euo pipefail
: "${PYTHON:?PYTHON must be the notebook interpreter (kaggle_runner sets it)}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINES="$(cd "$HERE/../.." && pwd)"
SRC="$BASELINES/third_party/LoopSplat"
BUILD="${TMPDIR:-/tmp}/loopsplat_build"

"$PYTHON" -m pip install -q uv
UV=("$PYTHON" -m uv pip install --system --python "$PYTHON" --break-system-packages --index-strategy unsafe-best-match)
"${UV[@]}" open3d roma einops plyfile pytorch_msssim torchmetrics imageio trimesh python-pycg evo rich scipy matplotlib tqdm
if "${UV[@]}" faiss-gpu-cu12; then
    echo "faiss: GPU build installed"
else
    echo "faiss: no faiss-gpu-cu12 wheel for this Python; installing faiss-cpu (the adapter uses faiss_shim.py on the GPU)"
    "${UV[@]}" faiss-cpu
fi
"$PYTHON" -c "import torch, open3d, roma, einops, plyfile, pytorch_msssim, torchmetrics, imageio, trimesh, pycg, evo, rich, faiss, cv2; print('torch', torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), '| open3d', open3d.__version__, '| faiss', faiss.__version__, 'GPU' if hasattr(faiss, 'StandardGpuResources') else 'CPU')"

ARCH="$("$PYTHON" -c "import torch; print('%d.%d' % torch.cuda.get_device_capability(0))")"
STAMP="$BUILD/.built_for_sm$ARCH"
if [[ -f "$STAMP" ]] && "$PYTHON" -c "import simple_knn._C, gaussian_rasterizer, diff_gaussian_rasterization" 2>/dev/null; then
    echo "LoopSplat extensions already built for sm $ARCH"
else
    rm -rf "$BUILD"
    mkdir -p "$BUILD"
    for m in simple-knn gaussian_rasterizer diff-gaussian-rasterization-w-pose; do
        cp -r "$SRC/_deps/$m" "$BUILD/$m"
    done
    for h in "$BUILD/gaussian_rasterizer/cuda_rasterizer/rasterizer_impl.h" \
             "$BUILD/diff-gaussian-rasterization-w-pose/cuda_rasterizer/rasterizer_impl.h"; do
        grep -q "<cstdint>" "$h" || sed -i '1i #include <cstdint>' "$h"
    done
    grep -q "<cfloat>" "$BUILD/simple-knn/simple_knn.cu" || sed -i '1i #include <cfloat>' "$BUILD/simple-knn/simple_knn.cu"
    # pip -v (not uv) for the builds: it streams the compiler output.
    for m in simple-knn gaussian_rasterizer diff-gaussian-rasterization-w-pose; do
        echo "=== building $m for sm $ARCH"
        TORCH_CUDA_ARCH_LIST="$ARCH" "$PYTHON" -m pip install -v --no-build-isolation --force-reinstall --no-deps "$BUILD/$m"
    done
    touch "$STAMP"
fi
"$PYTHON" -c "import simple_knn._C, gaussian_rasterizer, diff_gaussian_rasterization; print('extensions ok')"
"$PYTHON" -c "import json; print('LoopSplat', json.load(open('$SRC/.vendor.json'))['commit'])"
