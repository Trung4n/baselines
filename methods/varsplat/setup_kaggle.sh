#!/usr/bin/env bash
# Kaggle setup of VarSplat, called by common/kaggle_runner.setup("varsplat")
# from run_baseline.ipynb. Installs into the notebook's Python ($PYTHON) with
# uv; Kaggle's torch is kept (VarSplat was tested with torch 2.4.1 + CUDA
# 12.6; its code is LoopSplat's plus the variance terms, see
# methods/varsplat/README.md). Second call in a session: skips what is built.
#
# - Python packages of VarSplat's SLAM path (its requirements.txt without the
#   evaluation-only ones: evaluate_3d_reconstruction_lib, PyQt5, wandb; pins
#   without a wheel for this Python not kept), as methods/loopsplat;
# - faiss: the GPU build (faiss-gpu-cu12) when a wheel exists for this Python,
#   else faiss-cpu, and the adapter swaps in methods/loopsplat/faiss_shim.py
#   (exact torch GPU search) for the GPU index;
# - hloc: not installed; the adapter puts third_party/VarSplat/thirdparty/
#   Hierarchical-Localization on sys.path (only hloc.extractors.netvlad is
#   used; its weights are downloaded at the first run);
# - the three CUDA extensions of VarSplat's submodules (third_party/VarSplat/
#   thirdparty/), built for this GPU on COPIES, with '#include <cstdint>'
#   (rasterizers) and '#include <cfloat>' (simple-knn) added when missing;
# - a probe (methods/varsplat/probe.py): renders a small Gaussian model with
#   both rasterizers, colour variance on, and backpropagates the way tracking,
#   mapping and registration do, so an ABI or API mismatch shows here, not
#   an hour into a run.
set -euo pipefail
: "${PYTHON:?PYTHON must be the notebook interpreter (kaggle_runner sets it)}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINES="$(cd "$HERE/../.." && pwd)"
SRC="$BASELINES/third_party/VarSplat"
BUILD="${TMPDIR:-/tmp}/varsplat_build"
EXTS=(simple-knn gaussian_rasterizer_var diff-gaussian-rasterization-w-pose-var)

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
if [[ -f "$STAMP" ]] && "$PYTHON" -c "import torch, simple_knn._C, gaussian_rasterizer, diff_gaussian_rasterization" 2>/dev/null; then
    echo "VarSplat extensions already built for sm $ARCH"
else
    rm -rf "${BUILD:?}"
    mkdir -p "$BUILD"
    for m in "${EXTS[@]}"; do
        cp -r "$SRC/thirdparty/$m" "$BUILD/$m"
    done
    for h in "$BUILD/gaussian_rasterizer_var/cuda_rasterizer/rasterizer_impl.h" \
             "$BUILD/diff-gaussian-rasterization-w-pose-var/cuda_rasterizer/rasterizer_impl.h"; do
        grep -q "<cstdint>" "$h" || sed -i '1i #include <cstdint>' "$h"
    done
    grep -q "<cfloat>" "$BUILD/simple-knn/simple_knn.cu" || sed -i '1i #include <cfloat>' "$BUILD/simple-knn/simple_knn.cu"
    # pip -v (not uv) for the builds: it streams the compiler output.
    for m in "${EXTS[@]}"; do
        echo "=== building $m for sm $ARCH"
        TORCH_CUDA_ARCH_LIST="$ARCH" "$PYTHON" -m pip install -v --no-build-isolation --force-reinstall --no-deps "$BUILD/$m"
    done
    touch "$STAMP"
fi
"$PYTHON" -c "import torch, simple_knn._C, gaussian_rasterizer, diff_gaussian_rasterization; print('extensions ok')"
"$PYTHON" "$HERE/probe.py"
"$PYTHON" -c "import json; print('VarSplat', json.load(open('$SRC/.vendor.json'))['commit'])"
