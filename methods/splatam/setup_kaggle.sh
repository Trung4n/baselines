#!/usr/bin/env bash
# Kaggle setup of SplaTAM, called by common/kaggle_runner.setup("splatam")
# from run_baseline.ipynb. Installs into the notebook's Python ($PYTHON) with
# uv, as run-monogs does; Kaggle's torch is kept (SplaTAM's README: tested
# with torch 2.3 + CUDA 12.1). Second call in a session: skips what is done.
set -euo pipefail
: "${PYTHON:?PYTHON must be the notebook interpreter (kaggle_runner sets it)}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINES="$(cd "$HERE/../.." && pwd)"
DGR_SRC="$BASELINES/third_party/diff-gaussian-rasterization-w-depth"
BUILD="${TMPDIR:-/tmp}/splatam_dgr_build"

"$PYTHON" -m pip install -q uv
UV=("$PYTHON" -m uv pip install --system --python "$PYTHON" --break-system-packages --index-strategy unsafe-best-match)

# Packages on SplaTAM's SLAM path (scripts/splatam.py and the utils/,
# datasets/ modules it imports). open3d, faiss, cyclonedds serve viz and
# demos only: not installed.
"${UV[@]}" natsort kornia pytorch-msssim torchmetrics
"$PYTHON" -c "import torch, cv2, imageio, yaml, wandb, scipy, PIL, natsort, kornia, pytorch_msssim, torchmetrics; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))"

# Rasterizer with depth, the commit SplaTAM pins (third_party/.vendor.json),
# built for this GPU on a COPY: the one build fix, `#include <cstdint>` when
# missing (newer GCC), is not applied to the repo's sources.
ARCH="$("$PYTHON" -c "import torch; print('%d.%d' % torch.cuda.get_device_capability(0))")"
STAMP="$BUILD/.built_for_sm$ARCH"
if [[ -f "$STAMP" ]] && "$PYTHON" -c "import diff_gaussian_rasterization" 2>/dev/null; then
    echo "rasterizer already built for sm $ARCH"
else
    rm -rf "$BUILD"
    cp -r "$DGR_SRC" "$BUILD"
    grep -q "<cstdint>" "$BUILD/cuda_rasterizer/rasterizer_impl.h" \
        || sed -i '1i #include <cstdint>' "$BUILD/cuda_rasterizer/rasterizer_impl.h"
    # --no-build-isolation: build against the installed torch.
    TORCH_CUDA_ARCH_LIST="$ARCH" "${UV[@]}" --no-build-isolation --reinstall-package diff_gaussian_rasterization "$BUILD"
    touch "$STAMP"
fi
"$PYTHON" -c "import diff_gaussian_rasterization as d, importlib.metadata as m; print('rasterizer', d.__file__, m.version('diff_gaussian_rasterization'))"
"$PYTHON" -c "import json; print('SplaTAM', json.load(open('$BASELINES/third_party/SplaTAM/.vendor.json'))['commit'])"
