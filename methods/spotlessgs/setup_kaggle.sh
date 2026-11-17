#!/usr/bin/env bash
# Kaggle setup of SpotlessGS, called by common/kaggle_runner.setup("spotlessgs")
# from run_baseline.ipynb. Installs into the notebook's Python ($PYTHON) with
# uv; Kaggle's torch is kept. Second call in a session: skips what is built.
#
# - Python packages SpotlessGS imports (plyfile, scipy for sph_harm,
#   matplotlib, tensorboard: train.py always writes a SummaryWriter);
# - lietorch is NOT built: methods/spotlessgs/lietorch_shim (learnable SO3) is
#   put on PYTHONPATH by the adapter (README);
# - the two CUDA extensions: SpotlessGS commits Inria's diff-gaussian-
#   rasterization and simple-knn, identical to third_party/darkgs/submodules
#   (which also carries the '#include' fixes), so those are built, on COPIES;
# - a probe (methods/spotlessgs/probe.py).
set -euo pipefail
: "${PYTHON:?PYTHON must be the notebook interpreter (kaggle_runner sets it)}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINES="$(cd "$HERE/../.." && pwd)"
EXT_SRC="$BASELINES/third_party/darkgs/submodules"
BUILD="${TMPDIR:-/tmp}/spotlessgs_build"
EXTS=(simple-knn diff-gaussian-rasterization)

"$PYTHON" -m pip install -q uv
UV=("$PYTHON" -m uv pip install --system --python "$PYTHON" --break-system-packages --index-strategy unsafe-best-match)
"${UV[@]}" plyfile tqdm tensorboard scipy matplotlib pillow
"$PYTHON" -c "import torch, plyfile, scipy.special, matplotlib, tensorboard; print('torch', torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"

ARCH="$("$PYTHON" -c "import torch; print('%d.%d' % torch.cuda.get_device_capability(0))")"
STAMP="$BUILD/.built_for_sm$ARCH"
if [[ -f "$STAMP" ]] && "$PYTHON" -c "import torch, simple_knn._C, diff_gaussian_rasterization" 2>/dev/null; then
    echo "extensions already built for sm $ARCH"
else
    rm -rf "${BUILD:?}"
    mkdir -p "$BUILD"
    for m in "${EXTS[@]}"; do
        cp -r "$EXT_SRC/$m" "$BUILD/$m"
    done
    h="$BUILD/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h"
    grep -q "<cstdint>" "$h" || sed -i '1i #include <cstdint>' "$h"
    grep -q "<cfloat>" "$BUILD/simple-knn/simple_knn.cu" || sed -i '1i #include <cfloat>' "$BUILD/simple-knn/simple_knn.cu"
    for m in "${EXTS[@]}"; do
        echo "=== building $m for sm $ARCH"
        TORCH_CUDA_ARCH_LIST="$ARCH" "$PYTHON" -m pip install -v --no-build-isolation --force-reinstall --no-deps "$BUILD/$m"
    done
    touch "$STAMP"
fi
"$PYTHON" -c "import torch, simple_knn._C, diff_gaussian_rasterization; print('extensions ok')"
"$PYTHON" "$HERE/probe.py"
"$PYTHON" -c "import json; print('SpotlessGS', json.load(open('$BASELINES/third_party/SpotlessGS/.vendor.json'))['commit'])"
