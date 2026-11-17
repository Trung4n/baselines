#!/usr/bin/env bash
# Kaggle setup of DarkGS, called by common/kaggle_runner.setup("darkgs") from
# run_baseline.ipynb. Installs into the notebook's Python ($PYTHON) with uv;
# Kaggle's torch is kept. Second call in a session: skips what is built.
#
# - Python packages DarkGS's train.py imports (its requirements.txt without
#   the torch pins);
# - lietorch is NOT built: methods/darkgs/lietorch_shim is put on PYTHONPATH
#   by the adapter (README);
# - DarkGS's two CUDA extensions (third_party/darkgs/submodules: Inria's
#   diff-gaussian-rasterization and simple-knn at the upstream pins), built for
#   this GPU on COPIES with '#include <cstdint>' / '#include <cfloat>' added
#   when missing;
# - a probe (methods/darkgs/probe.py): DarkGS's code through the shim on a
#   synthetic plane with a calibrated lamp file.
set -euo pipefail
: "${PYTHON:?PYTHON must be the notebook interpreter (kaggle_runner sets it)}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINES="$(cd "$HERE/../.." && pwd)"
SRC="$BASELINES/third_party/darkgs"
BUILD="${TMPDIR:-/tmp}/darkgs_build"
EXTS=(simple-knn diff-gaussian-rasterization)

"$PYTHON" -m pip install -q uv
UV=("$PYTHON" -m uv pip install --system --python "$PYTHON" --break-system-packages --index-strategy unsafe-best-match)
"${UV[@]}" plyfile tqdm tensorboard wandb pillow scipy
"$PYTHON" -c "import torch, plyfile, tqdm, wandb, PIL; print('torch', torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"

ARCH="$("$PYTHON" -c "import torch; print('%d.%d' % torch.cuda.get_device_capability(0))")"
STAMP="$BUILD/.built_for_sm$ARCH"
if [[ -f "$STAMP" ]] && "$PYTHON" -c "import torch, simple_knn._C, diff_gaussian_rasterization" 2>/dev/null; then
    echo "DarkGS extensions already built for sm $ARCH"
else
    rm -rf "${BUILD:?}"
    mkdir -p "$BUILD"
    for m in "${EXTS[@]}"; do
        cp -r "$SRC/submodules/$m" "$BUILD/$m"
    done
    h="$BUILD/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h"
    grep -q "<cstdint>" "$h" || sed -i '1i #include <cstdint>' "$h"
    grep -q "<cfloat>" "$BUILD/simple-knn/simple_knn.cu" || sed -i '1i #include <cfloat>' "$BUILD/simple-knn/simple_knn.cu"
    # pip -v (not uv) for the builds: it streams the compiler output.
    for m in "${EXTS[@]}"; do
        echo "=== building $m for sm $ARCH"
        TORCH_CUDA_ARCH_LIST="$ARCH" "$PYTHON" -m pip install -v --no-build-isolation --force-reinstall --no-deps "$BUILD/$m"
    done
    touch "$STAMP"
fi
"$PYTHON" -c "import torch, simple_knn._C, diff_gaussian_rasterization; print('extensions ok')"
"$PYTHON" "$HERE/probe.py"
"$PYTHON" -c "import json; print('DarkGS', json.load(open('$SRC/.vendor.json'))['commit'])"
