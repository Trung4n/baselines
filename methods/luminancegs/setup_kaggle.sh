#!/usr/bin/env bash
# Kaggle setup of Luminance-GS (third_party/Luminance-GS, commit in adapter.COMMIT),
# called by common/kaggle_runner.setup("luminancegs") from run_baseline.ipynb.
# Installs into the notebook's Python ($PYTHON) with uv; Kaggle's torch is kept.
# Second call in a session: skips what is done.
#
# - glm at gsplat 1.0.0's submodule pin (adapter.GLM_COMMIT), fetched with git
#   to $TMPDIR/luminancegs_glm: the repository's bundled gsplat backend
#   includes it but the repository lost the submodule;
# - the trainer's requirements (examples/requirements.txt): pycolmap at the
#   rmbrualla fork pin (patched for numpy 2: common/install_pycolmap_fork.sh), nerfview==0.0.2, viser, tyro, einops, ...; NOT gsplat
#   (its kernels would be imported instead of the bundled ones) and NOT
#   numpy==1.24.3 (the notebook's numpy is kept; README);
# - ninja, for the JIT build;
# - a probe (methods/luminancegs/probe.py): makes the patched copy at its fixed
#   path and compiles the bundled backend there for this GPU (the run reuses
#   the build), then runs the dual rasterizer and imports the trainer.
set -euo pipefail
: "${PYTHON:?PYTHON must be the notebook interpreter (kaggle_runner sets it)}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP="${TMPDIR:-/tmp}"
GLM="$TMP/luminancegs_glm"
read -r GLM_URL GLM_COMMIT < <("$PYTHON" -c "import sys; sys.path.insert(0, '$HERE'); import adapter; print(adapter.GLM_URL, adapter.GLM_COMMIT)")

if [[ "$(git -C "$GLM" rev-parse HEAD 2>/dev/null || true)" != "$GLM_COMMIT" ]]; then
    rm -rf "${GLM:?}"
    git init -q "$GLM"
    git -C "$GLM" fetch -q --depth 1 "$GLM_URL" "$GLM_COMMIT"
    git -C "$GLM" -c advice.detachedHead=false checkout -q FETCH_HEAD
fi
[[ "$(git -C "$GLM" rev-parse HEAD)" == "$GLM_COMMIT" ]] || { echo "glm checkout is not at $GLM_COMMIT" >&2; exit 1; }
echo "glm $GLM_COMMIT at $GLM"

"$PYTHON" -m pip install -q uv
UV=("$PYTHON" -m uv pip install --system --python "$PYTHON" --break-system-packages --index-strategy unsafe-best-match)
"${UV[@]}" ninja viser "nerfview==0.0.2" "imageio[ffmpeg]" scikit-learn tqdm "torchmetrics[image]" opencv-python tyro \
    Pillow tensorboard matplotlib einops scipy rich
bash "$HERE/../../common/install_pycolmap_fork.sh"   # the rmbrualla fork at its pin, patched for numpy 2
if "$PYTHON" -c "import gsplat" 2>/dev/null; then
    echo "a gsplat package is installed: uninstalling it (the bundled backend must compile its own kernels)"
    "$PYTHON" -m pip uninstall -y gsplat
fi
"$PYTHON" -c "import torch, numpy, pycolmap, nerfview, viser, tyro, einops; print('torch', torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), '| numpy', numpy.__version__)"

ARCH="$("$PYTHON" -c "import torch; print('%d.%d' % torch.cuda.get_device_capability(0))")"
export TORCH_CUDA_ARCH_LIST="$ARCH" MAX_JOBS="${MAX_JOBS:-4}"
echo "=== probe (compiles the bundled gsplat backend for sm $ARCH on the first run in this session)"
"$PYTHON" "$HERE/probe.py"
