#!/usr/bin/env bash
# Kaggle setup of the bilateral-grid baseline (gsplat v1.5.3 simple_trainer),
# called by common/kaggle_runner.setup("bilagrid") from run_baseline.ipynb.
# Installs into the notebook's Python ($PYTHON); Kaggle's torch is kept.
#
# - gsplat==1.5.3 (the release of the vendored trainer, third_party/
#   gsplat_examples); its CUDA code is compiled on first use (needs nvcc and
#   ninja), which the probe triggers here, for this GPU;
# - the trainer's requirements (examples/requirements.txt) at its pins:
#   pycolmap (rmbrualla fork, patched for numpy 2 by common/install_pycolmap_fork.sh), nerfview, fused-ssim (CUDA, built here), viser,
#   tyro, ...; fused-bilagrid is not needed (lib_bilagrid.py is used);
# - a probe (methods/bilagrid/probe.py).
set -euo pipefail
: "${PYTHON:?PYTHON must be the notebook interpreter (kaggle_runner sets it)}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARCH="$("$PYTHON" -c "import torch; print('%d.%d' % torch.cuda.get_device_capability(0))")"
export TORCH_CUDA_ARCH_LIST="$ARCH" MAX_JOBS="${MAX_JOBS:-4}"

"$PYTHON" -m pip install -q uv
UV=("$PYTHON" -m uv pip install --system --python "$PYTHON" --break-system-packages --index-strategy unsafe-best-match)
"${UV[@]}" "gsplat==1.5.3" ninja viser "tyro>=0.8.8" "imageio[ffmpeg]" scikit-learn "torchmetrics[image]" tensorboard \
    tensorly pyyaml matplotlib opencv-python splines
"${UV[@]}" "nerfview @ git+https://github.com/nerfstudio-project/nerfview@4538024fe0d15fd1a0e4d760f3695fc44ca72787"
bash "$HERE/../../common/install_pycolmap_fork.sh"   # the rmbrualla fork at its pin, patched for numpy 2
if ! "$PYTHON" -c "import fused_ssim" 2>/dev/null; then
    echo "=== building fused-ssim for sm $ARCH"
    "$PYTHON" -m pip install -v --no-build-isolation \
        "git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5"
fi
"$PYTHON" -c "import torch, gsplat, pycolmap, nerfview, viser, tyro, fused_ssim; print('torch', torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), '| gsplat', gsplat.__version__)"
echo "=== probe (compiles gsplat's CUDA code on the first run in this session)"
"$PYTHON" "$HERE/probe.py"
