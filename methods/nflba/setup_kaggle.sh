#!/usr/bin/env bash
# Kaggle setup of NFL-BA, called by common/kaggle_runner.setup("nflba") from
# run_baseline.ipynb. Installs into the notebook's Python ($PYTHON) with uv;
# Kaggle's torch is kept (NFL-BA uses MonoGS's environment; see
# methods/nflba/README.md). Second call in a session: skips what is done.
#
# - NFL-BA's source at the pinned commit (adapter.COMMIT), fetched with git to
#   $TMPDIR/nflba_src: its repository has no licence, so it is not vendored in
#   this repo. Only Python files; refuses any other commit;
# - Python packages of MonoGS's SLAM path (its environment.yml without the GUI
#   ones: glfw, PyOpenGL, imgviz, PyGLM are imported only by gui/slam_gui.py,
#   which the adapter does not load);
# - MonoGS's two CUDA extensions at MonoGS's pinned commits, which this repo
#   already carries for LoopSplat (third_party/LoopSplat/_deps: simple-knn =
#   Inria 44f76429, diff-gaussian-rasterization-w-pose = 43e21bff, checked
#   identical 2026-11-17), built for this GPU on COPIES with the same
#   '#include <cstdint>' / '#include <cfloat>' fixes;
# - a probe (methods/nflba/probe.py): NFL-BA's code on a synthetic frame
#   (Gaussians from depth, render with and without the light model, a
#   tracking loop, a mapping step).
set -euo pipefail
: "${PYTHON:?PYTHON must be the notebook interpreter (kaggle_runner sets it)}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINES="$(cd "$HERE/../.." && pwd)"
DEPS="$BASELINES/third_party/LoopSplat/_deps"
TMP="${TMPDIR:-/tmp}"
NFL="$TMP/nflba_src"
BUILD="$TMP/nflba_build"
URL="$("$PYTHON" -c "import sys; sys.path.insert(0, '$HERE'); import adapter; print(adapter.UPSTREAM)")"
COMMIT="$("$PYTHON" -c "import sys; sys.path.insert(0, '$HERE'); import adapter; print(adapter.COMMIT)")"

if [[ "$(git -C "$NFL" rev-parse HEAD 2>/dev/null || true)" != "$COMMIT" ]]; then
    rm -rf "${NFL:?}"
    git init -q "$NFL"
    git -C "$NFL" fetch -q --depth 1 "$URL" "$COMMIT"
    git -C "$NFL" -c advice.detachedHead=false checkout -q FETCH_HEAD
fi
[[ "$(git -C "$NFL" rev-parse HEAD)" == "$COMMIT" ]] || { echo "NFL-BA checkout is not at $COMMIT" >&2; exit 1; }
echo "NFL-BA $COMMIT at $NFL ($(git -C "$NFL" ls-files | wc -l) files)"

"$PYTHON" -m pip install -q uv
UV=("$PYTHON" -m uv pip install --system --python "$PYTHON" --break-system-packages --index-strategy unsafe-best-match)
"${UV[@]}" munch plyfile trimesh evo open3d rich wandb torchmetrics tqdm matplotlib pyyaml
"$PYTHON" -c "import torch, munch, plyfile, trimesh, evo, open3d, rich, wandb, torchmetrics, cv2, yaml; print('torch', torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), '| open3d', open3d.__version__, '| evo', evo.__version__)"

ARCH="$("$PYTHON" -c "import torch; print('%d.%d' % torch.cuda.get_device_capability(0))")"
STAMP="$BUILD/.built_for_sm$ARCH"
if [[ -f "$STAMP" ]] && "$PYTHON" -c "import torch, simple_knn._C, diff_gaussian_rasterization" 2>/dev/null; then
    echo "MonoGS extensions already built for sm $ARCH"
else
    rm -rf "${BUILD:?}"
    mkdir -p "$BUILD"
    for m in simple-knn diff-gaussian-rasterization-w-pose; do
        cp -r "$DEPS/$m" "$BUILD/$m"
    done
    h="$BUILD/diff-gaussian-rasterization-w-pose/cuda_rasterizer/rasterizer_impl.h"
    grep -q "<cstdint>" "$h" || sed -i '1i #include <cstdint>' "$h"
    grep -q "<cfloat>" "$BUILD/simple-knn/simple_knn.cu" || sed -i '1i #include <cfloat>' "$BUILD/simple-knn/simple_knn.cu"
    # pip -v (not uv) for the builds: it streams the compiler output.
    for m in simple-knn diff-gaussian-rasterization-w-pose; do
        echo "=== building $m for sm $ARCH"
        TORCH_CUDA_ARCH_LIST="$ARCH" "$PYTHON" -m pip install -v --no-build-isolation --force-reinstall --no-deps "$BUILD/$m"
    done
    touch "$STAMP"
fi
"$PYTHON" -c "import torch, simple_knn._C, diff_gaussian_rasterization; print('extensions ok')"
"$PYTHON" "$HERE/probe.py" "$NFL"
