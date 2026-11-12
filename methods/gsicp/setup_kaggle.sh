#!/usr/bin/env bash
# Kaggle setup of GS-ICP SLAM, called by common/kaggle_runner.setup("gsicp")
# from run_baseline.ipynb. Installs into the notebook's Python ($PYTHON);
# Kaggle's torch is kept. Second call in a session: skips what is done.
#
# Builds COPIES of third_party/GS_ICP_SLAM/submodules (the sources stay as
# their commits, third_party/GS_ICP_SLAM/.vendor.json), with build-only fixes:
# - rasterizer: '#include <cstdint>' (newer GCC), simple-knn: '#include <cfloat>'
#   (FLT_MAX), each only when missing;
# - fast_gicp: its bundled pybind11 2.6 builds neither with Python >= 3.11 nor
#   with CMake >= 4 (cmake_minimum_required 3.4), so the copy's CMakeLists
#   takes the pybind11 installed here (find_package) instead of
#   add_subdirectory(thirdparty/pybind11). The bindings code is unchanged.
set -euo pipefail
: "${PYTHON:?PYTHON must be the notebook interpreter (kaggle_runner sets it)}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINES="$(cd "$HERE/../.." && pwd)"
SRC="$BASELINES/third_party/GS_ICP_SLAM"
BUILD="${TMPDIR:-/tmp}/gsicp_build"

# System libraries for fast_gicp: PCL and Eigen (its CMake find_package's them).
# Plain apt-get output on purpose: progress lines keep the runner's stall watchdog quiet.
if ! dpkg -s libpcl-dev >/dev/null 2>&1 || ! dpkg -s libeigen3-dev >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends libpcl-dev libeigen3-dev
fi

"$PYTHON" -m pip install -q uv
UV=("$PYTHON" -m uv pip install --system --python "$PYTHON" --break-system-packages --index-strategy unsafe-best-match)
# GS-ICP's requirements.txt, except opencv-contrib-python: only cv2.imread,
# cvtColor and resize are used, which Kaggle's opencv-python provides
# (installing contrib next to it would shadow it). cmake and pybind11 for the
# fast_gicp build.
"${UV[@]}" open3d scipy tqdm torchmetrics plyfile rerun-sdk "pybind11>=2.11"
# CMake: the image's own if any, else a 3.x one (CMake 4 drops support for
# projects declaring cmake_minimum_required < 3.5, which some find-modules of
# PCL's dependencies still do); the policy floor below covers a CMake 4 anyway.
command -v cmake >/dev/null || "${UV[@]}" "cmake>=3.18,<4"
export CMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --version | head -1
"$PYTHON" -c "import torch, cv2, open3d, scipy, tqdm, torchmetrics, plyfile, rerun, pybind11; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0), '| open3d', open3d.__version__, '| pybind11', pybind11.__version__)"

ARCH="$("$PYTHON" -c "import torch; print('%d.%d' % torch.cuda.get_device_capability(0))")"
STAMP="$BUILD/.built_for_sm$ARCH"
if [[ -f "$STAMP" ]] && "$PYTHON" -c "import diff_gaussian_rasterization, simple_knn._C, pygicp" 2>/dev/null; then
    echo "GS-ICP extensions already built for sm $ARCH"
else
    rm -rf "$BUILD"
    mkdir -p "$BUILD"
    for m in diff-gaussian-rasterization simple-knn fast_gicp; do
        cp -r "$SRC/submodules/$m" "$BUILD/$m"
    done
    grep -q "<cstdint>" "$BUILD/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h" \
        || sed -i '1i #include <cstdint>' "$BUILD/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h"
    grep -q "<cfloat>" "$BUILD/simple-knn/simple_knn.cu" \
        || sed -i '1i #include <cfloat>' "$BUILD/simple-knn/simple_knn.cu"
    PYBIND11_DIR="$("$PYTHON" -m pybind11 --cmakedir)"
    grep -q "add_subdirectory(thirdparty/pybind11)" "$BUILD/fast_gicp/CMakeLists.txt"
    sed -i "s|add_subdirectory(thirdparty/pybind11)|find_package(pybind11 CONFIG REQUIRED PATHS \"$PYBIND11_DIR\" NO_DEFAULT_PATH)|" \
        "$BUILD/fast_gicp/CMakeLists.txt"

    # pip -v (not uv) for the builds: it streams the compiler output.
    # --no-build-isolation: build against the installed torch / pybind11.
    for m in diff-gaussian-rasterization simple-knn; do
        echo "=== building $m for sm $ARCH"
        TORCH_CUDA_ARCH_LIST="$ARCH" "$PYTHON" -m pip install -v --no-build-isolation --force-reinstall --no-deps "$BUILD/$m"
    done
    echo "=== building fast_gicp (pygicp)"
    CMAKE_BUILD_PARALLEL_LEVEL="$(nproc)" "$PYTHON" -m pip install -v --no-build-isolation --force-reinstall --no-deps "$BUILD/fast_gicp"
    touch "$STAMP"
fi
"$PYTHON" -c "import diff_gaussian_rasterization, simple_knn._C, pygicp; r = pygicp.FastGICP(); print('extensions ok:', diff_gaussian_rasterization.__file__, pygicp.__file__)"
"$PYTHON" -c "import json; print('GS-ICP SLAM', json.load(open('$BASELINES/third_party/GS_ICP_SLAM/.vendor.json'))['commit'])"
