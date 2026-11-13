#!/usr/bin/env bash
# Kaggle setup of Photo-SLAM (RGB-D), called by common/kaggle_runner.setup("photoslam")
# from run_baseline.ipynb. Builds into ${TMPDIR:-/tmp}/photoslam_build
# (methods/photoslam/adapter.py looks there). Second call in a session: skips
# what is built.
#
# Dependencies Photo-SLAM needs that Ubuntu does not package:
# - OpenCV 4.10.0 + opencv_contrib 4.10.0 built WITH CUDA (Photo-SLAM's RGB-D
#   path uses cv::cuda::GpuMat / resize; it includes cudaimgproc, cudawarping,
#   cudastereo). Authors tested 4.7/4.8; 4.10 is the first release with fixes
#   for CUDA 12.4+ and GCC 13. Built with only the modules Photo-SLAM and its
#   ORB-SLAM3 include, for this GPU's compute capability only.
#   Cache: the installed OpenCV is packed to
#   /kaggle/working/photoslam_cache/opencv-<ver>-cuda<cuda>-sm<arch>.tar.gz;
#   add that file to a Kaggle dataset and attach it as input, and the next
#   session unpacks it from /kaggle/input instead of building (about an hour).
# - LibTorch C++ (cxx11 ABI): Photo-SLAM supports up to 2.1.2. CUDA 11.x ->
#   2.0.1+cu118 (the authors' setup), CUDA 12.x -> 2.1.2+cu121. Other CUDA
#   majors are refused.
# Build-only changes, on copies (third_party/Photo-SLAM stays as its commit):
# - CXXFLAGS '-include cstdint' (GCC 13), CMAKE_POLICY_VERSION_MINIMUM=3.5
#   (CMake 4 and the cmake_minimum_required 2.8 of ORB-SLAM3/DBoW2/g2o);
# - when nvcc is older than 12.4 (no GCC 13 support), g++-12 (CUDA 12.0-12.3)
#   or g++-11 (CUDA 11.x) as the CUDA host compiler;
# - -rpath-link to the system library directory (the NVIDIA GL libraries of
#   Kaggle, see methods/orbslam3/README.md); the driver runs without
#   LD_LIBRARY_PATH;
# - the driver methods/photoslam/photoslam_monogs.cpp added as one executable
#   (linked like examples/replica_rgbd, without the viewer); only it and what it
#   needs are built.
set -euo pipefail
: "${PYTHON:?PYTHON must be the notebook interpreter (kaggle_runner sets it)}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINES="$(cd "$HERE/../.." && pwd)"
BUILD="${TMPDIR:-/tmp}/photoslam_build"
DEPS="$BUILD/deps"
JOBS="$(nproc)"
OPENCV_VER=4.10.0
CACHE_OUT=/kaggle/working/photoslam_cache

PKGS=(build-essential cmake pkg-config wget unzip libeigen3-dev libboost-serialization-dev libssl-dev
      libjsoncpp-dev libglfw3-dev libglm-dev libgl1-mesa-dev libopengl-dev libjpeg-dev libpng-dev libtiff-dev)
missing=()
for p in "${PKGS[@]}"; do dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p"); done
if (( ${#missing[@]} )); then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends "${missing[@]}"
fi
export CMAKE_POLICY_VERSION_MINIMUM=3.5
export CXXFLAGS="${CXXFLAGS:-} -include cstdint"
echo "LD_LIBRARY_PATH was: ${LD_LIBRARY_PATH:-<unset>} (unset for Photo-SLAM)"
unset LD_LIBRARY_PATH
MULTIARCH_LIB="/usr/lib/$(gcc -print-multiarch)"

# CUDA toolkit, GPU, host compiler, LibTorch
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
NVCC_VER="$(nvcc --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')"
CUDA_MAJOR="${NVCC_VER%%.*}"; CUDA_MINOR="${NVCC_VER##*.}"
ARCH="$("$PYTHON" -c "import torch; print('%d.%d' % torch.cuda.get_device_capability(0))")"
ARCH_NUM="${ARCH/./}"
echo "nvcc $NVCC_VER at $CUDA_HOME, GPU sm $ARCH, $(gcc --version | head -1)"
HOSTCXX=""
if (( CUDA_MAJOR == 11 )); then
    HOSTCXX=g++-11; TORCH_URL="https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcu118.zip"
elif (( CUDA_MAJOR == 12 )); then
    (( CUDA_MINOR < 4 )) && HOSTCXX=g++-12
    TORCH_URL="https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.2%2Bcu121.zip"
else
    echo "CUDA $NVCC_VER: Photo-SLAM needs LibTorch <= 2.1.2, built for CUDA 11.8 or 12.1; no build for CUDA $CUDA_MAJOR" >&2
    exit 1
fi
if [[ -n "$HOSTCXX" ]]; then
    dpkg -s "$HOSTCXX" >/dev/null 2>&1 || apt-get install -y --no-install-recommends "$HOSTCXX"
    HOSTCXX="$(command -v "$HOSTCXX")"
    echo "CUDA host compiler: $HOSTCXX"
fi
CUDA_HOST_ARGS=()
[[ -n "$HOSTCXX" ]] && CUDA_HOST_ARGS=(-DCMAKE_CUDA_HOST_COMPILER="$HOSTCXX" -DCUDA_HOST_COMPILER="$HOSTCXX")

mkdir -p "$DEPS"
STAMP="$BUILD/.built"
if [[ -f "$STAMP" && -x "$BUILD/Photo-SLAM/bin/photoslam_monogs" ]]; then
    echo "Photo-SLAM already built in $BUILD"
else
    # ---------------------------------------------------------------- LibTorch
    if [[ ! -f "$DEPS/libtorch/share/cmake/Torch/TorchConfig.cmake" ]]; then
        echo "=== LibTorch: $TORCH_URL"
        wget -q --show-progress --progress=dot:giga -O "$DEPS/libtorch.zip" "$TORCH_URL"
        unzip -q "$DEPS/libtorch.zip" -d "$DEPS"
        rm -f "$DEPS/libtorch.zip"
    fi
    cat "$DEPS/libtorch/build-version" 2>/dev/null | sed 's/^/LibTorch /' || true

    # ---------------------------------------------------------------- OpenCV with CUDA
    OCV_TAG="opencv-$OPENCV_VER-cuda$NVCC_VER-sm$ARCH_NUM"
    OCV="$DEPS/opencv"
    if [[ ! -f "$OCV/lib/cmake/opencv4/OpenCVConfig.cmake" ]]; then
        cached="$( (ls /kaggle/input/*/"$OCV_TAG.tar.gz" /kaggle/input/*/*/"$OCV_TAG.tar.gz" \
                       /kaggle/input/*/*/*/"$OCV_TAG.tar.gz" "$CACHE_OUT/$OCV_TAG.tar.gz" 2>/dev/null || true) | head -1)"
        if [[ -n "$cached" ]]; then
            echo "=== OpenCV: unpacking the cached build $cached"
            tar -xzf "$cached" -C "$DEPS"
        else
            echo "=== OpenCV $OPENCV_VER + contrib with CUDA (sm $ARCH): no cached $OCV_TAG.tar.gz, building (about an hour)"
            for repo in opencv opencv_contrib; do
                [[ -d "$DEPS/$repo-$OPENCV_VER" ]] || {
                    wget -q -O "$DEPS/$repo.tar.gz" "https://github.com/opencv/$repo/archive/refs/tags/$OPENCV_VER.tar.gz"
                    sha256sum "$DEPS/$repo.tar.gz"
                    tar -xzf "$DEPS/$repo.tar.gz" -C "$DEPS"
                    rm -f "$DEPS/$repo.tar.gz"
                }
            done
            cmake -S "$DEPS/opencv-$OPENCV_VER" -B "$DEPS/opencv-build" -DCMAKE_BUILD_TYPE=Release \
                -DCMAKE_INSTALL_PREFIX="$OCV" -DOPENCV_EXTRA_MODULES_PATH="$DEPS/opencv_contrib-$OPENCV_VER/modules" \
                -DBUILD_LIST=core,imgproc,imgcodecs,highgui,features2d,calib3d,flann,cudev,cudaarithm,cudafilters,cudaimgproc,cudawarping,cudastereo \
                -DWITH_CUDA=ON -DCUDA_ARCH_BIN="$ARCH" -DCUDA_ARCH_PTX="" -DCUDA_TOOLKIT_ROOT_DIR="$CUDA_HOME" \
                -DWITH_CUDNN=OFF -DOPENCV_DNN_CUDA=OFF -DWITH_NVCUVID=OFF -DWITH_NVCUVENC=OFF -DWITH_CUFFT=OFF \
                -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF -DBUILD_JAVA=OFF \
                -DBUILD_opencv_python3=OFF -DBUILD_opencv_python2=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF \
                -DWITH_GTK=OFF -DWITH_QT=OFF -DWITH_OPENGL=OFF -DWITH_IPP=OFF -DWITH_ADE=OFF -DWITH_PROTOBUF=OFF \
                -DWITH_EIGEN=ON -DENABLE_PRECOMPILED_HEADERS=OFF -DOPENCV_GENERATE_PKGCONFIG=ON \
                "${CUDA_HOST_ARGS[@]}"
            cmake --build "$DEPS/opencv-build" -j "$JOBS"
            cmake --install "$DEPS/opencv-build"
            rm -rf "$DEPS/opencv-build"
            if [[ -d /kaggle/working ]]; then
                mkdir -p "$CACHE_OUT"
                tar -czf "$CACHE_OUT/$OCV_TAG.tar.gz" -C "$DEPS" opencv
                echo "cached: $CACHE_OUT/$OCV_TAG.tar.gz ($(du -h "$CACHE_OUT/$OCV_TAG.tar.gz" | cut -f1));" \
                     "add it to a Kaggle dataset and attach it to skip this build next time"
            fi
        fi
    fi
    OpenCV_DIR="$OCV/lib/cmake/opencv4"
    grep -m1 "OpenCV_VERSION " "$OpenCV_DIR/OpenCVConfig-version.cmake" 2>/dev/null || true
    grep -q "cudawarping" "$OpenCV_DIR/OpenCVModules.cmake" || { echo "OpenCV in $OCV has no CUDA modules" >&2; exit 1; }

    # ---------------------------------------------------------------- ORB-SLAM3 (Photo-SLAM's copy)
    P="$BUILD/Photo-SLAM"
    [[ -d "$P" ]] || cp -r "$BASELINES/third_party/Photo-SLAM" "$P"
    O="$P/ORB-SLAM3"
    for lib in DBoW2 g2o; do
        echo "=== $lib"
        cmake -S "$O/Thirdparty/$lib" -B "$O/Thirdparty/$lib/build" -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR="$OpenCV_DIR"
        cmake --build "$O/Thirdparty/$lib/build" -j "$JOBS"
    done
    [[ -f "$O/Vocabulary/ORBvoc.txt" ]] || tar -xf "$O/Vocabulary/ORBvoc.txt.tar.gz" -C "$O/Vocabulary"
    echo "=== ORB_SLAM3"
    cmake -S "$O" -B "$O/build" -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR="$OpenCV_DIR" \
        -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath-link,$MULTIARCH_LIB"
    cmake --build "$O/build" -j "$JOBS" --target ORB_SLAM3

    # ---------------------------------------------------------------- Photo-SLAM + driver
    echo "=== Photo-SLAM + photoslam_monogs"
    cp "$BASELINES/third_party/Photo-SLAM/CMakeLists.txt" "$P/CMakeLists.txt"
    cp "$HERE/photoslam_monogs.cpp" "$P/examples/photoslam_monogs.cpp"
    cat >> "$P/CMakeLists.txt" <<'EOF'

# baselines/methods/photoslam: the adapter's driver (replica_rgbd loop, no viewer, MonoGS dataset I/O)
add_executable(photoslam_monogs examples/photoslam_monogs.cpp)
target_link_libraries(photoslam_monogs
    gaussian_mapper
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so)
EOF
    cmake -S "$P" -B "$P/build" -DCMAKE_BUILD_TYPE=Release -DTorch_DIR="$DEPS/libtorch/share/cmake/Torch" \
        -DOpenCV_DIR="$OpenCV_DIR" -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" -DCMAKE_CUDA_ARCHITECTURES="$ARCH_NUM" \
        -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,$MULTIARCH_LIB" -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath-link,$MULTIARCH_LIB" \
        "${CUDA_HOST_ARGS[@]}"
    cmake --build "$P/build" -j "$JOBS" --target photoslam_monogs
    touch "$STAMP"
fi

DRV="$BUILD/Photo-SLAM/bin/photoslam_monogs"
ldd "$DRV" | grep -E "torch|opencv|ORB_SLAM3|gaussian|not found" | sed 's/^/  ldd: /' || true
out="$("$DRV" 2>&1 || true)"
if grep -q "Usage: photoslam_monogs" <<<"$out"; then
    echo "photoslam_monogs ok: $DRV"
else
    echo "$out"
    echo "photoslam_monogs does not start" >&2
    exit 1
fi
"$PYTHON" -c "import json; print('Photo-SLAM', json.load(open('$BASELINES/third_party/Photo-SLAM/.vendor.json'))['commit'])"
