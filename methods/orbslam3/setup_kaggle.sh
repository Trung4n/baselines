#!/usr/bin/env bash
# Kaggle setup of ORB-SLAM3 (RGB-D), called by common/kaggle_runner.setup("orbslam3")
# from run_baseline.ipynb. CPU only. Builds into ${TMPDIR:-/tmp}/orbslam3_build
# (methods/orbslam3/adapter.py looks there). Second call in a session: skips
# what is built.
#
# Builds COPIES of third_party/Pangolin (v0.8) and third_party/ORB_SLAM3 (v1.0);
# the sources there stay as their commits. Build-only changes on the copies:
# - ORB_SLAM3/CMakeLists.txt: -std=c++11 -> -std=c++17 (Pangolin 0.8 is C++17;
#   COMPILEDWITHC11 stays defined, as upstream), plus the driver
#   methods/orbslam3/rgbd_monogs.cc as one more executable;
# - ORB_SLAM3/src/LoopClosing.cc: 'mnFullBAIdx++;' -> 'mnFullBAIdx = true;'
#   (3 places; a bool member, ++ on bool is an error in C++17 and always set
#   true before, so behaviour is unchanged);
# - CXXFLAGS '-include cstdint' for every C++ build (GCC 13 no longer pulls in
#   <cstdint> transitively), CMAKE_POLICY_VERSION_MINIMUM=3.5 (CMake 4 refuses
#   cmake_minimum_required < 3.5, which DBoW2, g2o and ORB_SLAM3 declare);
# - only what the driver needs is built: Pangolin's libraries (no examples,
#   tools, Python console or optional video/image back-ends), DBoW2, g2o,
#   libORB_SLAM3 and rgbd_monogs (not the other examples, not Sophus's tests:
#   Sophus is header-only here).
set -euo pipefail
: "${PYTHON:?PYTHON must be the notebook interpreter (kaggle_runner sets it)}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINES="$(cd "$HERE/../.." && pwd)"
BUILD="${TMPDIR:-/tmp}/orbslam3_build"
JOBS="$(nproc)"

# System libraries. Plain apt-get output on purpose: progress lines keep the
# runner's stall watchdog quiet.
PKGS=(build-essential cmake pkg-config libeigen3-dev libopencv-dev libboost-serialization-dev libssl-dev
      libglew-dev libgl1-mesa-dev libegl1-mesa-dev libx11-dev)
missing=()
for p in "${PKGS[@]}"; do dpkg -s "$p" >/dev/null 2>&1 || missing+=("$p"); done
if (( ${#missing[@]} )); then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends "${missing[@]}"
fi
export CMAKE_POLICY_VERSION_MINIMUM=3.5
export CXXFLAGS="${CXXFLAGS:-} -include cstdint"
cmake --version | head -1
pkg-config --modversion opencv4 | sed 's/^/OpenCV /'
"$PYTHON" -c "import numpy, scipy, yaml; print('python deps ok')"

STAMP="$BUILD/.built"
if [[ -f "$STAMP" && -x "$BUILD/ORB_SLAM3/Examples/RGB-D/rgbd_monogs" ]]; then
    echo "ORB-SLAM3 already built in $BUILD"
else
    # Incremental: a failed earlier call in this session leaves its copies and
    # build trees; make only redoes what is missing. The files patched below
    # are restored from third_party/ first, so the patches apply to pristine
    # sources every time.
    mkdir -p "$BUILD"
    [[ -d "$BUILD/Pangolin" ]] || cp -r "$BASELINES/third_party/Pangolin" "$BUILD/Pangolin"
    [[ -d "$BUILD/ORB_SLAM3" ]] || cp -r "$BASELINES/third_party/ORB_SLAM3" "$BUILD/ORB_SLAM3"

    echo "=== Pangolin"
    cmake -S "$BUILD/Pangolin" -B "$BUILD/Pangolin/build" -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_EXAMPLES=OFF -DBUILD_TOOLS=OFF -DBUILD_TESTS=OFF -DBUILD_PANGOLIN_PYTHON=OFF \
        -DBUILD_PANGOLIN_FFMPEG=OFF -DBUILD_PANGOLIN_REALSENSE=OFF -DBUILD_PANGOLIN_REALSENSE2=OFF \
        -DBUILD_PANGOLIN_OPENNI=OFF -DBUILD_PANGOLIN_OPENNI2=OFF -DBUILD_PANGOLIN_LIBUVC=OFF \
        -DBUILD_PANGOLIN_LIBDC1394=OFF -DBUILD_PANGOLIN_V4L=OFF -DBUILD_PANGOLIN_LIBRAW=OFF \
        -DBUILD_PANGOLIN_LIBOPENEXR=OFF
    cmake --build "$BUILD/Pangolin/build" -j "$JOBS"

    O="$BUILD/ORB_SLAM3"
    for lib in DBoW2 g2o; do
        echo "=== $lib"
        cmake -S "$O/Thirdparty/$lib" -B "$O/Thirdparty/$lib/build" -DCMAKE_BUILD_TYPE=Release
        cmake --build "$O/Thirdparty/$lib/build" -j "$JOBS"
    done

    echo "=== vocabulary"
    [[ -f "$O/Vocabulary/ORBvoc.txt" ]] || tar -xf "$O/Vocabulary/ORBvoc.txt.tar.gz" -C "$O/Vocabulary"

    echo "=== ORB_SLAM3 + rgbd_monogs"
    cp "$BASELINES/third_party/ORB_SLAM3/CMakeLists.txt" "$O/CMakeLists.txt"
    cp "$BASELINES/third_party/ORB_SLAM3/src/LoopClosing.cc" "$O/src/LoopClosing.cc"
    # C++17 forbids ++ on a bool: LoopClosing.h declares 'bool mnFullBAIdx' and
    # LoopClosing.cc increments it in 3 places. For a bool, x++ always sets
    # true (C++14), so '= true' is the same behaviour.
    n="$(grep -c 'mnFullBAIdx++;' "$O/src/LoopClosing.cc" || true)"
    [[ "$n" == 3 ]] || { echo "expected 3 'mnFullBAIdx++;' in LoopClosing.cc, found $n" >&2; exit 1; }
    sed -i 's|mnFullBAIdx++;|mnFullBAIdx = true;  // baselines: was mnFullBAIdx++ on a bool (C++17)|' "$O/src/LoopClosing.cc"
    grep -q 'set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")' "$O/CMakeLists.txt"
    sed -i 's|set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")|set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")|' "$O/CMakeLists.txt"
    cp "$HERE/rgbd_monogs.cc" "$O/Examples/RGB-D/rgbd_monogs.cc"
    cat >> "$O/CMakeLists.txt" <<'EOF'

# baselines/methods/orbslam3: the adapter's driver (rgbd_tum loop, MonoGS dataset I/O)
add_executable(rgbd_monogs Examples/RGB-D/rgbd_monogs.cc)
target_link_libraries(rgbd_monogs ${PROJECT_NAME})
set_target_properties(rgbd_monogs PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D)
EOF
    cmake -S "$O" -B "$O/build" -DCMAKE_BUILD_TYPE=Release -DPangolin_DIR="$BUILD/Pangolin/build"
    cmake --build "$O/build" -j "$JOBS" --target ORB_SLAM3 rgbd_monogs
    touch "$STAMP"
fi

# The driver starts (links and loads) and prints its usage.
if "$BUILD/ORB_SLAM3/Examples/RGB-D/rgbd_monogs" 2>&1 | grep -q "Usage: rgbd_monogs"; then
    echo "rgbd_monogs ok: $BUILD/ORB_SLAM3/Examples/RGB-D/rgbd_monogs"
else
    "$BUILD/ORB_SLAM3/Examples/RGB-D/rgbd_monogs" || true
    echo "rgbd_monogs does not start" >&2
    exit 1
fi
ls -la "$BUILD/ORB_SLAM3/Vocabulary/ORBvoc.txt"
"$PYTHON" -c "import json; print('ORB-SLAM3', json.load(open('$BASELINES/third_party/ORB_SLAM3/.vendor.json'))['commit'], '| Pangolin', json.load(open('$BASELINES/third_party/Pangolin/.vendor.json'))['tag'])"
