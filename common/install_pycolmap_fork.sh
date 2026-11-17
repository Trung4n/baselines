#!/usr/bin/env bash
# Installs the pycolmap fork that gsplat's example trainers read COLMAP models
# with (rmbrualla/pycolmap, the commit pinned by gsplat's examples/
# requirements.txt), patched for numpy 2, into $PYTHON. Used by
# methods/bilagrid/setup_kaggle.sh and methods/luminancegs/setup_kaggle.sh.
#
# The fork defines   INVALID_POINT3D = np.uint64(-1)   at class level: numpy 1
# wrapped -1 to 2**64 - 1, numpy 2 raises OverflowError at import. Kaggle's
# Python 3.13 has no numpy 1.x wheels, so the pin numpy<2 of gsplat's examples
# cannot be met; the constant is written as np.uint64(2**64 - 1), the same
# value (it is only compared with point ids). Nothing else in the package is
# numpy-2 sensitive (grep of removed APIs, 2026-11-18). The fork's text reader
# is also broken under Python 3 (np.array(map(...))); the adapters write
# COLMAP's binary model, which it reads first.
#
# Fetched with git to $TMPDIR/pycolmap_fork, patched there, installed from it
# (--reinstall-package: an unpatched pycolmap of the same version may be there).
set -euo pipefail
: "${PYTHON:?PYTHON must be the notebook interpreter (kaggle_runner sets it)}"
URL="https://github.com/rmbrualla/pycolmap"
COMMIT="cc7ea4b7301720ac29287dbe450952511b32125e"
SRC="${TMPDIR:-/tmp}/pycolmap_fork"
LINE='    INVALID_POINT3D = np.uint64(-1)'
FIXED='    INVALID_POINT3D = np.uint64(2**64 - 1)  # baselines: was np.uint64(-1) (OverflowError under numpy 2), same value'

if [[ "$(git -C "$SRC" rev-parse HEAD 2>/dev/null || true)" != "$COMMIT" ]]; then
    rm -rf "${SRC:?}"
    git init -q "$SRC"
    git -C "$SRC" fetch -q --depth 1 "$URL" "$COMMIT"
    git -C "$SRC" -c advice.detachedHead=false checkout -q FETCH_HEAD
fi
[[ "$(git -C "$SRC" rev-parse HEAD)" == "$COMMIT" ]] || { echo "pycolmap fork is not at $COMMIT" >&2; exit 1; }
git -C "$SRC" checkout -q -- .   # back to the commit's files before patching (second call)
F="$SRC/pycolmap/scene_manager.py"
n="$(grep -cxF "$LINE" "$F" || true)"
[[ "$n" == 1 ]] || { echo "$F: expected one '$LINE', found $n" >&2; exit 1; }
"$PYTHON" - "$F" "$LINE" "$FIXED" <<'EOF'
import sys
path, line, fixed = sys.argv[1:4]
text = open(path, encoding="utf-8").read()
open(path, "w", encoding="utf-8").write(text.replace(line + "\n", fixed + "\n"))
EOF
grep -qxF "$FIXED" "$F" || { echo "$F: patch not applied" >&2; exit 1; }

"$PYTHON" -m pip install -q uv
"$PYTHON" -m uv pip install --system --python "$PYTHON" --break-system-packages --reinstall-package pycolmap "$SRC"
"$PYTHON" -c "import numpy, pycolmap; from pycolmap import SceneManager; assert SceneManager.INVALID_POINT3D == numpy.uint64(2**64 - 1); print('pycolmap fork', '$COMMIT'[:8], 'patched for numpy', numpy.__version__, 'from', pycolmap.__file__)"
