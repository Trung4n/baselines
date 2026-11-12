"""Runs any method of baselines/methods on Kaggle, from the one notebook
run_baseline.ipynb: the notebook sets METHOD and calls these functions.

Contract of a method folder methods/<method>/ (baselines/CLAUDE.md):
- setup_kaggle.sh: installs the method's dependencies into the notebook's
  Python (env PYTHON) with uv, builds its extensions for the local GPU;
  safe to call twice in one session;
- adapter.py run --dataset <folder> --work <dir> --checkpoint_every <n>
  [--resume] [--no_map]: runs the method on one MonoGS dataset folder and
  ends with <dir>/../<method>__<scene>_<variant>__<start>.zip (PROTOCOL.md,
  "Gói kết quả của một lần chạy"). An adapter rejects an option it cannot
  honour (e.g. checkpoints) with an error, never ignores it.

Not named kaggle.py: Kaggle images ship the `kaggle` API package.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile

COMMON = os.path.dirname(os.path.abspath(__file__))
BASELINES = os.path.dirname(COMMON)
METHODS = os.path.join(BASELINES, "methods")
LOG_NAME = "run.log"  # the adapters pack <work>/run.log into run/ of the zip
KILL_GRACE_S = 300  # after the timeout: SIGTERM, then SIGKILL this much later


def method_dir(method):
    d = os.path.join(METHODS, method)
    missing = [f for f in ("adapter.py", "setup_kaggle.sh") if not os.path.isfile(os.path.join(d, f))]
    if missing:
        have = sorted(m for m in os.listdir(METHODS) if os.path.isfile(os.path.join(METHODS, m, "adapter.py")))
        raise FileNotFoundError(f"methods/{method}: missing {missing}; methods with an adapter: {have}")
    return d


def scene_variant(dataset):
    """(scene, variant) as score.py names a run."""
    d = os.path.normpath(dataset)
    return os.path.basename(os.path.dirname(d)), os.path.basename(d)


def work_dir(method, dataset, out):
    scene, variant = scene_variant(dataset)
    return os.path.join(out, f"{method}__{scene}_{variant}")


def setup(method):
    """bash methods/<method>/setup_kaggle.sh with PYTHON = this kernel's
    interpreter; output streamed to the notebook."""
    script = os.path.join(method_dir(method), "setup_kaggle.sh")
    env = dict(os.environ, PYTHON=sys.executable)
    t0 = time.time()
    subprocess.run(["bash", script], env=env, check=True)
    print(f"setup {method}: {time.time() - t0:.0f} s")


def tail(path, n):
    with open(path, encoding="utf-8", errors="replace") as f:
        return "".join(f.readlines()[-n:])


def run(method, dataset, out, checkpoint_every, timeout_h, resume=False, no_map=False, tail_lines=25):
    """adapter.py run on one dataset, output to <out>/<method>__<scene>_<variant>/
    (log: run.log there), the zip next to it. Returns the zip path, None if
    the run failed or timed out (resume it in a later session)."""
    if not os.path.isdir(os.path.join(dataset, "results")):
        raise FileNotFoundError(f"{dataset}: no results/ (a MonoGS dataset folder?)")
    adapter = os.path.join(method_dir(method), "adapter.py")
    work = work_dir(method, dataset, out)
    os.makedirs(work, exist_ok=True)
    cmd = [sys.executable, adapter, "run", "--dataset", dataset, "--work", work,
           "--checkpoint_every", str(int(checkpoint_every))]
    cmd += ["--resume"] if resume else []
    cmd += ["--no_map"] if no_map else []
    zips_before = set(glob.glob(os.path.join(out, f"{os.path.basename(work)}__*.zip")))
    log = os.path.join(work, LOG_NAME)
    print(f"{method} on {dataset}\n  log: {log}")
    t0 = time.time()
    with open(log, "a" if resume else "w", encoding="utf-8") as f:
        f.write(f"$ {' '.join(cmd)}\n")
        f.flush()
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        try:
            code = proc.wait(timeout=timeout_h * 3600)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=KILL_GRACE_S)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            code = None
    print(tail(log, tail_lines))
    new = sorted(set(glob.glob(os.path.join(out, f"{os.path.basename(work)}__*.zip"))) - zips_before)
    status = "TIMEOUT" if code is None else f"exit {code}"
    print(f"{status} after {(time.time() - t0) / 3600:.2f} h")
    if code == 0 and len(new) == 1:
        print(f"zip: {new[0]}")
        return new[0]
    if code is None or code != 0:
        print(f"no zip: resume later with run(..., resume=True) from the checkpoints in {work}")
        return None
    raise RuntimeError(f"exit 0 but {len(new)} new zips for {work}: {new}")


def collect(method, out, remove_work=True):
    """Checks every zip of `method` in `out` (readable, summary.json) and,
    with remove_work, deletes the work folder of each checked run, so the
    notebook output holds the zips only."""
    zips = sorted(glob.glob(os.path.join(out, f"{method}__*__*.zip")))
    if not zips:
        print(f"no {method} zips in {out}")
    for z in zips:
        with zipfile.ZipFile(z) as f:
            bad = f.testzip()
            if bad is not None:
                raise ValueError(f"{z}: corrupt member {bad}")
            top = f.namelist()[0].split("/")[0]
            summary = json.loads(f.read(f"{top}/summary.json"))
        keys = ("frames", "keyframes", "gaussians", "fps", "sessions")
        print(f"{os.path.basename(z)}  {os.path.getsize(z) / 2**20:.0f} MiB  "
              + json.dumps({k: summary[k] for k in keys if k in summary}))
        work = os.path.join(out, top.rsplit("__", 1)[0])
        if remove_work and os.path.isdir(work):
            # Only the work folder this zip came from: a later run of the same
            # scene (unfinished, resumable) keeps its checkpoints.
            seg_path = os.path.join(work, "timing_segments.json")
            started = None
            if os.path.isfile(seg_path):
                with open(seg_path, encoding="utf-8") as f:
                    started = json.load(f)[0]["started_at"]
            if started == summary["started_at"]:
                shutil.rmtree(work)
                print(f"  removed {work}")
            else:
                print(f"  kept {work}: it holds another run (started {started})")
    return zips
