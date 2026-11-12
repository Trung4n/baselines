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
  honour (e.g. checkpoints) with an error, never ignores it. It prints
  progress often enough for the stall watchdog (stall_min) of run().

Output of setup and runs goes BOTH to the notebook (repetitive progress lines
thinned out, see Screen) and, complete, to a log file. Every child is stopped
after a total timeout, and after stall_min minutes without any output (a
hang). Children write to a pipe read here: a subprocess writing straight to
the kernel's stdout would not show in the notebook cell at all.

Not named kaggle.py: Kaggle images ship the `kaggle` API package.
"""
import codecs
import glob
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import zipfile

COMMON = os.path.dirname(os.path.abspath(__file__))
BASELINES = os.path.dirname(COMMON)
METHODS = os.path.join(BASELINES, "methods")
LOG_NAME = "run.log"  # the adapters pack <work>/run.log into run/ of the zip
KILL_GRACE_S = 60  # after a timeout: SIGTERM to the process group, SIGKILL this much later


def method_dir(method):
    d = os.path.join(METHODS, method)
    missing = [f for f in ("adapter.py", "setup_kaggle.sh", "kaggle.json") if not os.path.isfile(os.path.join(d, f))]
    if missing:
        have = sorted(m for m in os.listdir(METHODS) if os.path.isfile(os.path.join(METHODS, m, "adapter.py")))
        raise FileNotFoundError(f"methods/{method}: missing {missing}; methods with an adapter: {have}")
    return d


def method_info(method):
    """methods/<method>/kaggle.json: what the method supports. Required key
    "checkpoints" (bool): whether its adapter can save checkpoints and
    --resume. Kept in the repo, not in the notebook, so an older copy of the
    notebook still runs a method added later."""
    path = os.path.join(method_dir(method), "kaggle.json")
    with open(path, encoding="utf-8") as f:
        info = json.load(f)
    if not isinstance(info.get("checkpoints"), bool):
        raise KeyError(f"{path}: needs \"checkpoints\": true|false")
    return info


def scene_variant(dataset):
    """(scene, variant) as score.py names a run."""
    d = os.path.normpath(dataset)
    return os.path.basename(os.path.dirname(d)), os.path.basename(d)


def work_dir(method, dataset, out):
    scene, variant = scene_variant(dataset)
    return os.path.join(out, f"{method}__{scene}_{variant}")


# ---------------------------------------------------------------- streaming

class Screen:
    """Prints a child's output in the notebook without flooding it.

    Lines are grouped by their text with the numbers dropped ("Tracking Time
    Step: 12: 40%|...| 16/40" and "Selected Keyframes at Frame 12: [...]"
    each form one group, however they interleave). A group is printed at most
    once per progress_s seconds; the lines skipped in between are counted and
    the count is appended when the group prints again. tqdm's
    carriage-return updates are such lines. A line unlike any recent one (an
    error, a traceback, a warning) always prints. The log file keeps
    everything."""

    MAX_GROUPS = 1000  # forget the oldest groups beyond this many

    def __init__(self, progress_s):
        self.progress_s = progress_s
        self.buf = ""
        self.groups = {}  # key -> [last printed time, skipped since]

    @staticmethod
    def key(line):
        return re.sub(r"[\d.:/,]+", "#", line.strip())[:60]

    def line(self, text):
        if not text.strip():
            return
        k, now = self.key(text), time.time()
        g = self.groups.get(k)
        if g is not None and now - g[0] < self.progress_s:
            g[1] += 1
            return
        extra = f"   [+{g[1]} similar]" if g is not None and g[1] else ""
        print(text.rstrip() + extra, flush=True)
        self.groups.pop(k, None)
        self.groups[k] = [now, 0]
        if len(self.groups) > self.MAX_GROUPS:
            self.groups.pop(next(iter(self.groups)))

    def feed(self, text):
        self.buf += text
        parts = re.split(r"[\r\n]", self.buf)
        self.buf = parts.pop()
        for p in parts:
            self.line(p)

    def close(self):
        if self.buf:
            self.line(self.buf)
        skipped = sum(g[1] for g in self.groups.values())
        if skipped:
            print(f"   [{skipped} progress lines not shown, all in the log]", flush=True)
        self.buf, self.groups = "", {}


def _kill(proc):
    """SIGTERM to the child's whole process group (bash, uv, nvcc, python),
    then SIGKILL after KILL_GRACE_S."""
    for sig, wait in ((signal.SIGTERM, KILL_GRACE_S), (signal.SIGKILL, None)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


def stream(cmd, log, timeout_s, stall_s, progress_s, env=None, append=False):
    """Runs cmd with its output in the notebook (Screen) and, complete, in the
    log file. Stops it after timeout_s in total, or after stall_s without any
    output. Returns (exit code, or None when stopped; "ok" | "timeout" |
    "stall")."""
    env = dict(os.environ if env is None else env, PYTHONUNBUFFERED="1")
    screen = Screen(progress_s)
    chunks = queue.Queue()
    with open(log, "a" if append else "w", encoding="utf-8") as f:
        head = "$ " + " ".join(cmd)
        f.write(head + "\n")
        f.flush()
        print(f"{head}\n  log: {log}", flush=True)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                                start_new_session=True)

        def reader():
            while True:
                b = os.read(proc.stdout.fileno(), 65536)
                chunks.put(b)
                if not b:
                    return

        threading.Thread(target=reader, daemon=True).start()
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        t0 = last = time.time()
        reason = "ok"
        while True:
            try:
                b = chunks.get(timeout=1.0)
            except queue.Empty:
                b = None
            now = time.time()
            if b == b"":
                break
            if b:
                text = decoder.decode(b)
                f.write(text)
                f.flush()
                screen.feed(text)
                last = now
            if now - t0 > timeout_s:
                reason = "timeout"
            elif stall_s is not None and now - last > stall_s:
                reason = "stall"
            if reason != "ok":
                screen.close()
                what = (f"over {timeout_s / 3600:.2f} h in total" if reason == "timeout"
                        else f"no output for {stall_s / 60:.0f} min")
                msg = f"*** {reason.upper()}: {what}, stopping the process ***"
                f.write("\n" + msg + "\n")
                f.flush()
                print(msg, flush=True)
                _kill(proc)
                break
        screen.close()
        code = None
        if reason == "ok":  # output closed: the process ends now, or is stopped
            try:
                code = proc.wait(timeout=KILL_GRACE_S)
            except subprocess.TimeoutExpired:
                reason = "stall"
                print(f"*** STALL: output closed but the process still runs after {KILL_GRACE_S} s, "
                      "stopping it ***", flush=True)
                _kill(proc)
        f.write(f"\n[{reason}, exit {code}, {(time.time() - t0) / 3600:.2f} h]\n")
    return code, reason


# ---------------------------------------------------------------- notebook API

def setup(method, timeout_min, stall_min, progress_s=30):
    """bash methods/<method>/setup_kaggle.sh with PYTHON = this kernel's
    interpreter, shown in the notebook; raises if it fails or hangs."""
    script = os.path.join(method_dir(method), "setup_kaggle.sh")
    log = os.path.join(tempfile.gettempdir(), f"setup_{method}.log")
    t0 = time.time()
    code, reason = stream(["bash", script], log, timeout_min * 60, stall_min * 60, progress_s,
                          env=dict(os.environ, PYTHON=sys.executable))
    if code != 0:
        why = f"exit code {code}" if reason == "ok" else f"stopped ({reason})"
        raise RuntimeError(f"setup {method} FAILED: {why}; see the output above (log: {log})")
    print(f"=== setup {method}: ok, {time.time() - t0:.0f} s", flush=True)


def run(method, dataset, out, checkpoint_every, timeout_h, stall_min, progress_s=30, resume=False, no_map=False):
    """adapter.py run on one dataset, output to <out>/<method>__<scene>_<variant>/
    (complete log: run.log there), the zip next to it; output shown in the
    notebook as it comes. Stopped after timeout_h, or after stall_min without
    output. Returns the zip path; None if the run failed, timed out or hung
    (resume it with resume=True, from its checkpoints)."""
    if not os.path.isdir(os.path.join(dataset, "results")):
        raise FileNotFoundError(f"{dataset}: no results/ (a MonoGS dataset folder?)")
    info = method_info(method)
    if not info["checkpoints"]:
        if resume:
            raise ValueError(f"{method} has no checkpoints, nothing to resume ({info.get('why', '')}): run it again")
        if checkpoint_every:
            print(f"note: {method} has no checkpoints ({info.get('why', '')}): "
                  f"CKPT_EVERY={checkpoint_every} not used, run with --checkpoint_every 0", flush=True)
            checkpoint_every = 0
    adapter = os.path.join(method_dir(method), "adapter.py")
    work = work_dir(method, dataset, out)
    os.makedirs(work, exist_ok=True)
    cmd = [sys.executable, adapter, "run", "--dataset", dataset, "--work", work,
           "--checkpoint_every", str(int(checkpoint_every))]
    cmd += ["--resume"] if resume else []
    cmd += ["--no_map"] if no_map else []
    pattern = os.path.join(out, f"{os.path.basename(work)}__*.zip")
    zips_before = set(glob.glob(pattern))
    print(f"=== {method} on {dataset}", flush=True)
    t0 = time.time()
    code, reason = stream(cmd, os.path.join(work, LOG_NAME), timeout_h * 3600, stall_min * 60, progress_s,
                          append=resume)
    new = sorted(set(glob.glob(pattern)) - zips_before)
    status = ("OK" if code == 0 else f"FAILED, exit code {code}") if reason == "ok" else f"STOPPED ({reason})"
    print(f"=== {method} on {dataset}: {status}, {(time.time() - t0) / 3600:.2f} h", flush=True)
    if code == 0 and len(new) == 1:
        print(f"zip: {new[0]}", flush=True)
        return new[0]
    if code != 0:
        print("no zip: fix the error above, then "
              + (f"resume with run(..., resume=True) from the checkpoints in {work}" if info["checkpoints"]
                 else f"run it again ({method} has no checkpoints; delete {work} first)"), flush=True)
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
