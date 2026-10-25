"""Sample GPU power with nvidia-smi, and run llama-benchy with the sampler alongside."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from typing import IO, List, Optional, Sequence

from .power import OUR_HEADER

QUERY = "index,power.draw,temperature.gpu,memory.used"
RESERVED_FLAGS = ("--emit-progress", "--save-result", "--format")


def smi_base(smi_cmd: Optional[str]) -> List[str]:
    return shlex.split(smi_cmd) if smi_cmd else ["nvidia-smi"]


def smi_available(smi_cmd: Optional[str] = None) -> bool:
    base = smi_base(smi_cmd)
    return shutil.which(base[0]) is not None


def gpu_names(smi_cmd: Optional[str] = None) -> str:
    """'NVIDIA GeForce RTX 3090', or '2x NVIDIA GeForce RTX 3090' for a pair."""
    try:
        out = subprocess.run(smi_base(smi_cmd) + ["--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return ""
    names = [n.strip() for n in out.stdout.splitlines() if n.strip()]
    if out.returncode != 0 or not names:
        return ""
    counts = Counter(names)
    return " + ".join(n if c == 1 else f"{c}x {n}" for n, c in counts.items())


class PowerSampler:
    """Streams `nvidia-smi -lms` into our CSV layout, stamping each line with unix time on arrival."""

    def __init__(self, out: IO[str], interval_ms: int = 250, smi_cmd: Optional[str] = None,
                 gpus: Optional[Sequence[int]] = None):
        self.out = out
        cmd = smi_base(smi_cmd) + [f"--query-gpu={QUERY}", "--format=csv,noheader,nounits", f"-lms={interval_ms}"]
        if gpus:
            cmd.append("--id=" + ",".join(str(g) for g in gpus))
        self.cmd = cmd
        self.proc: Optional[subprocess.Popen] = None
        self.thread: Optional[threading.Thread] = None
        self.lines = 0

    def start(self) -> "PowerSampler":
        self.out.write(",".join(OUR_HEADER) + "\n")
        self.out.flush()
        self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     text=True, bufsize=1)
        self.thread = threading.Thread(target=self._pump, daemon=True)
        self.thread.start()
        return self

    def _pump(self) -> None:
        assert self.proc and self.proc.stdout
        for line in self.proc.stdout:
            now = time.time()
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) < 2:
                continue
            self.out.write(f"{now:.3f}," + ",".join(parts[:4]) + "\n")
            self.out.flush()
            self.lines += 1

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.thread:
            self.thread.join(timeout=5)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


def check_benchy_args(cmd: Sequence[str]) -> Optional[str]:
    if not cmd:
        return "give the llama-benchy command after --, e.g. -- llama-benchy --base-url http://localhost:8080/v1"
    for arg in cmd:
        for flag in RESERVED_FLAGS:
            if arg == flag or arg.startswith(flag + "="):
                return f"{flag} is set by record; leave it out of the llama-benchy command"
    return None


def run_benchy(cmd: Sequence[str], workdir: str, *, power: bool, interval_ms: int, idle_seconds: float,
               smi_cmd: Optional[str], gpus: Optional[Sequence[int]]):
    """Run llama-benchy writing report + progress into workdir. Returns (rc, report, progress, power_csv)."""
    report = os.path.join(workdir, "report.json")
    progress = os.path.join(workdir, "progress.jsonl")
    power_csv = os.path.join(workdir, "power.csv") if power else None
    full = list(cmd) + ["--emit-progress", progress, "--format", "json", "--save-result", report]
    sampler = None
    fh = None
    try:
        if power_csv:
            fh = open(power_csv, "w", encoding="utf-8", newline="\n")
            sampler = PowerSampler(fh, interval_ms, smi_cmd, gpus).start()
            time.sleep(max(idle_seconds, 0.0))
        rc = subprocess.call(full)
    except FileNotFoundError:
        print(f"amagi-benchy-archive: cannot run {cmd[0]!r}; is llama-benchy installed? "
              "(pipx install llama-benchy, or use uvx llama-benchy)", file=sys.stderr)
        rc = 127
    finally:
        if sampler:
            time.sleep(min(max(interval_ms / 1000.0, 0.2), 2.0))
            sampler.stop()
        if fh:
            fh.close()
    return rc, report, progress, power_csv
