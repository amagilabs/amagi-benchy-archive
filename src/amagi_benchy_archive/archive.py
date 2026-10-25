"""The archive: one JSON file per run under ``<archive>/runs/``.

Plain files rather than a database, so several machines can each record
locally and be combined with ``merge``, rsync, or git. A run's id is a hash
of its content, so ingesting the same files twice, or merging an archive
into itself, never makes a duplicate.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

from . import __version__
from .ingest import Parsed
from .power import Sample, baseline_power, window_power

RUN_SCHEMA = "amagi-benchy-archive.run.v1"
ENV_VAR = "AMAGI_BENCHY_ARCHIVE"
DEFAULT_DIR = "benchy-archive"


def archive_dir(explicit: Optional[str] = None) -> str:
    return explicit or os.environ.get(ENV_VAR) or DEFAULT_DIR


def iso(ts: Optional[float]) -> str:
    if ts is None:
        return ""
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _report_ts_to_iso(text: str) -> str:
    """llama-benchy writes '2026-02-07 19:09:00Z'."""
    text = (text or "").strip()
    for fmt in ("%Y-%m-%d %H:%M:%SZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            continue
    return ""


def build_run(parsed: Parsed, *, machine: str, gpu: str = "", engine: str = "", quant: str = "",
              tag: str = "", note: str = "", model: str = "",
              samples: Optional[Sequence[Sample]] = None, gpus: Optional[Sequence[int]] = None,
              sources: Optional[Dict[str, str]] = None, power_source: str = "") -> dict:
    if not machine:
        raise ValueError("machine name is required")
    started = iso(parsed.started_ts) or _report_ts_to_iso(parsed.report_timestamp)
    ended = iso(parsed.ended_ts) or _report_ts_to_iso(parsed.report_timestamp)
    run = {
        "schema": RUN_SCHEMA,
        "run_id": "",
        "machine": machine,
        "gpu": gpu,
        "engine": engine,
        "quant": quant,
        "model": model or parsed.model,
        "tag": tag,
        "note": note,
        "started": started,
        "ended": ended,
        "status": parsed.status,
        "llama_benchy_version": parsed.llama_benchy_version,
        "latency_mode": parsed.latency_mode,
        "latency_ms": parsed.latency_ms,
        "sources": dict(sources or {}),
        "archived_by": f"amagi-benchy-archive {__version__}",
        "power": None,
        "cells": [],
    }
    timed = [c for c in parsed.cells if c.start_ts is not None and c.end_ts is not None]
    if samples:
        first = min((c.start_ts for c in timed), default=None)
        run["power"] = {
            "source": power_source,
            "gpus": sorted({s.gpu for s in samples if gpus is None or s.gpu in gpus}),
            "samples": len(samples),
            "idle_w": baseline_power(samples, first, gpus) if first is not None else None,
            "joined": bool(timed),
        }
    for c in parsed.cells:
        cell = {
            "pp": c.pp, "tg": c.tg, "depth": c.depth, "concurrency": c.concurrency, "phase": c.phase,
            "label": c.label,
            "metrics": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in m.items()}
                        for k, m in sorted(c.metrics.items())},
            "start": iso(c.start_ts), "end": iso(c.end_ts),
            "requests": c.requests, "errors": c.errors,
            "power": None, "decode_power": None, "tok_per_j": None,
        }
        if samples and c.start_ts is not None and c.end_ts is not None:
            cell["power"] = window_power(samples, [(c.start_ts, c.end_ts)], gpus)
            if c.decode_windows:
                cell["decode_power"] = window_power(samples, c.decode_windows, gpus)
            watts = (cell["decode_power"] or cell["power"] or {}).get("avg_w")
            tg = c.metrics.get("tg_tps", {}).get("mean")
            if watts and tg:
                cell["tok_per_j"] = round(tg / watts, 4)
        run["cells"].append(cell)
    run["run_id"] = run_id(run)
    return run


def run_id(run: dict) -> str:
    """Stable id from what was measured, not from file names or archive time."""
    core = {k: run.get(k) for k in ("machine", "model", "started", "ended", "llama_benchy_version")}
    core["cells"] = [(c["label"], c["metrics"].get("tg_tps"), c["metrics"].get("pp_tps"))
                     for c in run.get("cells", [])]
    blob = json.dumps(core, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def runs_dir(root: str) -> str:
    return os.path.join(root, "runs")


def save_run(root: str, run: dict, raw_files: Optional[Dict[str, str]] = None) -> str:
    os.makedirs(runs_dir(root), exist_ok=True)
    path = os.path.join(runs_dir(root), f"{run['run_id']}.json")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(run, fh, indent=2, sort_keys=False)
        fh.write("\n")
    if raw_files:
        raw = os.path.join(root, "raw", run["run_id"])
        os.makedirs(raw, exist_ok=True)
        for kind, src in raw_files.items():
            if src and os.path.isfile(src):
                shutil.copyfile(src, os.path.join(raw, f"{kind}{os.path.splitext(src)[1] or '.txt'}"))
    return path


def load_runs(root: str) -> List[dict]:
    d = runs_dir(root)
    if not os.path.isdir(d):
        return []
    runs = []
    for name in sorted(os.listdir(d)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(d, name), encoding="utf-8") as fh:
            try:
                run = json.load(fh)
            except json.JSONDecodeError:
                continue
        if run.get("schema") == RUN_SCHEMA:
            runs.append(run)
    runs.sort(key=lambda r: (r.get("started") or "", r.get("run_id")))
    return runs


def find_run(root: str, prefix: str) -> dict:
    hits = [r for r in load_runs(root) if r["run_id"].startswith(prefix)]
    if not hits:
        raise KeyError(f"no run id starts with {prefix!r}")
    if len(hits) > 1:
        raise KeyError(f"{prefix!r} matches {len(hits)} runs; give more characters")
    return hits[0]


def merge_archives(dest: str, src: str) -> Dict[str, int]:
    """Copy runs (and their raw files) from src into dest. Returns counts."""
    added = skipped = 0
    for run in load_runs(src):
        target = os.path.join(runs_dir(dest), f"{run['run_id']}.json")
        if os.path.exists(target):
            skipped += 1
            continue
        save_run(dest, run)
        raw_src = os.path.join(src, "raw", run["run_id"])
        if os.path.isdir(raw_src):
            shutil.copytree(raw_src, os.path.join(dest, "raw", run["run_id"]), dirs_exist_ok=True)
        added += 1
    return {"added": added, "skipped": skipped}
