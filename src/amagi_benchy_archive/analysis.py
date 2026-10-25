"""Leaderboard, history and regressions over archived runs."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

# metric -> (label, higher is better)
METRICS: Dict[str, Tuple[str, bool]] = {
    "tg_tps": ("gen t/s", True),
    "pp_tps": ("prompt t/s", True),
    "peak_tps": ("peak t/s", True),
    "e2e_ttft_ms": ("TTFT ms", False),
    "est_ppt_ms": ("prompt ms", False),
    "tok_per_j": ("tok/J", True),
}
REGRESSION_METRICS = ("tg_tps", "pp_tps", "e2e_ttft_ms", "tok_per_j")


def system_key(run: dict) -> Tuple[str, str, str, str]:
    """Runs with the same key are the same setup measured at different times."""
    return (run.get("machine", ""), run.get("model", ""), run.get("engine", ""), run.get("quant", ""))


def system_label(run: dict) -> str:
    extra = ", ".join(x for x in (run.get("engine"), run.get("quant")) if x)
    return f"{run.get('model', '')} on {run.get('machine', '')}" + (f" ({extra})" if extra else "")


def metric_value(cell: dict, metric: str) -> Optional[float]:
    if metric == "tok_per_j":
        return cell.get("tok_per_j")
    m = cell.get("metrics", {}).get(metric)
    return m.get("mean") if m else None


def metric_std(cell: dict, metric: str) -> float:
    if metric == "tok_per_j":
        return 0.0
    m = cell.get("metrics", {}).get(metric)
    return float(m.get("std") or 0.0) if m else 0.0


def cell_watts(cell: dict) -> Optional[float]:
    p = cell.get("decode_power") or cell.get("power")
    return p.get("avg_w") if p else None


def matches(cell: dict, pp=None, tg=None, depth=None, concurrency=None, phase="std") -> bool:
    return ((pp is None or cell["pp"] == pp) and (tg is None or cell["tg"] == tg)
            and (depth is None or cell["depth"] == depth)
            and (concurrency is None or cell["concurrency"] == concurrency)
            and cell.get("phase", "std") == phase)


def common_shape(runs: Sequence[dict], depth=0, concurrency=1) -> Optional[Tuple[int, int]]:
    """The (pp, tg) present in the most runs at this depth/concurrency."""
    counts: Counter = Counter()
    for run in runs:
        shapes = {(c["pp"], c["tg"]) for c in run["cells"]
                  if matches(c, depth=depth, concurrency=concurrency) and metric_value(c, "tg_tps")}
        counts.update(shapes)
    if not counts:
        return None
    best = max(counts.values())
    return min(s for s, n in counts.items() if n == best)


@dataclass
class Row:
    run: dict
    cell: dict

    def value(self, metric: str) -> Optional[float]:
        return metric_value(self.cell, metric)


def leaderboard(runs: Sequence[dict], *, pp=None, tg=None, depth=0, concurrency=1,
                sort="tg_tps", pick="latest", machine=None, model=None) -> Tuple[List[Row], Optional[Tuple[int, int]]]:
    runs = [r for r in runs if (machine is None or r.get("machine") == machine)
            and (model is None or model.lower() in r.get("model", "").lower())]
    if pp is None or tg is None:
        shape = common_shape(runs, depth, concurrency)
        if shape is None:
            return [], None
        pp = shape[0] if pp is None else pp
        tg = shape[1] if tg is None else tg
    best: Dict[tuple, Row] = {}
    for run in runs:
        cell = next((c for c in run["cells"] if matches(c, pp, tg, depth, concurrency)
                     and metric_value(c, "tg_tps") is not None), None)
        if cell is None:
            continue
        key = system_key(run)
        row = Row(run, cell)
        cur = best.get(key)
        if cur is None:
            best[key] = row
        elif pick == "best":
            if _better(row.value(sort), cur.value(sort), METRICS[sort][1]):
                best[key] = row
        elif (run.get("started") or "") >= (cur.run.get("started") or ""):
            best[key] = row
    higher = METRICS[sort][1]
    rows = sorted(best.values(), key=lambda r: _sort_key(r.value(sort), higher))
    return rows, (pp, tg)


def _better(a, b, higher: bool) -> bool:
    if a is None:
        return False
    if b is None:
        return True
    return a > b if higher else a < b


def _sort_key(v, higher: bool):
    if v is None:
        return (1, 0.0)
    return (0, -v if higher else v)


def history(runs: Sequence[dict], *, pp, tg, depth=0, concurrency=1, metric="tg_tps") -> Dict[tuple, List[Tuple[str, float, str]]]:
    """system key -> [(started, value, run_id)] oldest first."""
    out: Dict[tuple, List[Tuple[str, float, str]]] = {}
    for run in sorted(runs, key=lambda r: r.get("started") or ""):
        for c in run["cells"]:
            if matches(c, pp, tg, depth, concurrency):
                v = metric_value(c, metric)
                if v is not None:
                    out.setdefault(system_key(run), []).append((run.get("started", ""), v, run["run_id"]))
    return out


@dataclass
class Change:
    system: str
    cell: str
    metric: str
    old: float
    new: float
    old_run: str
    new_run: str
    noise: float

    @property
    def pct(self) -> float:
        return (self.new - self.old) / self.old * 100.0 if self.old else math.inf

    @property
    def worse(self) -> bool:
        higher = METRICS[self.metric][1]
        return self.new < self.old if higher else self.new > self.old


def compare_runs(old: dict, new: dict, threshold_pct: float = 5.0,
                 metrics: Sequence[str] = REGRESSION_METRICS) -> List[Change]:
    """Changes between two runs of one setup that clear both the threshold and the noise.

    Noise is the two runs' standard deviations combined (sqrt(a^2 + b^2)); a
    move smaller than that is run-to-run jitter, whatever its percentage.
    """
    out = []
    old_cells = {c["label"]: c for c in old["cells"]}
    for c in new["cells"]:
        o = old_cells.get(c["label"])
        if o is None:
            continue
        for metric in metrics:
            a, b = metric_value(o, metric), metric_value(c, metric)
            if a is None or b is None or a == 0:
                continue
            noise = math.hypot(metric_std(o, metric), metric_std(c, metric))
            ch = Change(system_label(new), c["label"], metric, a, b, old["run_id"], new["run_id"], noise)
            if abs(ch.pct) >= threshold_pct and abs(b - a) > noise:
                out.append(ch)
    return out


def regressions(runs: Sequence[dict], threshold_pct: float = 5.0, machine=None, model=None,
                include_improvements=False) -> List[Change]:
    """Latest run of each setup against the run before it."""
    groups: Dict[tuple, List[dict]] = {}
    for run in runs:
        if machine and run.get("machine") != machine:
            continue
        if model and model.lower() not in run.get("model", "").lower():
            continue
        groups.setdefault(system_key(run), []).append(run)
    out: List[Change] = []
    for _, rs in sorted(groups.items()):
        rs = sorted(rs, key=lambda r: r.get("started") or "")
        if len(rs) < 2:
            continue
        for ch in compare_runs(rs[-2], rs[-1], threshold_pct):
            if ch.worse or include_improvements:
                out.append(ch)
    out.sort(key=lambda c: (not c.worse, -abs(c.pct)))
    return out
