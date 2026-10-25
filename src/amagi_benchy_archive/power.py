"""GPU power samples and the join against benchmark time windows.

Two CSV layouts are read:

* ours, written by ``amagi-benchy-archive power-log`` and ``record``:
  ``epoch_s,gpu,power_w,temp_c,mem_used_mib`` (unix seconds, so no time zone
  guessing), and
* plain ``nvidia-smi --query-gpu=timestamp,index,power.draw,... --format=csv``
  output. Its timestamps carry no zone, so they are read as local time unless
  ``utc_offset_hours`` says otherwise.
"""

from __future__ import annotations

import csv
import io
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

OUR_HEADER = ["epoch_s", "gpu", "power_w", "temp_c", "mem_used_mib"]


class PowerError(ValueError):
    pass


@dataclass(frozen=True)
class Sample:
    ts: float
    gpu: int
    watts: float
    temp_c: Optional[float] = None


def _num(text: str) -> Optional[float]:
    text = (text or "").strip()
    for unit in ("W", "MiB", "C", "%"):
        if text.endswith(unit):
            text = text[: -len(unit)].strip()
    if not text or text.startswith("[") or text.upper() in {"N/A", "NA"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _smi_time(text: str, tz: timezone) -> float:
    text = text.strip()
    for fmt in ("%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=tz).timestamp()
        except ValueError:
            continue
    raise PowerError(f"unrecognised nvidia-smi timestamp {text!r}")


def parse_power_csv(text: str, utc_offset_hours: Optional[float] = None) -> List[Sample]:
    rows = [r for r in csv.reader(io.StringIO(text)) if r and any(c.strip() for c in r)]
    if not rows:
        raise PowerError("power log is empty")
    header = [c.strip().lower() for c in rows[0]]
    if utc_offset_hours is None:
        tz = datetime.now().astimezone().tzinfo  # local
    else:
        tz = timezone(timedelta(hours=utc_offset_hours))
    samples: List[Sample] = []
    if header[: len(OUR_HEADER)] == OUR_HEADER:
        for r in rows[1:]:
            w = _num(r[2])
            if w is None:
                continue
            samples.append(Sample(float(r[0]), int(float(r[1])), w, _num(r[3]) if len(r) > 3 else None))
        return samples
    if any(h.startswith("timestamp") for h in header) and any(h.startswith("power.draw") for h in header):
        i_ts = next(i for i, h in enumerate(header) if h.startswith("timestamp"))
        i_w = next(i for i, h in enumerate(header) if h.startswith("power.draw"))
        i_gpu = next((i for i, h in enumerate(header) if h == "index"), None)
        i_t = next((i for i, h in enumerate(header) if h.startswith("temperature.gpu")), None)
        for r in rows[1:]:
            w = _num(r[i_w])
            if w is None:
                continue
            gpu = int(float(r[i_gpu])) if i_gpu is not None else 0
            temp = _num(r[i_t]) if i_t is not None else None
            samples.append(Sample(_smi_time(r[i_ts], tz), gpu, w, temp))  # type: ignore[arg-type]
        return samples
    raise PowerError(
        "unrecognised power log header; expected our epoch_s,gpu,power_w,... layout "
        "or nvidia-smi --format=csv with timestamp and power.draw columns"
    )


def load_power(path: str, utc_offset_hours: Optional[float] = None) -> List[Sample]:
    with open(path, encoding="utf-8") as fh:
        return parse_power_csv(fh.read(), utc_offset_hours)


def _by_gpu(samples: Iterable[Sample], gpus: Optional[Sequence[int]]) -> Dict[int, List[Sample]]:
    out: Dict[int, List[Sample]] = {}
    for s in samples:
        if gpus is not None and s.gpu not in gpus:
            continue
        out.setdefault(s.gpu, []).append(s)
    for v in out.values():
        v.sort(key=lambda s: s.ts)
    return out


def _in_windows(ts: float, windows: Sequence[Tuple[float, float]]) -> bool:
    return any(a <= ts <= b for a, b in windows)


def window_power(samples: Sequence[Sample], windows: Sequence[Tuple[float, float]],
                 gpus: Optional[Sequence[int]] = None, slack: float = 1.0) -> Optional[dict]:
    """Board power over the union of `windows`, summed across GPUs.

    Each GPU's samples inside the windows are averaged, then the per-GPU means
    are added. A window shorter than the sample interval may hold no sample; in
    that case the nearest sample within `slack` seconds of the window stands in.
    Returns None when no GPU has a usable sample.
    """
    windows = [(float(a), float(b)) for a, b in windows if a is not None and b is not None and b >= a]
    if not windows:
        return None
    groups = _by_gpu(samples, gpus)
    total_mean = 0.0
    total_peak = 0.0
    count = 0
    used = 0
    max_temp: Optional[float] = None
    for _, ss in sorted(groups.items()):
        inside = [s for s in ss if _in_windows(s.ts, windows)]
        if not inside:
            best = None
            for s in ss:
                gap = min(max(a - s.ts, 0.0, s.ts - b) for a, b in windows)
                if gap <= slack and (best is None or gap < best[0]):
                    best = (gap, s)
            if best is None:
                continue
            inside = [best[1]]
        used += 1
        total_mean += statistics.fmean(s.watts for s in inside)
        total_peak += max(s.watts for s in inside)
        count += len(inside)
        temps = [s.temp_c for s in inside if s.temp_c is not None]
        if temps:
            max_temp = max(temps) if max_temp is None else max(max_temp, max(temps))
    if not used:
        return None
    span = sum(b - a for a, b in windows)
    out = {
        "avg_w": round(total_mean, 2),
        "peak_w": round(total_peak, 2),
        "energy_j": round(total_mean * span, 2),
        "samples": count,
        "gpus": used,
    }
    if max_temp is not None:
        out["max_temp_c"] = max_temp
    return out


def baseline_power(samples: Sequence[Sample], before_ts: float,
                   gpus: Optional[Sequence[int]] = None, min_lead: float = 0.5) -> Optional[float]:
    """Mean summed board power from samples taken before `before_ts` (idle, model loaded)."""
    groups = _by_gpu(samples, gpus)
    total = 0.0
    used = 0
    for ss in groups.values():
        pre = [s.watts for s in ss if s.ts < before_ts - min_lead]
        if pre:
            total += statistics.fmean(pre)
            used += 1
    return round(total, 2) if used else None
