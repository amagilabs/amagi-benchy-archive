"""Read llama-benchy output into archive cells.

Two inputs are understood:

* the JSON report (``llama-benchy --format json --save-result X.json``), whose
  numbers llama-benchy computed itself, and
* the ``--emit-progress`` JSONL stream (schema ``llama-benchy-progress.v1``),
  which carries a timestamp on every event. The timestamps are what make the
  power join possible.

When both are given, the report's metrics win and the progress stream adds the
per-cell time windows. With only a progress stream, the metrics are rebuilt
from the events using the same formulas llama-benchy uses.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

PROGRESS_SCHEMA = "llama-benchy-progress.v1"

# report key -> archive metric name
REPORT_METRICS = {
    "pp_throughput": "pp_tps",
    "tg_throughput": "tg_tps",
    "peak_throughput": "peak_tps",
    "peak_req_throughput": "peak_req_tps",
    "pp_req_throughput": "pp_req_tps",
    "tg_req_throughput": "tg_req_tps",
    "ttfr": "ttfr_ms",
    "est_ppt": "est_ppt_ms",
    "e2e_ttft": "e2e_ttft_ms",
}


class IngestError(ValueError):
    """Input file is not something we can read."""


CellKey = Tuple[int, int, int, int, str]  # pp, tg, depth, concurrency, phase


@dataclass
class Cell:
    pp: int
    tg: int
    depth: int
    concurrency: int
    phase: str = "std"  # "std", or "ctx" for the prefix-caching context load
    metrics: Dict[str, dict] = field(default_factory=dict)
    start_ts: Optional[float] = None
    end_ts: Optional[float] = None
    decode_windows: List[Tuple[float, float]] = field(default_factory=list)
    requests: int = 0
    errors: int = 0

    @property
    def key(self) -> CellKey:
        return (self.pp, self.tg, self.depth, self.concurrency, self.phase)

    @property
    def label(self) -> str:
        return cell_label(self.key)


def cell_label(key: CellKey) -> str:
    pp, tg, depth, conc, phase = key
    text = f"pp{pp}/tg{tg}"
    if depth:
        text += f" @ d{depth}"
    if conc != 1:
        text += f" c{conc}"
    if phase != "std":
        text += f" [{phase}]"
    return text


@dataclass
class Parsed:
    model: str = ""
    llama_benchy_version: str = ""
    latency_mode: str = ""
    latency_ms: Optional[float] = None
    started_ts: Optional[float] = None
    ended_ts: Optional[float] = None
    report_timestamp: str = ""
    status: str = ""
    cells: List[Cell] = field(default_factory=list)


def summarize(values: Iterable[float], scale: float = 1.0) -> Optional[dict]:
    vals = [float(v) * scale for v in values if v is not None]
    if not vals:
        return None
    return {
        "mean": statistics.fmean(vals),
        "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        "n": len(vals),
    }


# ---------------------------------------------------------------- report


def parse_report(data: dict) -> Parsed:
    if not isinstance(data, dict) or "benchmarks" not in data:
        raise IngestError("not a llama-benchy JSON report (no 'benchmarks' key)")
    out = Parsed(
        model=str(data.get("model") or ""),
        llama_benchy_version=str(data.get("version") or ""),
        latency_mode=str(data.get("latency_mode") or ""),
        latency_ms=data.get("latency_ms"),
        report_timestamp=str(data.get("timestamp") or ""),
    )
    for b in data["benchmarks"]:
        cell = Cell(
            pp=int(b.get("prompt_size", 0)),
            tg=int(b.get("response_size", 0)),
            depth=int(b.get("context_size", 0)),
            concurrency=int(b.get("concurrency", 1)),
            phase="ctx" if b.get("is_context_prefill_phase") else "std",
        )
        for src, dst in REPORT_METRICS.items():
            m = b.get(src)
            if isinstance(m, dict) and m.get("mean") is not None:
                vals = m.get("values") or []
                cell.metrics[dst] = {
                    "mean": float(m["mean"]),
                    "std": float(m.get("std") or 0.0),
                    "n": len(vals) if vals else 1,
                }
        out.cells.append(cell)
    return out


# ---------------------------------------------------------------- progress


@dataclass
class _Req:
    rid: int
    start: dict
    first_response_ts: Optional[float] = None
    ttfr_s: Optional[float] = None
    first_token_ts: Optional[float] = None
    ttft_s: Optional[float] = None
    chunks: List[Tuple[float, int]] = field(default_factory=list)
    estimated: bool = False
    end_ts: Optional[float] = None
    total_tokens: int = 0
    prompt_tokens: int = 0
    error: str = ""

    @property
    def start_ts(self) -> float:
        return float(self.start["ts"])


def read_progress_lines(lines: Iterable[str]) -> List[dict]:
    events = []
    for n, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError as exc:
            raise IngestError(f"line {n}: not JSON ({exc.msg})") from None
        if not isinstance(ev, dict):
            continue
        schema = ev.get("schema")
        if schema != PROGRESS_SCHEMA:
            raise IngestError(f"line {n}: unknown schema {schema!r}, expected {PROGRESS_SCHEMA}")
        events.append(ev)
    if not events:
        raise IngestError("progress stream is empty")
    return events


def _peak_window(times: List[float], window: float = 1.0) -> Optional[float]:
    """Most tokens in any `window`-second span, as tokens/sec. A burst shorter than the
    window is rated over its own length, as llama-benchy does."""
    if not times:
        return None
    times = sorted(times)
    span = times[-1] - times[0]
    if 0 < span < window:
        return len(times) / span
    best = 0
    lo = 0
    for hi, t in enumerate(times):
        while lo < hi and times[lo] <= t - window:
            lo += 1
        best = max(best, hi - lo + 1)
    return best / window


def _token_times(r: "_Req") -> List[float]:
    """One timestamp per generated token, rebuilt the way llama-benchy's client does it:
    a chunk with exact token ids spreads its tokens back to the previous token, and
    estimated counts are spread evenly between the first and last chunk."""
    if not r.chunks:
        return []
    if not r.estimated:
        out: List[float] = []
        for ts, n in r.chunks:
            if n <= 0:
                continue
            if n == 1:
                out.append(ts)
                continue
            last = out[-1] if out else (r.first_token_ts if r.first_token_ts is not None else r.start_ts)
            out.extend(last + (ts - last) * (i + 1) / n for i in range(n))
        return out
    stamps = [ts for ts, _ in r.chunks]
    total = r.total_tokens
    if total <= 0:
        return stamps
    if total == 1 or len(stamps) == 1 or stamps[-1] <= stamps[0]:
        return [stamps[0]] * total
    step = (stamps[-1] - stamps[0]) / (total - 1)
    return [stamps[0] + step * i for i in range(total)]


def _decode_tokens(times: List[float]) -> Tuple[int, float]:
    """Tokens after the first token's timestamp, and the time they took."""
    if len(times) < 2:
        return 0, 0.0
    first = times[0]
    return sum(1 for t in times if t > first), times[-1] - first


def parse_progress(events: List[dict]) -> Parsed:
    out = Parsed()
    latency_s = 0.0
    reqs: Dict[int, _Req] = {}
    order: List[int] = []
    for ev in events:
        t = ev.get("type")
        ts = ev.get("ts")
        if t == "header":
            out.llama_benchy_version = str(ev.get("llama_benchy_version") or "")
            out.started_ts = ts
        elif t == "latency_measured":
            latency_s = float(ev.get("latency_s") or 0.0)
            out.latency_mode = str(ev.get("mode") or "")
            out.latency_ms = latency_s * 1000.0
        elif t == "request_start":
            rid = int(ev["request_id"])
            reqs[rid] = _Req(rid=rid, start=ev)
            order.append(rid)
            if not out.model:
                out.model = str(ev.get("model") or "")
        elif t == "request_first_response":
            r = reqs.get(int(ev.get("request_id", -1)))
            if r:
                r.first_response_ts = ts
                r.ttfr_s = ev.get("ttfr_s")
        elif t == "request_first_token":
            r = reqs.get(int(ev.get("request_id", -1)))
            if r:
                r.first_token_ts = ts
                r.ttft_s = ev.get("ttft_s")
        elif t == "tokens":
            r = reqs.get(int(ev.get("request_id", -1)))
            if r and ts is not None:
                r.chunks.append((float(ts), int(ev.get("count") or 0)))
                if ev.get("estimated"):
                    r.estimated = True
        elif t == "request_end":
            r = reqs.get(int(ev.get("request_id", -1)))
            if r:
                r.end_ts = ts
                r.total_tokens = int(ev.get("total_tokens") or 0)
                r.prompt_tokens = int(ev.get("prompt_tokens") or 0)
                r.error = str(ev.get("error") or "")
        elif t == "bench_complete":
            out.status = str(ev.get("status") or "")
            out.ended_ts = ts
    if out.started_ts is None and events:
        out.started_ts = events[0].get("ts")
    if out.ended_ts is None and events:
        out.ended_ts = events[-1].get("ts")
    if not reqs:
        return out

    # group: shape -> run_index -> [requests in start order]
    shapes: Dict[Tuple[int, int, int, int], Dict[int, List[_Req]]] = {}
    for rid in order:
        r = reqs[rid]
        s = r.start
        shape = (int(s.get("prompt_size", 0)), int(s.get("response_size", 0)),
                 int(s.get("context_size", 0)), int(s.get("concurrency", 1)))
        shapes.setdefault(shape, {}).setdefault(int(s.get("run_index", 0)), []).append(r)

    for (pp, tg, depth, conc), runs in shapes.items():
        # With --enable-prefix-caching every run sends the context load first,
        # then the real prompt, so a run holds 2 x concurrency requests.
        split = depth > 0 and all(len(v) == 2 * conc for v in runs.values())
        phases = {"std": {}}
        if split:
            phases = {"ctx": {}, "std": {}}
            for idx, rs in runs.items():
                phases["ctx"][idx] = rs[:conc]
                phases["std"][idx] = rs[conc:]
        else:
            phases["std"] = runs
        for phase, prun in phases.items():
            if phase == "ctx":
                expected = depth
            elif split:
                expected = pp
            else:
                expected = pp + depth
            out.cells.append(_cell_from_runs(pp, tg, depth, conc, phase, prun, expected, latency_s))
    return out


def _cell_from_runs(pp, tg, depth, conc, phase, runs: Dict[int, List[_Req]],
                    expected_pp: int, latency_s: float) -> Cell:
    cell = Cell(pp=pp, tg=tg, depth=depth, concurrency=conc, phase=phase)
    ttfr, ppt, ttft = [], [], []
    pp_req, tg_req = [], []
    pp_batch, tg_batch, peaks, peak_req = [], [], [], []
    starts, ends = [], []
    for _, rs in sorted(runs.items()):
        cell.requests += len(rs)
        ok = [r for r in rs if not r.error]
        cell.errors += len(rs) - len(ok)
        for r in rs:
            starts.append(r.start_ts)
            if r.end_ts is not None:
                ends.append(float(r.end_ts))
        if not ok:
            continue
        batch_prompt = 0
        batch_times: List[float] = []
        first_tokens, last_tokens = [], []
        batch_decode = 0
        for r in ok:
            prompt = expected_pp
            if r.prompt_tokens > 0 and abs(r.prompt_tokens - expected_pp) < expected_pp * 0.2:
                prompt = r.prompt_tokens
            batch_prompt += prompt
            if r.ttfr_s is not None:
                ttfr.append(r.ttfr_s)
            if r.ttft_s is not None:
                ttft.append(r.ttft_s)
                est = max(0.0, (r.ttfr_s if r.ttfr_s is not None else r.ttft_s) - latency_s)
                ppt.append(est)
                if est > 0:
                    pp_req.append(prompt / est)
            times = _token_times(r)
            n_after, dt = _decode_tokens(times)
            batch_decode += n_after
            if n_after > 0 and dt > 0:
                tg_req.append(n_after / dt)
            first = r.first_token_ts if r.first_token_ts is not None else (times[0] if times else None)
            if first is not None:
                first_tokens.append(float(first))
            if times:
                last_tokens.append(times[-1])
                cell.decode_windows.append((times[0], float(r.end_ts or times[-1])))
            elif r.end_ts is not None:
                last_tokens.append(float(r.end_ts))
            batch_times.extend(times)
            req_peak = _peak_window(times)
            if req_peak:
                peak_req.append(req_peak)
        if conc > 1 and first_tokens:
            t0 = min(r.start_ts for r in ok)
            pp_dur = max(first_tokens) - t0
            if pp_dur > 0:
                pp_batch.append(batch_prompt / pp_dur)
            tg_dur = max(last_tokens) - min(first_tokens) if last_tokens else 0
            if tg_dur > 0 and batch_decode > 0:
                tg_batch.append(batch_decode / tg_dur)
        peak = _peak_window(batch_times)
        if peak:
            peaks.append(peak)
    cell.start_ts = min(starts) if starts else None
    cell.end_ts = max(ends) if ends else None
    m = cell.metrics
    m["pp_req_tps"] = summarize(pp_req)
    m["tg_req_tps"] = summarize(tg_req)
    m["pp_tps"] = summarize(pp_batch) if conc > 1 else m["pp_req_tps"]
    m["tg_tps"] = summarize(tg_batch) if conc > 1 else m["tg_req_tps"]
    m["peak_tps"] = summarize(peaks)
    m["peak_req_tps"] = summarize(peak_req)
    m["ttfr_ms"] = summarize(ttfr, 1000.0)
    m["est_ppt_ms"] = summarize(ppt, 1000.0)
    m["e2e_ttft_ms"] = summarize(ttft, 1000.0)
    cell.metrics = {k: v for k, v in m.items() if v is not None}
    return cell


# ---------------------------------------------------------------- merge


def merge(report: Optional[Parsed], progress: Optional[Parsed]) -> Parsed:
    """Report metrics, progress timing. Either side may be missing."""
    if report is None and progress is None:
        raise IngestError("need a report, a progress stream, or both")
    if report is None:
        return progress  # type: ignore[return-value]
    if progress is None:
        return report
    by_key = {c.key: c for c in progress.cells}
    for c in report.cells:
        p = by_key.get(c.key)
        if p is None:
            continue
        c.start_ts, c.end_ts = p.start_ts, p.end_ts
        c.decode_windows = p.decode_windows
        c.requests, c.errors = p.requests, p.errors
    report.started_ts = progress.started_ts
    report.ended_ts = progress.ended_ts
    report.status = progress.status
    if not report.model:
        report.model = progress.model
    return report


def load_report(path: str) -> Parsed:
    with open(path, encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise IngestError(f"{path}: not JSON ({exc.msg})") from None
    return parse_report(data)


def load_progress(path: str) -> Parsed:
    with open(path, encoding="utf-8") as fh:
        return parse_progress(read_progress_lines(fh))


def sniff(path: str) -> str:
    """'report' or 'progress', from the first non-blank character pattern."""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{") and PROGRESS_SCHEMA in line:
                return "progress"
            return "report"
    raise IngestError(f"{path}: empty file")
