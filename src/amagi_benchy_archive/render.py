"""Text, markdown, CSV and JSON tables."""

from __future__ import annotations

import csv
import io
import json
from typing import List, Optional, Sequence

from .analysis import Change, Row, cell_watts
from .ingest import cell_label


def fmt(v: Optional[float], digits: int = 1) -> str:
    if v is None:
        return "-"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:,.{digits}f}"


def table(headers: Sequence[str], rows: Sequence[Sequence[str]], style: str = "text",
          right: Sequence[int] = ()) -> str:
    if style == "csv":
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow(headers)
        w.writerows(rows)
        return buf.getvalue()
    if style == "md":
        lines = ["| " + " | ".join(headers) + " |",
                 "|" + "|".join("--:" if i in right else "---" for i in range(len(headers))) + "|"]
        lines += ["| " + " | ".join(r) + " |" for r in rows]
        return "\n".join(lines) + "\n"
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h))
              for i, h in enumerate(headers)]

    def line(cells):
        return "  ".join(str(c).rjust(widths[i]) if i in right else str(c).ljust(widths[i])
                         for i, c in enumerate(cells)).rstrip()

    return "\n".join([line(headers), line(["-" * w for w in widths])] + [line(r) for r in rows]) + "\n"


LB_HEADERS = ["#", "model", "machine", "gpu", "gen t/s", "prompt t/s", "TTFT ms", "watts", "tok/J", "date", "run"]
LB_RIGHT = (0, 4, 5, 6, 7, 8)


def leaderboard_rows(rows: Sequence[Row]) -> List[List[str]]:
    out = []
    for i, r in enumerate(rows, 1):
        tg = r.cell["metrics"].get("tg_tps", {})
        gen = fmt(tg.get("mean"))
        if tg.get("n", 0) > 1 and tg.get("std"):
            gen += f" ±{fmt(tg['std'])}"
        out.append([
            str(i), r.run.get("model", ""), r.run.get("machine", ""), r.run.get("gpu", "") or "-",
            gen, fmt(r.value("pp_tps"), 0), fmt(r.value("e2e_ttft_ms"), 0),
            fmt(cell_watts(r.cell), 0), fmt(r.cell.get("tok_per_j"), 3),
            (r.run.get("started") or "")[:10], r.run["run_id"][:8],
        ])
    return out


def leaderboard_json(rows: Sequence[Row], shape, depth, concurrency) -> str:
    items = []
    for r in rows:
        items.append({
            "run_id": r.run["run_id"], "model": r.run.get("model"), "machine": r.run.get("machine"),
            "gpu": r.run.get("gpu"), "engine": r.run.get("engine"), "quant": r.run.get("quant"),
            "started": r.run.get("started"), "cell": r.cell["label"],
            "tg_tps": r.value("tg_tps"), "pp_tps": r.value("pp_tps"), "e2e_ttft_ms": r.value("e2e_ttft_ms"),
            "watts": cell_watts(r.cell), "tok_per_j": r.cell.get("tok_per_j"),
        })
    cell = cell_label((shape[0], shape[1], depth, concurrency, "std")) if shape else None
    return json.dumps({"cell": cell, "rows": items}, indent=2) + "\n"


CH_HEADERS = ["", "setup", "test", "metric", "before", "after", "change", "noise", "runs"]
CH_RIGHT = (4, 5, 6, 7)


def change_rows(changes: Sequence[Change]) -> List[List[str]]:
    from .analysis import METRICS
    out = []
    for c in changes:
        digits = 3 if c.metric == "tok_per_j" else 1
        out.append([
            "WORSE" if c.worse else "better", c.system, c.cell, METRICS[c.metric][0],
            fmt(c.old, digits), fmt(c.new, digits), f"{c.pct:+.1f}%", fmt(c.noise, digits),
            f"{c.old_run[:8]}->{c.new_run[:8]}",
        ])
    return out


RUN_HEADERS = ["run", "started", "machine", "gpu", "model", "cells", "power", "tag"]


def run_rows(runs: Sequence[dict]) -> List[List[str]]:
    out = []
    for r in runs:
        p = r.get("power")
        if p and p.get("joined"):
            power = f"yes ({p.get('samples')} samples)"
        elif p:
            power = "log, no timing"
        else:
            power = "-"
        out.append([r["run_id"][:8], r.get("started", ""), r.get("machine", ""), r.get("gpu") or "-",
                    r.get("model", ""), str(len(r.get("cells", []))), power, r.get("tag") or ""])
    return out


CELL_HEADERS = ["test", "gen t/s", "prompt t/s", "peak t/s", "TTFT ms", "watts", "peak W", "tok/J", "reqs", "errors"]
CELL_RIGHT = (1, 2, 3, 4, 5, 6, 7, 8, 9)


def cell_rows(run: dict) -> List[List[str]]:
    out = []
    for c in run["cells"]:
        m = c["metrics"]
        p = c.get("decode_power") or c.get("power") or {}
        out.append([
            c["label"], fmt(m.get("tg_tps", {}).get("mean")), fmt(m.get("pp_tps", {}).get("mean"), 0),
            fmt(m.get("peak_tps", {}).get("mean")), fmt(m.get("e2e_ttft_ms", {}).get("mean"), 0),
            fmt(p.get("avg_w"), 0), fmt(p.get("peak_w"), 0), fmt(c.get("tok_per_j"), 3),
            str(c.get("requests") or "-"), str(c.get("errors") or 0),
        ])
    return out
