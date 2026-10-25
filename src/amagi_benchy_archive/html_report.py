"""One self-contained HTML page: leaderboard, efficiency, regressions, history, runs.

No JavaScript and no external assets, so the file can be published as-is
(GitHub Pages, an S3 bucket, an email attachment). Hover text uses SVG
<title>, and every chart has a table next to it carrying the same numbers.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from html import escape
from typing import List, Optional, Sequence

from . import __version__
from .analysis import Change, Row, cell_watts, history, leaderboard, regressions, system_key, system_label
from .ingest import cell_label
from .render import (CELL_HEADERS, CH_HEADERS, LB_HEADERS, cell_rows, change_rows, fmt, leaderboard_rows)

CSS = """
.viz-root{color-scheme:light;--surface-1:#fcfcfb;--surface-2:#f3f2ef;--grid:#e4e3df;
--text-primary:#0b0b0b;--text-secondary:#52514e;--text-muted:#7a7975;--series-1:#2a78d6;
--bad:#c62828;--good:#1b7f3b}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;
--surface-1:#1a1a19;--surface-2:#242422;--grid:#383835;--text-primary:#ffffff;--text-secondary:#c3c2b7;
--text-muted:#9a998f;--series-1:#3987e5;--bad:#ff6b6b;--good:#5cd68a}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--surface-1:#1a1a19;--surface-2:#242422;--grid:#383835;
--text-primary:#ffffff;--text-secondary:#c3c2b7;--text-muted:#9a998f;--series-1:#3987e5;--bad:#ff6b6b;--good:#5cd68a}
*{box-sizing:border-box}
body{margin:0}
.viz-root{background:var(--surface-1);color:var(--text-primary);min-height:100vh;
font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;padding:32px 20px 48px}
main{max-width:980px;margin:0 auto}
h1{font-size:26px;margin:0 0 4px;letter-spacing:-.01em}
h2{font-size:18px;margin:40px 0 4px}
p.sub,p.note{color:var(--text-secondary);margin:0 0 12px}
p.note{font-size:13px}
table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0 0}
th,td{padding:6px 8px;border-bottom:1px solid var(--grid);text-align:left;white-space:nowrap}
th{color:var(--text-secondary);font-weight:600}
td.r,th.r{text-align:right;font-variant-numeric:tabular-nums}
.scroll{overflow-x:auto}
svg{display:block;max-width:100%;height:auto}
svg text{fill:var(--text-primary);font:12px system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
svg text.muted{fill:var(--text-secondary)}
svg .bar{fill:var(--series-1)}
svg .grid{stroke:var(--grid);stroke-width:1}
svg .line{stroke:var(--series-1);stroke-width:2;fill:none;stroke-linejoin:round;stroke-linecap:round}
svg .dot{fill:var(--series-1);stroke:var(--surface-1);stroke-width:2}
.tag{display:inline-block;font-size:12px;font-weight:600;padding:1px 6px;border-radius:4px;border:1px solid currentColor}
.worse{color:var(--bad)} .better{color:var(--good)}
.multiples{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px;margin-top:8px}
.card{background:var(--surface-2);border-radius:8px;padding:12px}
.card h3{font-size:13px;margin:0 0 6px;font-weight:600}
details{background:var(--surface-2);border-radius:8px;padding:8px 12px;margin:8px 0}
summary{cursor:pointer;font-size:14px}
footer{margin-top:48px;color:var(--text-muted);font-size:12px}
footer a{color:var(--text-secondary)}
"""


def _nice_axis(vmax: float, target: int = 5) -> tuple:
    """Round axis top, step and label decimals, about `target` intervals: 0-50 by 10."""
    if vmax <= 0:
        return 1.0, 0.25, 2
    mag = 10 ** math.floor(math.log10(vmax / target))
    best = None
    for scale in (mag / 10, mag, mag * 10):
        for m in (1, 2, 2.5, 5):
            step = m * scale
            top = math.ceil(vmax / step - 1e-9) * step
            score = (abs(top / step - target), top)
            if best is None or score < best[0]:
                best = (score, top, step, m)
    _, top, step, _ = best
    decimals = max(0, -math.floor(math.log10(step) + 1e-9))
    if abs(step * 10 ** decimals - round(step * 10 ** decimals)) > 1e-6:
        decimals += 1
    return top, step, decimals


def _bar_chart(items: Sequence[tuple], unit: str, digits: int = 1) -> str:
    """Horizontal bars. items: (label, value, tooltip)."""
    items = [i for i in items if i[1] is not None]
    if not items:
        return ""
    label_w, chart_w, bar_h, gap = 300, 560, 20, 12
    vmax, step, decimals = _nice_axis(max(v for _, v, _ in items))
    width = label_w + chart_w + 90
    height = len(items) * (bar_h + gap) + 24
    out = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(unit)} by setup">']
    for t in range(int(round(vmax / step)) + 1):
        x = label_w + chart_w * t * step / vmax
        out.append(f'<line class="grid" x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="{height - 20}"/>')
        out.append(f'<text class="muted" x="{x:.1f}" y="{height - 4}" text-anchor="middle">'
                   f'{fmt(t * step, decimals)}</text>')
    for i, (label, v, tip) in enumerate(items):
        y = i * (bar_h + gap) + 4
        w = max(chart_w * v / vmax, 2.0)
        r = min(4.0, w / 2)
        x0 = label_w
        # square at the baseline, 4px round at the data end
        path = (f"M{x0},{y} H{x0 + w - r:.1f} Q{x0 + w:.1f},{y} {x0 + w:.1f},{y + r} "
                f"V{y + bar_h - r} Q{x0 + w:.1f},{y + bar_h} {x0 + w - r:.1f},{y + bar_h} H{x0} Z")
        short = label if len(label) <= 44 else label[:43] + "…"
        out.append(f'<g><title>{escape(tip)}</title>'
                   f'<rect x="0" y="{y - 4}" width="{width}" height="{bar_h + gap}" fill="transparent"/>'
                   f'<text x="{label_w - 10}" y="{y + 14}" text-anchor="end">{escape(short)}</text>'
                   f'<path class="bar" d="{path}"/>'
                   f'<text x="{x0 + w + 6:.1f}" y="{y + 14}">{fmt(v, digits)}</text></g>')
    out.append("</svg>")
    return "".join(out)


def _sparkline(points: Sequence[tuple], unit: str) -> str:
    """points: (started, value, run_id). Line with an end dot and end label."""
    w, h, pad_l, pad_r, pad_y = 280, 100, 6, 56, 20
    vals = [p[1] for p in points]
    lo, hi = min(vals), max(vals)
    if hi == lo:
        lo, hi = lo * 0.95, hi * 1.05 or 1.0
    n = len(points)

    def xy(i, v):
        x = pad_l + (w - pad_l - pad_r) * (i / (n - 1) if n > 1 else 0.5)
        y = pad_y + (h - 2 * pad_y) * (1 - (v - lo) / (hi - lo))
        return x, y

    coords = [xy(i, v) for i, v in enumerate(vals)]
    d = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(coords))
    # fixed pixel size: stretched to the card width the labels would scale up with it
    out = [f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-label="{escape(unit)} over {n} runs">',
           f'<line class="grid" x1="{pad_l}" y1="{h - pad_y}" x2="{w - pad_r}" y2="{h - pad_y}"/>',
           f'<path class="line" d="{d}"/>']
    for (x, y), (started, v, rid) in zip(coords, points):
        out.append(f'<g><title>{escape(started[:16])} · {fmt(v)} {escape(unit)} · run {rid[:8]}</title>'
                   f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="transparent"/>'
                   f'<circle class="dot" cx="{x:.1f}" cy="{y:.1f}" r="4"/></g>')
    if n > 1:  # first value too, so the line's drop or rise reads in numbers, not just slope
        x0, y0 = coords[0]
        above = coords[1][1] >= y0  # put it on the side the line leaves from
        out.append(f'<text class="muted" x="{x0:.1f}" y="{y0 - 8 if above else y0 + 18:.1f}">{fmt(vals[0])}</text>')
    x, y = coords[-1]
    out.append(f'<text x="{x + 8:.1f}" y="{y + 4:.1f}">{fmt(vals[-1])}</text>')
    out.append(f'<text class="muted" x="{pad_l}" y="{h - 1}">{escape(points[0][0][:10])}</text>')
    out.append("</svg>")
    return "".join(out)


def _table(headers, rows, right=()) -> str:
    th = "".join(f'<th class="r">{escape(h)}</th>' if i in right else f"<th>{escape(h)}</th>"
                 for i, h in enumerate(headers))
    body = []
    for r in rows:
        tds = "".join(f'<td class="r">{escape(c)}</td>' if i in right else f"<td>{escape(c)}</td>"
                      for i, c in enumerate(r))
        body.append(f"<tr>{tds}</tr>")
    return f'<div class="scroll"><table><thead><tr>{th}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _tip(r: Row) -> str:
    parts = [system_label(r.run)]
    if r.run.get("gpu"):
        parts.append(r.run["gpu"])
    parts.append(f"gen {fmt(r.value('tg_tps'))} t/s")
    parts.append(f"prompt {fmt(r.value('pp_tps'), 0)} t/s")
    if r.value("e2e_ttft_ms") is not None:
        parts.append(f"TTFT {fmt(r.value('e2e_ttft_ms'), 0)} ms")
    if cell_watts(r.cell) is not None:
        parts.append(f"{fmt(cell_watts(r.cell), 0)} W")
    if r.cell.get("tok_per_j") is not None:
        parts.append(f"{fmt(r.cell['tok_per_j'], 3)} tok/J")
    parts.append(f"run {r.run['run_id'][:8]} · {(r.run.get('started') or '')[:10]}")
    return " · ".join(parts)


def render(runs: Sequence[dict], *, title: str = "Local LLM leaderboard", pp=None, tg=None,
           depth: int = 0, concurrency: int = 1, threshold: float = 5.0,
           generated: Optional[str] = None) -> str:
    rows, shape = leaderboard(runs, pp=pp, tg=tg, depth=depth, concurrency=concurrency)
    machines = sorted({r.get("machine", "") for r in runs})
    generated = generated or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    h: List[str] = [
        "<!doctype html>", '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        f"<title>{escape(title)}</title>",
        f'<meta name="generator" content="amagi-benchy-archive {__version__}">',
        f"<style>{CSS}</style></head><body><div class=\"viz-root\"><main>",
        f"<h1>{escape(title)}</h1>",
    ]
    if shape:
        test = cell_label((shape[0], shape[1], depth, concurrency, "std"))
        conc_text = "one request at a time" if concurrency == 1 else f"{concurrency} requests at once"
        h.append(f'<p class="sub">Test {escape(test)}: a {shape[0]}-token prompt, {shape[1]} generated tokens, '
                 f"{'no prior context' if depth == 0 else f'{depth} tokens of prior context'}, {conc_text}. "
                 f"{len(runs)} run{'s' if len(runs) != 1 else ''} from {len(machines)} "
                 f"machine{'s' if len(machines) != 1 else ''}, measured with llama-benchy.</p>")
    else:
        h.append('<p class="sub">No archived run has a usable generation-speed result yet.</p>')

    if rows:
        h.append("<h2>Generation speed</h2>")
        h.append('<p class="note">Tokens per second while the answer streams, latest run of each setup. '
                 "Hover a bar for the full row.</p>")
        h.append(_bar_chart([(f"{r.run.get('model', '')} · {r.run.get('machine', '')}", r.value("tg_tps"), _tip(r))
                             for r in rows], "gen t/s"))
        powered = [r for r in rows if r.cell.get("tok_per_j") is not None]
        if powered:
            powered.sort(key=lambda r: -r.cell["tok_per_j"])
            h.append("<h2>Tokens per joule</h2>")
            h.append('<p class="note">Generation speed divided by average GPU board power while tokens were '
                     "streaming, from nvidia-smi samples joined on the benchmark's own timestamps. "
                     "Board power only: CPU, RAM and PSU losses are not in it.</p>")
            h.append(_bar_chart([(f"{r.run.get('model', '')} · {r.run.get('machine', '')}", r.cell["tok_per_j"],
                                  _tip(r)) for r in powered], "tok/J", digits=3))
        h.append("<h2>Table</h2>")
        h.append(_table(LB_HEADERS, leaderboard_rows(rows), right=(0, 4, 5, 6, 7, 8)))

    changes: List[Change] = regressions(runs, threshold, include_improvements=True)
    h.append("<h2>Regressions</h2>")
    h.append(f'<p class="note">The latest run of each setup against the run before it, every test it shares. '
             f"A change is listed when it is at least {threshold:g}% and bigger than the two runs' "
             "combined standard deviation.</p>")
    if changes:
        crow = change_rows(changes)
        th = "".join(f'<th class="r">{escape(x)}</th>' if i in (4, 5, 6, 7) else f"<th>{escape(x)}</th>"
                     for i, x in enumerate(CH_HEADERS))
        body = []
        for c, r in zip(changes, crow):
            cls = "worse" if c.worse else "better"
            cells = [f'<td><span class="tag {cls}">{escape(r[0])}</span></td>']
            cells += [f'<td class="r">{escape(x)}</td>' if i in (4, 5, 6, 7) else f"<td>{escape(x)}</td>"
                      for i, x in enumerate(r) if i > 0]
            body.append("<tr>" + "".join(cells) + "</tr>")
        h.append(f'<div class="scroll"><table><thead><tr>{th}</tr></thead><tbody>{"".join(body)}</tbody></table></div>')
    else:
        repeat = sum(1 for k, v in _groups(runs).items() if len(v) > 1)
        if repeat:
            h.append(f'<p>Nothing moved past the threshold across {repeat} setup{"s" if repeat != 1 else ""} '
                     "with repeat runs.</p>")
        else:
            h.append("<p>No setup has two runs yet, so there is nothing to compare.</p>")

    if shape:
        hist = history(runs, pp=shape[0], tg=shape[1], depth=depth, concurrency=concurrency)
        multi = {k: v for k, v in hist.items() if len(v) > 1}
        if multi:
            h.append("<h2>History</h2>")
            h.append(f'<p class="note">Generation speed on {escape(cell_label((shape[0], shape[1], depth, concurrency, "std")))} '
                     "for every setup with more than one run, oldest to newest.</p>")
            h.append('<div class="multiples">')
            labels = {system_key(r): system_label(r) for r in runs}
            for key, pts in sorted(multi.items()):
                h.append(f'<div class="card"><h3>{escape(labels[key])}</h3>{_sparkline(pts, "gen t/s")}</div>')
            h.append("</div>")

    h.append("<h2>All runs</h2>")
    for run in sorted(runs, key=lambda r: r.get("started") or "", reverse=True):
        p = run.get("power") or {}
        bits = [run.get("started", "")[:16].replace("T", " "), system_label(run)]
        if run.get("gpu"):
            bits.append(run["gpu"])
        if p.get("idle_w") is not None:
            bits.append(f"idle {fmt(p['idle_w'], 0)} W")
        if run.get("tag"):
            bits.append(run["tag"])
        h.append(f"<details><summary>{escape(' · '.join(bits))} <code>{run['run_id'][:8]}</code></summary>")
        if run.get("note"):
            h.append(f'<p class="note">{escape(run["note"])}</p>')
        h.append(_table(CELL_HEADERS, cell_rows(run), right=tuple(range(1, 10))))
        h.append("</details>")

    h.append(f'<footer><p>Generated {escape(generated)} by '
             f'<a href="https://github.com/amagilabs/amagi-benchy-archive">amagi-benchy-archive</a> {__version__}. '
             'Built on <a href="https://github.com/eugr/llama-benchy">llama-benchy</a> by eugr (MIT) and '
             'llama-benchy-viz by Alex Ziskind (MIT) '
             '(<a href="https://github.com/alexziskind1/llama-benchy-viz-tui">TUI</a>, '
             '<a href="https://github.com/alexziskind1/llama-benchy-viz-web">web</a>).</p></footer>')
    h.append("</main></div></body></html>")
    return "\n".join(h) + "\n"


def _groups(runs):
    out = {}
    for r in runs:
        out.setdefault(system_key(r), []).append(r)
    return out
