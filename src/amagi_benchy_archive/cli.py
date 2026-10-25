"""amagi-benchy-archive command line."""

from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import sys
import tempfile
from typing import List, Optional

from . import __version__
from .analysis import METRICS, leaderboard, regressions
from .archive import archive_dir, build_run, find_run, load_runs, merge_archives, save_run
from .html_report import render as render_html
from .ingest import IngestError, cell_label, load_progress, load_report, merge, sniff
from .power import PowerError, load_power
from .record import PowerSampler, check_benchy_args, gpu_names, run_benchy, smi_available
from .render import (CELL_HEADERS, CELL_RIGHT, CH_HEADERS, CH_RIGHT, LB_HEADERS, LB_RIGHT, RUN_HEADERS,
                     cell_rows, change_rows, leaderboard_json, leaderboard_rows, run_rows, table)

PROG = "amagi-benchy-archive"


def _err(msg: str) -> int:
    print(f"{PROG}: {msg}", file=sys.stderr)
    return 2


def _gpus(text: Optional[str]):
    if not text:
        return None
    return [int(x) for x in text.split(",") if x.strip()]


def _meta_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--machine", help="name for this box (default: hostname)")
    p.add_argument("--gpu", default=None, help="GPU label (default: read from nvidia-smi when available)")
    p.add_argument("--engine", default="", help="e.g. 'llama.cpp b6000', 'vLLM 0.11'")
    p.add_argument("--quant", default="", help="e.g. Q4_K_M, AWQ")
    p.add_argument("--model", default="", help="override the model name llama-benchy recorded")
    p.add_argument("--tag", default="", help="short label, e.g. 'after driver update'")
    p.add_argument("--note", default="", help="free text stored with the run")
    p.add_argument("--gpus", default=None, help="GPU indexes to count for power, e.g. 0 or 0,1 (default: all)")


def _archive_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("-a", "--archive", default=None,
                   help="archive folder (default: $AMAGI_BENCHY_ARCHIVE or ./benchy-archive)")


def _filter_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--pp", type=int, default=None, help="prompt tokens of the test to rank on")
    p.add_argument("--tg", type=int, default=None, help="generated tokens of the test to rank on")
    p.add_argument("--depth", type=int, default=0, help="context depth (default 0)")
    p.add_argument("-c", "--concurrency", type=int, default=1, help="concurrency level (default 1)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=PROG,
        description="Keep every llama-benchy run from every machine in one archive, with GPU watts "
                    "joined in. Rank setups, catch regressions, publish one HTML page.")
    ap.add_argument("--version", action="version", version=f"{PROG} {__version__}")
    sub = ap.add_subparsers(dest="cmd", metavar="command")

    p = sub.add_parser("record", help="run llama-benchy with a power sampler, then archive the run",
                       description="Everything after -- is the llama-benchy command. record adds "
                                   "--emit-progress, --format json and --save-result itself.")
    _archive_arg(p)
    _meta_args(p)
    p.add_argument("--no-power", action="store_true", help="skip nvidia-smi sampling")
    p.add_argument("--interval-ms", type=int, default=250, help="power sample interval (default 250)")
    p.add_argument("--idle-seconds", type=float, default=3.0,
                   help="seconds of power sampled before the benchmark starts, stored as idle watts (default 3)")
    p.add_argument("--smi-cmd", default=None,
                   help="how to call nvidia-smi, e.g. 'ssh gpubox nvidia-smi' when the GPU is remote")
    p.add_argument("benchy", nargs=argparse.REMAINDER, help="-- llama-benchy ARGS...")

    p = sub.add_parser("ingest", help="archive llama-benchy output files you already have")
    _archive_arg(p)
    _meta_args(p)
    p.add_argument("files", nargs="*", help="report .json and/or --emit-progress .jsonl (type is detected)")
    p.add_argument("--report", default=None, help="llama-benchy JSON report")
    p.add_argument("--progress", default=None, help="llama-benchy --emit-progress JSONL")
    p.add_argument("--power", default=None, help="power CSV (from power-log, or nvidia-smi --format=csv)")
    p.add_argument("--power-utc-offset", type=float, default=None,
                   help="hours from UTC of nvidia-smi timestamps in --power (default: this machine's zone)")
    p.add_argument("--no-raw", action="store_true", help="do not copy the input files into the archive")

    p = sub.add_parser("power-log", help="write nvidia-smi power samples to CSV until Ctrl-C")
    p.add_argument("-o", "--output", default="power.csv", help="CSV path, '-' for stdout (default power.csv)")
    p.add_argument("--interval-ms", type=int, default=250)
    p.add_argument("--gpus", default=None, help="GPU indexes, e.g. 0,1")
    p.add_argument("--smi-cmd", default=None, help="how to call nvidia-smi (default: nvidia-smi)")
    p.add_argument("--seconds", type=float, default=None, help="stop after this long")

    p = sub.add_parser("runs", help="list archived runs")
    _archive_arg(p)
    p.add_argument("--format", choices=["text", "md", "csv", "json"], default="text")

    p = sub.add_parser("show", help="every test in one run")
    _archive_arg(p)
    p.add_argument("run_id", help="run id or a unique prefix")
    p.add_argument("--format", choices=["text", "md", "csv", "json"], default="text")

    p = sub.add_parser("leaderboard", help="rank setups on one test")
    _archive_arg(p)
    _filter_args(p)
    p.add_argument("--sort", choices=sorted(METRICS), default="tg_tps")
    p.add_argument("--pick", choices=["latest", "best"], default="latest",
                   help="which run of a setup to show (default latest)")
    p.add_argument("--machine", default=None)
    p.add_argument("--model", default=None, help="substring match")
    p.add_argument("--format", choices=["text", "md", "csv", "json"], default="text")

    p = sub.add_parser("regressions", help="latest run of each setup against the one before")
    _archive_arg(p)
    p.add_argument("--threshold", type=float, default=5.0, help="minimum change in percent (default 5)")
    p.add_argument("--machine", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--all", action="store_true", help="also list improvements")
    p.add_argument("--fail", action="store_true", help="exit 1 when a regression is found (for CI)")
    p.add_argument("--format", choices=["text", "md", "csv", "json"], default="text")

    p = sub.add_parser("html", help="write the leaderboard page")
    _archive_arg(p)
    _filter_args(p)
    p.add_argument("-o", "--output", default="index.html")
    p.add_argument("--title", default="Local LLM leaderboard")
    p.add_argument("--threshold", type=float, default=5.0)

    p = sub.add_parser("merge", help="copy runs from another archive folder into this one")
    _archive_arg(p)
    p.add_argument("source", help="archive folder to copy from (e.g. one synced from another machine)")
    return ap


# ------------------------------------------------------------------ commands


def _finish_ingest(args, parsed, sources, power_path, raw) -> int:
    samples = None
    if power_path:
        try:
            samples = load_power(power_path, getattr(args, "power_utc_offset", None))
        except (OSError, PowerError) as exc:
            return _err(f"power log: {exc}")
        if not samples:
            print(f"{PROG}: warning: power log has no usable samples", file=sys.stderr)
            samples = None
    if samples and not any(c.start_ts for c in parsed.cells):
        print(f"{PROG}: warning: no progress stream, so watts cannot be matched to tests "
              "(pass the --emit-progress JSONL too)", file=sys.stderr)
    gpu = args.gpu if args.gpu is not None else ""
    run = build_run(parsed, machine=args.machine or platform.node() or "unknown", gpu=gpu,
                    engine=args.engine, quant=args.quant, tag=args.tag, note=args.note, model=args.model,
                    samples=samples, gpus=_gpus(args.gpus), sources=sources,
                    power_source=os.path.basename(power_path) if power_path else "")
    if not run["cells"]:
        return _err("no tests found in the input")
    root = archive_dir(args.archive)
    path = save_run(root, run, raw)
    joined = sum(1 for c in run["cells"] if c.get("tok_per_j") is not None)
    msg = f"archived run {run['run_id']} ({run['model']} on {run['machine']}, {len(run['cells'])} tests"
    if samples:
        msg += f", watts on {joined}"
    print(msg + f") -> {path}")
    return 0


def cmd_ingest(args) -> int:
    report_path, progress_path = args.report, args.progress
    try:
        for f in args.files:
            kind = sniff(f)
            if kind == "progress":
                progress_path = f
            else:
                report_path = f
        if not report_path and not progress_path:
            return _err("give a llama-benchy report (.json), a progress stream (.jsonl), or both")
        report = load_report(report_path) if report_path else None
        progress = load_progress(progress_path) if progress_path else None
        parsed = merge(report, progress)
    except (OSError, IngestError) as exc:
        return _err(str(exc))
    sources = {k: os.path.basename(v) for k, v in
               (("report", report_path), ("progress", progress_path), ("power", args.power)) if v}
    raw = None if args.no_raw else {"report": report_path, "progress": progress_path, "power": args.power}
    return _finish_ingest(args, parsed, sources, args.power, raw)


def cmd_record(args) -> int:
    cmd = list(args.benchy)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    problem = check_benchy_args(cmd)
    if problem:
        return _err(problem)
    power = not args.no_power
    if power and not smi_available(args.smi_cmd):
        print(f"{PROG}: nvidia-smi not found, recording without power (use --no-power to silence)",
              file=sys.stderr)
        power = False
    if args.gpu is None and power:
        args.gpu = gpu_names(args.smi_cmd)
    with tempfile.TemporaryDirectory(prefix="benchy-") as work:
        rc, report, progress, power_csv = run_benchy(
            cmd, work, power=power, interval_ms=args.interval_ms, idle_seconds=args.idle_seconds,
            smi_cmd=args.smi_cmd, gpus=_gpus(args.gpus))
        have_report = os.path.isfile(report) and os.path.getsize(report) > 0
        have_progress = os.path.isfile(progress) and os.path.getsize(progress) > 0
        if not have_report and not have_progress:
            return _err(f"llama-benchy exited {rc} and wrote no results; nothing archived")
        try:
            parsed = merge(load_report(report) if have_report else None,
                           load_progress(progress) if have_progress else None)
        except IngestError as exc:
            return _err(str(exc))
        sources = {"report": "report.json" if have_report else "", "progress": "progress.jsonl" if have_progress else "",
                   "power": "power.csv" if power_csv else "", "command": " ".join(cmd[:1])}
        sources = {k: v for k, v in sources.items() if v}
        raw = {"report": report if have_report else None, "progress": progress if have_progress else None,
               "power": power_csv}
        out = _finish_ingest(args, parsed, sources, power_csv, raw)
    if rc != 0:
        print(f"{PROG}: note: llama-benchy exited {rc}", file=sys.stderr)
    return out if out else (1 if rc else 0)


def cmd_power_log(args) -> int:
    if not smi_available(args.smi_cmd):
        return _err("nvidia-smi not found")
    import time
    fh = sys.stdout if args.output == "-" else open(args.output, "w", encoding="utf-8", newline="\n")
    sampler = PowerSampler(fh, args.interval_ms, args.smi_cmd, _gpus(args.gpus)).start()
    if fh is not sys.stdout:
        print(f"{PROG}: logging to {args.output}, Ctrl-C to stop", file=sys.stderr)
    try:
        if args.seconds is not None:
            time.sleep(args.seconds)
        else:
            signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        sampler.stop()
        if fh is not sys.stdout:
            fh.close()
    print(f"{PROG}: {sampler.lines} samples", file=sys.stderr)
    return 0


def cmd_runs(args) -> int:
    runs = load_runs(archive_dir(args.archive))
    if args.format == "json":
        print(json.dumps([{k: r.get(k) for k in ("run_id", "started", "machine", "gpu", "model", "engine",
                                                 "quant", "tag")} for r in runs], indent=2))
        return 0
    if not runs:
        print(f"no runs in {archive_dir(args.archive)}")
        return 0
    sys.stdout.write(table(RUN_HEADERS, run_rows(runs), args.format, right=(5,)))
    return 0


def cmd_show(args) -> int:
    try:
        run = find_run(archive_dir(args.archive), args.run_id)
    except KeyError as exc:
        return _err(str(exc).strip("'\""))
    if args.format == "json":
        print(json.dumps(run, indent=2))
        return 0
    if args.format == "text":
        p = run.get("power") or {}
        print(f"run {run['run_id']}  {run.get('started', '')}  {run.get('model', '')} on {run.get('machine', '')}")
        extra = [x for x in (run.get("gpu"), run.get("engine"), run.get("quant"), run.get("tag")) if x]
        if extra:
            print("  " + " | ".join(extra))
        if p.get("idle_w") is not None:
            print(f"  idle before the run: {p['idle_w']:.0f} W")
        print()
    sys.stdout.write(table(CELL_HEADERS, cell_rows(run), args.format, right=CELL_RIGHT))
    return 0


def cmd_leaderboard(args) -> int:
    runs = load_runs(archive_dir(args.archive))
    rows, shape = leaderboard(runs, pp=args.pp, tg=args.tg, depth=args.depth, concurrency=args.concurrency,
                              sort=args.sort, pick=args.pick, machine=args.machine, model=args.model)
    if args.format == "json":
        sys.stdout.write(leaderboard_json(rows, shape, args.depth, args.concurrency))
        return 0
    if not rows:
        print("no runs match")
        return 0
    if args.format == "text":
        print(f"test {cell_label((shape[0], shape[1], args.depth, args.concurrency, 'std'))}, "
              f"{args.pick} run per setup, sorted by {METRICS[args.sort][0]}\n")
    sys.stdout.write(table(LB_HEADERS, leaderboard_rows(rows), args.format, right=LB_RIGHT))
    return 0


def cmd_regressions(args) -> int:
    runs = load_runs(archive_dir(args.archive))
    changes = regressions(runs, args.threshold, args.machine, args.model, include_improvements=args.all)
    worse = [c for c in changes if c.worse]
    if args.format == "json":
        print(json.dumps([{"setup": c.system, "test": c.cell, "metric": c.metric, "before": c.old,
                           "after": c.new, "pct": round(c.pct, 2), "noise": round(c.noise, 4),
                           "worse": c.worse, "before_run": c.old_run, "after_run": c.new_run}
                          for c in changes], indent=2))
    elif not changes:
        print(f"no change of {args.threshold:g}% or more beyond run-to-run noise")
    else:
        sys.stdout.write(table(CH_HEADERS, change_rows(changes), args.format, right=CH_RIGHT))
    return 1 if (args.fail and worse) else 0


def cmd_html(args) -> int:
    runs = load_runs(archive_dir(args.archive))
    if not runs:
        return _err(f"no runs in {archive_dir(args.archive)}")
    page = render_html(runs, title=args.title, pp=args.pp, tg=args.tg, depth=args.depth,
                       concurrency=args.concurrency, threshold=args.threshold)
    parent = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(parent, exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(page)
    print(f"wrote {args.output} ({len(runs)} runs)")
    return 0


def cmd_merge(args) -> int:
    if not os.path.isdir(args.source):
        return _err(f"{args.source} is not a folder")
    counts = merge_archives(archive_dir(args.archive), args.source)
    print(f"added {counts['added']} runs, {counts['skipped']} already present")
    return 0


COMMANDS = {
    "record": cmd_record, "ingest": cmd_ingest, "power-log": cmd_power_log, "runs": cmd_runs,
    "show": cmd_show, "leaderboard": cmd_leaderboard, "regressions": cmd_regressions,
    "html": cmd_html, "merge": cmd_merge,
}


def main(argv: Optional[List[str]] = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (ValueError, OSError):
            pass
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.cmd:
        parser.print_help()
        return 0
    return COMMANDS[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
