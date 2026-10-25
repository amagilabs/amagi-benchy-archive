import json
import os
import subprocess
import sys

import pytest

from amagi_benchy_archive.cli import main
from amagi_benchy_archive.record import check_benchy_args
from helpers import request, stream, write_jsonl

FIX = os.path.join(os.path.dirname(__file__), "fixtures")


def _ingest_progress(tmp_path, arc, name, t0, chunk_dt, machine="box-a", power=None):
    p = write_jsonl(tmp_path / f"{name}.jsonl",
                    stream(*[request(i, t0 + i, run_index=i, chunk_dt=chunk_dt) for i in range(3)], t_start=t0 - 1))
    args = ["ingest", str(p), "-a", str(arc), "--machine", machine, "--gpu", "RTX 3090"]
    if power:
        args += ["--power", str(power)]
    return main(args)


def test_ingest_report_only(tmp_path, capsys):
    arc = tmp_path / "arc"
    rc = main(["ingest", os.path.join(FIX, "llama_benchy_sample.json"), "-a", str(arc), "--machine", "spark"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "archived run" in out and "2 tests" in out
    runs = os.listdir(arc / "runs")
    assert len(runs) == 1
    assert os.path.isfile(arc / "raw" / runs[0][:-5] / "report.json")


def test_ingest_with_power_and_full_flow(tmp_path, capsys):
    arc = tmp_path / "arc"
    power = tmp_path / "power.csv"
    lines = ["epoch_s,gpu,power_w,temp_c,mem_used_mib"]
    lines += [f"{998 + k * 0.25:.2f},0,{'40' if k < 4 else '280'},60,9000" for k in range(40)]
    power.write_text("\n".join(lines) + "\n")
    assert _ingest_progress(tmp_path, arc, "a", 1000.0, 0.1, power=power) == 0
    assert "watts on 1" in capsys.readouterr().out
    assert _ingest_progress(tmp_path, arc, "b", 5000.0, 0.125) == 0
    assert _ingest_progress(tmp_path, arc, "c", 3000.0, 0.05, machine="box-b") == 0
    capsys.readouterr()

    assert main(["runs", "-a", str(arc)]) == 0
    out = capsys.readouterr().out
    assert out.count("box-a") == 2 and "box-b" in out and "yes (40 samples)" in out

    assert main(["leaderboard", "-a", str(arc), "--format", "json"]) == 0
    lb = json.loads(capsys.readouterr().out)
    assert lb["cell"] == "pp512/tg4"
    assert [r["machine"] for r in lb["rows"]] == ["box-b", "box-a"]
    assert lb["rows"][1]["tg_tps"] == pytest.approx(8.0)

    assert main(["leaderboard", "-a", str(arc), "--format", "md"]) == 0
    assert capsys.readouterr().out.startswith("| # | model |")

    assert main(["regressions", "-a", str(arc), "--fail"]) == 1
    out = capsys.readouterr().out
    assert "WORSE" in out and "-20.0%" in out
    assert main(["regressions", "-a", str(arc), "--threshold", "50"]) == 0
    assert "no change" in capsys.readouterr().out

    run_id = sorted(os.listdir(arc / "runs"))[0][:6]
    assert main(["show", run_id, "-a", str(arc)]) == 0
    assert "pp512/tg4" in capsys.readouterr().out

    out_html = tmp_path / "site" / "index.html"
    assert main(["html", "-a", str(arc), "-o", str(out_html), "--title", "Box board"]) == 0
    page = out_html.read_text(encoding="utf-8")
    assert "<h1>Box board</h1>" in page and "Regressions" in page
    # the run with watts is box-a's older run, so the latest-per-setup board has no tok/J chart
    assert "Tokens per joule" not in page


def test_merge_command(tmp_path, capsys):
    a, b = tmp_path / "a", tmp_path / "b"
    _ingest_progress(tmp_path, a, "x", 1000.0, 0.1, machine="box-a")
    _ingest_progress(tmp_path, b, "y", 2000.0, 0.1, machine="box-b")
    capsys.readouterr()
    assert main(["merge", str(b), "-a", str(a)]) == 0
    assert "added 1 runs, 0 already present" in capsys.readouterr().out
    assert main(["merge", str(b), "-a", str(a)]) == 0
    assert "added 0 runs, 1 already present" in capsys.readouterr().out


def test_errors_are_clean(tmp_path, capsys):
    assert main(["ingest", "-a", str(tmp_path)]) == 2
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"schema": "llama-benchy-progress.v1", "type": "header", "ts": 1}\n{oops\n')
    assert main(["ingest", str(bad), "-a", str(tmp_path)]) == 2
    assert "line 2" in capsys.readouterr().err
    assert main(["show", "nope", "-a", str(tmp_path)]) == 2
    assert main(["html", "-a", str(tmp_path / "empty")]) == 2


def test_record_rejects_reserved_flags():
    assert check_benchy_args([]) is not None
    assert "set by record" in check_benchy_args(["llama-benchy", "--format", "md"])
    assert "set by record" in check_benchy_args(["llama-benchy", "--save-result=x.json"])
    assert check_benchy_args(["llama-benchy", "--base-url", "http://h/v1"]) is None


def test_record_runs_a_fake_benchy(tmp_path, capsys):
    """record hands its own --emit-progress/--save-result paths to the command and archives what it wrote."""
    fake = tmp_path / "fake_benchy.py"
    events = stream(request(0, 1000.0), request(1, 1001.0, run_index=1))
    src = write_jsonl(tmp_path / "canned.jsonl", events)
    fake.write_text(
        "import shutil, sys\n"
        "a = sys.argv\n"
        "prog = a[a.index('--emit-progress') + 1]\n"
        "a[a.index('--save-result') + 1]\n"
        "assert a[a.index('--format') + 1] == 'json'\n"
        f"shutil.copyfile({str(src)!r}, prog)\n"
    )
    arc = tmp_path / "arc"
    rc = main(["record", "-a", str(arc), "--machine", "box", "--no-power", "--gpu", "test",
               "--", sys.executable, str(fake), "--base-url", "http://h/v1"])
    assert rc == 0
    assert "archived run" in capsys.readouterr().out
    (run_file,) = os.listdir(arc / "runs")
    run = json.loads((arc / "runs" / run_file).read_text())
    assert run["sources"]["progress"] == "progress.jsonl"
    assert run["cells"][0]["metrics"]["tg_tps"]["mean"] == pytest.approx(10.0)


def test_record_reports_missing_results(tmp_path, capsys):
    rc = main(["record", "-a", str(tmp_path), "--no-power", "--machine", "m", "--",
               sys.executable, "-c", "import sys; sys.exit(3)"])
    assert rc == 2
    assert "wrote no results" in capsys.readouterr().err


def test_console_script_installed():
    out = subprocess.run([sys.executable, "-m", "amagi_benchy_archive", "--version"],
                         capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.startswith("amagi-benchy-archive ")
