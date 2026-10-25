import json
import os

import pytest

from amagi_benchy_archive.analysis import compare_runs, leaderboard, regressions
from amagi_benchy_archive.archive import build_run, find_run, load_runs, merge_archives, save_run
from amagi_benchy_archive.html_report import render
from amagi_benchy_archive.ingest import parse_progress
from amagi_benchy_archive.power import Sample
from helpers import request, stream


def make_run(machine="box-a", model="qwen", t0=1000.0, chunk_dt=0.1, watts=None, runs=3, std_jitter=0.0):
    reqs = []
    for i in range(runs):
        dt = chunk_dt * (1 + (std_jitter if i % 2 else -std_jitter))
        reqs.append(request(i, t0 + i * 2, run_index=i, chunk_dt=dt, model=model))
    parsed = parse_progress(stream(*reqs, t_start=t0 - 1))
    samples = None
    if watts is not None:
        samples = [Sample(t0 - 1 + k * 0.25, 0, watts if k >= 4 else 50.0) for k in range(4 * (runs * 2 + 3))]
    return build_run(parsed, machine=machine, gpu="RTX 3090", samples=samples, power_source="power.csv")


def test_build_run_power_join_and_idle():
    run = make_run(watts=250.0)
    c = run["cells"][0]
    assert c["metrics"]["tg_tps"]["mean"] == pytest.approx(10.0)
    assert c["decode_power"]["avg_w"] == 250.0
    assert c["tok_per_j"] == pytest.approx(10.0 / 250.0)
    assert run["power"]["idle_w"] == 50.0
    assert run["power"]["joined"] is True


def test_run_id_is_stable_and_dedupes(tmp_path):
    a = make_run()
    b = make_run()
    assert a["run_id"] == b["run_id"]
    save_run(str(tmp_path), a)
    save_run(str(tmp_path), b)
    assert len(load_runs(str(tmp_path))) == 1
    assert make_run(machine="box-b")["run_id"] != a["run_id"]


def test_find_run_prefix(tmp_path):
    a = make_run()
    save_run(str(tmp_path), a)
    assert find_run(str(tmp_path), a["run_id"][:4])["run_id"] == a["run_id"]
    with pytest.raises(KeyError):
        find_run(str(tmp_path), "zzzz")


def test_merge_archives(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    save_run(str(src), make_run(machine="box-a"))
    save_run(str(src), make_run(machine="box-b"))
    save_run(str(dst), make_run(machine="box-a"))
    assert merge_archives(str(dst), str(src)) == {"added": 1, "skipped": 1}
    assert len(load_runs(str(dst))) == 2


def test_raw_files_copied(tmp_path):
    raw = tmp_path / "in.jsonl"
    raw.write_text("x")
    run = make_run()
    save_run(str(tmp_path / "arc"), run, {"progress": str(raw), "power": None})
    assert (tmp_path / "arc" / "raw" / run["run_id"] / "progress.jsonl").read_text() == "x"


def test_leaderboard_ranks_latest_per_setup():
    runs = [
        make_run("box-a", "qwen", t0=1000, chunk_dt=0.1),      # 10 t/s
        make_run("box-a", "qwen", t0=5000, chunk_dt=0.05),     # 20 t/s, newer
        make_run("box-b", "qwen", t0=2000, chunk_dt=0.075),    # 13.3 t/s
    ]
    rows, shape = leaderboard(runs)
    assert shape == (512, 4)
    assert [(r.run["machine"], round(r.value("tg_tps"), 1)) for r in rows] == [("box-a", 20.0), ("box-b", 13.3)]
    rows, _ = leaderboard(runs, pick="best", sort="e2e_ttft_ms")
    assert len(rows) == 2


def test_compare_respects_threshold_and_noise():
    old = make_run(t0=1000, chunk_dt=0.1)
    new = make_run(t0=5000, chunk_dt=0.125)  # 10 -> 8 t/s, -20%
    ch = compare_runs(old, new, threshold_pct=5)
    tg = [c for c in ch if c.metric == "tg_tps"]
    assert len(tg) == 1 and tg[0].worse and tg[0].pct == pytest.approx(-20.0)
    assert not any(c.metric == "tg_tps" for c in compare_runs(old, new, threshold_pct=25))


def test_noisy_runs_not_flagged():
    old = make_run(t0=1000, chunk_dt=0.1, std_jitter=0.3)
    new = make_run(t0=5000, chunk_dt=0.106, std_jitter=0.3)  # ~6% slower but std is ~30%
    assert [c for c in compare_runs(old, new, 5) if c.metric == "tg_tps"] == []


def test_regressions_only_latest_pair_and_worse_by_default():
    runs = [make_run(t0=1000, chunk_dt=0.2), make_run(t0=2000, chunk_dt=0.1), make_run(t0=3000, chunk_dt=0.125)]
    ch = regressions(runs)
    tg = [c for c in ch if c.metric == "tg_tps"]
    assert len(tg) == 1
    assert tg[0].old == pytest.approx(10.0) and tg[0].new == pytest.approx(8.0)
    better = regressions([runs[0], runs[1]], include_improvements=True)
    assert any(not c.worse and c.metric == "tg_tps" for c in better)
    assert [c for c in regressions([runs[0], runs[1]]) if c.metric == "tg_tps"] == []


def test_html_render_contains_sections_and_credit():
    runs = [make_run("box-a", t0=1000, watts=250.0), make_run("box-a", t0=3000, chunk_dt=0.125, watts=250.0),
            make_run("box-b <script>", t0=2000)]
    page = render(runs, title="Test board", generated="2026-10-25 00:00 UTC")
    assert "<h1>Test board</h1>" in page
    assert "Generation speed" in page and "Tokens per joule" in page and "Regressions" in page
    assert "History" in page
    assert "WORSE" in page
    assert "<script>" not in page  # escaped
    assert "box-b &lt;script&gt;" in page
    assert "Built on <a href=\"https://github.com/eugr/llama-benchy\">llama-benchy</a> by eugr (MIT)" in page
    assert "llama-benchy-viz by Alex Ziskind (MIT)" in page


def test_html_render_empty_archive_message():
    page = render([])
    assert "No archived run" in page
