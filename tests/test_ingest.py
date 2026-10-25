import json
import os

import pytest

from amagi_benchy_archive.ingest import (IngestError, cell_label, load_report, merge, parse_progress,
                                         read_progress_lines, sniff)
from helpers import ev, request, stream, write_jsonl

FIX = os.path.join(os.path.dirname(__file__), "fixtures")


def test_report_fixture_from_upstream():
    p = load_report(os.path.join(FIX, "llama_benchy_sample.json"))
    assert p.model == "test"
    assert p.latency_mode == "api"
    assert [c.label for c in p.cells] == ["pp2048/tg32", "pp2048/tg32 c2"]
    c1 = p.cells[0]
    assert c1.metrics["tg_tps"]["mean"] == pytest.approx(49.97785098352942)
    assert c1.metrics["tg_tps"]["n"] == 2
    assert c1.metrics["pp_tps"]["mean"] == pytest.approx(987.0987514675164)
    assert c1.metrics["e2e_ttft_ms"]["mean"] == pytest.approx(2076.0429790243506)
    assert c1.start_ts is None  # a report has no per-test timing


def test_report_rejects_other_json():
    from amagi_benchy_archive.ingest import parse_report
    with pytest.raises(IngestError):
        parse_report({"hello": 1})


def test_progress_single_request_math():
    # 4 chunks of 1 token, 0.1 s apart: 3 tokens after the first in 0.3 s = 10 t/s.
    events = stream(request(0, 1000.0, pp=512, ttfr=0.25), latency=0.05)
    p = parse_progress(events)
    assert p.llama_benchy_version == "0.3.0"
    assert p.latency_ms == pytest.approx(50.0)
    assert p.status == "ok"
    (c,) = p.cells
    assert c.key == (512, 4, 0, 1, "std")
    assert c.metrics["tg_tps"]["mean"] == pytest.approx(10.0)
    # est_ppt = ttfr - latency = 0.2 s -> 512 / 0.2 = 2560 t/s
    assert c.metrics["est_ppt_ms"]["mean"] == pytest.approx(200.0)
    assert c.metrics["pp_tps"]["mean"] == pytest.approx(2560.0)
    assert c.metrics["e2e_ttft_ms"]["mean"] == pytest.approx(250.0)
    assert c.start_ts == 1000.0
    assert c.end_ts == pytest.approx(1000.55)
    assert c.decode_windows == [(pytest.approx(1000.25), pytest.approx(1000.55))]


def test_progress_multi_token_chunks_counted():
    # speculative decoding style: chunks of 1, 3, 3, 3 -> 9 tokens after first in 0.3 s
    events = stream(request(0, 1000.0, chunk_tokens=(1, 3, 3, 3), tg=10))
    (c,) = parse_progress(events).cells
    assert c.metrics["tg_tps"]["mean"] == pytest.approx(30.0)
    # a burst shorter than 1 s is rated over its own length: 10 tokens / 0.3 s
    assert c.metrics["peak_tps"]["mean"] == pytest.approx(10 / 0.3)
    assert c.metrics["peak_req_tps"]["mean"] == pytest.approx(10 / 0.3)


def test_progress_estimated_counts_spread_over_reported_total():
    # server sent no token ids: 4 chunks over 0.3 s, request_end says 7 tokens.
    # llama-benchy spreads 7 timestamps evenly -> 6 after the first in 0.3 s = 20 t/s.
    events = stream(request(0, 1000.0, chunk_tokens=(1, 1, 1, 1), estimated=True, total_tokens=7))
    (c,) = parse_progress(events).cells
    assert c.metrics["tg_tps"]["mean"] == pytest.approx(20.0)


def test_peak_uses_sliding_one_second_window():
    # 31 single-token chunks 0.1 s apart (3 s): any 1 s window holds 10 of them.
    events = stream(request(0, 1000.0, chunk_tokens=(1,) * 31, tg=31))
    (c,) = parse_progress(events).cells
    assert c.metrics["peak_tps"]["mean"] == pytest.approx(10.0)


def test_progress_runs_mean_and_std():
    events = stream(
        request(0, 1000.0, run_index=0, chunk_dt=0.1),   # 10 t/s
        request(1, 1010.0, run_index=1, chunk_dt=0.05),  # 20 t/s
    )
    (c,) = parse_progress(events).cells
    m = c.metrics["tg_tps"]
    assert m["mean"] == pytest.approx(15.0)
    assert m["std"] == pytest.approx(5.0)
    assert m["n"] == 2
    assert c.requests == 2


def test_progress_concurrency_uses_batch_throughput():
    # two requests at once, each 3 tokens after first over 0.3 s, overlapping fully.
    events = stream(
        request(0, 1000.0, conc=2, chunk_tokens=(1, 1, 1, 1)),
        request(1, 1000.0, conc=2, chunk_tokens=(1, 1, 1, 1)),
    )
    (c,) = parse_progress(events).cells
    assert c.concurrency == 2
    # batch: tokens after each request's first (3 + 3) / (last token - first token = 0.3 s)
    assert c.metrics["tg_tps"]["mean"] == pytest.approx(6 / 0.3)
    assert c.metrics["tg_req_tps"]["mean"] == pytest.approx(10.0)


def test_progress_prefix_caching_splits_ctx_phase():
    events = stream(
        request(0, 1000.0, pp=256, depth=4096, ttfr=1.0, prompt_tokens=4096),
        request(1, 1002.0, pp=256, depth=4096, ttfr=0.1, prompt_tokens=256),
    )
    cells = {c.phase: c for c in parse_progress(events).cells}
    assert set(cells) == {"ctx", "std"}
    assert cells["ctx"].metrics["pp_tps"]["mean"] == pytest.approx(4096.0)
    assert cells["std"].metrics["pp_tps"]["mean"] == pytest.approx(2560.0)
    assert cell_label(cells["ctx"].key) == "pp256/tg4 @ d4096 [ctx]"


def test_progress_failed_requests_counted_not_measured():
    events = stream(request(0, 1000.0), request(1, 1001.0, run_index=1, error="HTTP 500"))
    (c,) = parse_progress(events).cells
    assert c.errors == 1
    assert c.requests == 2
    assert c.metrics["tg_tps"]["n"] == 1


def test_progress_rejects_wrong_schema():
    with pytest.raises(IngestError, match="unknown schema"):
        read_progress_lines([json.dumps({"schema": "other.v9", "type": "header", "ts": 1})])


def test_progress_rejects_garbage_line():
    with pytest.raises(IngestError, match="line 2"):
        read_progress_lines([json.dumps(ev("header", 1.0)), "{not json"])


def test_unknown_event_types_ignored():
    events = stream(request(0, 1000.0))
    events.insert(2, ev("something_new", 1000.0, foo=1))
    (c,) = parse_progress(events).cells
    assert c.metrics["tg_tps"]["mean"] == pytest.approx(10.0)


def test_merge_report_metrics_with_progress_timing(tmp_path):
    report = load_report(os.path.join(FIX, "llama_benchy_sample.json"))
    events = stream(request(0, 1000.0, pp=2048, tg=32), request(1, 1005.0, pp=2048, tg=32, conc=2),
                    request(2, 1005.0, pp=2048, tg=32, conc=2))
    progress = parse_progress(events)
    merged = merge(report, progress)
    c1 = merged.cells[0]
    assert c1.metrics["tg_tps"]["mean"] == pytest.approx(49.97785098352942)  # report wins
    assert c1.start_ts == 1000.0  # progress adds timing
    assert merged.cells[1].start_ts == 1005.0
    assert merged.started_ts == 1000.0


def test_sniff(tmp_path):
    jl = write_jsonl(tmp_path / "p.jsonl", stream(request(0, 1.0)))
    assert sniff(str(jl)) == "progress"
    assert sniff(os.path.join(FIX, "llama_benchy_sample.json")) == "report"


def test_real_run_progress_matches_llama_benchy_report():
    """Both files come from one real llama-benchy 0.4.1.dev run against its tests/mock_server.py.
    Rebuilding the numbers from the JSONL alone must land on the report's own figures."""
    report = load_report(os.path.join(FIX, "mock_run_report.json"))
    progress = parse_progress(read_progress_lines(open(os.path.join(FIX, "mock_run_progress.jsonl"),
                                                       encoding="utf-8")))
    ours = {c.key: c for c in progress.cells}
    assert set(ours) == {c.key for c in report.cells}
    assert len(ours) == 12
    for c in report.cells:
        q = ours[c.key]
        for m in ("ttfr_ms", "est_ppt_ms", "e2e_ttft_ms", "pp_req_tps"):
            assert q.metrics[m]["mean"] == pytest.approx(c.metrics[m]["mean"], rel=1e-6), (c.label, m)
        for m in ("pp_tps", "tg_tps", "tg_req_tps", "peak_tps"):
            # token timestamps in the JSONL are taken when the event is written, a hair after
            # the client's own clock read, so allow a small drift.
            assert q.metrics[m]["mean"] == pytest.approx(c.metrics[m]["mean"], rel=0.02), (c.label, m)
