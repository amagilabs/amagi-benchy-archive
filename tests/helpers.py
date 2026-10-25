"""Build llama-benchy-progress.v1 streams with known timing, so expected numbers are exact."""

import json

SCHEMA = "llama-benchy-progress.v1"


def ev(type_, ts, **kw):
    return dict(schema=SCHEMA, type=type_, ts=ts, **kw)


def request(rid, t0, *, pp=512, tg=4, depth=0, conc=1, run_index=0, ttfr=0.2, chunk_dt=0.1,
            chunk_tokens=(1, 1, 1, 1), prompt_tokens=None, error="", model="test/model"):
    """One request: start at t0, first response after ttfr, then one chunk every chunk_dt."""
    out = [ev("request_start", t0, request_id=rid, model=model, base_url="http://x/v1", prompt_size=pp,
              response_size=tg, context_size=depth, concurrency=conc, run_index=run_index, target_label="")]
    if error:
        out.append(ev("request_end", t0 + ttfr, request_id=rid, total_tokens=0, prompt_tokens=0,
                      decode_seconds=0.0, error=error))
        return out
    out.append(ev("request_first_response", t0 + ttfr, request_id=rid, ttfr_s=ttfr))
    out.append(ev("request_first_token", t0 + ttfr, request_id=rid, ttft_s=ttfr))
    t = t0 + ttfr
    for i, n in enumerate(chunk_tokens):
        out.append(ev("tokens", t + i * chunk_dt, request_id=rid, count=n, snippet="x"))
    end = t + (len(chunk_tokens) - 1) * chunk_dt
    out.append(ev("request_end", end, request_id=rid, total_tokens=sum(chunk_tokens),
                  prompt_tokens=pp + depth if prompt_tokens is None else prompt_tokens,
                  decode_seconds=end - t, error=""))
    return out


def stream(*requests, latency=0.0, t_start=1000.0, t_end=None, status="ok", version="0.3.0"):
    events = [ev("header", t_start, llama_benchy_version=version),
              ev("latency_measured", t_start + 0.01, latency_s=latency, mode="api")]
    for r in requests:
        events.extend(r)
    last = max(e["ts"] for e in events)
    events.append(ev("bench_complete", t_end or last + 0.01, status=status))
    return events


def write_jsonl(path, events):
    with open(path, "w", encoding="utf-8") as fh:
        for e in events:
            fh.write(json.dumps(e) + "\n")
    return path
