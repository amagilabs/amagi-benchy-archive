import pytest

from amagi_benchy_archive.power import (PowerError, Sample, baseline_power, parse_power_csv, window_power)

OURS = """epoch_s,gpu,power_w,temp_c,mem_used_mib
100.0,0,100.0,50,1000
101.0,0,300.0,60,1000
102.0,0,300.0,62,1000
103.0,0,100.0,55,1000
101.0,1,200.0,58,1000
102.0,1,[N/A],58,1000
"""

SMI = """timestamp, index, power.draw [W], temperature.gpu
2026/10/25 05:44:28.000, 0, 112.37 W, 60
2026/10/25 05:44:29.000, 0, 280.00 W, 70
"""


def test_parse_ours_skips_na():
    s = parse_power_csv(OURS)
    assert len(s) == 5
    assert s[0] == Sample(100.0, 0, 100.0, 50.0)


def test_parse_nvidia_smi_with_offset():
    s = parse_power_csv(SMI, utc_offset_hours=0)
    assert [x.watts for x in s] == [112.37, 280.0]
    # 2026-10-25 05:44:28 UTC
    assert s[0].ts == pytest.approx(1792907068.0)
    assert s[1].ts - s[0].ts == pytest.approx(1.0)
    shifted = parse_power_csv(SMI, utc_offset_hours=-4)
    assert shifted[0].ts - s[0].ts == pytest.approx(4 * 3600)


def test_parse_unknown_header():
    with pytest.raises(PowerError):
        parse_power_csv("a,b\n1,2\n")


def test_window_mean_summed_across_gpus():
    s = parse_power_csv(OURS)
    p = window_power(s, [(100.5, 102.5)])
    # gpu0: 300, 300 -> 300 ; gpu1: 200 -> 200 ; sum 500
    assert p["avg_w"] == 500.0
    assert p["peak_w"] == 500.0
    assert p["gpus"] == 2
    assert p["energy_j"] == pytest.approx(1000.0)
    assert p["max_temp_c"] == 62


def test_window_gpu_filter():
    s = parse_power_csv(OURS)
    assert window_power(s, [(100.5, 102.5)], gpus=[0])["avg_w"] == 300.0


def test_window_union_of_decode_windows():
    s = parse_power_csv(OURS)
    p = window_power(s, [(99.5, 100.5), (102.5, 103.5)], gpus=[0])
    assert p["avg_w"] == 100.0
    assert p["energy_j"] == pytest.approx(200.0)


def test_short_window_uses_nearest_sample_within_slack():
    s = parse_power_csv(OURS)
    p = window_power(s, [(101.2, 101.4)], gpus=[0])
    assert p["avg_w"] == 300.0
    assert window_power(s, [(110.0, 110.1)], gpus=[0]) is None


def test_baseline_is_mean_before_start():
    s = parse_power_csv(OURS)
    assert baseline_power(s, 101.0, gpus=[0]) == 100.0
    assert baseline_power(s, 100.0, gpus=[0]) is None
