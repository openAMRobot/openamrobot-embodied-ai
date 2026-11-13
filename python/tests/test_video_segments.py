# SPDX-License-Identifier: MIT
"""Guard adjacent frame selection at repeating fractional boundaries."""

import pytest
from oa_embodied_ai.dataset_qa.video import segment_positions


def test_repeating_30hz_boundary_has_no_leak_or_gap():
    probe = {
        "streams": [{"time_base": "1/15360"}],
        "frames": [
            {
                "best_effort_timestamp": i * 512,
                "best_effort_timestamp_time": f"{i / 30:.6f}",
            }
            for i in range(8)
        ],
    }
    first = segment_positions(probe, 0, 4 / 30)
    second = segment_positions(probe, 4 / 30, 8 / 30)
    assert first == [0, 1, 2, 3]
    assert second == [4, 5, 6, 7]
    assert first + second == list(range(8))
    # Old decimal-text comparison assigns the boundary to the wrong episode.
    assert [
        i
        for i, f in enumerate(probe["frames"])
        if 0 <= float(f["best_effort_timestamp_time"]) < 4 / 30
    ] != first


@pytest.mark.parametrize(
    "start,end",
    [(None, 1), (0, None), (-1, 1), (1, 0), (float("nan"), 1), (0, float("inf"))],
)
def test_invalid_segment_bounds_reject(start, end):
    with pytest.raises(ValueError):
        segment_positions(
            {
                "streams": [{"time_base": "1/1000"}],
                "frames": [{"best_effort_timestamp": 0}],
            },
            start,
            end,
        )


@pytest.mark.parametrize(
    "probe",
    [
        {"frames": [{"best_effort_timestamp_time": "0"}]},
        {
            "streams": [{"time_base": "1/1000"}],
            "frames": [{"best_effort_timestamp": 1}, {"best_effort_timestamp": 1}],
        },
        {
            "streams": [{"time_base": "1/1000"}],
            "frames": [{"best_effort_timestamp": "0"}],
        },
        {"streams": [{"time_base": "0/1"}], "frames": [{"best_effort_timestamp": 0}]},
    ],
)
def test_bad_pts_or_time_base_reject(probe):
    with pytest.raises(ValueError):
        segment_positions(probe, 0, 1)
