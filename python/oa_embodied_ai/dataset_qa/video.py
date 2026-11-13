# SPDX-License-Identifier: MIT
"""Shared native-video frame selection using integer presentation timestamps."""

from fractions import Fraction

PROBE_ENTRIES = "frame=best_effort_timestamp,width,height:stream=time_base"


def segment_positions(probe, start=None, end=None):
    """Return ordinals in a half-open native segment, quantized to video ticks.

    ffprobe's *_time text is typically rounded to six decimals; comparing it to
    native float boundaries misassigns frames at 30 Hz. Integer PTS and the
    stream's rational time base are authoritative here. Native float boundaries
    are rounded once to the nearest representable tick (ties to even). This
    handles serialization noise without adding an arbitrary timing tolerance.
    """
    frames = probe.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Video has no decoded frames")
    if start is None and end is None:
        return list(range(len(frames)))
    if start is None or end is None or isinstance(start, bool) or isinstance(end, bool):
        raise ValueError("Video segment requires finite start and end timestamps")
    try:
        streams = probe["streams"]
        if len(streams) != 1:
            raise ValueError("Video probe requires exactly one selected stream")
        time_base = Fraction(streams[0]["time_base"])
        lo, hi = Fraction(str(start)), Fraction(str(end))
        if time_base <= 0 or lo < 0 or hi <= lo:
            raise ValueError("Invalid video time base or segment interval")
        lo_tick, hi_tick = round(lo / time_base), round(hi / time_base)
        pts = [frame["best_effort_timestamp"] for frame in frames]
    except (KeyError, TypeError, ZeroDivisionError) as exc:
        raise ValueError(
            "Video segment requires integer PTS and rational time_base"
        ) from exc
    if hi_tick <= lo_tick or any(type(p) is not int for p in pts):
        raise ValueError("Video segment has no representable interval or invalid PTS")
    if any(a >= b for a, b in zip(pts, pts[1:], strict=False)):
        raise ValueError("Nonmonotonic or duplicate video presentation timestamps")
    return [i for i, p in enumerate(pts) if lo_tick <= p < hi_tick]
