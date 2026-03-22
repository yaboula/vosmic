"""Tests for OverlapAddStitcher — cross-fade, reset, flush."""

from __future__ import annotations

import numpy as np

from src.inference.stitcher import OverlapAddStitcher


class TestFirstCallBehavior:
    def test_first_call_stores_tail(self) -> None:
        st = OverlapAddStitcher(overlap_ratio=0.5)
        chunk = np.ones(100, dtype=np.float32)
        result = st.stitch(chunk)
        assert len(result) == 50  # first half returned, tail stored

    def test_empty_chunk(self) -> None:
        st = OverlapAddStitcher()
        result = st.stitch(np.array([], dtype=np.float32))
        assert len(result) == 0


class TestCrossFade:
    def test_second_call_crossfades(self) -> None:
        st = OverlapAddStitcher(overlap_ratio=0.5)
        chunk1 = np.ones(100, dtype=np.float32)
        chunk2 = np.ones(100, dtype=np.float32) * 2.0

        st.stitch(chunk1)
        result = st.stitch(chunk2)

        assert len(result) > 0
        # The crossfaded region should be between 1.0 and 2.0
        assert not np.all(result[:50] == 1.0)
        assert not np.all(result[:50] == 2.0)

    def test_continuous_chunks_no_clicks(self) -> None:
        st = OverlapAddStitcher(overlap_ratio=0.25)
        outputs = []
        for i in range(10):
            chunk = np.full(256, float(i), dtype=np.float32)
            outputs.append(st.stitch(chunk))

        all_audio = np.concatenate(outputs)
        diffs = np.abs(np.diff(all_audio))
        max_jump = float(np.max(diffs))
        # With proper crossfade, jumps should be gradual
        assert max_jump < 2.0, f"Max sample jump {max_jump} suggests click artifact"


class TestResetAndFlush:
    def test_reset_clears_state(self) -> None:
        st = OverlapAddStitcher()
        st.stitch(np.ones(100, dtype=np.float32))
        st.reset()

        # After reset, next call behaves as first call
        result = st.stitch(np.ones(100, dtype=np.float32))
        assert len(result) == 50  # first-call behavior again

    def test_flush_returns_tail(self) -> None:
        st = OverlapAddStitcher(overlap_ratio=0.5)
        st.stitch(np.ones(100, dtype=np.float32))
        tail = st.flush()
        assert len(tail) == 50
        assert np.all(tail == 1.0)

    def test_flush_empty_after_reset(self) -> None:
        st = OverlapAddStitcher()
        st.reset()
        tail = st.flush()
        assert len(tail) == 0
