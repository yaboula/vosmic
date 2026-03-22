"""Tests for LockFreeRingBuffer — SPSC lock-free ring buffer (TDD)."""

from __future__ import annotations

import threading
import time

import numpy as np

from src.audio_io.ring_buffer import LockFreeRingBuffer, next_power_of_two


class TestNextPowerOfTwo:
    def test_exact_power(self) -> None:
        assert next_power_of_two(4096) == 4096

    def test_rounds_up(self) -> None:
        assert next_power_of_two(3000) == 4096

    def test_rounds_up_large(self) -> None:
        assert next_power_of_two(5000) == 8192

    def test_one(self) -> None:
        assert next_power_of_two(1) == 1

    def test_two(self) -> None:
        assert next_power_of_two(2) == 2


class TestWriteAndReadBasic:
    def test_write_and_read(self) -> None:
        buf = LockFreeRingBuffer(1024)
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        assert buf.write(data) is True
        assert buf.available_read == 4
        result = buf.read(4)
        assert np.array_equal(result, data)
        assert buf.available_read == 0

    def test_write_zero_samples(self) -> None:
        buf = LockFreeRingBuffer(1024)
        empty = np.array([], dtype=np.float32)
        assert buf.write(empty) is True
        assert buf.available_read == 0

    def test_multiple_writes_then_read(self) -> None:
        buf = LockFreeRingBuffer(1024)
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 4.0], dtype=np.float32)
        buf.write(a)
        buf.write(b)
        assert buf.available_read == 4
        result = buf.read(4)
        assert np.array_equal(result, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))

    def test_partial_read(self) -> None:
        buf = LockFreeRingBuffer(1024)
        buf.write(np.arange(10, dtype=np.float32))
        result = buf.read(5)
        assert len(result) == 5
        assert np.array_equal(result, np.arange(5, dtype=np.float32))
        assert buf.available_read == 5

    def test_read_more_than_available_returns_what_exists(self) -> None:
        buf = LockFreeRingBuffer(1024)
        buf.write(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        result = buf.read(100)
        assert len(result) == 3
        assert np.array_equal(result, np.array([1.0, 2.0, 3.0], dtype=np.float32))


class TestOverflow:
    def test_overflow_returns_false(self) -> None:
        buf = LockFreeRingBuffer(4)
        data = np.ones(5, dtype=np.float32)
        assert buf.write(data) is False
        assert buf.available_read == 0

    def test_exact_capacity_succeeds(self) -> None:
        buf = LockFreeRingBuffer(8)
        data = np.ones(8, dtype=np.float32)
        assert buf.write(data) is True
        assert buf.available_read == 8
        assert buf.available_write == 0


class TestEmptyRead:
    def test_empty_read_returns_empty(self) -> None:
        buf = LockFreeRingBuffer(1024)
        result = buf.read(100)
        assert len(result) == 0
        assert result.dtype == np.float32


class TestWrapAround:
    def test_wrap_around_correctness(self) -> None:
        buf = LockFreeRingBuffer(8)
        buf.write(np.arange(6, dtype=np.float32))
        buf.read(6)
        new_data = np.array([10, 20, 30, 40, 50, 60], dtype=np.float32)
        assert buf.write(new_data) is True
        result = buf.read(6)
        assert np.array_equal(result, new_data)

    def test_many_wrap_arounds(self) -> None:
        buf = LockFreeRingBuffer(16)
        chunk = np.arange(10, dtype=np.float32)
        for i in range(100):
            assert buf.write(chunk) is True
            result = buf.read(10)
            assert np.array_equal(result, chunk), f"Failed at iteration {i}"


class TestPeekAndClear:
    def test_peek_does_not_advance_pointer(self) -> None:
        buf = LockFreeRingBuffer(1024)
        buf.write(np.array([1.0, 2.0, 3.0], dtype=np.float32))
        peeked = buf.peek(3)
        assert buf.available_read == 3  # unchanged
        assert np.array_equal(peeked, np.array([1.0, 2.0, 3.0], dtype=np.float32))

    def test_clear_empties_buffer(self) -> None:
        buf = LockFreeRingBuffer(1024)
        buf.write(np.ones(100, dtype=np.float32))
        assert buf.available_read == 100
        buf.clear()
        assert buf.available_read == 0

    def test_len_matches_available_read(self) -> None:
        buf = LockFreeRingBuffer(1024)
        buf.write(np.ones(42, dtype=np.float32))
        assert len(buf) == 42


class TestConcurrentSPSC:
    def test_concurrent_write_read(self) -> None:
        """Simulate SPSC with two threads — 256K total samples."""
        buf = LockFreeRingBuffer(4096)
        num_chunks = 1000
        chunk_size = 256
        errors: list[str] = []

        def producer() -> None:
            for i in range(num_chunks):
                chunk = np.full(chunk_size, float(i), dtype=np.float32)
                while not buf.write(chunk):
                    time.sleep(0.0001)

        def consumer() -> None:
            total_read = 0
            target = num_chunks * chunk_size
            while total_read < target:
                data = buf.read(chunk_size)
                if len(data) > 0:
                    total_read += len(data)
                else:
                    time.sleep(0.0001)
            if total_read != target:
                errors.append(f"Expected {target} samples, got {total_read}")

        t_prod = threading.Thread(target=producer)
        t_cons = threading.Thread(target=consumer)
        t_prod.start()
        t_cons.start()
        t_prod.join(timeout=10)
        t_cons.join(timeout=10)
        assert not t_prod.is_alive(), "Producer thread hung"
        assert not t_cons.is_alive(), "Consumer thread hung"
        assert len(errors) == 0, errors


class TestPerformanceNoAllocation:
    def test_no_excessive_allocation(self) -> None:
        """Verify read/write hot path does not leak memory."""
        import tracemalloc

        buf = LockFreeRingBuffer(4096)
        data = np.ones(256, dtype=np.float32)

        buf.write(data)
        buf.read(256)

        tracemalloc.start()
        snapshot1 = tracemalloc.take_snapshot()

        for _ in range(10_000):
            buf.write(data)
            buf.read(256)

        snapshot2 = tracemalloc.take_snapshot()
        tracemalloc.stop()

        diff = snapshot2.compare_to(snapshot1, "lineno")
        total_leaked = sum(stat.size_diff for stat in diff if stat.size_diff > 0)
        assert total_leaked < 4096, f"Leaked {total_leaked} bytes in 10K cycles"


class TestCapacityAndProperties:
    def test_capacity_rounded_to_power_of_two(self) -> None:
        buf = LockFreeRingBuffer(3000)
        assert buf.capacity == 4096

    def test_available_write_full_buffer(self) -> None:
        buf = LockFreeRingBuffer(8)
        buf.write(np.ones(8, dtype=np.float32))
        assert buf.available_write == 0
        assert buf.available_read == 8

    def test_available_after_read(self) -> None:
        buf = LockFreeRingBuffer(8)
        buf.write(np.ones(8, dtype=np.float32))
        buf.read(4)
        assert buf.available_read == 4
        assert buf.available_write == 4
