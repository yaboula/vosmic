"""Tests for ContentEncoder — stub fallback and factory."""

from __future__ import annotations

import numpy as np

from src.dsp.content_encoder import ContentEncoder, StubContentEncoder


class TestStubEncoder:
    def test_output_shape(self) -> None:
        enc = StubContentEncoder()
        audio = np.random.randn(1600).astype(np.float32) * 0.1
        embeddings = enc.encode(audio, sample_rate=16000)
        assert embeddings.ndim == 2
        assert embeddings.shape[1] == 256

    def test_empty_input(self) -> None:
        enc = StubContentEncoder()
        embeddings = enc.encode(np.array([], dtype=np.float32))
        assert embeddings.shape == (0, 256)

    def test_deterministic(self) -> None:
        enc = StubContentEncoder()
        audio = np.random.randn(1600).astype(np.float32) * 0.1
        e1 = enc.encode(audio, sample_rate=16000)
        enc2 = StubContentEncoder()
        e2 = enc2.encode(audio, sample_rate=16000)
        assert np.allclose(e1, e2), "Same seed should give same result"

    def test_l2_normalized(self) -> None:
        enc = StubContentEncoder()
        audio = np.random.randn(3200).astype(np.float32) * 0.1
        embeddings = enc.encode(audio, sample_rate=16000)
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_resamples_from_48k(self) -> None:
        enc = StubContentEncoder()
        audio = np.random.randn(4800).astype(np.float32) * 0.1
        embeddings = enc.encode(audio, sample_rate=48000)
        assert embeddings.ndim == 2
        assert embeddings.shape[1] == 256


class TestContentEncoderFactory:
    def test_falls_back_to_stub(self) -> None:
        enc = ContentEncoder(model="contentvec", device="cuda")
        assert enc.model_name == "stub"

    def test_encoding_dim(self) -> None:
        enc = ContentEncoder()
        assert enc.embedding_dim == 256

    def test_encode_via_factory(self) -> None:
        enc = ContentEncoder()
        audio = np.random.randn(4800).astype(np.float32) * 0.1
        embeddings = enc.encode(audio, sample_rate=48000)
        assert embeddings.shape[1] == 256
