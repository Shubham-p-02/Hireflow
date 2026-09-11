"""Unit tests for core/vector_store.py's is_ready()/initialize() state handling
— no real Pinecone or embedding calls (Pinecone client and get_embeddings are
mocked)."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.vector_store import VectorStore


class TestIsReady:
    def test_false_before_initialize(self):
        vs = VectorStore()
        assert vs.is_ready() is False

    def test_false_when_api_key_missing(self):
        with patch("core.vector_store.PINECONE_API_KEY", ""):
            vs = VectorStore()
            vs.initialize()
        assert vs.is_ready() is False

    def test_false_when_ensure_index_fails(self):
        # Regression: initialize() used to return False here without ever
        # setting self._ready, leaving is_ready() returning None instead of
        # False (violates its -> bool contract, and reads as "unknown"
        # rather than "not ready" in status reporting).
        with patch("core.vector_store.PINECONE_API_KEY", "fake-key"), \
             patch("core.vector_store.Pinecone") as MockPinecone:
            MockPinecone.return_value.list_indexes.side_effect = Exception("network blocked")
            vs = VectorStore()
            result = vs.initialize()
        assert result is False
        assert vs.is_ready() is False

    def test_false_when_embeddings_unavailable(self):
        with patch("core.vector_store.PINECONE_API_KEY", "fake-key"), \
             patch("core.vector_store.Pinecone") as MockPinecone, \
             patch("core.vector_store.get_embeddings", return_value=None):
            mock_index_info = MagicMock(dimension=384)
            MockPinecone.return_value.list_indexes.return_value = [MagicMock(name="hireflow")]
            MockPinecone.return_value.describe_index.return_value = mock_index_info
            vs = VectorStore()
            with patch.object(vs, "ensure_index", return_value=True):
                result = vs.initialize()
        assert result is False
        assert vs.is_ready() is False

    def test_true_on_full_success(self):
        with patch("core.vector_store.PINECONE_API_KEY", "fake-key"), \
             patch("core.vector_store.Pinecone") as MockPinecone, \
             patch("core.vector_store.get_embeddings", return_value=MagicMock()):
            vs = VectorStore()
            with patch.object(vs, "ensure_index", return_value=True):
                result = vs.initialize()
        assert result is True
        assert vs.is_ready() is True

    def test_re_initialize_after_success_then_failure_resets_to_false(self):
        vs = VectorStore()
        vs._ready = True  # simulate a prior successful initialize()
        with patch("core.vector_store.PINECONE_API_KEY", ""):
            vs.initialize()
        assert vs.is_ready() is False
