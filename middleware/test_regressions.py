"""Isolated regression tests for the middleware safety boundaries."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from middleware import read_path, write_path, consolidator
from middleware.memory_schema import MemoryObject


class FakeCollection:
    def __init__(self, query_result=None):
        self.query_result = query_result or {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
            "ids": [[]],
        }
        self.where = None

    def count(self):
        return 1

    def query(self, **kwargs):
        self.where = kwargs["where"]
        return self.query_result


class MiddlewareRegressionTests(unittest.TestCase):
    def test_read_retrieval_filters_by_session(self):
        memory = MemoryObject(
            text="private fact",
            role="user",
            session_id="session-a",
            importance_score=1.0,
        )
        fake = FakeCollection({
            "documents": [[memory.text]],
            "metadatas": [[memory.to_metadata()]],
            "distances": [[0.0]],
            "ids": [[memory.id]],
        })
        with patch.object(read_path, "collection", fake), patch.object(
            read_path, "embed", return_value=[0.1]
        ):
            read_path.retrieve_memories("fact", session_id="session-a")

        self.assertEqual(
            fake.where,
            {"$and": [
                {"status": {"$eq": "active"}},
                {"session_id": {"$eq": "session-a"}},
            ]},
        )

    def test_write_similarity_filters_by_session(self):
        fake = FakeCollection()
        with patch.object(write_path, "collection", fake):
            write_path.get_top_similar([0.1], session_id="session-b")

        self.assertEqual(
            fake.where,
            {"$and": [
                {"status": {"$eq": "active"}},
                {"session_id": {"$eq": "session-b"}},
            ]},
        )

    def test_invalid_llm_fields_use_safe_fallback(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(
                message=SimpleNamespace(content='{"importance":"high", "contradiction":false, "reasoning":"x"}')
            )]
        )
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response)))
        with patch.object(write_path, "groq_client", client):
            result = write_path.job3_llm_evaluate("fact", [])

        self.assertEqual(result["importance"], 5)
        self.assertFalse(result["contradiction"])

    def test_consolidator_reports_failed_deletion(self):
        fake = SimpleNamespace(delete=unittest.mock.Mock(side_effect=RuntimeError("locked")))
        member = MemoryObject("fact", "user", "session-a", 0.5)
        with patch.object(consolidator, "collection", fake):
            self.assertFalse(consolidator.delete_originals([member]))


if __name__ == "__main__":
    unittest.main()