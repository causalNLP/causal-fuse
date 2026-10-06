from __future__ import annotations

import math
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from cd_llm.core.finite_support import graph_edges
from cd_llm.experiments.neuropathic_causal_order_subnetwork import (
    MAX_BENCHMARK_RUNTIME_SECONDS,
    METHODS,
    SUBNETWORK_EDGE_COUNT,
    SUBNETWORK_LAYER_COUNTS,
    SUBNETWORK_NODE_COUNT,
    SUBNETWORK_ORIGINAL_INDICES,
    TRIPLET_COUNT,
    _cache_purpose_summary,
    _protocol,
    _retry_io,
    build_subnetwork_spec,
    slice_for_task,
)


class NeuropathicCausalOrderSubnetworkTests(unittest.TestCase):
    def test_subnetwork_is_the_locked_ancestor_closure(self) -> None:
        spec = build_subnetwork_spec()
        self.assertEqual(spec.node_count, SUBNETWORK_NODE_COUNT)
        self.assertEqual(spec.edge_count, SUBNETWORK_EDGE_COUNT)
        self.assertEqual(spec.layer_counts, SUBNETWORK_LAYER_COUNTS)
        self.assertEqual(spec.original_indices, SUBNETWORK_ORIGINAL_INDICES)
        self.assertEqual(spec.node_names[-1], "L Nackbesvär")
        self.assertEqual(len(graph_edges(spec.target)["directed"]), 30)

    def test_protocol_is_exhaustive_nonthinking_qwen(self) -> None:
        protocol = _protocol({"run_id": "test", "model_revision": "test"})
        causal_order = protocol["causal_order"]
        self.assertEqual(TRIPLET_COUNT, math.comb(20, 3))
        self.assertEqual(causal_order["triplet_count"], 1140)
        self.assertTrue(causal_order["shared_across_data_seeds_and_sample_sizes"])
        self.assertFalse(causal_order["thinking"])
        self.assertEqual(causal_order["temperature"], 0.0)
        self.assertFalse(causal_order["repair"])
        self.assertEqual(causal_order["triplet_parse_retry_max_tokens"], 7680)
        self.assertFalse(causal_order["semantic_retry"])
        self.assertEqual(MAX_BENCHMARK_RUNTIME_SECONDS, 3600.0)
        self.assertEqual(METHODS, ("PC", "GES", "Causal Order+PC (Qwen3.5-9B)"))

    def test_task_mapping_covers_twenty_seeds_and_two_sizes(self) -> None:
        self.assertEqual(slice_for_task(0), (0, 200))
        self.assertEqual(slice_for_task(1), (0, 1000))
        self.assertEqual(slice_for_task(38), (19, 200))
        self.assertEqual(slice_for_task(39), (19, 1000))
        with self.assertRaises(ValueError):
            slice_for_task(40)

    def test_cache_summary_separates_request_purposes(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "cache.sqlite3"
            connection = sqlite3.connect(path)
            connection.execute(
                "CREATE TABLE responses (purpose TEXT, prompt_tokens INTEGER, "
                "completion_tokens INTEGER, attempts INTEGER, latency_seconds REAL)"
            )
            connection.executemany(
                "INSERT INTO responses VALUES (?, ?, ?, ?, ?)",
                (("triplet", 10, 20, 1, 0.5), ("triplet", 11, 21, 1, 0.6), ("vote_tiebreaker", 30, 4, 2, 1.0)),
            )
            connection.commit()
            connection.close()
            summary = _cache_purpose_summary(path)
        self.assertEqual(summary[0]["purpose"], "triplet")
        self.assertEqual(summary[0]["request_count"], 2)
        self.assertEqual(summary[0]["prompt_tokens"], 21)
        self.assertEqual(summary[0]["completion_tokens"], 41)
        self.assertEqual(summary[1]["purpose"], "vote_tiebreaker")
        self.assertEqual(summary[1]["attempts"], 2)

    def test_retry_io_recovers_from_transient_filesystem_error(self) -> None:
        operation = Mock(side_effect=(OSError(5, "Input/output error"), "ok"))
        with patch("cd_llm.experiments.neuropathic_causal_order_subnetwork.time.sleep") as sleep:
            value, attempts = _retry_io(operation)
        self.assertEqual(value, "ok")
        self.assertEqual(attempts, 2)
        sleep.assert_called_once_with(1.0)


if __name__ == "__main__":
    unittest.main()
