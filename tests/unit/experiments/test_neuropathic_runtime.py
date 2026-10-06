from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from cd_llm.experiments.neuropathic_runtime import (
    BDEU_EQUIVALENT_SAMPLE_SIZE,
    CPC_ALPHA,
    NODE_COUNT,
    SAMPLE_SIZE,
    load_binary_dataset,
    run_cpc,
    run_ges,
)
from cd_llm.core.finite_support import PartialGraph


class FakeGraph:
    def get_graph_edges(self):
        return []


class NeuropathicRuntimeTests(unittest.TestCase):
    def test_load_binary_dataset_preserves_raw_zero_one_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "dataset.csv"
            metadata_path = root / "dataset.json"
            values = np.zeros((SAMPLE_SIZE, NODE_COUNT), dtype=int)
            values[0, 0] = 1
            pd.DataFrame(values, columns=[str(index) for index in range(NODE_COUNT)]).to_csv(
                data_path
            )
            metadata_path.write_text(
                json.dumps(
                    {
                        "sample_size": SAMPLE_SIZE,
                        "level": 0,
                        "seed": 0,
                        "observed_indices": list(range(NODE_COUNT)),
                        "hidden_indices": [],
                    }
                ),
                encoding="utf-8",
            )

            data, names, summary = load_binary_dataset(data_path, metadata_path)

        self.assertEqual(data.dtype, np.int64)
        self.assertEqual(data.shape, (SAMPLE_SIZE, NODE_COUNT))
        self.assertEqual(data[0, 0], 1)
        self.assertEqual(set(np.unique(data)), {0, 1})
        self.assertEqual(len(names), NODE_COUNT)
        self.assertEqual(summary.level, 0)
        self.assertEqual(summary.seed, 0)

    def test_cpc_uses_g_square_stable_fas_and_unmodified_binary_data(self) -> None:
        data = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
        observed: dict[str, object] = {}

        def fake_cpc(received, settings, node_names):
            observed["data"] = received.copy()
            observed["settings"] = settings
            observed["node_names"] = node_names
            return PartialGraph.from_matrix(np.zeros((2, 2), dtype=int), "cpdag")

        graph, settings = run_cpc(data, ["X", "Y"], runner=fake_cpc)

        np.testing.assert_array_equal(observed["data"], data)
        backend_settings = observed["settings"]
        self.assertTrue(backend_settings.stable_fas)
        self.assertEqual(backend_settings.alpha, CPC_ALPHA)
        self.assertEqual(backend_settings.independence_test, "g_square")
        self.assertEqual(backend_settings.variable_type, "discrete")
        self.assertEqual(observed["node_names"], ("X", "Y"))
        self.assertIsInstance(graph, PartialGraph)
        self.assertEqual(settings["algorithm"], "tetrad_cpc")
        self.assertTrue(settings["diagnostics"]["certified"])
        self.assertFalse(settings["standardized"])

    def test_ges_uses_bdeu_with_default_equivalent_sample_size(self) -> None:
        data = np.asarray([[0, 1], [1, 0]], dtype=np.int64)
        observed: dict[str, object] = {}

        def fake_ges(received, **kwargs):
            observed["data"] = received.copy()
            observed.update(kwargs)
            return {"G": FakeGraph(), "score": 3.5}

        graph, settings = run_ges(data, ["X", "Y"], runner=fake_ges)

        np.testing.assert_array_equal(observed["data"], data)
        self.assertEqual(observed["score_func"], "local_score_BDeu")
        self.assertIsNone(observed["maxP"])
        self.assertEqual(settings["score_backend"], "reference")
        self.assertIsNone(settings["score_diagnostics"])
        self.assertEqual(settings["equivalent_sample_size"], BDEU_EQUIVALENT_SAMPLE_SIZE)
        self.assertFalse(settings["standardized"])
        self.assertIsInstance(graph, FakeGraph)

    def test_prefactored_ges_matches_reference_backend(self) -> None:
        data = np.asarray(
            [
                [0, 0, 0],
                [0, 1, 1],
                [1, 0, 1],
                [1, 1, 0],
                [1, 0, 1],
                [0, 1, 0],
            ],
            dtype=np.int64,
        )
        names = ["X", "Y", "Z"]

        reference_graph, reference = run_ges(data, names)
        optimized_graph, optimized = run_ges(
            data, names, score_backend="prefactored"
        )

        np.testing.assert_array_equal(reference_graph.graph, optimized_graph.graph)
        self.assertAlmostEqual(reference["final_score"], optimized["final_score"])
        self.assertEqual(optimized["score_backend"], "prefactored")
        self.assertGreater(optimized["score_diagnostics"]["misses"], 0)


if __name__ == "__main__":
    unittest.main()
