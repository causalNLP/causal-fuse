from __future__ import annotations

import json
from itertools import combinations
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from cd_llm.baselines.guess2graph2026 import (
    CachedOpenAISkeletonExpert,
    DAGDSeparationOracle,
    LLMExpertSettings,
    edges_to_dag,
    expert_edge_order,
    parse_dag_response,
    parse_skeleton_response,
    run_gpc_guess_dag,
    run_gpc_guess_skeleton,
    run_pc_guess_skeleton,
)
from cd_llm.experiments.neuropathic_guess2graph import (
    ceg_fuse_skeleton_metrics,
    random_dag,
    target_ceg_graph_from_dag,
)


class Guess2GraphTests(unittest.TestCase):
    def test_explicit_rng_is_deterministic_and_prioritizes_non_edges(self) -> None:
        guess = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        edges = [(0, 1), (0, 2), (1, 2)]
        first = expert_edge_order(edges, guess, np.random.default_rng(3))
        second = expert_edge_order(edges, guess, np.random.default_rng(3))
        self.assertEqual(first, second)
        self.assertEqual(set(first[:2]), {(0, 2), (1, 2)})
        self.assertEqual(first[-1], (0, 1))

    def test_guided_methods_return_valid_skeletons(self) -> None:
        generator = np.random.default_rng(4)
        left = generator.binomial(1, 0.5, size=250)
        middle = np.bitwise_xor(left, generator.binomial(1, 0.1, size=250))
        right = np.bitwise_xor(middle, generator.binomial(1, 0.1, size=250))
        data = np.column_stack([left, middle, right])
        guess = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        for method in (run_pc_guess_skeleton, run_gpc_guess_skeleton):
            result = method(
                data,
                guess,
                alpha=0.05,
                ci_test="chisq",
                rng=np.random.default_rng(5),
            )
            self.assertEqual(result.shape, (3, 3))
            self.assertTrue(np.array_equal(result, result.T))
            self.assertFalse(np.diag(result).any())

    def test_llm_expert_uses_generic_metadata_and_cache(self) -> None:
        client = _FakeClient({"edges": [["V01", "V02"]]})
        with tempfile.TemporaryDirectory() as directory:
            expert = CachedOpenAISkeletonExpert(
                cache_dir=Path(directory),
                settings=LLMExpertSettings(model="gpt-5.6-terra"),
                client=client,
                domain_context="a clinical system",
                descriptions={"V01": "first measurement", "V02": "second measurement"},
            )
            first = expert.guess(["V01", "V02"], trial_seed=7)
            second = expert.guess(["V01", "V02"], trial_seed=7)
        self.assertEqual(client.calls, 1)
        self.assertEqual(first.edges, second.edges)
        self.assertIn("V01: first measurement", first.prompt)
        self.assertNotIn("neuropathic pain", first.prompt)

    def test_current_point_metrics_score_unoriented_output(self) -> None:
        prediction = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
        target = target_ceg_graph_from_dag(
            [("A", "B"), ("B", "C")], [0, 1, 2], ["A", "B", "C"]
        )
        metrics = ceg_fuse_skeleton_metrics(prediction, target)
        self.assertEqual(metrics["structural_hamming_distance"], 5)
        self.assertAlmostEqual(metrics["expected_distance"], 2.5)
        self.assertEqual(metrics["adjacency_f1"], 0.5)
        self.assertEqual(metrics["orientation_f1"], 0.0)

    def test_random_control_is_reproducible_and_acyclic(self) -> None:
        first = random_dag(11, 0.2, 17)
        second = random_dag(11, 0.2, 17)
        self.assertTrue(np.array_equal(first, second))

    def test_parser_rejects_unknown_nodes(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown variable"):
            parse_skeleton_response({"edges": [["V01", "missing"]]}, ["V01", "V02"])

    def test_dag_prompt_explicitly_prevents_reciprocal_edges_and_cycles(self) -> None:
        from cd_llm.baselines.guess2graph2026 import build_dag_prompt
        prompt = build_dag_prompt(["V01", "V02"])
        self.assertIn("Never include both [A, B] and [B, A]", prompt)
        self.assertIn("every remaining directed cycle", prompt)
        self.assertIn("must not be included in the response", prompt)

    def test_dag_parser_preserves_direction_and_rejects_cycles(self) -> None:
        nodes = ["V01", "V02", "V03"]
        edges = parse_dag_response(
            {"directed_edges": [["V01", "V02"], ["V02", "V03"]]}, nodes)
        self.assertEqual(edges, (("V01", "V02"), ("V02", "V03")))
        dag = edges_to_dag(nodes, edges)
        self.assertEqual(dag[0, 1], 1)
        self.assertEqual(dag[1, 0], 0)
        with self.assertRaisesRegex(ValueError, "cycle"):
            parse_dag_response(
                {"directed_edges": [["V01", "V02"], ["V02", "V01"]]}, nodes)

    def test_dag_d_separation_chain_fork_and_collider(self) -> None:
        chain = DAGDSeparationOracle(np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]]))
        self.assertFalse(chain(0, 2, ()))
        self.assertTrue(chain(0, 2, (1,)))
        fork = DAGDSeparationOracle(np.array([[0, 0, 0], [1, 0, 1], [0, 0, 0]]))
        self.assertFalse(fork(0, 2, ()))
        self.assertTrue(fork(0, 2, (1,)))
        collider = DAGDSeparationOracle(np.array([[0, 1, 0], [0, 0, 0], [0, 1, 0]]))
        self.assertTrue(collider(0, 2, ()))
        self.assertFalse(collider(0, 2, (1,)))

    def test_dag_d_separation_matches_networkx(self) -> None:
        import networkx as nx
        from networkx.algorithms.d_separation import is_d_separator
        generator = np.random.default_rng(12)
        for _ in range(8):
            dag = np.triu((generator.random((5, 5)) < .3).astype(int), 1)
            oracle = DAGDSeparationOracle(dag)
            graph = nx.DiGraph()
            graph.add_nodes_from(range(5))
            graph.add_edges_from(zip(*np.nonzero(dag)))
            for left, right in combinations(range(5), 2):
                remaining = [node for node in range(5) if node not in (left, right)]
                for size in range(len(remaining) + 1):
                    for subset in combinations(remaining, size):
                        self.assertEqual(
                            oracle(left, right, subset),
                            is_d_separator(graph, {left}, {right}, set(subset)),
                        )

    def test_dag_guided_gpc_returns_undirected_skeleton(self) -> None:
        generator = np.random.default_rng(9)
        left = generator.binomial(1, .5, 400)
        middle = np.bitwise_xor(left, generator.binomial(1, .1, 400))
        right = np.bitwise_xor(middle, generator.binomial(1, .1, 400))
        data = np.column_stack([left, middle, right])
        dag = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
        result = run_gpc_guess_dag(data, dag, ci_test="chisq", rng=np.random.default_rng(3))
        self.assertTrue(np.array_equal(result, result.T))
        self.assertFalse(np.diag(result).any())


class _FakeClient:
    def __init__(self, payload) -> None:
        self.payload = payload
        self.calls = 0
        self.chat = SimpleNamespace(completions=self)

    def create(self, **_kwargs):
        self.calls += 1
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(self.payload)))]
        )


if __name__ == "__main__":
    unittest.main()
