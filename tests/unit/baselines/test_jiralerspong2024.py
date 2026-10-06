from __future__ import annotations

import unittest

import pandas as pd

from cd_llm.baselines.jiralerspong2024 import (
    Completion,
    creates_cycle,
    parse_answer,
    run_bfs,
)


class BFS2024Tests(unittest.TestCase):
    def test_parse_answer_accepts_empty_and_deduplicates(self) -> None:
        names = ["A", "B"]
        self.assertEqual(parse_answer("reason <Answer>None</Answer>", names), [])
        self.assertEqual(
            parse_answer("<Answer>A, B, A</Answer>", names), ["A", "B"]
        )

    def test_parse_answer_can_apply_published_unknown_drop_policy(self) -> None:
        with self.assertWarns(RuntimeWarning):
            parsed = parse_answer(
                "<Answer>A, invented</Answer>", ["A", "B"], drop_unknown=True
            )
        self.assertEqual(parsed, ["A"])

    def test_bfs_returns_acyclic_edges_and_usage(self) -> None:
        frame = pd.DataFrame({"A": [0, 1], "B": [0, 1], "C": [1, 0]})
        answers = iter(
            (
                "<Answer>A</Answer>",
                "<Answer>B</Answer>",
                "<Answer>C</Answer>",
                "<Answer>None</Answer>",
            )
        )

        def complete(_messages):
            return Completion(next(answers), 10, 2)

        edges, usage = run_bfs(
            frame,
            complete,
            include_statistics=True,
            descriptions={"A": "a", "B": "b", "C": "c"},
            domain_context="a generic clinical system",
        )

        self.assertEqual(edges, [(0, 1), (1, 2)])
        self.assertEqual(
            usage, {"query_count": 4, "input_tokens": 40, "output_tokens": 8}
        )

    def test_bfs_retries_one_missing_answer_block(self) -> None:
        frame = pd.DataFrame({"A": [0, 1]})
        answers = iter(("A", "<Answer>A</Answer>", "<Answer>None</Answer>"))

        def complete(_messages):
            return Completion(next(answers), 5, 1)

        edges, usage = run_bfs(
            frame,
            complete,
            include_statistics=False,
            descriptions={"A": "variable A"},
        )
        self.assertEqual(edges, [])
        self.assertEqual(
            usage, {"query_count": 3, "input_tokens": 15, "output_tokens": 3}
        )

    def test_empty_root_policy_can_score_native_failure_as_empty_graph(self) -> None:
        frame = pd.DataFrame({"A": [0, 1], "B": [1, 0]})

        def complete(_messages):
            return Completion("<Answer>None</Answer>", 5, 1)

        with self.assertRaisesRegex(ValueError, "no root"):
            run_bfs(frame, complete, include_statistics=False)
        edges, usage = run_bfs(
            frame,
            complete,
            include_statistics=False,
            empty_root_policy="empty_graph",
        )
        self.assertEqual(edges, [])
        self.assertEqual(usage["empty_root_response"], 1)

    def test_bfs_retries_description_when_opaque_identifier_is_required(self) -> None:
        frame = pd.DataFrame({"V01": [0, 1], "V02": [1, 0]})
        answers = iter(
            (
                "<Answer>first measurement</Answer>",
                "<Answer>V01</Answer>",
                "<Answer>None</Answer>",
            )
        )
        observed_messages = []

        def complete(messages):
            observed_messages.append(tuple(message["content"] for message in messages))
            return Completion(next(answers), 5, 1)

        edges, usage = run_bfs(
            frame,
            complete,
            include_statistics=False,
            descriptions={"V01": "first measurement", "V02": "second measurement"},
            opaque_identifiers=True,
        )
        self.assertEqual(edges, [])
        self.assertEqual(usage["query_count"], 3)
        self.assertIn("only the opaque identifiers", observed_messages[0][-1])
        self.assertIn("only exact identifiers", observed_messages[1][-1])

    def test_cycle_check_rejects_back_edge_but_allows_cross_edge(self) -> None:
        edges = [(0, 1), (1, 2)]
        self.assertTrue(creates_cycle(edges, 2, 0, 3))
        self.assertFalse(creates_cycle(edges, 0, 2, 3))


if __name__ == "__main__":
    unittest.main()
