from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd

from cd_llm.baselines.causal_order2025.openrouter_backend import ResponseCache
from cd_llm.core.finite_support import graph_edges
from cd_llm.experiments.neuropathic_decontamination import (
    BFS_METHOD,
    CLAIM_FUSE_METHODS,
    CLAIM_FUSE_METHOD,
    CONFIDENCE_STRATEGIES,
    DATA_SEEDS,
    GPC_METHOD,
    METHODS,
    MODEL,
    OPAQUE_NAMES,
    SAMPLE_SIZES,
    CachedTerraCompleter,
    _guess_skeleton,
    _run_bfs,
    audit_model_text,
    build_spec,
    prepare,
)


class NeuropathicDecontaminationTests(unittest.TestCase):
    def test_locked_terra_method_set_excludes_logprob_and_triplet_baselines(self) -> None:
        self.assertEqual(MODEL, "openai/gpt-5.6-terra")
        self.assertEqual(
            METHODS,
            (
                GPC_METHOD,
                BFS_METHOD,
                CLAIM_FUSE_METHOD,
                CLAIM_FUSE_METHODS["consistency"],
                CLAIM_FUSE_METHODS["grounded"],
            ),
        )
        self.assertEqual(CONFIDENCE_STRATEGIES, ("stated", "consistency", "grounded"))
        self.assertFalse(any("SCP" in method for method in METHODS))
        self.assertFalse(any("Long" in method for method in METHODS))
        self.assertFalse(any("Causal Order" in method for method in METHODS))

    def test_opaque_mapping_is_deterministic_and_complete(self) -> None:
        first = build_spec()
        second = build_spec()
        self.assertEqual(first, second)
        self.assertEqual(first.names, OPAQUE_NAMES)
        self.assertEqual(set(first.permutation), set(range(20)))
        self.assertEqual(set(first.descriptions), set(OPAQUE_NAMES))
        self.assertEqual(len(graph_edges(first.target)["directed"]), 30)

    def test_prompt_audit_rejects_source_identity(self) -> None:
        spec = build_spec()
        audit_model_text("V01 is a clinical finding.", spec)
        with self.assertRaisesRegex(ValueError, "protected benchmark identity"):
            audit_model_text(spec.original_names[0], spec)
        with self.assertRaisesRegex(ValueError, "protected benchmark identity"):
            audit_model_text("This concerns neuropathic pain.", spec)

    def test_prepare_writes_nested_opaque_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            results_root = Path(temporary)
            benchmark = prepare("20260902T000000Z", results_root=results_root)
            for seed in DATA_SEEDS:
                small = pd.read_csv(benchmark / "data" / f"seed_{seed:02d}_n200.csv")
                large = pd.read_csv(benchmark / "data" / f"seed_{seed:02d}_n1000.csv")
                self.assertEqual(tuple(small.columns), OPAQUE_NAMES)
                self.assertEqual(tuple(large.columns), OPAQUE_NAMES)
                self.assertEqual(len(small), SAMPLE_SIZES[0])
                self.assertEqual(len(large), SAMPLE_SIZES[1])
                self.assertTrue(small.equals(large.iloc[:200].reset_index(drop=True)))

    def test_terra_completer_persists_exact_responses(self) -> None:
        response = _FakeResponse('{"edges": [["V01", "V02"]]}')
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=_FakeCompletions(response))
        )
        with tempfile.TemporaryDirectory() as temporary:
            cache = ResponseCache(Path(temporary) / "requests.sqlite3")
            try:
                completer = CachedTerraCompleter(cache, client=client)
                kwargs = {
                    "purpose": "gpc_guess:seed_00",
                    "seed": 0,
                    "max_completion_tokens": 256,
                }
                first = completer.complete([{"role": "user", "content": "prompt"}], **kwargs)
                second = completer.complete([{"role": "user", "content": "prompt"}], **kwargs)
            finally:
                cache.close()
        self.assertFalse(first["cached"])
        self.assertTrue(second["cached"])
        self.assertEqual(client.chat.completions.calls, 1)

    def test_terra_completer_checks_cost_before_call(self) -> None:
        response = _FakeResponse("unused")
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=_FakeCompletions(response))
        )
        with tempfile.TemporaryDirectory() as temporary:
            cache = ResponseCache(Path(temporary) / "requests.sqlite3")
            try:
                completer = CachedTerraCompleter(cache, client=client, cost_cap_usd=1e-8)
                with self.assertRaisesRegex(RuntimeError, "safety cap"):
                    completer.complete(
                        [{"role": "user", "content": "prompt"}],
                        purpose="bfs_pearson:seed_00:n200",
                        seed=0,
                        max_completion_tokens=256,
                    )
            finally:
                cache.close()
        self.assertEqual(client.chat.completions.calls, 0)

    def test_bfs_runner_requires_opaque_identifiers(self) -> None:
        spec = build_spec()
        frame = pd.DataFrame({name: [0, 1] for name in OPAQUE_NAMES})
        usage = {"query_count": 0, "input_tokens": 0, "output_tokens": 0}
        with tempfile.TemporaryDirectory() as temporary:
            with patch(
                "cd_llm.experiments.neuropathic_decontamination.run_bfs",
                return_value=([], usage),
            ) as mocked_run_bfs:
                _run_bfs(
                    Path(temporary),
                    frame,
                    spec,
                    seed=0,
                    sample_size=200,
                    router=SimpleNamespace(),
                )
        self.assertTrue(mocked_run_bfs.call_args.kwargs["opaque_identifiers"])

    def test_gpc_prompt_uses_opaque_metadata_without_extra_interface_flags(self) -> None:
        spec = build_spec()
        router = SimpleNamespace(
            complete=MagicMock(
                return_value={
                    "content": '{"edges": [["V01", "V02"]]}',
                    "request_key": "request-key",
                    "cached": False,
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                }
            )
        )
        skeleton, request = _guess_skeleton(router, spec, seed=0)
        messages = router.complete.call_args.args[0]
        self.assertEqual(skeleton[0, 1], 1)
        self.assertEqual(request["request_key"], "request-key")
        self.assertIn("V01:", messages[1]["content"])
        self.assertNotIn("neuropathic pain", messages[1]["content"].casefold())


class _FakeCompletions:
    def __init__(self, response) -> None:
        self.response = response
        self.calls = 0

    def create(self, **_kwargs):
        self.calls += 1
        return self.response


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.id = "request-1"
        self.model = MODEL
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]
        self._payload = {
            "id": self.id,
            "model": self.model,
            "choices": [{"message": {"content": content}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "cost": 0.00008,
            },
        }

    def model_dump(self):
        return self._payload


if __name__ == "__main__":
    unittest.main()
