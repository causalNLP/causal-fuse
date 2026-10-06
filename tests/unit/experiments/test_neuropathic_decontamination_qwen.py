from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cd_llm.baselines.causal_order2025 import QWEN35_MODEL, QWEN35_REVISION
from cd_llm.baselines.scp2025 import build_decision_prompt
from cd_llm.experiments.neuropathic_decontamination_qwen import (
    CLAIM_FUSE_METHODS,
    CONFIDENCE_STRATEGIES,
    DATA_SEEDS,
    METHODS,
    SAMPLE_SIZES,
    SCP_CONDITIONS,
    SMOKE_SCP_PAIRS,
    STUDY,
    _audit_scp_decision_prompt,
    _audit_supplied_messages,
    _result_path,
    audit_saved_requests,
    build_parser,
    prepare,
    validate_runtime_manifest,
)


class NeuropathicDecontaminationQwenTests(unittest.TestCase):
    def test_locked_method_and_condition_sets(self) -> None:
        self.assertEqual(CONFIDENCE_STRATEGIES, ("stated", "consistency", "grounded"))
        self.assertEqual(len(METHODS), 8)
        self.assertEqual(len(set(METHODS)), 8)
        self.assertEqual(tuple(CLAIM_FUSE_METHODS), CONFIDENCE_STRATEGIES)
        self.assertEqual(len(SCP_CONDITIONS), 20)
        self.assertEqual(SCP_CONDITIONS[0], (DATA_SEEDS[0], SAMPLE_SIZES[0]))
        self.assertEqual(SCP_CONDITIONS[-1], (DATA_SEEDS[-1], SAMPLE_SIZES[-1]))

    def test_smoke_uses_exactly_sixteen_ordered_pairs(self) -> None:
        self.assertEqual(len(SMOKE_SCP_PAIRS), 16)
        self.assertEqual(len(set(SMOKE_SCP_PAIRS)), 16)
        self.assertTrue(all(left != right for left, right in SMOKE_SCP_PAIRS))

    def test_result_paths_separate_confidence_strategies(self) -> None:
        root = Path("/tmp/result")
        paths = {_result_path(root, method, 0, 200) for method in CLAIM_FUSE_METHODS.values()}
        self.assertEqual(len(paths), 3)

    def test_parser_requires_task_index_only_at_runtime(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--stage", "scp", "--run-id", "20260903T000000Z", "--task-index", "19"])
        self.assertEqual(args.task_index, 19)

    def test_prepare_references_locked_inputs_by_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            benchmark = prepare("20260903T000000Z", results_root=Path(temporary))
            manifest = __import__("json").loads((benchmark / "evaluation_manifest.json").read_text())
        self.assertEqual(manifest["study"], STUDY)
        self.assertEqual(manifest["model"], QWEN35_MODEL)
        self.assertEqual(manifest["model_revision"], QWEN35_REVISION)
        self.assertEqual(len(manifest["datasets"]), 20)
        self.assertEqual(len(manifest["source_statistics"]), 20)
        self.assertTrue(all(Path(item["path"]).is_absolute() for item in manifest["datasets"]))

    def test_runtime_manifest_rejects_wrong_revision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint"
            image = root / "image.sif"
            checkpoint.mkdir()
            image.touch()
            manifest = root / "manifest.json"
            manifest.write_text(
                '{"model":"Qwen/Qwen3.5-9B","model_revision":"wrong",'
                f'"checkpoint_path":"{checkpoint}","image_path":"{image}"}}'
            )
            with self.assertRaisesRegex(ValueError, "runtime identity"):
                validate_runtime_manifest(manifest)

    def test_bfs_audit_distinguishes_supplied_text_from_model_history(self) -> None:
        spec = __import__(
            "cd_llm.experiments.neuropathic_decontamination",
            fromlist=["build_spec"],
        ).build_spec()
        _audit_supplied_messages(
            (
                {"role": "user", "content": "This study contains V01."},
                {"role": "assistant", "content": "This benchmark resembles neuropathic pain."},
            ),
            spec,
        )
        with self.assertRaisesRegex(ValueError, "protected benchmark identity"):
            _audit_supplied_messages((
                {"role": "user", "content": "This benchmark contains V01."},
                {"role": "assistant", "content": "This is harmless model history."},
            ), spec)

    def test_scp_audit_allows_generated_protected_terms_only(self) -> None:
        spec = __import__(
            "cd_llm.experiments.neuropathic_decontamination",
            fromlist=["build_spec"],
        ).build_spec()
        safe_knowledge = "This clinical study contains V01 and V02."
        generated = f"This neuropathic pain benchmark resembles {spec.original_names[0]}."
        decision = build_decision_prompt(
            knowledge_prompt=safe_knowledge,
            explanation=generated,
            source="V01",
            target="V02",
        )
        self.assertEqual(_audit_scp_decision_prompt(decision, spec), generated)

        leaked_decision = build_decision_prompt(
            knowledge_prompt=f"This dataset is {spec.original_names[0]}.",
            explanation="Generated domain reasoning.",
            source="V01",
            target="V02",
        )
        with self.assertRaisesRegex(ValueError, "protected benchmark identity"):
            _audit_scp_decision_prompt(leaked_decision, spec)

    def test_saved_scp_audit_tracks_generated_terms_diagnostically(self) -> None:
        spec = __import__(
            "cd_llm.experiments.neuropathic_decontamination",
            fromlist=["build_spec"],
        ).build_spec()
        knowledge = "This clinical study contains V01 and V02."
        generated = "This appears to concern neuropathic pain in a benchmark."
        decision = build_decision_prompt(
            knowledge_prompt=knowledge,
            explanation=generated,
            source="V01",
            target="V02",
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "caches" / "scp" / "seed_00_n200"
            cache.mkdir(parents=True)
            (cache / "explanation.json").write_text(json.dumps({
                "request": {"kind": "explanation", "prompt": knowledge},
                "text": generated,
            }))
            (cache / "decision.json").write_text(json.dumps({
                "request": {"kind": "yes_probability", "prompt": decision},
                "probability": 0.5,
            }))
            audit = audit_saved_requests(root, spec)

        self.assertEqual(audit["status"], "passed")
        self.assertEqual(audit["unique_prompt_count"], 2)
        self.assertIn("neuropathic pain", audit["generated_protected_terms"])
        self.assertIn("benchmark", audit["generated_protected_terms"])


if __name__ == "__main__":
    unittest.main()
