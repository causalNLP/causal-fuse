from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cd_llm.experiments.neuropathic_active_smc_qwen import (
    ACTIVE_BUDGET,
    BASELINE_RUN_ID,
    COMPLETION_LIMIT,
    METHOD_LABELS,
    QUERY_POOL_SIZE,
    SELECTION_CONFIDENCE,
    SMC_CONFIG,
    STRATEGIES,
    prepare_document_corpus,
    slice_for_task,
)
from cd_llm.methods.llm_claim_confidence import DOCUMENT_URLS, load_document_corpus


class NeuropathicActiveSMCQwenTests(unittest.TestCase):
    def test_protocol_constants_lock_ges_only_active_configuration(self) -> None:
        self.assertEqual(BASELINE_RUN_ID, "20260825T082523Z")
        self.assertEqual(COMPLETION_LIMIT, 200)
        self.assertEqual(SMC_CONFIG.particle_count, 384)
        self.assertEqual(SMC_CONFIG.initialization_updates, 192)
        self.assertEqual(ACTIVE_BUDGET, 20)
        self.assertEqual(SELECTION_CONFIDENCE, 0.9)
        self.assertEqual(QUERY_POOL_SIZE, 570)
        self.assertEqual(STRATEGIES, ("stated", "consistency", "grounded"))
        self.assertEqual(set(METHOD_LABELS), set(STRATEGIES))

    def test_task_mapping_covers_twenty_seeds_and_two_sample_sizes(self) -> None:
        self.assertEqual(slice_for_task(0), (0, 200))
        self.assertEqual(slice_for_task(1), (0, 1_000))
        self.assertEqual(slice_for_task(39), (19, 1_000))
        with self.assertRaisesRegex(ValueError, r"\[0, 39\]"):
            slice_for_task(40)

    def test_document_preparation_uses_only_the_four_whitelisted_reviews(self) -> None:
        html = b"<html><head><title>Clinical review</title></head><body><p>This is a sufficiently long clinical review passage about cervical radiculopathy and neuropathic pain mechanisms for deterministic testing.</p></body></html>"
        requested: list[str] = []

        def fetcher(url: str) -> bytes:
            requested.append(url)
            return html

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "corpus.json"
            prepare_document_corpus(output, fetcher=fetcher)
            sources = load_document_corpus(output)
        self.assertEqual(tuple(requested), DOCUMENT_URLS)
        self.assertEqual(tuple(source.url for source in sources), DOCUMENT_URLS)
        self.assertTrue(all("target" not in source.title.lower() for source in sources))


if __name__ == "__main__":
    unittest.main()
