import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from cd_llm.experiments import alarm_bfs_openrouter as study


BIF = """
network unknown {}
variable ROOT { type discrete [ 2 ] { FALSE, TRUE }; }
variable CHILD { type discrete [ 2 ] { FALSE, TRUE }; }
variable LEAF { type discrete [ 2 ] { FALSE, TRUE }; }
probability ( ROOT ) { table 0.7, 0.3; }
probability ( CHILD | ROOT ) { (FALSE) 1, 0; (TRUE) 0, 1; }
probability ( LEAF | ROOT, CHILD ) {
 (FALSE, FALSE) 1, 0; (FALSE, TRUE) 0, 1;
 (TRUE, FALSE) 0, 1; (TRUE, TRUE) 1, 0;
}
"""


class AlarmTests(unittest.TestCase):
    def test_sampler_follows_conditionals_and_seed(self):
        parsed = study.parse_bif(BIF)
        first = study.sample_states(parsed, 2000, 4)
        np.testing.assert_array_equal(first, study.sample_states(parsed, 2000, 4))
        self.assertFalse(np.array_equal(first, study.sample_states(parsed, 2000, 5)))
        np.testing.assert_array_equal(first[:, 0], first[:, 1])
        self.assertTrue((first[:, 2] == 0).all())
        self.assertAlmostEqual(first[:, 0].mean(), .3, delta=.04)

    def test_parser_rejects_missing_and_invalid_conditionals(self):
        for bad in (BIF.replace("(TRUE) 0, 1;", ""),
                    BIF.replace("table 0.7, 0.3", "table 0.7, 0.8"),
                    BIF.replace("(TRUE) 0, 1;", "(TRUE) -1, 2;")):
            with self.assertRaises(ValueError):
                study.parse_bif(bad)

    def test_encoding_preserves_binary_and_ordinal_meanings(self):
        levels = {"A": ["TRUE", "FALSE"], "B": ["HIGH", "LOW", "NORMAL"],
                  "INTUBATION": ["NORMAL", "ESOPHAGEAL", "ONESIDED"]}
        codes = study.encoding(levels)
        self.assertEqual(codes["A"], {"TRUE": 1, "FALSE": 0})
        self.assertEqual(codes["B"], {"HIGH": 2, "LOW": 0, "NORMAL": 1})
        np.testing.assert_array_equal(
            study.encode_states(np.array([[0, 0, 2], [1, 2, 0]]), list(levels), levels, codes),
            [[1, 2, 2], [0, 1, 0]])

    def test_masking_roundtrip_and_correlation_alignment(self):
        names = ["ROOT", "CHILD", "LEAF"]
        values = np.array([[0, 0, 1], [1, 1, 1], [0, 0, 1], [1, 1, 1]])
        spec = study.presentation(names)
        frame = study.transform(values, names, spec)
        inverse = np.argsort(spec["order"])
        np.testing.assert_array_equal(frame.to_numpy()[:, inverse], values)
        labels = spec["mapping"]
        self.assertAlmostEqual(frame.corr().loc[labels["ROOT"], labels["CHILD"]], 1.)
        self.assertTrue(np.isnan(frame.corr().loc[labels["LEAF"], labels["ROOT"]]))
        self.assertEqual(set(frame.columns), {"V01", "V02", "V03"})

    def test_prompt_audit_rejects_identifier_and_benchmark_leaks(self):
        names = ["V01", "V02"]
        good = [{"role": "user", "content": "V01: Pressure in central veins.\nV02: 0.30\nV01 causes V02"}]
        study.audit_messages(good, names, ["CVP", "CO"])
        for text in ("ALARM", "CVP: pressure", "V99 causes V01", "CO causes V02"):
            with self.assertRaises(ValueError):
                study.audit_messages([{"role": "user", "content": text}], names, ["CVP", "CO"])

    def test_metrics_distinguish_direct_edges_and_ancestry(self):
        reference = dict(nodes=3, edges=[[0, 1], [1, 2]])
        scores = study.score_edges([[0, 1], [1, 2], [0, 2]], reference)
        self.assertAlmostEqual(scores["directed_edge_f1"], .8)
        self.assertEqual(scores["ancestry"]["f1"], 1.)
        self.assertEqual(scores["indirect_shortcut_edges"], 1)
        self.assertEqual(study.score_edges(reference["edges"], reference)["graph_distance"], 0.)
        self.assertEqual(study.score_edges([], reference)["edge_count"], 0)

    def test_provider_request_is_pinned_and_reasoning_disabled(self):
        request = study.request_for([{"role": "user", "content": "hello"}], 3)
        self.assertEqual(request["model"], study.MODEL)
        self.assertEqual(request["seed"], 3)
        self.assertFalse(request["extra_body"]["reasoning"]["enabled"])
        self.assertEqual(request["extra_body"]["provider"]["only"], ["deepinfra/bf16"])
        self.assertFalse(request["extra_body"]["provider"]["allow_fallbacks"])

    def test_cache_reuses_response_and_budget_blocks_dispatch(self):
        response = dict(model=study.MODEL, provider="DeepInfra",
                        usage=dict(prompt_tokens=100, completion_tokens=20, cost=.000013),
                        choices=[dict(finish_reason="stop", message=dict(content="<Answer>None</Answer>"))])
        calls = []
        async def fake_send(request):
            calls.append(request)
            return response
        with tempfile.TemporaryDirectory() as folder:
            transport = study.Transport(Path(folder), "unused")
            with patch.object(transport, "send", side_effect=fake_send):
                messages = [{"role": "user", "content": "hello"}]
                self.assertEqual(transport.query(0, 0, messages), response)
                self.assertEqual(transport.query(0, 0, messages), response)
                self.assertEqual(len(calls), 1)
                restarted = study.Transport(Path(folder), "unused")
                self.assertAlmostEqual(restarted.spent, .000013)
                transport.query(0, 0, [{"role": "user", "content": "changed"}])
                self.assertEqual(len(calls), 2)
                transport.spent = .999999
                self.assertEqual(transport.query(0, 1, messages), response)
                with self.assertRaisesRegex(RuntimeError, "ceiling"):
                    transport.query(0, 2, [{"role": "user", "content": "uncached budget test"}])
                self.assertEqual(len(calls), 2)

    def test_truncated_example_answer_cannot_be_parsed(self):
        text = "An example is <Answer>None</Answer>. I am still considering the answer"
        truncated = dict(message=dict(content=text), finish_reason="length")
        transformed = study.completion_text(truncated)
        with self.assertRaises(ValueError):
            study.parse_answer(transformed, ["V01"], drop_unknown=True)
        self.assertIn("I am still considering", transformed)
        finished = dict(message=dict(content="<Answer>V01</Answer>"), finish_reason="stop")
        self.assertEqual(study.parse_answer(study.completion_text(finished), ["V01"]), ["V01"])

    def test_compacted_retry_preserves_the_previous_assessment(self):
        messages = [
            dict(role="system", content="clinical context"),
            dict(role="user", content="root question"),
            dict(role="assistant", content="old reasoning " * 4000),
            dict(role="user", content="Select the variables that are caused by V01."),
            dict(role="assistant", content="V01 influences V02, but the answer was cut short"),
            dict(role="user", content="Your previous response did not follow the required answer format."),
        ]
        compact = study.bfs_messages(messages)
        self.assertEqual([m["role"] for m in compact], ["system", "user", "user", "assistant", "user"])
        self.assertEqual(compact[-2], messages[-2])
        self.assertEqual(compact[-3], messages[-3])
        self.assertIn("no explanation or additional reasoning", compact[-1]["content"])
        self.assertNotIn("old reasoning", str(compact))
        ordinary = messages[:-1]
        self.assertEqual(study.bfs_messages(ordinary), study._compact_bfs_messages(ordinary))

    def test_legacy_cache_is_reused_without_billing_again(self):
        messages = [dict(role="user", content="unchanged")]
        response = dict(model=study.MODEL, provider="DeepInfra", usage=dict(prompt_tokens=10, completion_tokens=2),
                        choices=[dict(message=dict(content="<Answer>None</Answer>"), finish_reason="stop")])
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            study.write_json(root / "requests/r0_q003_a0.json", dict(
                request=study.request_for(messages, 0), response=response, reserved_usd=.01))
            transport = study.Transport(root, "unused")
            before = transport.spent
            with patch.object(transport, "send") as send:
                self.assertEqual(transport.query(0, 9, messages), response)
                send.assert_not_called()
            self.assertEqual(transport.spent, before)
            self.assertTrue((root / "cache_uses/r0_q009.json").exists())

    def test_live_report_distinguishes_running_and_waiting(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            study.write_json(root / "evaluation_only/reference.json", dict(nodes=3, edges=[[0, 1]]))
            study.write_json(root / "status/0.json", dict(state="running"))
            study.write_json(root / "progress/0.json", dict(queries=2, visited=["V01"], diagnostics=[
                dict(finish_reason="length", format_retry=False),
                dict(finish_reason="stop", format_retry=True)]))
            with patch.object(study, "root_for", return_value=root), patch.object(study, "validate"):
                result = study.report("20260909T215919Z", quiet=True)
            self.assertEqual(result["completed"], 0)
            self.assertIsNone(result["summary"]["directed_edge_f1"]["mean"])
            text = (root / "report.md").read_text()
            self.assertIn("| 0 | running | 2 | 1 | 1 | 1 |", text)
            self.assertIn("| 1 | waiting |", text)

    def test_uncertain_request_remains_reserved_and_is_not_repeated(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            messages = [{"role": "user", "content": "hello"}]
            request = study.request_for(messages, 0)
            reservation = study.reserved_cost(request, study.PRICES)
            study.write_json(root / "requests/r0_q000_a0.json",
                             dict(request=request, reserved_usd=reservation, started=0))
            transport = study.Transport(root, "unused")
            with patch.object(transport, "send") as send:
                with self.assertRaises(RuntimeError):
                    transport.query(0, 0, messages)
                send.assert_not_called()
            self.assertEqual(transport.spent, reservation)


if __name__ == "__main__":
    unittest.main()
