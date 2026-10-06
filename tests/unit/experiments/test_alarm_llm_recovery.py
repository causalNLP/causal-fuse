import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from cd_llm.experiments.alarm_llm_recovery import RecoveryTransport, IdentifierRepair, check_error_envelope
from cd_llm.experiments import alarm_llm_transport as network


class RecoveryTests(unittest.TestCase):
    def test_error_envelope_retries_and_preserves_ledger(self):
        good = dict(model=network.MODEL, provider="Parasail", choices=[dict(
            finish_reason="stop", message=dict(content="ok"))], usage=dict(prompt_tokens=1, completion_tokens=1))
        replies = iter([{"error": {"code": 502, "message": "upstream failed"}}, good])
        with tempfile.TemporaryDirectory() as tmp, patch("time.sleep"):
            client = RecoveryTransport(tmp, "test", sender=lambda _: next(replies))
            result, _ = client.query(dict(model=network.MODEL, messages=[dict(role="user", content="test")], max_tokens=10))
            self.assertEqual(result, good)
            with client.connect() as db:
                states = db.execute("select state,error from requests order by id").fetchall()
            self.assertEqual([x[0] for x in states], ["failed", "complete"])
            self.assertEqual(json.loads(states[0][1])["status"], 502)

    def test_wrong_model_still_rejected_without_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = RecoveryTransport(tmp, "test", sender=lambda _: dict(model="wrong", provider="Parasail", choices=[{}]))
            with self.assertRaises(RuntimeError):
                client.query(dict(model=network.MODEL, messages=[dict(role="user", content="test")], max_tokens=1))
            self.assertEqual(client.summary()["requests"], 1)

    def test_nontransient_error_retains_status(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            check_error_envelope({"error": {"code": 401}})
        self.assertEqual(caught.exception.code, 401)

    def test_identifier_repair_cannot_change_valid_claims(self):
        for final_edges, accepted in [([["V01", "V02"]], True), ([], False)]:
            with self.subTest(accepted=accepted), tempfile.TemporaryDirectory() as tmp:
                class Fake:
                    root = Path(tmp)
                    calls = 0
                    def query(self, request):
                        self.calls += 1
                        edges = final_edges if self.calls == 3 else [["V01", "V02"], ["V01", "V00"]]
                        return dict(choices=[dict(finish_reason="stop", message=dict(content=json.dumps({"edges": edges})))]), f"fake#{self.calls}"
                client = Fake()
                repair = IdentifierRepair(client, ["V01", "V02"])
                req = dict(messages=[dict(role="user", content="guess")])
                if accepted:
                    repair.query(req)
                    self.assertEqual(len(list((Path(tmp)/"identifier_repairs").glob("*.json"))), 1)
                else:
                    with self.assertRaisesRegex(ValueError, "changed valid"):
                        repair.query(req)
                self.assertEqual(client.calls, 3)


if __name__ == "__main__":
    unittest.main()
