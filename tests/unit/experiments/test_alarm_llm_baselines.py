import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from concurrent.futures import ThreadPoolExecutor

from cd_llm.experiments import alarm_llm_transport as transport
from cd_llm.experiments.alarm_llm_baselines import METHODS, TASKS, select_distribution
from cd_llm.experiments.alarm_llm_repair import GPCRepair


def request(seed=0):
    return dict(model=transport.MODEL, messages=[dict(role="user", content="Choose A or B")],
                seed=seed, temperature=0, max_tokens=1, logprobs=True, top_logprobs=20)


def response():
    return dict(model=transport.MODEL, provider="Parasail",
        choices=[dict(message=dict(content="A"), finish_reason="length")],
        usage=dict(prompt_tokens=10, completion_tokens=1, cost=.00000125))


class TransportTests(unittest.TestCase):
    def test_provider_pinned_and_native_settings_retained(self):
        supplied = request()
        supplied["provider"] = {"only": ["wrong"]}
        supplied["chat_template_kwargs"] = {"enable_thinking": True}
        effective = transport.effective_request(supplied)
        self.assertEqual(effective["provider"]["only"], ["parasail/bf16"])
        self.assertFalse(effective["provider"]["allow_fallbacks"])
        self.assertFalse(effective["reasoning"]["enabled"])
        self.assertEqual(effective["top_logprobs"],20)
        self.assertEqual(effective["max_tokens"],1)

    def test_cache_deduplicates_parallel_calls_but_preserves_seeds(self):
        calls=[]
        with tempfile.TemporaryDirectory() as tmp:
            client=transport.Transport(tmp,"test",sender=lambda r:(calls.append(r) or response()))
            with ThreadPoolExecutor(4) as pool:
                records=list(pool.map(lambda _:client.query(request()),range(4)))
            self.assertEqual(len(calls),1)
            self.assertEqual(len({r[1] for r in records}),1)
            client.query(request(1))
            self.assertEqual(len(calls),2)
            self.assertEqual(client.summary()["cached_uses"]["test"],3)

    def test_cap_reserves_before_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            client=transport.Transport(tmp,"test",sender=lambda _:self.fail("must not dispatch"))
            with patch.object(transport,"CAP",0):
                with self.assertRaisesRegex(RuntimeError,"ceiling"):
                    client.query(request())
            self.assertEqual(client.summary()["requests"],0)

    def test_unknown_completion_is_not_redispatched(self):
        with tempfile.TemporaryDirectory() as tmp:
            client=transport.Transport(tmp,"test",sender=lambda _:self.fail("must not dispatch"))
            req=transport.effective_request(request())
            with client.connect() as db:
                db.execute("INSERT INTO requests(key,method,request,state,reserved,charged,started) VALUES(?,?,?,'dispatched',1,1,0)",
                    (transport.digest(req),"test",json.dumps(req)))
            with self.assertRaisesRegex(RuntimeError,"uncertain"):
                client.query(request())
            self.assertEqual(client.summary()["accounted_usd"],1)

    def test_wrong_provider_is_audited_and_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            client=transport.Transport(tmp,"test",sender=lambda _:dict(response(),provider="DeepInfra"))
            with self.assertRaises(RuntimeError):
                client.query(request())
            with client.connect() as db:
                state,raw=db.execute("SELECT state,response FROM requests").fetchone()
            self.assertEqual(state,"invalid")
            self.assertEqual(json.loads(raw)["provider"],"DeepInfra")

    def test_long_one_token_completion_is_valid_transport(self):
        self.assertEqual(transport.checked_response(response())["choices"][0]["message"]["content"],"A")

    def test_scp_not_scheduled(self):
        self.assertNotIn("scp",METHODS)
        self.assertEqual(len(TASKS),20)
        self.assertEqual({s for m,s in TASKS},set(range(5)))

    def test_map_tie_uses_canonical_labels(self):
        records=[dict(edges=[[0,1]],weight=.5),dict(edges=[[1,0]],weight=.5)]
        self.assertEqual(select_distribution(records,[1,0])["edges"],[[1,0]])
        self.assertEqual(select_distribution(records,[0,1])["edges"],[[0,1]])

    def test_gpc_repair_preserves_assessment_and_is_bounded(self):
        class Fake:
            def __init__(self, valid):
                self.calls=[]
                self.valid=valid
            def query(self,req):
                self.calls.append(req)
                text='{"edges": [["V01", "V00"]]}' if len(self.calls)==1 or not self.valid else '{"edges": []}'
                return dict(choices=[dict(finish_reason="stop",message=dict(content=text))]),"fake"
        fake=Fake(True)
        GPCRepair(fake,["V01","V02"]).query(request())
        self.assertEqual(len(fake.calls),2)
        self.assertEqual(fake.calls[1]["messages"][-2]["role"],"assistant")
        self.assertIn("Do not add replacement edges",fake.calls[1]["messages"][-1]["content"])
        bad=Fake(False)
        with self.assertRaises(ValueError):
            GPCRepair(bad,["V01","V02"]).query(request())
        self.assertEqual(len(bad.calls),2)


if __name__ == "__main__":
    unittest.main()
