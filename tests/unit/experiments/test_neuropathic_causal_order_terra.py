"""Protocol and accounting regression tests; no external API calls."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cd_llm.experiments import neuropathic_causal_order_terra as study


def manifest():
    return dict(response_models=[study.MODEL], pricing=dict(prompt="0.000002",
        completion="0.000012", input_cache_write="0.0000025", input_cache_read="0.0000002"))


class Client:
    def __init__(self, *, deny=False):
        self.chat = SimpleNamespace(completions=self)
        self.calls = []
        self.deny = deny

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.deny:
            error = RuntimeError("secret provider error")
            error.status_code = 403
            raise error
        await asyncio.sleep(0)
        value = dict(model=study.MODEL,provider="OpenAI",id=f"test-{len(self.calls)}",
            choices=[dict(message=dict(content="<Answer>A</Answer>"),finish_reason="stop")],
            usage=dict(prompt_tokens=20,completion_tokens=5,total_tokens=25,cost=.0001))
        return SimpleNamespace(model_dump=lambda:value)


def test_graph_and_pair_answers_are_separate():
    assert study.graph_answer("<Answer>A</Answer>") is None
    assert study.pair_answer("<Answer>A</Answer>") == "A"
    assert study.pair_answer("<Answer>B</Answer>") == "B"
    assert study.pair_answer("<Answer>C</Answer>") == "C"
    assert study.pair_answer("<Answer>[]</Answer>") is None
    assert study.graph_answer("<Answer>\n[('V01','V02'), 'V03']\n</Answer>")[1] == [("V01","V02"),("V03",)]
    assert study.graph_answer("<Answer>[]</Answer>")[1] == []
    assert study.graph_answer("<Answer>malformed</Answer>") is None
    assert study.graph_answer("<Answer>[]</Answer><Answer>[('V01','V02')]</Answer>")[1] == [("V01","V02")]


def test_parse_gate_and_pilot_coverage():
    assert study.passes_gate(1083,1140)
    assert not study.passes_gate(1082,1140)
    assert study.passes_gate(19,20)
    assert not study.passes_gate(18,20)
    assert not study.passes_gate(0,0)
    pilot = study.pilot_triplets()
    assert len(set(pilot)) == 20
    assert set(n for group in pilot for n in group) == set(study.NAMES)
    assert set(pilot) <= set(study.TRIPLETS)


def test_request_settings_and_cache_identity():
    messages=[dict(role="system",content="system"),dict(role="assistant",content="question")]
    body=study.request_body(messages,4096)
    assert body["messages"][-1]["role"] == "user"
    assert messages[-1]["role"] == "assistant"
    assert body["reasoning"] == {"effort":"none"}
    assert body["provider"] == dict(only=["openai"],allow_fallbacks=False,require_parameters=True)
    assert "temperature" not in body
    assert "logprobs" not in body
    original=study.fingerprint(body)
    for field,value in (("model","other"),("seed",1),("max_tokens",7680),
                         ("provider",dict(only=["azure"])),("reasoning",dict(effort="low"))):
        assert study.fingerprint(dict(body,**{field:value})) != original


def test_provider_and_reasoning_validation():
    value=dict(model=study.MODEL,provider="OpenAI",choices=[dict(message=dict(content="ok"))],
        usage=dict(prompt_tokens=20,completion_tokens=5))
    study.validate_response(value,manifest())
    for wrong in (dict(value,provider="Azure"),dict(value,model="other"),
                  dict(value,usage=dict(prompt_tokens=20,completion_tokens=5,completion_tokens_details=dict(reasoning_tokens=1))),
                  dict(value,usage=dict(prompt_tokens=None,completion_tokens=5))):
        with pytest.raises(ValueError):
            study.validate_response(wrong,manifest())


def test_concurrent_cache_reuse_and_accounting(tmp_path):
    async def check():
        client=Client()
        transport=study.Transport(tmp_path,manifest(),client=client)
        task=([dict(role="user",content="question")],4096,"triplet")
        records=await transport.many([task,task])
        assert len(client.calls)==1
        assert len(records)==2
        assert transport.accounting()["attempts"]==1
        assert transport.accounting()["budget_used_usd"]==pytest.approx(.0001)
        assert client.calls[0]["extra_body"]["reasoning"] == dict(effort="none")
        assert "temperature" not in client.calls[0]
        await transport.close()
        resumed=study.Transport(tmp_path,manifest(),client=Client())
        await resumed.request(*task)
        assert not resumed.client.calls
        assert resumed.accounting()["attempts"]==1
        await resumed.close()
    asyncio.run(check())


def test_reservations_survive_restart_and_enforce_limit(tmp_path):
    async def check():
        transport=study.Transport(tmp_path,manifest(),client=Client())
        body=study.request_body([dict(role="user",content="question")],4096)
        with patch.object(study,"LIMIT",.11):
            transport.reserve(body)
            before=transport.accounting()["budget_used_usd"]
            assert before>0
            await transport.close()
            resumed=study.Transport(tmp_path,manifest(),client=Client())
            assert resumed.accounting()["budget_used_usd"]==before
            resumed.reserve(body)
            with pytest.raises(study.SpendingLimit):
                await resumed.request([dict(role="user",content="new question")],4096,"triplet")
            assert not resumed.client.calls
            await resumed.close()
    asyncio.run(check())


def test_queued_requests_are_not_all_reserved(tmp_path):
    async def check():
        transport=study.Transport(tmp_path,manifest(),client=Client())
        tasks=[([dict(role="user",content=f"question {i}")],100,"triplet") for i in range(50)]
        with patch.object(study,"LIMIT",.02):
            result=await transport.many(tasks)
            assert len(result)==50
            assert transport.accounting()["budget_used_usd"]==pytest.approx(.005)
        await transport.close()
    asyncio.run(check())


def test_failed_request_retains_reservation_without_secret(tmp_path):
    async def check():
        transport=study.Transport(tmp_path,manifest(),client=Client(deny=True))
        with pytest.raises(RuntimeError,match="403") as caught:
            await transport.request([dict(role="user",content="question")],4096,"triplet")
        assert "secret" not in str(caught.value)
        assert transport.accounting()["unresolved_reservations_usd"]>0
        assert transport.accounting()["accounted_cost_usd"]==0
        await transport.close()
    asyncio.run(check())


def test_deadline_blocks_uncached_requests(tmp_path):
    async def check():
        transport=study.Transport(tmp_path,manifest(),client=Client(),deadline=1)
        with pytest.raises(study.CheckpointRequested):
            await transport.request([dict(role="user",content="question")],4096,"triplet")
        assert not transport.client.calls
        assert transport.accounting()["attempts"]==0
        await transport.close()
    asyncio.run(check())


def test_partial_cache_cannot_replay():
    with pytest.raises(ValueError,match="incomplete"):
        study.selected_rows([],dict(triplets=[dict(messages=[dict(role="user",content="q")])]))
