"""Terra-specific transport, replay, and reporting checks without network calls."""
import asyncio
from dataclasses import asdict
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from cd_llm.experiments import neuropathic_terra_smc as study
from cd_llm.experiments import neuropathic_ancestry_prompt_ablation as parser
from cd_llm.experiments.neuropathic_grid_interpretation import inference_arguments
from cd_llm.methods.active_query import QueryResponse


def manifest():
    return dict(settings=study.settings(), response_models=[study.MODEL, study.MODEL+"-20260709"],
        pricing=dict(prompt="0.000002", completion="0.000012", input_cache_write="0.0000025", input_cache_read="0.0000002"))


class Client:
    def __init__(self, fail=False):
        self.chat = SimpleNamespace(completions=self)
        self.calls, self.fail = [], fail

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            error = RuntimeError("private provider body")
            error.status_code = 403
            raise error
        await asyncio.sleep(.01)
        value = dict(model=study.MODEL, provider="OpenAI", id="fake",
            choices=[dict(message=dict(content='{"answer":"yes","confidence":0.99,"justification":"test"}'))],
            usage=dict(prompt_tokens=20,completion_tokens=10,total_tokens=30,cost=.00016))
        return SimpleNamespace(model_dump=lambda:value)


def test_grid_and_inference_seeds_match_qwen():
    pairs = list(zip(study.tasks(),study.original.tasks(),strict=True))
    assert len(pairs)==120
    assert len(study.queries())==950
    assert len(study.pilot_queries())==6
    for task,reference in pairs:
        assert task["id"].replace("terra_","qwen_",1)==reference["id"]
        assert inference_arguments(task,[],[],"literal")==inference_arguments(reference,[],[],"literal")
    assert study.BUDGETS==(10,20,50,100)


def test_cache_isolation_and_generation_settings():
    body=study.terra.request_body([dict(role="user",content="question")],256)
    assert body["reasoning"]==dict(effort="none") and body["seed"]==0
    assert "temperature" not in body and body["max_tokens"]==256
    assert body["provider"]==dict(only=["openai"],allow_fallbacks=False,require_parameters=True)
    for field,value in (("model","qwen/qwen3.5-9b"),("provider",dict(only=["other"])),
                        ("seed",1),("max_tokens",512),("reasoning",dict(effort="low")),
                        ("messages",[dict(role="user",content="different")])):
        assert study.terra.fingerprint(body)!=study.terra.fingerprint(dict(body,**{field:value}))


def test_alias_validation_and_no_reasoning_tokens():
    value=dict(model=study.MODEL+"-20260709",provider="OpenAI",choices=[dict(message=dict(content="ok"))],
               usage=dict(prompt_tokens=20,completion_tokens=10))
    study.terra.validate_response(value,manifest())
    for bad in (dict(value,model="other"),dict(value,provider="DeepInfra"),
                dict(value,usage=dict(prompt_tokens=20,completion_tokens=10,completion_tokens_details=dict(reasoning_tokens=1)))):
        with pytest.raises(ValueError):
            study.terra.validate_response(bad,manifest())


def test_duplicate_requests_and_offline_replay(tmp_path):
    async def run():
        chat=study.Chat(tmp_path,manifest(),client=Client())
        messages=[dict(role="user",content="question")]
        kwargs=dict(strategy="stated",condition="new",purpose="test")
        answers=await asyncio.gather(*(chat.request(messages,**kwargs) for _ in range(2)))
        assert len(chat.client.calls)==1
        assert answers[0]==answers[1]
        assert chat.accounting()["accounted_cost_usd"]==pytest.approx(.00016)
        await chat.close()
        raw=study.sqlite_snapshot(tmp_path/"cache/responses.sqlite3")
        with study.original.offline_guard():
            replay=await study.OfflineChat(raw,manifest()).request(messages,**kwargs)
        assert replay==answers[0]
        resumed=study.Chat(tmp_path,manifest(),client=Client(),deadline=1)
        assert await resumed.request(messages,**kwargs)==answers[0]
        assert not resumed.client.calls
        assert resumed.accounting()["attempts"]==1
        await resumed.close()
    asyncio.run(run())


def test_concurrent_reservations_never_exceed_limit(tmp_path):
    async def run():
        chat=study.Chat(tmp_path,manifest(),client=Client())
        with patch.object(study.terra,"LIMIT",.006):
            results=await asyncio.gather(*(chat.request([dict(role="user",content=f"question {i}")],
                strategy="stated",condition="new",purpose="test") for i in range(4)),return_exceptions=True)
            assert sum(isinstance(r,study.terra.SpendingLimit) for r in results)==3
            assert len(chat.client.calls)==1
            assert chat.accounting()["budget_used_usd"]<=.006
        await chat.close()
    asyncio.run(run())


def test_uncertain_request_remains_reserved_across_restart(tmp_path):
    async def run():
        chat=study.Chat(tmp_path,manifest(),client=Client(fail=True))
        with pytest.raises(RuntimeError,match="403") as error:
            await chat.request([dict(role="user",content="q")],strategy="stated",condition="new",purpose="test")
        assert "private" not in str(error.value)
        reserved=chat.accounting()["unresolved_reservations_usd"]
        assert reserved>0
        await chat.close()
        resumed=study.Chat(tmp_path,manifest(),client=Client())
        assert resumed.accounting()["budget_used_usd"]==reserved
        assert resumed.accounting()["attempts"]==1
        await resumed.close()
    asyncio.run(run())


def test_deadline_then_resume_collects_uncached_question(tmp_path):
    async def run():
        chat=study.Chat(tmp_path,manifest(),client=Client(),deadline=1)
        kwargs=dict(strategy="stated",condition="new",purpose="test")
        with pytest.raises(study.terra.CheckpointRequested):
            await chat.request([dict(role="user",content="q")],**kwargs)
        assert chat.accounting()["attempts"]==0 and not chat.client.calls
        await chat.close()
        resumed=study.Chat(tmp_path,manifest(),client=Client())
        await resumed.request([dict(role="user",content="q")],**kwargs)
        assert len(resumed.client.calls)==1
        await resumed.close()
    asyncio.run(run())


def test_pilot_allows_abstention_but_rejects_malformed_consistency():
    rows=[dict(strategy="consistency",decisions=[dict(status="abstain")]*8) for _ in range(12)]
    study.validate_api_pilot(rows)
    rows[0]["decisions"]=[dict(status="invalid")]*8
    with pytest.raises(ValueError,match="malformed"):
        study.validate_api_pilot(rows)


def test_six_vote_threshold_and_raw_confidence_preserved():
    for count in (5,6,8):
        votes=[dict(answer=1,status="valid")]*count+[dict(answer=None,status="abstain") ]*(8-count)
        answer,confidence=parser.pool_votes(votes)
        assert (answer is not None)==(count>=6)
        response=study.capped.cap_response(QueryResponse(answer,confidence,treatment="soft_evidence"))
        assert response.audit["raw_reported_confidence"]==confidence
        assert response.reported_confidence==(None if confidence is None else min(confidence,.8))


def test_report_requires_ten_seeds_and_uses_pooled_calibration():
    rows=[]
    pooled={}
    for b in study.BUDGETS:
        for n in (200,1000):
            for s in study.original.STRATEGIES:
                for g in study.original.GROUPS:
                    key=b,n,s,g
                    pooled[key]=[[(1,.99),(0,.6)],[(1,.8),(0,.6)]]
                    for seed in range(10):
                        row=dict(budget=b,sample_size=n,strategy=s,group=g,seed=seed)
                        row.update({k:float(seed+1) for k in study.METRICS})
                        row.update({k:0 for k in study.original.COUNTS})
                        row.update(attempted=b,answered=b-1)
                        rows.append(row)
    summary=study.summarize(rows,pooled)
    assert len(summary)==48
    assert summary[0]["accuracy"]==.5
    assert summary[0]["raw_brier"]==pytest.approx((.01**2+.6**2)/2)
    assert summary[0]["expected_distance_mean"]==5.5
    with pytest.raises(ValueError,match="ten unique seeds"):
        study.summarize(rows[:-1],pooled)
