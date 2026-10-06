import json
import time
from unittest.mock import patch
import numpy as np
import pytest
from cd_llm.experiments.alarm_matched_store import Store,Transport,MODEL,Checkpoint
from cd_llm.experiments.alarm_matched_methods import parse_order,long_expert
from cd_llm.experiments.alarm_matched_gpc import search


def response(text="<Answer>C</Answer>",finish="stop"):
    return dict(model=MODEL,choices=[dict(finish_reason=finish,message=dict(content=text))],
        usage=dict(prompt_tokens=17,completion_tokens=3))


def test_restart_isolation_and_logical_cost(tmp_path):
    store=Store(tmp_path)
    calls=[]
    def sender(req): calls.append(req); return response()
    request=dict(model=MODEL,messages=[dict(role="user",content="abc")],max_tokens=600)
    t=Transport(store,"first","runtime1",online=True,sender=sender)
    t.query(request)
    Transport(store,"first","runtime1").query(request)
    Transport(store,"second","runtime1").query(request)
    assert len(calls)==1
    assert store.usage("")["total_tokens"]==40
    for change in (dict(seed=1),dict(temperature=.7),dict(max_tokens=601),dict(messages=[dict(role="user",content="different")])):
        with pytest.raises(RuntimeError,match="offline cache miss"):
            Transport(store,"third","runtime1").query(dict(request,**change))
    with pytest.raises(RuntimeError,match="offline cache miss"):
        Transport(store,"third","runtime2").query(request)


def test_interrupted_attempt_remains_visible(tmp_path):
    store=Store(tmp_path)
    def crash(req): raise KeyboardInterrupt()
    request=dict(messages=[dict(role="user",content="q")],max_tokens=1)
    with pytest.raises(KeyboardInterrupt): Transport(store,"a","r",online=True,sender=crash).query(request)
    Transport(store,"a","r",online=True,sender=lambda _:response()).query(request)
    with store.db() as db:
        assert dict(db.execute("SELECT state,COUNT(*) FROM attempts GROUP BY state"))==dict(dispatched=1,complete=1)


def test_order_parsers_are_separate():
    assert parse_order(response())=="C"
    assert parse_order(response(),["V01","V02","V03"]) is None
    assert parse_order(response("<Answer>[('V01','V02')]</Answer>"),["V01","V02","V03"]) is not None
    assert parse_order(response("<Answer>[('V01','V02')]</Answer>")) is None
    assert parse_order(response("<Answer>[('V01','V02'),('V02','V01')]</Answer>"),["V01","V02"]) is None
    assert parse_order(response("<Answer>[('V01','XX')]</Answer>"),["V01","V02"]) is None
    assert parse_order(response("<Answer>A</Answer>","length")) is None


def test_cached_order_fallback_preserves_abstention_for_cycle_guard():
    from cd_llm.experiments.alarm_matched_baselines import cached_order_fallback,MissingOrderExpertPair
    class Cached:
        values={"pair:1:2":dict(parsed="A"),"pair:2:3":dict(parsed="B"),"pair:3:4":dict(parsed="C")}
        def get(self,kind,key): return self.values.get(key)
    store=Cached()
    assert cached_order_fallback(store,1,2)==(1,2)
    assert cached_order_fallback(store,2,3)==(3,2)
    assert cached_order_fallback(store,3,4) is None
    with pytest.raises(MissingOrderExpertPair): cached_order_fallback(store,4,5)


def test_long_native_probability_is_not_capped(tmp_path):
    import math
    r=response("A","length")
    r["choices"][0]["logprobs"]=dict(content=[dict(top_logprobs=[dict(token="A",logprob=math.log(.97)),dict(token="B",logprob=math.log(.03))])])
    expert=long_expert(Transport(Store(tmp_path),"long","runtime",online=True,sender=lambda _:r),"clinical context")
    arc,prob,_=expert(0,1,"V01","V02")
    assert prob==pytest.approx(.97)
    assert set(arc)=={0,1}


def test_gpc_resume_inside_edge_matches_native(tmp_path):
    from cd_llm.experiments.alarm_local_support import gpc_skeleton
    rng=np.random.default_rng(84)
    data=rng.integers(0,2,(100,5))
    guess=np.ones((5,5),int)-np.eye(5,dtype=int)
    baseline=gpc_skeleton(data,guess,np.random.default_rng(91))
    root=tmp_path/"run"; root.mkdir()
    store=Store(root)
    with pytest.raises(Checkpoint): search(data,guess,np.random.default_rng(91),store,"a",deadline=0)
    resumed=search(data,guess,np.random.default_rng(91),store,"a")
    assert np.array_equal(resumed,baseline)
    # Force several tests within the same edge before stopping.
    clock=[0.]
    def test(a,b,z): clock[0]+=1; return 0.0
    root2=tmp_path/"deep"; root2.mkdir(); deep=Store(root2)
    with patch("cd_llm.experiments.alarm_matched_gpc.time.time",lambda:clock[0]):
        with pytest.raises(Checkpoint): search(data,guess,np.random.default_rng(91),deep,"a",deadline=4,tester=test)
        state=deep.get("gpc_progress","a")
        assert state["edge"]==0 and state["tests"]==4
        matrix=search(data,guess,np.random.default_rng(91),deep,"a",tester=test)
    assert np.array_equal(matrix,guess)


def test_summary_does_not_drop_failed_seeds():
    from cd_llm.experiments.alarm_matched_report import summarize,METRICS
    rows=[dict(method="pc",sample_size=200,budget=0,status="complete",**{k:1. for k in METRICS}) for _ in range(9)]
    rows.append(dict(method="pc",sample_size=200,budget=0,status="infeasible"))
    result=summarize(rows)[0]
    assert result["complete"]==9 and result["expected_distance_mean"] is None


def test_scp_native_trials_and_repairs_are_charged(tmp_path):
    import math
    from cd_llm.experiments.alarm_matched_methods import scp_pair
    calls=[]
    def sender(req):
        calls.append(req)
        decision=req.get("logprobs",False)
        invalid=decision and len(calls)==2
        token="perhaps" if invalid else "yes"
        r=response("<yes>" if decision else "Brief explanation.")
        r.update(id="test",created=0,object="chat.completion")
        r["usage"]["total_tokens"]=20
        r["choices"][0].update(index=0)
        r["choices"][0]["message"]["role"]="assistant"
        if decision:
            r["choices"][0]["logprobs"]=dict(content=[dict(token=token,logprob=math.log(.96),bytes=None,
                top_logprobs=[dict(token="yes",logprob=math.log(.96),bytes=None),dict(token="no",logprob=math.log(.04),bytes=None)])])
        return r
    store=Store(tmp_path)
    t=Transport(store,"scp:pair","r",online=True,sender=sender)
    row=scp_pair(t,"Frozen knowledge prompt","V01","V02")
    assert row["mean_yes_probability"]==pytest.approx(.96)
    assert len(row["trials"])==5
    assert [r["seed"] for r in calls if r.get("logprobs")]==[0,1,10,20,30,40]
    assert all(r["temperature"]==.7 for r in calls)
    assert store.usage("")["logical_requests"]==7
    assert store.usage("")["total_tokens"]==140
    replay=scp_pair(Transport(store,"scp:pair","r"),"Frozen knowledge prompt","V01","V02")
    assert replay==row and len(calls)==7


def test_concurrent_shared_cache_has_one_generation(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    store=Store(tmp_path)
    calls=[]
    def sender(req): calls.append(req); return response()
    def one(i):
        Transport(store,f"unit:{i}","r",online=True,sender=sender).query(dict(messages=[dict(role="user",content="same")],max_tokens=600))
    with ThreadPoolExecutor(4) as pool: list(pool.map(one,range(8)))
    assert len(calls)==1 and store.usage("")["logical_requests"]==8


def test_parsing_gate_boundary():
    from cd_llm.experiments.neuropathic_causal_order_terra import passes_gate
    assert passes_gate(7382,7770)
    assert not passes_gate(7381,7770)


def test_persistent_journal_and_infrastructure_failure(tmp_path):
    from cd_llm.experiments.alarm_matched_recovery import scientific_failure
    store=Store(tmp_path)
    with store.db() as db:
        assert db.execute("PRAGMA journal_mode").fetchone()[0]=="persist"
        assert db.execute("PRAGMA synchronous").fetchone()[0]==2
    for method in ("long","bfs","order","scp"):
        assert not scientific_failure(method,OSError(122,"Disk quota exceeded"))
        assert not scientific_failure(method,RuntimeError("offline cache miss"))
    assert scientific_failure("long",ValueError("invalid native Long A/B answer"))


def test_completion_excludes_technical_and_unvalidated_failures(tmp_path):
    from cd_llm.experiments.alarm_matched_recovery import validated_completion
    from cd_llm.experiments.alarm_matched_baselines import METHODS
    store=Store(tmp_path)
    for i in range(20):
        for method in METHODS: store.put("prediction",f"{method}:{i}",dict(status="complete"))
    assert validated_completion(store)
    for status in ("failed","pending","infeasible"):
        store.put("prediction","long:0",dict(status=status))
        assert not validated_completion(store)
    store.put("prediction","long:0",dict(status="infeasible",scientific_infeasibility=True))
    assert validated_completion(store)


def test_submission_reconciles_crash_and_missing_array_tasks(tmp_path,monkeypatch):
    from cd_llm.experiments import alarm_matched_recovery as recovery
    store=Store(tmp_path)
    store.put("dispatch","old",dict(stage="gpc",conditions=[1,3]))
    monkeypatch.setattr(recovery,"queue",lambda:[("91_1","old")])
    calls=[]
    def launch(array,token): calls.append((array,token)); return "92"
    recovery.submit_unique(tmp_path,"gpc","1,3%4",None,None,launch)
    assert calls[0][0]=="3%4"
    assert store.get("dispatch","old")["job"]=="91"
    assert recovery.submit_unique(tmp_path,"gpc",None,None,1,launch)=="91"
    assert len(calls)==1
    # A stale marker without any active job must not prevent another attempt.
    monkeypatch.setattr(recovery,"queue",lambda:[])
    recovery.submit_unique(tmp_path,"gpc",None,None,1,launch)
    assert len(calls)==2


def test_array_checkpoint_schedules_successor(tmp_path,monkeypatch):
    from cd_llm.experiments import alarm_matched_recovery as recovery
    store=Store(tmp_path)
    store.put("dispatch","old",dict(stage="gpc",conditions=[1,3],job="91"))
    monkeypatch.setattr(recovery,"queue",lambda:[("91_1","old"),("91_3","old")])
    monkeypatch.setenv("SLURM_JOB_ID","94")
    monkeypatch.setenv("SLURM_ARRAY_JOB_ID","91")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID","1")
    calls=[]
    def launch(array,token): calls.append(token); return "95"
    assert recovery.submit_unique(tmp_path,"gpc",None,"afterok:94",1,launch)=="95"
    assert len(calls)==1
    token=calls[0]
    monkeypatch.setattr(recovery,"queue",lambda:[("91_1","old"),("95",token)])
    recovery.submit_unique(tmp_path,"gpc",None,"afterok:94",1,launch)
    assert len(calls)==1


def test_submission_failure_keeps_reconcilable_intent(tmp_path,monkeypatch):
    from cd_llm.experiments import alarm_matched_recovery as recovery
    monkeypatch.setattr(recovery,"queue",lambda:[])
    def crash(array,token): raise RuntimeError("connection lost after acceptance")
    with pytest.raises(RuntimeError): recovery.submit_unique(tmp_path,"collect",None,None,None,crash)
    token=next(iter(Store(tmp_path).items("dispatch")))
    monkeypatch.setattr(recovery,"queue",lambda:[("101",token)])
    assert recovery.submit_unique(tmp_path,"collect",None,None,None,crash)=="101"


def test_method_temperature_validation():
    from cd_llm.experiments.alarm_matched_store import canonical_request,validate_request_temperature
    settings=dict(temperature=0,scp=dict(temperature=.7))
    assert canonical_request({})["temperature"]==0
    request=canonical_request(dict(temperature=.7))
    assert request["temperature"]==.7
    validate_request_temperature(request,"scp:0:0:1",settings)
    for unit in ("bfs:0","long:0","order:triplet:1","gpc:shared","pilot:long"):
        validate_request_temperature(canonical_request({}),unit,settings)
        with pytest.raises(ValueError,match="temperature mismatch"):
            validate_request_temperature(request,unit,settings)
    with pytest.raises(ValueError,match="temperature mismatch"):
        validate_request_temperature(canonical_request({}),"scp:0:0:1",settings)


def test_scp_temperature_migration_preserves_other_work_and_new_replay(tmp_path):
    from cd_llm.experiments.alarm_matched_recovery import migrate_scp_temperature_records
    store=Store(tmp_path)
    request=dict(messages=[dict(role="user",content="same")],temperature=0,max_tokens=1)
    old,_=Transport(store,"scp:0:0:1","r",online=True,sender=lambda _:response()).query(request)
    Transport(store,"long:0","r").query(request)
    retained=[("bootstrap","0",dict(matrices=[1])),("gpc_progress","1",dict(tests=10000)),
              ("prediction","gpc:0",dict(status="complete")),("collected","bfs:0",dict(status="complete"))]
    for kind,key,value in retained: store.put(kind,key,value)
    store.put("scp","0:0:1",dict(trials=[.5]*5))
    store.put("prediction","scp:0",dict(status="complete"))
    store.put("gate","pilot",dict(passed=True))
    store.put("gate","audit",dict(passed=True))
    with store.db() as db:
        raw=list(db.execute("SELECT * FROM responses"))
        attempts=list(db.execute("SELECT * FROM attempts"))
    first=migrate_scp_temperature_records(store)
    assert first["archived_pairs"]==first["archived_predictions"]==first["archived_logical_requests"]==1
    for kind,key,value in retained: assert store.get(kind,key)==value
    for kind,key in (("scp","0:0:1"),("prediction","scp:0"),("gate","pilot"),("gate","audit")):
        assert store.get(kind,key) is None
        assert store.get(first["archive_prefix"]+kind,key) is not None
    assert store.usage("scp:")["total_tokens"]==0
    assert store.usage("long:0")["total_tokens"]==20
    assert store.usage(first["archive_prefix"])["total_tokens"]==20
    with store.db() as db:
        assert list(db.execute("SELECT * FROM responses"))==raw
        assert list(db.execute("SELECT * FROM attempts"))==attempts
    new_request=dict(request,temperature=.7)
    with pytest.raises(RuntimeError,match="offline cache miss"):
        Transport(store,"scp:0:0:1","r").query(new_request)
    Transport(store,"scp:0:0:1","r",online=True,sender=lambda _:response()).query(new_request)
    store.put("scp","0:0:1",dict(trials=[.96]*5))
    assert migrate_scp_temperature_records(store)==first
    assert store.get("scp","0:0:1")==dict(trials=[.96]*5)
    assert Transport(store,"scp:0:0:1","r").query(new_request)[0]==old
    assert store.usage("scp:")["total_tokens"]==20


def test_scp_migration_rejects_unexpected_source_temperature(tmp_path):
    from cd_llm.experiments.alarm_matched_recovery import migrate_scp_temperature_records
    store=Store(tmp_path)
    Transport(store,"scp:0:0:1","r",online=True,sender=lambda _:response()).query(dict(temperature=.7,max_tokens=1))
    store.put("scp","0:0:1",dict(trials=[.96]*5))
    with pytest.raises(ValueError,match="zero-temperature source"):
        migrate_scp_temperature_records(store)
    assert store.get("scp","0:0:1") is not None
    assert store.usage("scp:")["logical_requests"]==1


def test_scp_amendment_backup_and_restart(tmp_path,monkeypatch):
    import tarfile
    import sqlite3
    from cd_llm.experiments import alarm_matched_baselines as baseline,alarm_matched_recovery as recovery
    from cd_llm.experiments.common.artifacts import write_json,read_json,sha256
    root=tmp_path/"run"; root.mkdir()
    code=tmp_path/"workspace"
    name="code/src/cd_llm/experiments/alarm_matched_methods.py"
    source=code/name; source.parent.mkdir(parents=True); source.write_text("before")
    folder=root/"amendments/scp_temperature_0p7"; folder.mkdir(parents=True)
    original=dict(inputs={},implementation={name:sha256(source)},settings=dict(temperature=0,scp=dict(bootstraps=100)))
    write_json(folder/"manifest_before.json",original)
    write_json(root/"manifest.json",original)
    with tarfile.open(folder/"implementation_before.tar.gz","w:gz") as archive:
        archive.add(source,arcname=name)
    source.write_text("after")
    monkeypatch.setattr(baseline,"ROOT",code)
    monkeypatch.setattr(baseline,"CODE_ROOT",code)
    monkeypatch.setattr(recovery,"queue",lambda:[])
    store=Store(root)
    store.put("scp","0:0:1",dict(trials=[.5]*5))
    store.put("prediction","gpc:0",dict(status="complete"))
    write_json(root/"completion.json",dict(complete=True))
    amendment=recovery.amend_scp_temperature(root,.7)
    assert read_json(root/"manifest.json")["settings"]==dict(temperature=0,scp=dict(bootstraps=100,temperature=.7))
    assert not (root/"completion.json").exists()
    with sqlite3.connect(folder/"cache_before.sqlite3") as db:
        assert db.execute("SELECT COUNT(*) FROM items WHERE kind='scp'").fetchone()[0]==1
    # Simulate interruption after the manifest is written, before final marker.
    (folder/"complete.json").unlink()
    assert recovery.amend_scp_temperature(root,.7)==amendment
    assert recovery.amend_scp_temperature(root,.7)==amendment
    assert store.get("prediction","gpc:0")==dict(status="complete")
