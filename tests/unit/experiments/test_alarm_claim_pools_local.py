import asyncio
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from cd_llm.experiments import alarm_claim_pools_local as study
from cd_llm.experiments import alarm_claim_pools_local_transport as transport


class Client:
    def __init__(self, outputs=None):
        self.outputs=iter(outputs or ['{"answer":"yes","confidence":1.0}'])
        self.chat=SimpleNamespace(completions=self)
        self.calls=[]
        self.active=self.peak=0

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        self.active+=1
        self.peak=max(self.peak,self.active)
        await asyncio.sleep(0)
        self.active-=1
        response=dict(id=str(len(self.calls)),model=study.MODEL,
            choices=[dict(message=dict(content=next(self.outputs)),finish_reason='stop')],
            usage=dict(prompt_tokens=100,completion_tokens=10,total_tokens=110))
        return SimpleNamespace(model_dump=lambda:response)

    async def close(self):
        pass


class TransportTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.path=Path(self.temp.name)/'responses.sqlite3'
        self.db=study.ResponseCache(self.path)
        self.m=dict(settings=study.settings(),runtime=dict(model_revision='revision'),runtime_fingerprint='runtime')
        self.inputs=study.prompts.load_inputs(study.SOURCE)
        self.query=next(q for q in study.dag_query_pool(37,study.GROUPS['all']) if q.kind=='ancestry')

    def tearDown(self):
        self.db.close()
        self.temp.cleanup()

    def chat(self,client,**kwargs):
        return transport.LocalChat(self.db,self.m,'http://127.0.0.1:22000/v1',client=client,**kwargs)

    async def test_raw_cap_and_offline_replay_and_restart(self):
        client=Client()
        chat=self.chat(client)
        row=await study.prompts.elicit(self.query,'stated',chat,self.inputs)
        answer=study.response_for(row,self.m)
        self.assertEqual(answer.reported_confidence,.8)
        self.assertEqual(answer.audit['raw_reported_confidence'],1.)
        self.assertEqual(answer.audit['provider'],'local-vllm')
        self.assertEqual(answer.audit['model_revision'],'revision')
        with study.offline_guard():
            replay=await study.prompts.elicit(self.query,'stated',
                transport.OfflineChat(study.sqlite_snapshot(self.path),self.m),self.inputs)
        self.assertEqual(row,replay)
        self.db.close()
        self.db=study.ResponseCache(self.path)
        restarted=self.chat(client,deadline=0)
        self.assertEqual(row,await study.prompts.elicit(self.query,'stated',restarted,self.inputs))
        self.assertEqual(len(client.calls),1)
        self.assertEqual(restarted.accounting()['attempts'],1)
        self.assertEqual(restarted.accounting()['actual_api_spending_usd'],0.)
        self.assertEqual(client.calls[0]['extra_body'],dict(chat_template_kwargs=dict(enable_thinking=False)))
        for key in ('seed','max_tokens','temperature'):
            self.assertEqual(client.calls[0][key],study.settings()[key])

    async def test_repair_and_abstention_usage(self):
        chat=self.chat(Client(['invalid','{"answer":"abstain"}']))
        row=await study.prompts.elicit(self.query,'stated',chat,self.inputs)
        study.valid_outcome(row)
        self.assertIsNone(row['answer'])
        self.assertEqual((row['prompt_tokens'],row['completion_tokens']),(200,20))
        self.assertEqual(len(row['request_keys']),2)
        self.assertEqual(chat.accounting()['returned_prompt_tokens'],200)

    async def test_final_brace_recovery_preserves_both_requests_and_raw_content(self):
        content='{"answer":"no","confidence":0.95,"justification":"No direct mechanism."'
        client=Client([content,content])
        chat=self.chat(client)
        row=await study.recovery.elicit(self.query,'stated',chat,self.inputs)
        study.valid_outcome(row)
        self.assertEqual((row['answer'],row['reported_confidence']),(0,.95))
        self.assertEqual(row['decisions'][0]['status'],'malformed')
        self.assertEqual(row['decisions'][1]['syntactic_recovery'],study.recovery.POLICY)
        self.assertEqual((row['prompt_tokens'],row['completion_tokens']),(200,20))
        self.assertEqual(len(client.calls),2)
        raw=study.sqlite_snapshot(self.path)
        self.assertTrue(all(r['record']['content']==content for r in raw.values()))
        with study.offline_guard():
            replay=await study.recovery.elicit(self.query,'stated',transport.OfflineChat(raw,self.m),self.inputs)
        self.assertEqual(row,replay)
        answer=study.response_for(row,self.m)
        self.assertEqual(answer.reported_confidence,.8)
        self.assertEqual(answer.audit['raw_reported_confidence'],.95)

    async def test_complete_response_unchanged_by_recovery(self):
        client=Client()
        chat=self.chat(client)
        strict=await study.prompts.elicit(self.query,'stated',chat,self.inputs)
        recovered=await study.recovery.elicit(self.query,'stated',chat,self.inputs)
        self.assertEqual(strict,recovered)
        self.assertEqual(len(client.calls),1)

    def test_recovery_rejects_ambiguous_truncated_and_invalid_fields(self):
        valid='{"answer":"no","confidence":0.95,"justification":"Clear."'
        def record(content,finish='stop'):
            return dict(content=content,request_key='key',response=dict(choices=[dict(finish_reason=finish)]))
        self.assertIsNotNone(study.recovery.recover_final_brace(record(valid)))
        for content in (valid+'}', valid[:-2],
                '{"answer":"no","answer":"yes","confidence":0.95,"justification":"X"',
                '{"answer":"maybe","confidence":0.95,"justification":"X"',
                '{"answer":"no","confidence":true,"justification":"X"',
                '{"answer":"no","confidence":0.2,"justification":"X"',
                '{"answer":"no","confidence":NaN,"justification":"X"',
                '{"answer":"no","confidence":0.9', 'text '+valid, valid+' extra',
                '{"answer":"no","confidence":0.9,"justification":{"x":"nested"}'):
            with self.subTest(content=content):
                self.assertIsNone(study.recovery.recover_final_brace(record(content)))
        for finish in ('length',None,'error'):
            self.assertIsNone(study.recovery.recover_final_brace(record(valid,finish)))

    async def test_expired_deadline_and_concurrency(self):
        client=Client(['{"answer":"yes","confidence":.8}']*12)
        chat=self.chat(client,deadline=0)
        with self.assertRaises(transport.CheckpointRequested):
            await study.prompts.elicit(self.query,'stated',chat,self.inputs)
        self.assertEqual(chat.accounting()['attempts'],0)
        chat.deadline=float('inf')
        await asyncio.gather(*(chat.request([dict(role='user',content=str(i))],strategy='stated',condition='new',purpose=str(i)) for i in range(12)))
        self.assertEqual(len(client.calls),12)
        self.assertEqual(client.peak,4)

    async def test_ledger_response_recovers_without_generation(self):
        client=Client()
        chat=self.chat(client)
        row=await study.prompts.elicit(self.query,'stated',chat,self.inputs)
        self.db.connection.execute('DELETE FROM responses')
        self.db.connection.commit()
        replay=await study.prompts.elicit(self.query,'stated',self.chat(client,deadline=0),self.inputs)
        self.assertEqual(row,replay)
        self.assertEqual(len(client.calls),1)

    async def test_unresolved_attempt_is_preserved_after_restart(self):
        client=Client()
        chat=self.chat(client)
        self.db.connection.execute("INSERT INTO local_attempts(request_key,purpose,state) VALUES ('unresolved','test','pending')")
        self.db.connection.commit()
        self.db.close()
        self.db=study.ResponseCache(self.path)
        restarted=self.chat(client)
        self.assertEqual(restarted.accounting()['unresolved_attempts'],1)
        self.assertEqual(restarted.accounting()['attempts'],1)

    def test_cache_isolation_and_loopback_only(self):
        messages=[dict(role='user',content='question')]
        key=lambda m:study._request_key(transport.request_identity(messages,m))
        baseline=key(self.m)
        for keyname,value in (('model','other'),('seed',1),('temperature',.1),('max_tokens',512),('extra_body',{})):
            other=copy.deepcopy(self.m)
            other['settings'][keyname]=value
            self.assertNotEqual(baseline,key(other))
        for other in (dict(self.m,runtime_fingerprint='other'),dict(self.m,runtime=dict(model_revision='other'))):
            self.assertNotEqual(baseline,key(other))
        hosted=study.original.api.request_record(messages,'stated','new',dict(settings=study.original.settings(),seeds=dict(stated=0)))
        self.assertNotEqual(baseline,study._request_key(hosted))
        for url in ('https://openrouter.ai/api/v1','http://127.0.0.1:22@external/v1','http://localhost:8000/v1'):
            with self.assertRaises(ValueError):
                transport.LocalChat(self.db,self.m,url,client=Client())

    async def test_wrong_model_and_reasoning_rejected(self):
        response=(await Client().create(model=study.MODEL)).model_dump()
        transport.validate_response(response,self.m)
        for changes in (dict(model='other'),dict(usage={}),
                        dict(choices=[dict(message=dict(content='x',reasoning_content='hidden'))])):
            with self.assertRaises(ValueError):
                transport.validate_response(dict(response,**changes),self.m)


class WorkflowTests(unittest.TestCase):
    def test_scientific_kernel_is_shared(self):
        self.assertIs(study.arguments,study.original.arguments)
        self.assertIs(study.active_query_smc,study.original.active_query_smc)
        self.assertIs(study.validate_record,study.original.validate_record)
        self.assertEqual(len(study.tasks()),60)
        args=study.arguments(1000,200,(),())
        self.assertEqual((args['confidence'],args['soft_confidence_cap']),(.9,.99))
        self.assertEqual(args['budgets'],(10,20,50,100))
        self.assertEqual(args['budget_unit'],'attempted')

    def test_chain_uses_current_jobs_and_never_publishes(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with patch.dict(study.os.environ,{'SLURM_JOB_ID':'123'}),patch.object(study,'launch',return_value='456') as launch:
                study.chain(root,'audit')
                self.assertEqual(launch.call_args_list[0].args,(root,'run-block','123','0-1%2'))
                self.assertEqual(launch.call_args_list[1].args,(root,'validate-pilot','456'))
                launch.reset_mock()
                study.chain(root,'report')
                launch.assert_not_called()

    def test_checkpoint_resubmits_collection_only(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            study.write_json(root/'status.json',dict(stage='checkpointed'))
            with patch.dict(study.os.environ,{'SLURM_JOB_ID':'123'}),patch.object(study,'launch') as launch:
                study.chain(root,'collect')
                launch.assert_called_once_with(root,'collect','123')


if __name__=='__main__':
    unittest.main()
