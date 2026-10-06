import asyncio
import copy
from dataclasses import asdict
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from cd_llm.baselines.causal_order2025.openrouter_backend import ResponseCache, _request_key
from cd_llm.core.finite_support import graph_from_edges
from cd_llm.experiments import alarm_claim_pools as study
from cd_llm.experiments import alarm_claim_pools_report as report
from cd_llm.methods import active_query as active
from cd_llm.methods.adaptive_pruning import GaussianBICScore


class Client:
    def __init__(self, outputs=None):
        self.outputs = iter(outputs or ['{"answer":"yes","confidence":1.0}'])
        self.calls = []
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        await asyncio.sleep(0)
        response = dict(id=str(len(self.calls)), provider='DeepInfra', model=kwargs['model'],
            choices=[dict(message=dict(content=next(self.outputs)), finish_reason='stop')],
            usage=dict(prompt_tokens=100, completion_tokens=10, cost=.00004,
                       completion_tokens_details=dict(reasoning_tokens=0)))
        return SimpleNamespace(model_dump=lambda: response)

    async def close(self):
        pass


class CollectionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)/'cache.sqlite3'
        self.db = ResponseCache(self.path)
        self.manifest = dict(settings=study.settings(), seeds=dict(stated=0),
            response_provider='DeepInfra', require_token_usage=True, cost_limit_usd=25.,
            pricing=dict(prompt='0.00000015', completion='0.000001875'))
        self.inputs = dict(metadata=dict(domain_context='clinical physiology',
            nodes=[dict(id=f'V{i:02}',description=f'description {i}') for i in range(37)]),
            templates=study.templates())
        self.query = next(q for q in study.dag_query_pool(37,study.GROUPS['all']) if q.kind=='ancestry')

    def tearDown(self):
        self.db.close()
        self.temp.cleanup()

    async def test_raw_confidence_offline_replay_and_expired_cache_hit(self):
        client = Client()
        chat = study.Chat(self.db,self.manifest,client=client)
        row = await study.prompts.elicit(self.query,'stated',chat,self.inputs)
        study.valid_outcome(row)
        capped = study.cap_response(study.prompts.response_for(row,self.manifest),.8)
        self.assertEqual((row['reported_confidence'],capped.reported_confidence), (1.,.8))
        self.assertEqual(capped.audit['raw_reported_confidence'],1.)
        with study.offline_guard():
            replay = await study.prompts.elicit(self.query,'stated',
                study.api.OfflineChat(study.sqlite_snapshot(self.path),self.manifest),self.inputs)
        self.assertEqual(row,replay)
        chat.deadline = 0
        self.assertEqual(row,await study.prompts.elicit(self.query,'stated',chat,self.inputs))
        self.assertEqual(len(client.calls),1)
        self.assertEqual(client.calls[0]['seed'],0)
        self.assertEqual(client.calls[0]['max_tokens'],256)
        self.assertEqual(client.calls[0]['temperature'],0)

    async def test_repair_and_abstention_are_charged(self):
        client = Client(['invalid','{"answer":"abstain"}'])
        row = await study.prompts.elicit(self.query,'stated',study.Chat(self.db,self.manifest,client=client),self.inputs)
        study.valid_outcome(row)
        self.assertIsNone(row['answer'])
        self.assertEqual((row['prompt_tokens'],row['completion_tokens']),(200,20))
        self.assertEqual(len(row['request_keys']),2)
        invalid = copy.deepcopy(row)
        invalid['decisions'][-1]['status']='invalid'
        with self.assertRaisesRegex(ValueError,'unparseable'):
            study.valid_outcome(invalid)

    async def test_restart_retains_pending_cost_and_blocks_four_workers(self):
        chat = study.Chat(self.db,self.manifest,client=Client())
        request = study.api.request_record([dict(role='user',content='test')],'stated','new',self.manifest)
        chat.reserve(request)
        before = chat.accounting()['budget_used_usd']
        self.db.close()
        self.db = ResponseCache(self.path)
        client = Client()
        chat = study.Chat(self.db,self.manifest,client=client)
        self.assertEqual(chat.accounting()['budget_used_usd'],before)
        self.db.connection.execute("INSERT INTO api_attempts(request_key,reserved_usd,state) VALUES ('unknown',?,'uncertain')",(25-before-.000001,))
        self.db.connection.commit()
        outcomes = await asyncio.gather(*(chat.request([dict(role='user',content=f'question {i}')],strategy='stated',condition='new',purpose='test') for i in range(4)),return_exceptions=True)
        self.assertTrue(all(isinstance(r,RuntimeError) for r in outcomes))
        self.assertEqual(client.calls,[])
        self.assertLessEqual(chat.accounting()['budget_used_usd'],25)

    async def test_deadline_stops_before_spending(self):
        client = Client()
        chat = study.Chat(self.db,self.manifest,client=client,deadline=0)
        with self.assertRaises(study.CheckpointRequested):
            await study.prompts.elicit(self.query,'stated',chat,self.inputs)
        self.assertEqual(chat.accounting()['attempts'],0)
        self.assertEqual(client.calls,[])

    def test_model_provider_generation_and_prompt_isolate_cache(self):
        messages = [dict(role='user',content='question')]
        def key(m,ms=messages):
            return _request_key(study.api.request_record(ms,'stated','new',m))
        original=key(self.manifest)
        for field,value in (('model','qwen/other'),('provider',dict(only=['other'])),('max_tokens',512),('temperature',.1),('reasoning',dict(enabled=True))):
            other=copy.deepcopy(self.manifest)
            other['settings'][field]=value
            self.assertNotEqual(original,key(other))
        other=copy.deepcopy(self.manifest)
        other['seeds']['stated']=1
        self.assertNotEqual(original,key(other))
        self.assertNotEqual(original,key(self.manifest,[dict(role='user',content='different')]))

    def test_provider_and_reasoning_validation(self):
        response=dict(provider='DeepInfra',model=study.MODEL,
            choices=[dict(message=dict(content='yes'))],usage=dict(prompt_tokens=100,completion_tokens=10))
        study.api.validate_response(response,self.manifest)
        for change in (dict(model='other'),dict(provider='Other'),dict(usage={}),
                       dict(choices=[dict(message=dict(reasoning='hidden'))])):
            with self.assertRaises(RuntimeError):
                study.api.validate_response(dict(response,**change),self.manifest)


class InferenceTests(unittest.TestCase):
    def test_prepare_freezes_nested_observations_and_column_alignment(self):
        endpoint=dict(pricing=dict(prompt='0.00000015',completion='0.000001875'))
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(study,'SEEDS',(1000,)), \
             patch.object(study.hosted,'endpoints',return_value=({},endpoint)):
            root=Path(directory)
            manifest=study.prepare(root)
            self.assertEqual(manifest,study.verify(root))
            np.testing.assert_array_equal(np.load(root/'inputs/alarm_s1000_n200.npy'),
                                          np.load(root/'inputs/alarm_s1000_n1000.npy')[:200])
            for condition in manifest['conditions']:
                data=np.load(root/condition['data_path'])
                score=study.grid.common_score(data,condition)
                self.assertEqual(tuple(score.cardinalities),tuple(condition['observed_cardinalities']))
                for node,count in enumerate(score.cardinalities):
                    self.assertEqual(set(score.data[:,node]),set(range(count)))
            names,levels,parents,_,_=study.data_source.parse_bif((root/'generation/alarm.bif').read_text())
            presentation=study.read_json(root/'inputs/presentation.json')
            nodes=study.prompts.load_inputs(root)['metadata']['nodes']
            self.assertEqual([n['id'] for n in nodes],[presentation['mapping'][names[i]] for i in presentation['order']])
            truth=study.read_json(root/'evaluation/truth_graph.json')['graph']['directed']
            recovered={(names[presentation['order'][a]],names[presentation['order'][b]]) for a,b in truth}
            self.assertEqual(recovered,{(p,c) for c,ps in parents.items() for p in ps})
            (root/'prompts.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'frozen inputs changed'):
                study.verify(root)

    def test_registration_and_pool_union(self):
        self.assertEqual(len(study.tasks()),60)
        self.assertEqual({t['seed'] for t in study.tasks()},set(range(1000,1010)))
        pools={g:set(study.dag_query_pool(37,families)) for g,families in study.GROUPS.items()}
        self.assertEqual([len(pools[g]) for g in ('group_i','group_ii','all')],[1998,1332,3330])
        self.assertEqual(pools['all'],pools['group_i']|pools['group_ii'])
        self.assertFalse(pools['group_i']&pools['group_ii'])
        self.assertNotIn('anatomical level',str(study.templates()))

    def small(self,budgets=(1,2,3)):
        empty,edge=(graph_from_edges(3,directed=e) for e in ([],[(0,1)]))
        score=GaussianBICScore(np.random.default_rng(7).normal(size=(30,3)),max_parents=2)
        responses=iter(((1,1.),(None,None),(0,.95)))
        def provider(q):
            a,c=next(responses)
            return study.cap_response(active.QueryResponse(a,c,treatment='soft_evidence'),.8)
        pool=tuple(active.ClaimQuery('directed',a,b,str(i)) for i,(a,b) in enumerate(((0,1),(1,0),(0,2))))
        args=study.arguments(1000,200,(empty,edge),(.5,.5))
        args.update(budgets=budgets,ess_threshold_fraction=.01)
        return active.active_query_smc(None,score,provider,query_pool=pool,**args)

    def test_cap_in_weighting_and_mh_abstention_and_checkpoints(self):
        with patch.object(active,'_rejuvenate',wraps=active._rejuvenate) as mutate:
            prefixes=self.small()
        self.assertEqual([s.effective_confidence for s in prefixes.steps],[.8,None,.8])
        self.assertTrue(all(s.selection_effective_confidence==.9 for s in prefixes.steps))
        self.assertAlmostEqual(prefixes.distributions[1][graph_from_edges(3,directed=[(0,1)])],.8)
        self.assertEqual(prefixes.distributions[1],prefixes.distributions[2])
        self.assertEqual((prefixes.query_count,prefixes.answered_query_count),(3,2))
        self.assertEqual(mutate.call_count,1)
        self.assertEqual(mutate.call_args.args[12],.99)
        self.assertEqual([n.reported_confidence for n in mutate.call_args.args[3].nodes],[.8,.8])
        proposal=active.EvidenceGuidedDAGProposal(mutate.call_args.args[2],mutate.call_args.args[3],
            node_count=3,soft_confidence_cap=mutate.call_args.args[12])
        self.assertEqual(proposal.effective_confidences,(.8,.8))
        terminal=self.small((3,))
        self.assertEqual(terminal.distributions[3],prefixes.distributions[3])
        for a,b in zip(terminal.steps,prefixes.steps):
            left,right=asdict(a),asdict(b)
            left.pop('cumulative_seconds'); right.pop('cumulative_seconds')
            self.assertEqual(left,right)

    def test_scoring_37_nodes_and_indirect_ancestry(self):
        truth=graph_from_edges(37,directed=[(0,1),(1,2)])
        step=dict(query=asdict(active.ClaimQuery('ancestry',0,2,'test')),valid_response=True,
            answer=1,effective_confidence=.8,response_audit=dict(raw_reported_confidence=1.,request_keys=['a']),
            prompt_tokens=100,completion_tokens=10)
        record=dict(distributions={'1':[dict(graph=study.graph_edges(truth),weight=1.)]},trace=[step])
        metrics,events=report.score_record(record,truth,1)
        self.assertEqual(metrics['expected_distance'],0)
        self.assertEqual(metrics['total_tokens'],110)
        self.assertTrue(events[0]['indirect_ancestry'])

    def test_paired_statistics_and_incomplete_aggregation(self):
        row=report.bootstrap([-1.]*10,'constant')
        self.assertEqual((row['mean'],row['low'],row['high']),(-1.,-1.,-1.))
        rows=report.holm([dict(p_value=p) for p in (.04,.001,.03,.2)])
        np.testing.assert_allclose([r['holm_p_value'] for r in rows],[.09,.004,.09,.2])
        with self.assertRaisesRegex(ValueError,'ten paired'):
            report.bootstrap([1.]*9,'incomplete')
        with self.assertRaisesRegex(ValueError,'incomplete condition'):
            report.aggregate([dict(sample_size=200,group='all',budget=100,seed=1000)])


if __name__=='__main__':
    unittest.main()
