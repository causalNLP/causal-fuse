import unittest
from unittest.mock import patch

import numpy as np

from cd_llm.core.finite_support import graph_from_edges, has_edge
from cd_llm.methods.active_query import (
    ClaimQuery, QueryResponse, active_query_smc, group_i_query_pool,
    _select_uniform_candidate_index, _systematic_resample,
)
from cd_llm.methods.graph_scores import BDeuScore


class AlternatingQueryTests(unittest.TestCase):
    def setUp(self):
        self.empty=graph_from_edges(5)
        self.edge=graph_from_edges(5,directed=[(0,1)])
        self.score=BDeuScore(np.zeros((8,5),dtype=int),cardinalities=[2]*5,max_parents=4)

    def run_query(self, **kwargs):
        config=dict(budgets=(20,),confidence=.7,treatment="soft_evidence",seed=12,
                    acquisition_seed=45,rejuvenation_steps=0,policy="alternating")
        config.update(kwargs)
        return active_query_smc({self.empty:.5,self.edge:.5},self.score,
            lambda q: QueryResponse(int(q.kind=="directed" and (q.source,q.target)==(0,1)),.7),**config)

    def test_schedule_counts_and_no_repeated_queries(self):
        for period,random_count in ((2,10),(4,5)):
            result=self.run_query(exploration_period=period)
            self.assertEqual([s.selection_rule for s in result.steps],
                ["random" if i%period==0 else "eig" for i in range(20)])
            self.assertEqual(sum(s.selection_rule=="random" for s in result.steps),random_count)
            self.assertEqual(len({s.query for s in result.steps}),20)

    def test_abstention_consumes_slot_and_keeps_particles(self):
        seen=[]
        def answer(q):
            seen.append(q)
            return QueryResponse(None,None) if len(seen)==1 else QueryResponse(0,.7)
        result=active_query_smc(None,self.score,answer,policy="alternating",
            initial_particles=[self.empty]*2,initial_weights=[.5,.5],
            budgets=(1,2),rejuvenation_steps=0,confidence=.7,treatment="soft_evidence")
        self.assertEqual([s.selection_rule for s in result.steps],["random","eig"])
        self.assertEqual(result.distributions[1],{self.empty:1})
        self.assertEqual(len(set(seen)),2)

    def test_uniform_mapping_includes_every_available_zero_eig_question(self):
        class Draw:
            def __init__(self,index): self.index=index
            def integers(self,high):
                self.bounds=(0,high)
                return self.index
        available=np.array([1,4,8,12])
        self.assertEqual([_select_uniform_candidate_index(available,Draw(i)) for i in range(4)],list(available))
        result=active_query_smc(None,self.score,lambda q: QueryResponse(0,.7),
            initial_particles=[self.empty]*2,initial_weights=[.5,.5],
            policy="alternating",budgets=(1,),rejuvenation_steps=0,confidence=.7)
        self.assertEqual(result.steps[0].selection_rule,"random")
        self.assertEqual(result.steps[0].expected_information_gain,0)

    def test_acquisition_rng_is_independent_of_resampling(self):
        a=self.run_query(policy="random",budgets=(30,),ess_threshold_fraction=1)
        def consume(weights,rng):
            result=_systematic_resample(weights,rng)
            rng.random(1000)
            return result
        with patch("cd_llm.methods.active_query._systematic_resample",side_effect=consume) as hook:
            b=self.run_query(policy="random",budgets=(30,),ess_threshold_fraction=1)
        self.assertTrue(hook.called)
        self.assertEqual([s.query for s in a.steps],[s.query for s in b.steps])

    def test_fixed_seed_reproduces_and_eig_ignores_acquisition_seed(self):
        for policy in ("eig","alternating","random"):
            a=self.run_query(policy=policy)
            b=self.run_query(policy=policy)
            self.assertEqual(a.distributions,b.distributions)
            self.assertEqual([s.query for s in a.steps],[s.query for s in b.steps])
        a=self.run_query(policy="eig",acquisition_seed=None)
        b=self.run_query(policy="eig",acquisition_seed=999)
        self.assertEqual(a.distributions,b.distributions)
        self.assertEqual([s.query for s in a.steps],[s.query for s in b.steps])

    def test_mutation_can_introduce_a_unanimously_absent_edge(self):
        data=np.tile([[0,0],[1,1]],(50,1))
        score=BDeuScore(data,cardinalities=[2,2],max_parents=1)
        empty=graph_from_edges(2)
        result=active_query_smc(None,score,lambda q: QueryResponse(1,.99),
            initial_particles=[empty]*32,initial_weights=[1/32]*32,
            query_pool=[ClaimQuery("directed",0,1,"d")],budgets=(1,),
            policy="alternating",seed=15,confidence=.7,treatment="soft_evidence",
            rejuvenate_after_every_query=True,rejuvenation_steps=2)
        self.assertEqual(result.steps[0].predicate_probability,0)
        self.assertEqual(result.steps[0].rejuvenation_attempted_moves,64)
        self.assertTrue(any(has_edge(g,0,1) for g in result.particles))

    def test_invalid_schedule_and_seed_are_rejected(self):
        for value in (True,0,1,2.5):
            with self.assertRaises(ValueError): self.run_query(exploration_period=value)
        with self.assertRaises(ValueError): self.run_query(acquisition_seed=-1)


if __name__=="__main__":
    unittest.main()
