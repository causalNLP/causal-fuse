import math
import unittest
from unittest.mock import patch

import numpy as np
from cd_llm.core.finite_support import Claim, graph_from_edges
from cd_llm.methods.source_pair_evidence import SourcePairEvidence
from cd_llm.methods.active_query import active_query_smc, ClaimQuery, QueryResponse
from cd_llm.methods.adaptive_pruning import GaussianBICScore
from cd_llm.methods.evidence_guided import EvidenceGuidedDAGProposal


class SourcePairTests(unittest.TestCase):
    def setUp(self):
        self.graphs = (graph_from_edges(3), graph_from_edges(3,[(0,1)]), graph_from_edges(3,[(1,0)]))
        self.score = GaussianBICScore(np.random.default_rng(4).normal(size=(30,3)), max_parents=2)

    def test_nested_claims_preserve_direction_without_multiplication(self):
        p=SourcePairEvidence('model/context')
        p.add(Claim('d','I','directed',1,0,1),.95)
        p.add(Claim('a','I','adjacency',1,1,0),.9)
        self.assertEqual(len(p.groups),1)
        group=p.groups[0,1]
        self.assertEqual(group.allowed_states,frozenset((1,)))
        self.assertEqual(group.confidence,.9)
        self.assertAlmostEqual(math.exp(p.log_value(self.graphs[1])-p.log_value(self.graphs[0])),9)
        claims,evidence=p.effective_evidence()
        proposal=EvidenceGuidedDAGProposal(claims,evidence,node_count=3)
        differences=[proposal.evidence_score(g)-p.log_value(g) for g in self.graphs]
        self.assertTrue(all(math.isclose(x,differences[0]) for x in differences))

    def test_negative_edges_do_not_exclude_indirect_paths(self):
        p=SourcePairEvidence('model/context')
        p.add(Claim('d','I','directed',0,0,1),.9)
        p.add(Claim('r','I','directed',0,1,0),.8)
        chain=graph_from_edges(3,[(0,2),(2,1)])
        self.assertEqual(p.groups[0,1].allowed_states,frozenset((0,)))
        self.assertAlmostEqual(p.log_value(chain),math.log(.8))

    def test_conflicting_factor_is_removed_in_both_weights_and_mutation(self):
        pool=(ClaimQuery('directed',0,1,'forward'),ClaimQuery('directed',1,0,'reverse'))
        mutation=[]
        def capture(*args):
            mutation.append((args[2],args[3]));return args[1]
        with patch('cd_llm.methods.active_query._rejuvenate',side_effect=capture):
            result=active_query_smc(dict.fromkeys(self.graphs,1/3),self.score,
                lambda q:QueryResponse(1,.9),query_pool=pool,budgets=(2,),confidence=.9,
                ess_threshold_fraction=.01,rejuvenation_steps=1,evidence_aggregation='per_source_pair',
                evidence_source_family='model/context')
        for g in self.graphs:self.assertAlmostEqual(result.distributions[2][g],1/3)
        self.assertTrue(result.steps[-1].source_pair_update['after']['contradiction'])
        self.assertIsNone(result.steps[-1].applied_claim)
        self.assertEqual(len(mutation),1)
        self.assertEqual(mutation[0][0],())
        self.assertEqual(mutation[0][1].nodes,())

    def test_active_nested_update_matches_one_joint_factor(self):
        pool=(ClaimQuery('directed',0,1,'forward'),ClaimQuery('adjacency',0,1,'adjacent'))
        result=active_query_smc(dict.fromkeys(self.graphs,1/3),self.score,
            lambda q:QueryResponse(1,.95 if q.kind=='directed' else .9),
            query_pool=pool,budgets=(2,),confidence=.9,rejuvenation_steps=0,
            ess_threshold_fraction=.01,evidence_aggregation='per_source_pair',evidence_source_family='model/context')
        self.assertAlmostEqual(result.distributions[2][self.graphs[1]],9/11)
        self.assertAlmostEqual(result.distributions[2][self.graphs[0]],1/11)
        self.assertAlmostEqual(result.distributions[2][self.graphs[2]],1/11)

    def test_abstention_does_not_replace_a_valid_factor(self):
        pool=(ClaimQuery('directed',0,1,'forward'),ClaimQuery('adjacency',0,1,'adjacent'))
        calls=[]
        def answer(q):
            calls.append(q)
            return QueryResponse(1,.9) if len(calls)==1 else QueryResponse(None,None)
        result=active_query_smc(dict.fromkeys(self.graphs,1/3),self.score,answer,
            query_pool=pool,budgets=(1,2),confidence=.9,rejuvenation_steps=0,
            ess_threshold_fraction=.01,evidence_aggregation='per_source_pair',evidence_source_family='model/context')
        self.assertEqual(result.distributions[1],result.distributions[2])
        self.assertEqual(result.abstention_count,1)

    def test_unsupported_hard_and_ancestral_settings_rejected(self):
        kwargs=dict(query_pool=(ClaimQuery('ancestry',0,1,'ancestor'),),budgets=(1,),evidence_aggregation='per_source_pair',evidence_source_family='source')
        with self.assertRaises(ValueError):active_query_smc(dict.fromkeys(self.graphs,1/3),self.score,lambda q:1,**kwargs)
        kwargs['query_pool']=(ClaimQuery('directed',0,1,'direct'),)
        with self.assertRaises(ValueError):active_query_smc(dict.fromkeys(self.graphs,1/3),self.score,lambda q:1,treatment='hard_constraint',**kwargs)


if __name__=='__main__':unittest.main()
