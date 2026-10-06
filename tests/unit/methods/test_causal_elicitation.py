from __future__ import annotations

import json
import unittest
from dataclasses import asdict

import numpy as np

from cd_llm.core.finite_support import EvidenceGraph, graph_from_edges
from cd_llm.methods.causal_elicitation import (
    CausalReport, PairLikelihood, REPORTS, ResponseChannel,
    aggregate_reports, fuse_distribution,
    parse_report, report_factor, state_posterior, validate_factors,
)
from cd_llm.methods.evidence_guided import EvidenceGuidedDAGProposal, evidence_guided_tempered_smc
from cd_llm.methods.adaptive_pruning import GaussianBICScore


def calibration_row(state, index, relation='no_forward_direct', confidence=0.9):
    return dict(split='calibration', truth_kind='generating_graph', state=state, group_id=f'{state}-{index}',
                report=asdict(CausalReport(relation, confidence, f'{state}-{index}')))


def candidate_graphs():
    common = ((0, 3), (1, 3))
    return tuple(graph_from_edges(4, (*edges, *common)) for edges in (
        ((0, 1), (0, 2), (2, 1)), ((0, 2), (2, 1)),
        ((1, 0), (1, 2), (2, 0)), ((1, 2), (2, 0)), ((2, 0), (2, 1)),
    ))


class ReportTests(unittest.TestCase):
    def test_ancestry_does_not_imply_no_direct_edge(self):
        factor = report_factor(CausalReport('forward_ancestry', 0.8, 'paper'), 0, 1)
        posterior = state_posterior((0.4, 0.4, 0, 0.2, 0), factor)
        np.testing.assert_allclose(posterior, (8/17, 8/17, 0, 1/17, 0))
        self.assertEqual(factor.log_values[0], factor.log_values[1])

    def test_unknown_is_neutral_and_malformed_is_visible(self):
        for report in (CausalReport('unknown', None, 'p'), CausalReport('unknown', None, 'p', valid=False)):
            prior = (0.01, 0.09, 0.05, 0.15, 0.70)
            np.testing.assert_allclose(state_posterior(prior, report_factor(report, 0, 1)), prior)
        invalid = parse_report('{"relation":"forward_direct","confidence":0.9,"quote":"invented"}',
                               source_id='p', evidence='No information.', allowed=REPORTS)
        self.assertFalse(invalid.valid)
        self.assertFalse(invalid.usable)

    def test_full_json_and_exact_quotes_are_required(self):
        content = json.dumps(dict(relation='forward_direct', confidence=.9, quote='A changes B.'))
        self.assertTrue(parse_report(content, source_id='p', evidence='A changes B.', allowed=REPORTS).usable)
        self.assertFalse(parse_report('```json\n'+content+'\n```', source_id='p', evidence='A changes B.', allowed=REPORTS).valid)

    def test_order_disagreement_abstains_without_counting_independent_evidence(self):
        a = CausalReport('forward_direct', .9, 'p')
        b = CausalReport('no_forward_direct', .9, 'p')
        self.assertFalse(aggregate_reports((a,b)).usable)
        self.assertTrue(aggregate_reports((a,a)).usable)
        with self.assertRaises(ValueError):
            validate_factors((report_factor(a, 0, 1), report_factor(a, 0, 2)))


class ChannelTests(unittest.TestCase):
    def test_constant_negative_source_is_neutral_with_unequal_class_counts(self):
        rows = [calibration_row(s,i) for s,n in enumerate((10,20,30,40,90)) for i in range(n)]
        channel = ResponseChannel.fit(rows)
        factor = channel.factor(CausalReport('no_forward_direct', .9, 'new'), 0, 1)
        self.assertFalse(factor.informative)
        prior = (.05, .1, .05, .1, .7)
        np.testing.assert_allclose(state_posterior(prior, factor), prior)

    def test_informative_source_has_asymmetric_likelihoods(self):
        rows = [calibration_row(s,i, 'forward_direct' if s == 0 else 'no_forward_direct') for s in range(5) for i in range(30)]
        factor = ResponseChannel.fit(rows).factor(CausalReport('forward_direct', .9, 'new'), 0, 1)
        self.assertGreater(state_posterior((.2,)*5, factor)[0], .8)

    def test_unseen_symbols_are_neutral(self):
        rows = [calibration_row(s,i) for s in range(5) for i in range(10)]
        self.assertFalse(ResponseChannel.fit(rows).factor(CausalReport('reverse_direct', .6, 'new'), 0, 1).informative)

    def test_cannot_fit_passage_annotations_or_test_cases(self):
        rows = [calibration_row(s,0) for s in range(5)]
        rows[0]['split'] = 'test'
        with self.assertRaises(ValueError):
            ResponseChannel.fit(rows)
        rows[0]['split'] = 'calibration'
        rows[0]['truth_kind'] = 'passage_annotation'
        with self.assertRaises(ValueError):
            ResponseChannel.fit(rows)

    def test_duplicate_sources_and_missing_states_rejected(self):
        rows = [calibration_row(s,0) for s in range(5)]
        with self.assertRaises(ValueError):
            ResponseChannel.fit(rows + [rows[0]])
        with self.assertRaises(ValueError):
            ResponseChannel.fit(rows[:-1])


class InferenceTests(unittest.TestCase):
    def test_joint_factor_changes_mh_target_and_cache_without_duplication(self):
        factor = PairLikelihood(0, 1, (0, -1, -2, -3, -4), 'p')
        proposal = EvidenceGuidedDAGProposal((), EvidenceGraph('target', ()), node_count=4,
                                           gamma=0, pair_likelihoods=(factor,))
        for _ in range(3):
            for state, graph in enumerate(candidate_graphs()):
                proposal.evidence_states(graph)
                self.assertAlmostEqual(proposal.evidence_score(graph), -state)

    def test_tempered_smc_equals_exact_weighting_without_mutation(self):
        graphs = candidate_graphs()
        factor = report_factor(CausalReport('forward_ancestry', .8, 'p'), 0, 1)
        prior = dict(zip(graphs, (.4, .4, .05, .1, .05)))
        score = GaussianBICScore(np.random.default_rng(3).normal(size=(30,4)), max_parents=3)
        result = evidence_guided_tempered_smc(score, graphs, (), EvidenceGraph('target', ()), 2,
                                              initial_weights=tuple(prior.values()), temperatures=(0,.3,1),
                                              gamma=0, ess_threshold_fraction=.01, rejuvenation_steps=0,
                                              pair_likelihoods=(factor,))
        exact = fuse_distribution(prior, (factor,))
        np.testing.assert_allclose([result.distribution[g] for g in graphs], [exact[g] for g in graphs])

    def test_log_space_updates_preserve_zero_support(self):
        graphs = candidate_graphs()
        prior = {graphs[0]: 0, graphs[1]: 1}
        factor = PairLikelihood(0,1,(0,-10000,0,0,0),'p')
        self.assertEqual(fuse_distribution(prior,(factor,)), prior)
        np.testing.assert_allclose(state_posterior((0,1,0,0,0),factor),(0,1,0,0,0))

    def test_matched_data_only_control_runs_the_same_final_mutation(self):
        graphs = candidate_graphs()
        score = GaussianBICScore(np.random.default_rng(7).normal(size=(30,4)), max_parents=3)
        args = dict(temperatures=(0,1),gamma=0,ess_threshold_fraction=.01,rejuvenation_steps=2,
                    run_without_evidence=True)
        control = evidence_guided_tempered_smc(score,graphs,(),EvidenceGraph('target',()),3,**args)
        unknown = report_factor(CausalReport('unknown',None,'p'),0,1)
        neutral = evidence_guided_tempered_smc(score,graphs,(),EvidenceGraph('target',()),3,
                                              pair_likelihoods=(unknown,),**args)
        self.assertGreater(control.attempted_moves,0)
        self.assertEqual(control.distribution,neutral.distribution)




if __name__ == '__main__':
    unittest.main()
