import unittest
from dataclasses import asdict
from pathlib import Path
import tempfile
from unittest.mock import patch

from cd_llm.experiments.neuropathic_grid_interpretation import inference_arguments, make_tasks, validate_population


class GridInterpretationTests(unittest.TestCase):
    def test_full_matched_design(self):
        tasks=make_tasks()
        self.assertEqual(len(tasks),120)
        self.assertEqual(len({t["id"] for t in tasks}),120)
        for b in (20,50,100):
            selected=[t for t in tasks if t["budget"]==b]
            self.assertEqual(len(selected),40)
            self.assertEqual({(t["strategy"],t["sample_size"],t["seed"]) for t in selected},
                {(s,n,seed) for s in ("stated","consistency") for n in (200,1000) for seed in range(10)})

    def test_budgets_are_independent_and_seeds_match(self):
        selected=[t for t in make_tasks() if t["strategy"]=="stated" and t["sample_size"]==200 and t["seed"]==0]
        seeds=set()
        particles,weights=("sentinel",),(1.0,)
        for t in selected:
            a=inference_arguments(t,particles,weights,"literal")
            b=inference_arguments(t,particles,weights,"ancestry")
            self.assertEqual(a["budgets"],(t["budget"],))
            self.assertFalse(a["rejuvenate_after_every_query"])
            self.assertEqual(a["confidence"],0.9)
            self.assertEqual(a["rejuvenation_steps"],2)
            self.assertIs(a["initial_particles"],b["initial_particles"])
            self.assertEqual({k:v for k,v in a.items() if k!="direct_yes_interpretation"},
                             {k:v for k,v in b.items() if k!="direct_yes_interpretation"})
            seeds.add(a["seed"])
        self.assertEqual(len(seeds),1)

    def test_population_validation(self):
        population=dict(particles=[[(0,1),(1,2)]]*384,weights=[1/384]*384)
        graphs,weights=validate_population(population)
        self.assertEqual(len(graphs),384)
        self.assertAlmostEqual(weights.sum(),1)
        with self.assertRaises(ValueError):
            validate_population(dict(population,weights=[0.0]*384))
        with self.assertRaises(ValueError):
            validate_population(dict(population,particles=[[(0,1),(1,0)]]*384))

    def test_unregistered_treatment_rejected(self):
        task=make_tasks()[0]
        with self.assertRaises(ValueError):
            inference_arguments(task,(),(),"discard")
        with self.assertRaises(ValueError):
            inference_arguments(dict(task,budget=200),(),(),"literal")

    def test_partial_report_handles_no_affirmative_answers(self):
        from cd_llm.core.finite_support import Claim, graph_edges, graph_from_edges
        from cd_llm.experiments.common.artifacts import write_json
        from cd_llm.experiments.neuropathic_grid_interpretation_report import report
        task=make_tasks()[0]
        truth=graph_from_edges(20,directed=[(i,i+7) for i in range(7)]+[(i+7,14) for i in range(7)])
        empty=graph_from_edges(20)
        trace=[]
        for slot,(a,b) in enumerate([(0,j) for j in range(1,20)]+[(1,0)],1):
            trace.append(dict(budget=slot,query=dict(kind="directed",source=a,target=b,query_id=f"q{slot}"),
                answer=0,reported_confidence=0.7,applied_claim=asdict(Claim(f"c{slot}","I","directed",0,a,b))))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            write_json(root/'evaluation/truth_graph.json',dict(graph=graph_edges(truth)))
            for arm in ('literal','ancestry'):
                write_json(root/f"records/{arm}/{task['id']}.json",dict(task=task,trace=trace,
                    initialization_sha256="shared",answered_queries=20,
                    distribution=[dict(graph=graph_edges(empty),weight=1.0)]))
            with patch('cd_llm.experiments.neuropathic_grid_interpretation_report.verify',return_value=dict(tasks=[task])):
                summary=report(root)
            self.assertFalse(summary['complete'])
            self.assertEqual(summary['paired_comparisons'],1)
            self.assertTrue(all(r['direct_yes_accuracy'] is None for r in summary['summaries']))
            self.assertTrue(all(r['low'] is None for r in summary['differences']))


if __name__=="__main__":
    unittest.main()
