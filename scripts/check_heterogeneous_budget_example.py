"""Check the finite decision-tree witness in Theorem 2; no expert requests."""
import argparse
from functools import lru_cache
from pathlib import Path

from cd_llm.core.finite_support import graph_from_edges
from cd_llm.experiments.common.artifacts import write_json
from cd_llm.methods.active_query import dag_query_pool, query_values


def verify():
    # X,A,B,Y,Z,W have indices 0,...,5.
    base=[(0,1),(2,3)]
    graphs=tuple(graph_from_edges(6,directed=base+extra) for extra in
                 ([(1,3)],[(0,2)],[(4,5)],[]))
    result={}
    for label,families in (('local',('local_edge',)),('mixed',('local_edge','ancestry'))):
        queries=dag_query_pool(6,families)
        patterns=tuple(tuple(int(v) for v in query_values(graphs,q)) for q in queries)
        @lru_cache(None)
        def depth(support):
            if len(support)<=1:
                return 0
            costs=[]
            for pattern in patterns:
                yes=tuple(i for i in support if pattern[i])
                no=tuple(i for i in support if not pattern[i])
                if yes and no:
                    costs.append(1+max(depth(yes),depth(no)))
            return min(costs,default=float('inf'))
        result[label]=dict(minimum_worst_case_queries=depth(tuple(range(4))),
            unique_query_partitions=sorted(set(patterns)))
    assert result['local']['minimum_worst_case_queries']==3
    assert result['mixed']['minimum_worst_case_queries']==2
    return dict(passed=True,node_order=['X','A','B','Y','Z','W'],pools=result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    write_json(parser.parse_args().output,verify())
