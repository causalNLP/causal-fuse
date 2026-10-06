"""Memory-conscious gPC execution without restricting its conditioning search."""
from itertools import combinations
from math import comb
from functools import lru_cache
import numpy as np
from causallearn.utils.cit import CIT
from cd_llm.baselines.guess2graph2026.paper_methods import expert_edge_order, neighbors, validate_guessed_skeleton


def unrank_combination(pool, size, rank):
    if not 0 <= size <= len(pool) or not 0 <= rank < comb(len(pool), size):
        raise ValueError("invalid combination rank")
    output, first = [], 0
    for remaining in range(size, 0, -1):
        for index in range(first, len(pool)-remaining+1):
            block = comb(len(pool)-index-1, remaining-1)
            if rank < block:
                output.append(pool[index]); first = index+1; break
            rank -= block
    return tuple(output)


def shuffled_subsets(pool, size, rng, materialize_limit=100000):
    count = comb(len(pool), size)
    if count <= materialize_limit:
        values = list(combinations(pool, size))
        rng.shuffle(values)
        yield from values
        return
    # Lazy Fisher-Yates draws a uniform permutation without first allocating C(n,k) tuples.
    swaps = {}
    for index in range(count):
        selected = int(rng.integers(index, count))
        rank = swaps.get(selected, selected)
        replacement = swaps.pop(index, index)
        if selected != index: swaps[selected] = replacement
        yield unrank_combination(pool, size, rank)


def gpc_skeleton(data, guess, rng, progress=None, resume=None):
    guess = validate_guessed_skeleton(guess, data.shape[1])
    cit = CIT(data, "gsq")
    # Eviction changes runtime only, not p-values or the random search order.
    @lru_cache(maxsize=100000)
    def independence(a,b,conditioning):
        value = cit(a,b,list(conditioning))
        if len(cit.pvalue_cache) > 100000:
            cit.pvalue_cache.clear()
        return value
    count = data.shape[1]
    current = np.ones((count,count), dtype=int)-np.eye(count,dtype=int)
    edges = expert_edge_order(list(combinations(range(count),2)),guess,rng)
    start, tests = 0, 0
    if resume is not None:
        current = np.array(resume["matrix"],dtype=int)
        assert resume["edges"] == [list(e) for e in edges]
        start, tests = resume["completed_edges"], resume["tests"]
        rng.bit_generator.state = resume["rng_state"]
    for index in range(start,len(edges)):
        left,right = edges[index]
        if current[left,right]:
            pools = [neighbors(current,left,exclude=right),neighbors(current,right,exclude=left)]
            pool = min(pools,key=len)
            removed = False
            for size in range(len(pool)+1):
                for conditioning in shuffled_subsets(pool,size,rng):
                    tests += 1
                    if independence(left,right,conditioning) > .05:
                        current[left,right] = current[right,left] = 0
                        removed = True
                        break
                if removed: break
        if progress:
            progress(dict(completed_edges=index+1, total_edges=len(edges), tests=tests,
                matrix=current.tolist(), edges=edges, rng_state=rng.bit_generator.state))
    return current
