"""Classify graph queries against a directed reference graph."""


def reachability(count, edges):
    rows = [set() for _ in range(count)]
    for a, b in edges:
        rows[a].add(b)
    for k in range(count):
        for a in range(count):
            if k in rows[a]:
                rows[a].update(rows[k])
    if any(a in rows[a] for a in range(count)):
        raise ValueError("reference graph is cyclic")
    return rows


def classify(query, edges, closure):
    a, b = query["source"], query["target"]
    if query["kind"] == "directed":
        if (a, b) in edges:
            return "direct"
        return "indirect" if b in closure[a] else "no_forward_path"
    if query["kind"] == "adjacency":
        if (a, b) in edges or (b, a) in edges:
            return "adjacent"
        return "indirect_either_direction" if b in closure[a] or a in closure[b] else "no_path_either_direction"
    raise ValueError("unexpected question type")
