# BFS neuropathic baseline

The retained BFS experiment evaluates the 20-node neuropathic subnetwork used by the paper. It uses the archived nested observations, fixed variable descriptions, and the same structural metrics as the other point-estimate baselines.

```bash
../.venv/bin/cd-llm experiment neuropathic-bfs --help
```

The driver records the exact model request, response cache entry, parsed graph, token usage, runtime, and structural metrics. Generated artifacts belong under a timestamped workspace result directory.
