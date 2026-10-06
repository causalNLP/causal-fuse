# Long et al. neuropathic baseline

The retained Long et al. experiment evaluates the 20-node neuropathic subnetwork used by the paper. It learns an observational PC graph, queries ambiguous edge orientations through the baseline expert protocol, and reports the resulting graph distribution with the shared structural metrics.

```bash
../.venv/bin/cd-llm experiment neuropathic-long --help
```

The archived observations and truth graph are loaded through the same 20-node benchmark interface used by the other neuropathic baselines. Request records and numerical outputs belong under a timestamped workspace result directory.
