# Reproduction inputs

These three archives contain only fixed experiment inputs: observations, masked variable metadata, reference graphs used for evaluation, prompt templates, and data-only initial particle populations. They contain no model responses, response cache, experiment record, numerical result, or paper figure.

`protocol.json` pins the dataset seeds, hosted routing settings, local model revision, precision, and serving version. The experiment `prepare` stages verify the archive SHA-256 values before extracting them into a timestamped workspace-level result directory.

The hosted model entries pin the public model slug, provider route, request seed, temperature, reasoning setting, and output limit. A hosted provider may update the weights behind the same slug. The local ALARM run is the path that pins the exact Qwen checkpoint revision.
