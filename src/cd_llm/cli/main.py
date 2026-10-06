from __future__ import annotations

import argparse
import importlib
import inspect
import sys
from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any


COMMANDS: dict[str, dict[str, str]] = {
    "data": {
        "generate-synthetic": "cd_llm.data.generation.synthetic",
        "generate-neuropathic": "cd_llm.data.generation.neuropathic",
        "visualize-synthetic": "cd_llm.data.generation.visualize",
    },
    "experiment": {
        "claim-fuse-smc": "cd_llm.experiments.claim_fuse_smc_sidecar",
        "claim-fuse-smc-eig": "cd_llm.experiments.claim_fuse_smc_eig",
        "synthetic-pairwise-expert-benchmark": "cd_llm.experiments.synthetic_pairwise_expert_benchmark",
        "synthetic-eca-original": "cd_llm.experiments.synthetic_eca_original",
        "synthetic-eca-confidence-cap": "cd_llm.experiments.synthetic_eca_confidence_cap",
        "synthetic-claim-robustness": "cd_llm.experiments.synthetic_claim_robustness",
        "synthetic-pc-grid-ablation": "cd_llm.experiments.synthetic_pc_grid_ablation",
        "latent-mag-validation": "cd_llm.experiments.latent_mag_validation",
        "neuropathic-decontamination": "cd_llm.experiments.neuropathic_decontamination",
        "neuropathic-decontamination-qwen": "cd_llm.experiments.neuropathic_decontamination_qwen",
        "neuropathic-new-prompt-smc": "cd_llm.experiments.neuropathic_new_prompt_smc",
        "neuropathic-terra-smc": "cd_llm.experiments.neuropathic_terra_smc",
        "neuropathic-causal-order-terra": "cd_llm.experiments.neuropathic_causal_order_terra",
        "neuropathic-gpc-dag-cpdag": "cd_llm.experiments.neuropathic_gpc_dag_cpdag",
        "neuropathic-bfs": "cd_llm.experiments.neuropathic_bfs2024",
        "neuropathic-guess2graph": "cd_llm.experiments.neuropathic_guess2graph",
        "neuropathic-long": "cd_llm.experiments.neuropathic_long2023",
        "neuropathic-scp": "cd_llm.experiments.neuropathic_scp2025",
        "alarm-claim-pools": "cd_llm.experiments.alarm_claim_pools",
        "alarm-claim-pools-local": "cd_llm.experiments.alarm_claim_pools_local",
        "alarm-claim-pools-consistency": "cd_llm.experiments.alarm_claim_pools_consistency",
        "alarm-matched-baselines": "cd_llm.experiments.alarm_matched_baselines",
        "alarm-gpc-dag": "cd_llm.experiments.alarm_gpc_dag",
        "alarm-gpc-cpdag-replay": "cd_llm.experiments.alarm_gpc_cpdag_replay",
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cd-llm",
        description="Run Claim-Fuse data and experiment commands.",
    )
    categories = parser.add_subparsers(dest="category", required=True)
    for category, commands in COMMANDS.items():
        category_parser = categories.add_parser(category, add_help=False)
        category_parser.add_argument("command", choices=tuple(commands))
    return parser


@contextmanager
def _argv(arguments: Sequence[str]):
    previous = sys.argv
    sys.argv = [previous[0], *arguments]
    try:
        yield
    finally:
        sys.argv = previous


def _run_module(module_name: str, arguments: Sequence[str]) -> int:
    module = importlib.import_module(module_name)
    entry = getattr(module, "main")
    parameters = inspect.signature(entry).parameters
    if parameters:
        result: Any = entry(arguments)
    else:
        with _argv(arguments):
            result = entry()
    return int(result) if result is not None else 0


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    namespace, remainder = parser.parse_known_args(arguments)
    module_name = COMMANDS[namespace.category][namespace.command]
    return _run_module(module_name, remainder)


if __name__ == "__main__":
    raise SystemExit(main())
