#!/usr/bin/env python3
"""Run the complete experiment and export workflows used in Section 4."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from typing import Callable, Mapping, Sequence


CODE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = CODE_ROOT.parent
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class Command:
    label: str
    argv: tuple[str, ...]
    env: tuple[tuple[str, str], ...] = ()


def _cli(*args: object) -> tuple[str, ...]:
    return (sys.executable, "-m", "cd_llm.cli.main", "experiment", *(str(arg) for arg in args))


def _script(name: str, *args: object) -> tuple[str, ...]:
    return (sys.executable, str(CODE_ROOT / "scripts" / name), *(str(arg) for arg in args))


def _value(env: Mapping[str, str], name: str, dry_run: bool) -> str:
    value = env.get(name)
    if value:
        return value
    if dry_run:
        return f"${name}"
    raise ValueError(f"{name} must be set for this workflow")


def _paths(run_id: str) -> tuple[Path, Path, Path]:
    result_root = WORKSPACE_ROOT / "results" / run_id
    generated_root = WORKSPACE_ROOT / "generated" / "runtime" / run_id
    export_root = result_root / "paper_exports"
    return result_root, generated_root, export_root


def synthetic_claim_pools(run_id: str, env: Mapping[str, str], dry_run: bool) -> list[Command]:
    del env, dry_run
    result_root, _, _ = _paths(run_id)
    commands = [Command("prepare", _cli("synthetic-pc-grid-ablation", "--stage", "prepare", "--run-id", run_id))]
    commands.extend(Command(f"compute-{task}", _cli("synthetic-pc-grid-ablation", "--stage", "compute", "--run-id", run_id, "--task-index", task)) for task in range(180))
    commands.append(Command("aggregate", _cli("synthetic-pc-grid-ablation", "--stage", "aggregate", "--run-id", run_id)))
    commands.append(Command("figure", (sys.executable, str(CODE_ROOT / "figures" / "synthetic_pc_grid_ablation.py"), "--result-root", str(result_root / "synthetic_pc_grid_ablation"))))
    return commands


def neuropathic_claim_fuse_qwen(run_id: str, env: Mapping[str, str], dry_run: bool) -> list[Command]:
    _value(env, "OPENROUTER_API_KEY", dry_run)
    result_root, _, _ = _paths(run_id)
    source = result_root / "neuropathic_new_prompt_smc"
    commands = [Command(stage, _cli("neuropathic-new-prompt-smc", stage, "--run-id", run_id)) for stage in ("prepare", "pilot", "collect", "audit")]
    commands.extend(Command(f"qwen-block-{block}", _cli("neuropathic-new-prompt-smc", "run-block", "--run-id", run_id, "--block", block)) for block in range(2))
    commands.append(Command("qwen-validate-pilot", _cli("neuropathic-new-prompt-smc", "validate-pilot", "--run-id", run_id)))
    commands.extend(Command(f"qwen-block-{block}", _cli("neuropathic-new-prompt-smc", "run-block", "--run-id", run_id, "--block", block)) for block in range(2, 20))
    commands.append(Command("qwen-report", _cli("neuropathic-new-prompt-smc", "report", "--run-id", run_id)))
    commands.append(Command("cap-prepare", _cli("neuropathic-confidence-cap", "prepare", "--run-id", run_id, "--source-root", source)))
    commands.extend(Command(f"cap-block-{block}", _cli("neuropathic-confidence-cap", "run-block", "--run-id", run_id, "--block", block)) for block in range(2))
    commands.append(Command("cap-validate-pilot", _cli("neuropathic-confidence-cap", "validate-pilot", "--run-id", run_id)))
    commands.extend(Command(f"cap-block-{block}", _cli("neuropathic-confidence-cap", "run-block", "--run-id", run_id, "--block", block)) for block in range(2, 20))
    commands.append(Command("cap-report", _cli("neuropathic-confidence-cap", "report", "--run-id", run_id)))
    return commands


def neuropathic_claim_fuse_terra(run_id: str, env: Mapping[str, str], dry_run: bool) -> list[Command]:
    _value(env, "OPENROUTER_API_KEY", dry_run)
    result_root, _, _ = _paths(run_id)
    commands = [
        Command("prepare", _cli("neuropathic-terra-smc", "prepare", "--run-id", run_id, "--source-root", result_root / "neuropathic_confidence_cap")),
        Command("collect", _cli("neuropathic-terra-smc", "collect", "--run-id", run_id)),
        Command("audit", _cli("neuropathic-terra-smc", "audit", "--run-id", run_id)),
    ]
    commands.extend(Command(f"block-{block}", _cli("neuropathic-terra-smc", "run-block", "--run-id", run_id, "--block", block)) for block in range(2))
    commands.append(Command("validate-pilot", _cli("neuropathic-terra-smc", "validate-pilot", "--run-id", run_id)))
    commands.extend(Command(f"block-{block}", _cli("neuropathic-terra-smc", "run-block", "--run-id", run_id, "--block", block)) for block in range(2, 20))
    commands.append(Command("report", _cli("neuropathic-terra-smc", "report", "--run-id", run_id)))
    return commands


def neuropathic_gpc(run_id: str, env: Mapping[str, str], dry_run: bool) -> list[Command]:
    base_url = _value(env, "QWEN_BASE_URL", dry_run)
    _value(env, "OPENROUTER_API_KEY", dry_run)
    result_root, _, _ = _paths(run_id)
    root = result_root / "neuropathic_gpc_dag_cpdag"
    commands = [
        Command("prepare", _cli("neuropathic-gpc-dag-cpdag", "prepare", "--root", root)),
        Command("collect-qwen", _cli("neuropathic-gpc-dag-cpdag", "collect-qwen", "--root", root, "--base-url", base_url)),
        Command("collect-terra", _cli("neuropathic-gpc-dag-cpdag", "collect-terra", "--root", root)),
        Command("audit", _cli("neuropathic-gpc-dag-cpdag", "audit", "--root", root)),
    ]
    commands.extend(Command(f"infer-{task}", _cli("neuropathic-gpc-dag-cpdag", "infer", "--root", root, "--index", task)) for task in range(40))
    commands.append(Command("report", _cli("neuropathic-gpc-dag-cpdag", "report", "--root", root)))
    return commands


def neuropathic_causal_order_qwen(run_id: str, env: Mapping[str, str], dry_run: bool) -> list[Command]:
    base_url = _value(env, "QWEN_BASE_URL", dry_run)
    checkpoint = _value(env, "QWEN_CHECKPOINT", dry_run)
    executable = env.get("VLLM_EXECUTABLE") or shutil.which("vllm")
    if not executable:
        if not dry_run:
            raise ValueError("vllm is not on PATH; set VLLM_EXECUTABLE to its executable path")
        executable = "$VLLM_EXECUTABLE"
    result_root, generated_root, _ = _paths(run_id)
    commands = [Command(f"baseline-{task}", _cli("neuropathic-causal-order", "--stage", "baseline", "--run-id", run_id, "--task-index", task, "--data-root", result_root / "neuropathic_new_prompt_smc" / "inputs")) for task in range(20)]
    commands.extend([
        Command("prepare-manifest", _cli("neuropathic-causal-order", "--stage", "prepare-manifest", "--run-id", run_id, "--image-path", executable, "--checkpoint-path", checkpoint)),
        Command("qwen", _cli("neuropathic-causal-order", "--stage", "qwen", "--run-id", run_id, "--runtime-manifest", generated_root / "qwen35_neuropathic20_manifest.json", "--base-url", base_url)),
        Command("aggregate", _cli("neuropathic-causal-order", "--stage", "aggregate", "--run-id", run_id)),
    ])
    return commands


def neuropathic_baselines(run_id: str, env: Mapping[str, str], dry_run: bool) -> list[Command]:
    base_url = _value(env, "QWEN_BASE_URL", dry_run)
    _value(env, "OPENROUTER_API_KEY", dry_run)
    result_root, _, _ = _paths(run_id)
    seeds = tuple(str(seed) for seed in range(10))
    common = ("--run-id", run_id, "--seeds", *seeds, "--sample-sizes", "200", "1000", "--temperature", "0.7")
    commands = [
        Command("bfs-qwen", _cli("neuropathic-bfs", *common, "--model", "Qwen/Qwen3.5-9B", "--base-url", base_url, "--output-dir", result_root / "neuropathic_bfs_qwen")),
        Command("bfs-terra", _cli("neuropathic-bfs", *common, "--model", "openai/gpt-5.6-terra", "--base-url", "https://openrouter.ai/api/v1", "--output-dir", result_root / "neuropathic_bfs_terra")),
        Command("long-qwen", _cli("neuropathic-long", "--run-id", run_id, "--seeds", *seeds, "--sample-sizes", "200", "1000", "--strategy", "srisk", "--tolerance", "0.1", "--max-extensions", "50000", "--prompt-seed", "0", "--model", "Qwen/Qwen3.5-9B", "--base-url", base_url, "--output-dir", result_root / "neuropathic_long_qwen")),
    ]
    for sample_size in (200, 1000):
        commands.append(Command(f"scp-qwen-{sample_size}", _cli(
            "neuropathic-scp", "--run-id", run_id, "--levels", "0", "--patterns", "2",
            "--module-size", "20", "--seeds", *seeds, "--sample-size", sample_size,
            "--bootstrap-samples", "100", "--alpha", "0.05", "--forbid-threshold", "0.05",
            "--require-threshold", "0.95", "--temperature", "0.7", "--probability-samples", "5",
            "--top-logprobs", "5", "--probability-protocol", "paper_logprobs", "--llm-model",
            "Qwen/Qwen3.5-9B", "--llm-base-url", base_url, "--acknowledge-neuropathic-adaptation",
            "--generate-missing", "--data-dir", result_root / "neuropathic_scp_data",
            "--output-dir", result_root / f"neuropathic_scp_qwen_n{sample_size}",
        )))
    commands.extend(Command(f"terra-order-{stage}", _cli("neuropathic-causal-order-terra", stage, "--run-id", run_id)) for stage in ("prepare", "pilot", "collect", "audit", "integrate", "report"))
    return commands


def alarm_claim_fuse(run_id: str, env: Mapping[str, str], dry_run: bool) -> list[Command]:
    base_url = _value(env, "QWEN_BASE_URL", dry_run)
    result_root, _, _ = _paths(run_id)
    local_env = (("ALARM_LOCAL_BASE_URL", base_url),)
    commands = [Command(f"stated-{stage}", _cli("alarm-claim-pools-local", stage, "--run-id", run_id), local_env) for stage in ("prepare", "runtime-check", "collect", "audit")]
    commands.extend(Command(f"stated-block-{block}", _cli("alarm-claim-pools-local", "run-block", "--run-id", run_id, "--block", block), local_env) for block in range(2))
    commands.append(Command("stated-validate-pilot", _cli("alarm-claim-pools-local", "validate-pilot", "--run-id", run_id), local_env))
    commands.extend(Command(f"stated-block-{block}", _cli("alarm-claim-pools-local", "run-block", "--run-id", run_id, "--block", block), local_env) for block in range(2, 20))
    commands.append(Command("stated-report", _cli("alarm-claim-pools-local", "report", "--run-id", run_id), local_env))
    commands.append(Command("consistency-prepare", _cli("alarm-claim-pools-consistency", "prepare", "--run-id", run_id, "--source-root", result_root / "alarm_claim_pools_local"), local_env))
    commands.append(Command("consistency-pilot", _cli("alarm-claim-pools-consistency", "pilot", "--run-id", run_id), local_env))
    commands.extend(Command(f"consistency-collect-{shard}", _cli("alarm-claim-pools-consistency", "collect", "--run-id", run_id, "--shard", shard), local_env) for shard in range(4))
    commands.append(Command("consistency-merge", _cli("alarm-claim-pools-consistency", "merge", "--run-id", run_id), local_env))
    commands.extend(Command(f"consistency-block-{block}", _cli("alarm-claim-pools-consistency", "run-block", "--run-id", run_id, "--block", block), local_env) for block in range(2))
    commands.append(Command("consistency-validate-pilot", _cli("alarm-claim-pools-consistency", "validate-inference-pilot", "--run-id", run_id), local_env))
    commands.extend(Command(f"consistency-block-{block}", _cli("alarm-claim-pools-consistency", "run-block", "--run-id", run_id, "--block", block), local_env) for block in range(2, 20))
    commands.append(Command("consistency-report", _cli("alarm-claim-pools-consistency", "report", "--run-id", run_id), local_env))
    return commands


def alarm_baselines(run_id: str, env: Mapping[str, str], dry_run: bool) -> list[Command]:
    base_url = _value(env, "QWEN_BASE_URL", dry_run)
    result_root, _, _ = _paths(run_id)
    root = result_root / "alarm_matched_baselines"
    child_env = (("ALARM_MATCHED_BASE_URL", base_url),)
    commands = [Command("prepare", _cli("alarm-matched-baselines", "prepare", "--root", root, "--source-root", result_root / "alarm_claim_pools_local"), child_env)]
    commands.extend(Command(f"bootstrap-{condition}", _cli("alarm-matched-baselines", "bootstrap", "--root", root, "--condition", condition), child_env) for condition in range(20))
    commands.append(Command("pilot", _cli("alarm-matched-baselines", "pilot", "--root", root), child_env))
    for method in ("bfs", "long", "order", "scp"):
        commands.extend(Command(f"collect-{method}-{condition}", _cli("alarm-matched-baselines", "collect", "--root", root, "--method", method, "--condition", condition), child_env) for condition in range(20))
    commands.append(Command("audit", _cli("alarm-matched-baselines", "audit", "--root", root), child_env))
    for condition in range(20):
        commands.append(Command(f"fit-{condition}", _cli("alarm-matched-baselines", "fit", "--root", root, "--condition", condition), child_env))
        commands.append(Command(f"gpc-{condition}", _cli("alarm-matched-baselines", "gpc", "--root", root, "--condition", condition), child_env))
    commands.append(Command("report", _cli("alarm-matched-baselines", "report", "--root", root), child_env))
    return commands


def export_main_results(run_id: str, env: Mapping[str, str], dry_run: bool) -> list[Command]:
    del env, dry_run
    result_root, _, export_root = _paths(run_id)
    gpc = result_root / "neuropathic_gpc_dag_cpdag"
    alarm = result_root / "alarm_matched_baselines"
    commands = [
        Command("rq2", _script("export_rq2_figure_data.py", "--synthetic", result_root / "synthetic_pc_grid_ablation" / "summary.csv", "--neuropathic", result_root / "neuropathic_confidence_cap" / "summary.csv", "--alarm", result_root / "alarm_claim_pools_consistency" / "combined_summary.csv", "--output-dir", export_root / "rq2")),
        Command("claim-types", _script("export_claim_type_table.py", "--neuropathic-qwen", result_root / "neuropathic_confidence_cap", "--neuropathic-terra", result_root / "neuropathic_terra_smc", "--alarm-stated", result_root / "alarm_claim_pools_local", "--alarm-consistency", result_root / "alarm_claim_pools_consistency", "--output", export_root / "claim_type_accuracy_proportion.csv")),
    ]
    rq1 = ["--neuropathic-source", f"data={result_root / 'neuropathic_causal_order_subnetwork' / 'summary.csv'}", "--neuropathic-source", f"qwen={result_root / 'neuropathic_causal_order_subnetwork' / 'summary.csv'}", "--neuropathic-source", f"qwen={gpc / 'summary.csv'}", "--neuropathic-source", f"qwen:BFS+Pearson={result_root / 'neuropathic_bfs_qwen' / 'summary.csv'}", "--neuropathic-source", f"qwen:PC+Long-Srisk={result_root / 'neuropathic_long_qwen' / 'summary.csv'}", "--neuropathic-source", f"qwen:SCP Pattern 2={result_root / 'neuropathic_scp_qwen_n200' / 'summary.csv'}", "--neuropathic-source", f"qwen:SCP Pattern 2={result_root / 'neuropathic_scp_qwen_n1000' / 'summary.csv'}", "--neuropathic-source", f"qwen={result_root / 'neuropathic_confidence_cap' / 'summary.csv'}", "--neuropathic-source", f"terra={gpc / 'summary.csv'}", "--neuropathic-source", f"terra:BFS+Pearson={result_root / 'neuropathic_bfs_terra' / 'summary.csv'}", "--neuropathic-source", f"terra={result_root / 'neuropathic_causal_order_terra' / 'summary.csv'}", "--neuropathic-source", f"terra={result_root / 'neuropathic_terra_smc' / 'summary.csv'}", "--alarm-source", f"data={alarm / 'summary.csv'}", "--alarm-source", f"qwen={alarm / 'summary.csv'}", "--alarm-source", f"qwen={result_root / 'alarm_claim_pools_consistency' / 'combined_summary.csv'}", "--output-dir", export_root / "rq1"]
    commands.append(Command("rq1", _script("build_rq1_tables.py", *rq1)))
    return commands


def export_cost_results(run_id: str, env: Mapping[str, str], dry_run: bool) -> list[Command]:
    del env, dry_run
    result_root, _, export_root = _paths(run_id)
    return [Command("rq3", _script(
        "export_all_method_cost_data.py",
        "--neuropathic-claim-fuse", result_root / "neuropathic_confidence_cap",
        "--neuropathic-cache", result_root / "neuropathic_new_prompt_smc",
        "--neuropathic-gpc", result_root / "neuropathic_gpc_dag_cpdag",
        "--neuropathic-bfs", result_root / "neuropathic_bfs_qwen",
        "--neuropathic-long", result_root / "neuropathic_long_qwen",
        "--neuropathic-order", result_root / "neuropathic_causal_order_subnetwork",
        "--neuropathic-scp", result_root / "neuropathic_scp_qwen_n200",
        "--neuropathic-scp", result_root / "neuropathic_scp_qwen_n1000",
        "--alarm-stated", result_root / "alarm_claim_pools_local",
        "--alarm-consistency", result_root / "alarm_claim_pools_consistency",
        "--alarm-baselines", result_root / "alarm_matched_baselines",
        "--output-dir", export_root / "rq3",
    ))]


Workflow = Callable[[str, Mapping[str, str], bool], list[Command]]
WORKFLOWS: dict[str, Workflow] = {
    "synthetic-claim-pools": synthetic_claim_pools,
    "neuropathic-claim-fuse-qwen": neuropathic_claim_fuse_qwen,
    "neuropathic-claim-fuse-terra": neuropathic_claim_fuse_terra,
    "neuropathic-gpc": neuropathic_gpc,
    "neuropathic-causal-order-qwen": neuropathic_causal_order_qwen,
    "neuropathic-baselines": neuropathic_baselines,
    "alarm-claim-fuse": alarm_claim_fuse,
    "alarm-baselines": alarm_baselines,
    "export-main-results": export_main_results,
    "export-cost-results": export_cost_results,
}


def run_commands(commands: Sequence[Command], *, workflow: str, run_id: str, dry_run: bool, base_env: Mapping[str, str] | None = None) -> None:
    environment = dict(os.environ if base_env is None else base_env)
    source_root = str(CODE_ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (source_root, environment.get("PYTHONPATH")) if value
    )
    log_root = WORKSPACE_ROOT / "out" / run_id / workflow
    for index, command in enumerate(commands, start=1):
        display = shlex.join(command.argv)
        if dry_run:
            print(f"[{index:03d}/{len(commands):03d}] {command.label}: {display}")
            continue
        log_root.mkdir(parents=True, exist_ok=True)
        log_path = log_root / f"{index:03d}-{command.label}.log"
        child_env = environment | dict(command.env)
        print(f"[{index:03d}/{len(commands):03d}] {command.label}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command.argv, cwd=CODE_ROOT, env=child_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, command.argv)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workflow", choices=WORKFLOWS)
    parser.add_argument("--run-id", required=True, help="shared identifier used under ../results and ../out")
    parser.add_argument("--dry-run", action="store_true", help="print every child command without running it")
    args = parser.parse_args(argv)
    if not RUN_ID_PATTERN.fullmatch(args.run_id):
        parser.error("--run-id must contain only letters, digits, dots, underscores, and hyphens")
    try:
        commands = WORKFLOWS[args.workflow](args.run_id, os.environ, args.dry_run)
        run_commands(commands, workflow=args.workflow, run_id=args.run_id, dry_run=args.dry_run)
    except (ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return error.returncode if isinstance(error, subprocess.CalledProcessError) else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
