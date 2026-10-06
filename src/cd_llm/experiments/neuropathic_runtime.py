from __future__ import annotations

import argparse
import json
import resource
import signal
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from cd_llm.experiments.common.artifacts import run_id, sha256, write_json
from cd_llm.core.cpc import fit_cpc
from cd_llm.core.finite_support import Endpoint, PartialGraph
from cd_llm.methods.fast_bdeu import patched_causal_learn_bdeu
from cd_llm.paths import WORKSPACE_ROOT


SAMPLE_SIZE = 1000
LEVEL = 0
SEED = 0
NODE_COUNT = 222
CPC_ALPHA = 0.01
BDEU_EQUIVALENT_SAMPLE_SIZE = 1
BDEU_STRUCTURE_PRIOR = 1

DEFAULT_DATA_PATH = (
    WORKSPACE_ROOT
    / "generated"
    / "neuropathic"
    / "neuropathic_n1000_level0_seed0.csv"
)
DEFAULT_METADATA_PATH = DEFAULT_DATA_PATH.with_suffix(".json")
DEFAULT_OUTPUT_ROOT = WORKSPACE_ROOT / "results"


class RuntimeLimitExceeded(TimeoutError):
    pass


@dataclass(frozen=True)
class DatasetSummary:
    data_path: str
    metadata_path: str
    data_sha256: str
    sample_size: int
    node_count: int
    level: int
    seed: int
    minimum_prevalence: float
    median_prevalence: float
    maximum_prevalence: float
    variables_below_one_percent: int
    variables_below_five_percent: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Time binary-data CPC or GES on the complete 222-node neuropathic "
            "simulator dataset."
        )
    )
    parser.add_argument("--method", choices=("cpc", "ges"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=0,
        help="Optional in-process runtime limit. Zero disables the limit.",
    )
    return parser


def load_binary_dataset(
    data_path: Path,
    metadata_path: Path,
) -> tuple[np.ndarray, list[str], DatasetSummary]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_metadata = {
        "sample_size": SAMPLE_SIZE,
        "level": LEVEL,
        "seed": SEED,
        "observed_indices": list(range(NODE_COUNT)),
        "hidden_indices": [],
    }
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"metadata {key!r} differs: {metadata.get(key)!r} versus {expected!r}"
            )

    frame = pd.read_csv(data_path, index_col=0)
    if frame.shape != (SAMPLE_SIZE, NODE_COUNT):
        raise ValueError(
            f"dataset shape differs: {frame.shape} versus {(SAMPLE_SIZE, NODE_COUNT)}"
        )
    data = frame.to_numpy(dtype=np.int64, copy=True)
    if np.any((data != 0) & (data != 1)):
        raise ValueError("neuropathic runtime benchmark requires raw binary 0/1 data")
    prevalence = np.mean(data, axis=0)
    summary = DatasetSummary(
        data_path=str(data_path.resolve()),
        metadata_path=str(metadata_path.resolve()),
        data_sha256=sha256(data_path),
        sample_size=SAMPLE_SIZE,
        node_count=NODE_COUNT,
        level=LEVEL,
        seed=SEED,
        minimum_prevalence=float(np.min(prevalence)),
        median_prevalence=float(np.median(prevalence)),
        maximum_prevalence=float(np.max(prevalence)),
        variables_below_one_percent=int(np.sum(prevalence < 0.01)),
        variables_below_five_percent=int(np.sum(prevalence < 0.05)),
    )
    return data, [str(column) for column in frame.columns], summary


def run_cpc(
    data: np.ndarray,
    node_names: Sequence[str],
    *,
    alpha: float = CPC_ALPHA,
    runner: Callable[..., Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    result = fit_cpc(
        data,
        alpha=alpha,
        node_names=list(node_names),
        variable_type="discrete",
        backend_runner=runner,
    )
    return result.graph, {
        "algorithm": "tetrad_cpc",
        "conditional_independence_test": "g_square",
        "alpha": alpha,
        "standardized": False,
        "diagnostics": asdict(result.diagnostics),
    }


def run_pc(
    data: np.ndarray,
    node_names: Sequence[str],
    *,
    alpha: float = CPC_ALPHA,
    show_progress: bool = True,
    runner: Callable[..., Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Compatibility alias for callers migrating from the legacy PC adapter."""

    del show_progress
    return run_cpc(data, node_names, alpha=alpha, runner=runner)


def run_ges(
    data: np.ndarray,
    node_names: Sequence[str],
    *,
    runner: Callable[..., Any] | None = None,
    score_backend: Literal["reference", "prefactored"] = "reference",
) -> tuple[Any, dict[str, Any]]:
    if score_backend not in ("reference", "prefactored"):
        raise ValueError("unknown GES BDeu score backend")
    if runner is not None and score_backend != "reference":
        raise ValueError("a custom GES runner only supports the reference backend")
    if runner is None:
        from causallearn.search.ScoreBased.GES import ges as runner

    score_diagnostics: dict[str, Any] | None = None
    if score_backend == "reference":
        record = runner(
            data,
            score_func="local_score_BDeu",
            maxP=None,
            node_names=list(node_names),
        )
    else:
        with patched_causal_learn_bdeu(
            data,
            equivalent_sample_size=BDEU_EQUIVALENT_SAMPLE_SIZE,
            structure_prior=BDEU_STRUCTURE_PRIOR,
        ) as scorer:
            record = runner(
                data,
                score_func="local_score_BDeu",
                maxP=None,
                node_names=list(node_names),
            )
            score_diagnostics = scorer.cache_info()
    return record["G"], {
        "algorithm": "ges",
        "score": "local_score_BDeu",
        "score_backend": score_backend,
        "score_diagnostics": score_diagnostics,
        "equivalent_sample_size": BDEU_EQUIVALENT_SAMPLE_SIZE,
        "structure_prior": BDEU_STRUCTURE_PRIOR,
        "max_parents": None,
        "standardized": False,
        "final_score": float(record["score"]),
    }


def _timeout_handler(_signum: int, _frame: Any) -> None:
    raise RuntimeLimitExceeded("runtime limit exceeded")


def _package_versions() -> dict[str, str | None]:
    values: dict[str, str | None] = {}
    for package in (
        "causal-learn",
        "JPype1",
        "numpy",
        "pandas",
        "py-tetrad",
        "scipy",
    ):
        try:
            values[package] = version(package)
        except PackageNotFoundError:
            values[package] = None
    return values


def benchmark_method(
    method: str,
    data: np.ndarray,
    node_names: Sequence[str],
    *,
    timeout_seconds: int,
) -> tuple[dict[str, Any], int]:
    if timeout_seconds < 0:
        raise ValueError("timeout seconds must be nonnegative")
    previous_handler = signal.getsignal(signal.SIGALRM)
    started = time.perf_counter()
    status = "succeeded"
    error_type = None
    error_message = None
    graph = None
    settings: dict[str, Any] = {}
    try:
        if timeout_seconds:
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(timeout_seconds)
        if method == "cpc":
            graph, settings = run_cpc(data, node_names)
        elif method == "ges":
            graph, settings = run_ges(data, node_names)
        else:
            raise ValueError(f"unsupported method: {method}")
    except RuntimeLimitExceeded as error:
        status = "timed_out"
        error_type = type(error).__name__
        error_message = str(error)
    except Exception as error:
        status = "failed"
        error_type = type(error).__name__
        error_message = str(error)
    finally:
        if timeout_seconds:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous_handler)

    runtime_seconds = time.perf_counter() - started
    if graph is None:
        edges = None
    elif isinstance(graph, PartialGraph):
        edges = [
            {
                "first": first,
                "second": second,
                "first_mark": int(graph.mark(first, second)),
                "second_mark": int(graph.mark(second, first)),
            }
            for first in range(graph.node_count)
            for second in range(first + 1, graph.node_count)
            if graph.mark(first, second) != Endpoint.NULL
        ]
    else:
        edges = [str(edge) for edge in graph.get_graph_edges()]
    payload = {
        "method": method,
        "status": status,
        "runtime_seconds": runtime_seconds,
        "timeout_seconds": timeout_seconds,
        "edge_count": len(edges) if edges is not None else None,
        "edges": edges,
        "settings": settings,
        "error_type": error_type,
        "error_message": error_message,
        "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    return payload, 0 if status == "succeeded" else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    selected_run_id = run_id(args.run_id)
    data, node_names, dataset_summary = load_binary_dataset(
        args.data_path,
        args.metadata_path,
    )
    result, exit_code = benchmark_method(
        args.method,
        data,
        node_names,
        timeout_seconds=args.timeout_seconds,
    )
    payload = {
        "study": "neuropathic_cpc_ges_runtime",
        "run_id": selected_run_id,
        "dataset": asdict(dataset_summary),
        "input_transformation": "none_raw_integer_binary",
        "package_versions": _package_versions(),
        **result,
    }
    output_path = (
        args.output_root
        / selected_run_id
        / "neuropathic_runtime"
        / f"{args.method}.json"
    )
    write_json(output_path, payload)
    print(
        f"method={args.method} status={payload['status']} "
        f"runtime_seconds={payload['runtime_seconds']:.3f} output={output_path}",
        flush=True,
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
