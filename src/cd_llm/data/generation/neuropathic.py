from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import List, Sequence

import numpy as np

from cd_llm.paths import RESOURCE_ROOT, WORKSPACE_ROOT


DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "generated" / "neuropathic"
DEFAULT_MODEL_PATH = RESOURCE_ROOT / "neuropathic" / "bnm.pickle"


CANONICAL_VARIABLE_NAMES: List[str] = [
    "DLS C1-C2", "DLS C2-C3", "DLS C3-C4", "DLS C4-C5", "DLS C5-C6", "DLS C6-C7", "DLS C7-C8", "DLS C8-T1",
    "DLS L1-L2", "DLS L2-L3", "DLS L3-L4", "DLS L4-L5", "DLS L5-S1", "DLS S1-S2", "DLS T1-T2", "DLS T10-T11",
    "DLS T11-T12", "DLS T12-L1", "DLS T2-T3", "DLS T3-T4", "DLS T4-T5", "DLS T5-T6", "DLS T6-T7", "DLS T7-T8",
    "DLS T8-T9", "DLS T9-T10", "Kraniocervikal ledskada", "L C2 Radikulopati", "R C2 Radikulopati",
    "L C3 Radikulopati", "R C3 Radikulopati", "L C4 Radikulopati", "R C4 Radikulopati", "L C5 Radikulopati",
    "R C5 Radikulopati", "L C6 Radikulopati", "R C6 Radikulopati", "L C7 Radikulopati", "R C7 Radikulopati",
    "L C8 Radikulopati", "R C8 Radikulopati", "L T1 Radikulopati", "R T1 Radikulopati", "L T2 Radikulopati",
    "R T2 Radikulopati", "L T3 Radikulopati", "R T3 Radikulopati", "L T4 Radikulopati", "R T4 Radikulopati",
    "L T5 Radikulopati", "R T5 Radikulopati", "L T6 Radikulopati", "R T6 Radikulopati", "L T7 Radikulopati",
    "R T7 Radikulopati", "L T8 Radikulopati", "R T8 Radikulopati", "L T9 Radikulopati", "R T9 Radikulopati",
    "L T10 Radikulopati", "R T10 Radikulopati", "L T11 Radikulopati", "R T11 Radikulopati", "L T12 Radikulopati",
    "R T12 Radikulopati", "L L1 Radikulopati", "R L1 Radikulopati", "L L2 Radikulopati", "R L2 Radikulopati",
    "L L3 Radikulopati", "R L3 Radikulopati", "L L4 Radikulopati", "R L4 Radikulopati", "L L5 Radikulopati",
    "R L5 Radikulopati", "L S1 Radikulopati", "R S1 Radikulopati", "L S2 Radikulopati", "R S2 Radikulopati", "IBS",
    "L Nackbesvär", "Nackbesvär", "R Nackbesvär", "L Tinnitus", "L Ögonbesvär", "L Öronbesvär", "R Tinnitus",
    "R Ögonbesvär", "R Öronbesvär", "Huvudvärk", "L Käkbesvär", "L Pannhuvudvärk", "Munbesvär", "Pannhuvudvärk",
    "R Pannhuvudvärk", "R PFS", "Svalgbesvär", "R Käkbesvär", "Bakhuvudvärk", "R Bakhuvudvärk", "L Nyckelbensbesvär",
    "R Nyckelbensbesvär", "Central bröstsmärta", "L Central bröstsmärta", "L Centrala bröstbesvär",
    "R Främre Axelbesvär", "L Axel impingement", "R Axel impingement", "L Axelbesvär", "L Skulderbesvär",
    "R Axelbesvär", "R Skulderbesvär", "L Övre armsbesvär", "L Övre armbågsbesvär", "Interskapulära besvär",
    "L Interskapulära besvär", "R Interskapulära besvär", "L Laterala armbågsbesvär", "L Laterala armsbesvär",
    "R Laterala armbågsbesvär", "L Armbågsbesvär", "R Armbågsbesvär", "L Armbesvär", "L Tumbesvär", "R Tumbesvär",
    "L Handledsbesvär", "R Handledsbesvär", "L Under armsbesvär", "R Under armsbesvär", "L Handbesvär", "R Handbesvär",
    "L Armvecksbesvär", "R Armbesvär", "R Armvecksbesvär", "L Mediala armbågsbesvär", "R Mediala armbågsbesvär",
    "L Fingerbesvär", "R Fingerbesvär", "L Lillfingerbesvär", "R Lillfingerbesvär", "L Ljumskbesvär",
    "L Mediala ljumskbesvär", "L Laterala ljumskbesvär", "Centrala Ljumskbesvär", "R Laterala ljumskbesvär",
    "R Ljumskbesvär", "L Adduktortendalgi", "R Adduktortendalgi", "L Höftbesvär", "L Bakhuvudvärk", "Ryggsbesvär",
    "L Lumbago", "Lumbago", "R Lumbago", "L Främre lårbesvär", "R Främre lårbesvär", "R Lårbesvär", "L Benbesvär",
    "L Lårbesvär", "R Benbesvär", "R Mediala vadbesvär", "L PFS", "L Höftkamsbesvär", "R Höftbesvär",
    "R Höftkamsbesvär", "L Mediala knäledsbesvär", "L Främre knäbesvär", "R Mediala knäledsbesvär",
    "R Främre knäbesvär", "L Tibiaperialgi", "R Tibiaperialgi", "L Laterala vadbesvär", "L Knäbesvär", "R Knäbesvär",
    "L Tåledbesvär", "L Stortårbesvär", "R Stortårbesvär", "L Fotbesvär", "L Fotledsbesvär", "R Fotledsbesvär",
    "L Fotvalvsbesvär", "R Fotvalvsbesvär", "R Mortonbesvär", "R Tåledbesvär", "L Ischias", "R Ischias",
    "L Skinkbesvär", "L Vadbesvär", "R Skinkbesvär", "L Tårbesvär", "R Fotbesvär", "R Tårbesvär", "R Vadbesvär",
    "R Dorsala knäledsbesvär", "L Dorsala knäledsbesvär", "L Laterala knäbesvär", "R Laterala knäbesvär",
    "L Lilltåbesvär", "L Laterala Fotbesvär", "R Laterala Fotbesvär", "R Hälbesvär", "Hälbesvär", "L Hälbesvär",
    "Coccydyni", "L Bakre lårbesvär", "R Bakre lårbesvär", "L Achillesbesvär", "L Achillestendalgi", "L Achillodyni",
    "R Achillesbesvär", "R Achillestendalgi", "R Achillodyni", "Bröstryggsbesvär", "Bröstbesvär", "L Bröstbesvär",
    "R Bröstbesvär", "Torakal Dysfunktion", "Övre bukbesvär", "Laterala bukbesvär", "Bukbesvär", "L Nedre bukbesvär",
    "Nedre bukbesvär",
]

LEVEL2_HIDDEN_PATTERN_LEVELS = ["C2", "C4", "C6", "C8", "T2", "T4", "T6", "T8", "T10", "T12", "L2", "L4", "S1"]
PATTERN_LEVELS = [
    "C2", "C3", "C4", "C5", "C6", "C7", "C8",
    "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12",
    "L1", "L2", "L3", "L4", "L5", "S1", "S2",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate neuropathic simulator datasets under the Level 0-3 hidden-confounding scheme."
    )
    parser.add_argument(
        "--sample-sizes",
        type=int,
        nargs="+",
        required=True,
        help="One or more sample sizes to generate.",
    )
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="Hidden-confounding levels to generate. Choices: 0 1 2 3.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(range(20)),
        help="One or more random seeds. Default: 0 through 19.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where generated CSVs and metadata will be written.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the serialized pgmpy Bayesian network.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    invalid_levels = sorted(set(args.levels) - {0, 1, 2, 3})
    if invalid_levels:
        parser.error(f"Unsupported levels: {invalid_levels}. Valid levels are 0, 1, 2, 3.")
    return args


def load_model(model_path: Path):
    from cd_llm.data.loaders.neuropathic import load_simulator_model

    return load_simulator_model(model_path)


def sample_binary_bn(model, ordered_nodes: Sequence[str], sample_size: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    row_count = sample_size
    col_count = len(ordered_nodes)
    samples = np.zeros((row_count, col_count), dtype=int)
    rv_map = {rv: idx for idx, rv in enumerate(ordered_nodes)}

    for row in range(row_count):
        while samples[row].sum() == 0:
            samples[row] = 0
            for rv in ordered_nodes:
                cpd = model.get_cpds(rv)
                parents = list(cpd.variables[1:])
                parent_values = tuple(int(samples[row, rv_map[parent]]) for parent in parents)
                probabilities = np.asarray(cpd.values, dtype=float)
                if parent_values:
                    probabilities = probabilities[(slice(None), *parent_values)]
                probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
                probabilities = probabilities / probabilities.sum()
                chosen_value = rng.choice(int(cpd.variable_card), p=probabilities)
                samples[row, rv_map[rv]] = int(chosen_value)
    return samples


def hidden_indices_for_level(level: int) -> List[int]:
    if level == 0:
        return []
    if level == 1:
        return list(range(27))
    if level == 2:
        level_to_pair_start = {level_name: 27 + 2 * idx for idx, level_name in enumerate(PATTERN_LEVELS)}
        hidden = list(range(27))
        for level_name in LEVEL2_HIDDEN_PATTERN_LEVELS:
            pair_start = level_to_pair_start[level_name]
            hidden.extend([pair_start, pair_start + 1])
        return sorted(hidden)
    if level == 3:
        return list(range(79))
    raise ValueError(f"Unsupported level: {level}")


def observed_indices_for_level(level: int) -> List[int]:
    hidden = set(hidden_indices_for_level(level))
    return [index for index in range(len(CANONICAL_VARIABLE_NAMES)) if index not in hidden]


def mask_columns(data: np.ndarray, observed_indices: Sequence[int]) -> np.ndarray:
    return data[:, observed_indices]


def ensure_model_order(model) -> List[str]:
    ordered_names = list(CANONICAL_VARIABLE_NAMES)
    model_nodes = set(model.nodes())
    if model_nodes != set(ordered_names):
        missing = sorted(set(ordered_names) - model_nodes)
        extra = sorted(model_nodes - set(ordered_names))
        raise ValueError(
            f"Model node set does not match the canonical simulator order. Missing={missing[:5]} Extra={extra[:5]}"
        )
    return ordered_names


def write_csv(path: Path, data: np.ndarray, observed_indices: Sequence[int]) -> None:
    header = [""] + [str(index) for index in observed_indices]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row_index, row in enumerate(data):
            writer.writerow([row_index] + row.astype(int).tolist())


def write_metadata(path: Path, *, sample_size: int, level: int, seed: int, observed_indices: Sequence[int]) -> None:
    payload = {
        "sample_size": sample_size,
        "level": level,
        "seed": seed,
        "observed_indices": list(observed_indices),
        "hidden_indices": hidden_indices_for_level(level),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def generate_dataset(model, sample_size: int, level: int, seed: int) -> tuple[np.ndarray, List[int]]:
    ordered_nodes = ensure_model_order(model)
    full_data = sample_binary_bn(model, ordered_nodes, sample_size, seed)
    observed_indices = observed_indices_for_level(level)
    observed_data = mask_columns(full_data, observed_indices)
    return observed_data, observed_indices


def output_stem(sample_size: int, level: int, seed: int) -> str:
    return f"neuropathic_n{sample_size}_level{level}_seed{seed}"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    model = load_model(args.model_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        (sample_size, level, seed)
        for sample_size in args.sample_sizes
        for level in args.levels
        for seed in args.seeds
    ]
    total_tasks = len(tasks)
    start_time = time.monotonic()

    for completed_tasks, (sample_size, level, seed) in enumerate(tasks, start=1):
        stem = output_stem(sample_size, level, seed)
        csv_path = args.output_dir / f"{stem}.csv"
        meta_path = args.output_dir / f"{stem}.json"
        if not args.overwrite and (csv_path.exists() or meta_path.exists()):
            raise SystemExit(f"Refusing to overwrite existing output for {stem}. Use --overwrite.")

        observed_data, observed_indices = generate_dataset(model, sample_size, level, seed)
        write_csv(csv_path, observed_data, observed_indices)
        write_metadata(
            meta_path,
            sample_size=sample_size,
            level=level,
            seed=seed,
            observed_indices=observed_indices,
        )

        elapsed_seconds = time.monotonic() - start_time
        average_seconds = elapsed_seconds / completed_tasks
        remaining_tasks = total_tasks - completed_tasks
        remaining_seconds = average_seconds * remaining_tasks
        print(
            f"[{completed_tasks}/{total_tasks}] Wrote {csv_path} "
            f"(ETA: {format_duration(remaining_seconds)})"
        )
    return 0


def format_duration(seconds: float) -> str:
    rounded = max(0, int(round(seconds)))
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


if __name__ == "__main__":
    raise SystemExit(main())
