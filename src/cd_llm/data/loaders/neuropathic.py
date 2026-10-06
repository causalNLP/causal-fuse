from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from cd_llm.data.generation.neuropathic import CANONICAL_VARIABLE_NAMES


PATHOPHYSIOLOGICAL_END = 27
RADICULOPATHY_END = 79

_BASE_TERM_TRANSLATIONS = {
    "Achillesbesvär": "Achilles tendon discomfort",
    "Achillestendalgi": "Achilles tendon pain",
    "Achillodyni": "achillodynia (Achilles pain)",
    "Adduktortendalgi": "adductor tendon pain",
    "Armbesvär": "arm discomfort",
    "Armbågsbesvär": "elbow discomfort",
    "Armvecksbesvär": "elbow-crease discomfort",
    "Axel impingement": "shoulder impingement",
    "Axelbesvär": "shoulder discomfort",
    "Bakhuvudvärk": "back-of-head headache",
    "Bakre lårbesvär": "posterior thigh discomfort",
    "Benbesvär": "leg discomfort",
    "Bröstbesvär": "chest discomfort",
    "Bröstryggsbesvär": "thoracic-spine discomfort",
    "Bukbesvär": "abdominal discomfort",
    "Central bröstsmärta": "central chest pain",
    "Centrala Ljumskbesvär": "central groin discomfort",
    "Centrala bröstbesvär": "central chest discomfort",
    "Coccydyni": "coccydynia (tailbone pain)",
    "Dorsala knäledsbesvär": "posterior knee-joint discomfort",
    "Fingerbesvär": "finger discomfort",
    "Fotbesvär": "foot discomfort",
    "Fotledsbesvär": "ankle discomfort",
    "Fotvalvsbesvär": "foot-arch discomfort",
    "Främre Axelbesvär": "anterior shoulder discomfort",
    "Främre knäbesvär": "anterior knee discomfort",
    "Främre lårbesvär": "anterior thigh discomfort",
    "Handbesvär": "hand discomfort",
    "Handledsbesvär": "wrist discomfort",
    "Huvudvärk": "headache",
    "Hälbesvär": "heel discomfort",
    "Höftbesvär": "hip discomfort",
    "Höftkamsbesvär": "iliac-crest discomfort",
    "IBS": "irritable bowel syndrome",
    "Interskapulära besvär": "interscapular discomfort",
    "Ischias": "sciatica",
    "Knäbesvär": "knee discomfort",
    "Kraniocervikal ledskada": "craniocervical joint injury",
    "Käkbesvär": "jaw discomfort",
    "Laterala Fotbesvär": "lateral foot discomfort",
    "Laterala armbågsbesvär": "lateral elbow discomfort",
    "Laterala armsbesvär": "lateral arm discomfort",
    "Laterala bukbesvär": "lateral abdominal discomfort",
    "Laterala knäbesvär": "lateral knee discomfort",
    "Laterala ljumskbesvär": "lateral groin discomfort",
    "Laterala vadbesvär": "lateral calf discomfort",
    "Lillfingerbesvär": "little-finger discomfort",
    "Lilltåbesvär": "little-toe discomfort",
    "Ljumskbesvär": "groin discomfort",
    "Lumbago": "lumbago (low back pain)",
    "Lårbesvär": "thigh discomfort",
    "Mediala armbågsbesvär": "medial elbow discomfort",
    "Mediala knäledsbesvär": "medial knee-joint discomfort",
    "Mediala ljumskbesvär": "medial groin discomfort",
    "Mediala vadbesvär": "medial calf discomfort",
    "Mortonbesvär": "Morton's neuroma discomfort",
    "Munbesvär": "mouth discomfort",
    "Nackbesvär": "neck discomfort",
    "Nedre bukbesvär": "lower abdominal discomfort",
    "Nyckelbensbesvär": "clavicle discomfort",
    "PFS": "patellofemoral syndrome",
    "Pannhuvudvärk": "frontal headache",
    "Ryggsbesvär": "back discomfort",
    "Skinkbesvär": "buttock discomfort",
    "Skulderbesvär": "shoulder discomfort",
    "Stortårbesvär": "big-toe discomfort",
    "Svalgbesvär": "throat discomfort",
    "Tibiaperialgi": "shin pain",
    "Tinnitus": "tinnitus",
    "Torakal Dysfunktion": "thoracic dysfunction",
    "Tumbesvär": "thumb discomfort",
    "Tåledbesvär": "toe-joint discomfort",
    "Tårbesvär": "toe discomfort",
    "Under armsbesvär": "forearm discomfort",
    "Vadbesvär": "calf discomfort",
    "Ögonbesvär": "eye discomfort",
    "Öronbesvär": "ear discomfort",
    "Övre armbågsbesvär": "upper elbow discomfort",
    "Övre armsbesvär": "upper arm discomfort",
    "Övre bukbesvär": "upper abdominal discomfort",
}


class _GenericPickleObject:
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.state = state


class _StubBayesianModel(_GenericPickleObject):
    def nodes(self) -> List[str]:
        node_dict = getattr(self, "node", None) or getattr(self, "_node", None)
        if node_dict is None:
            raise AttributeError("Could not recover node dictionary from simulator pickle.")
        return list(node_dict.keys())

    def edges(self) -> List[Tuple[str, str]]:
        adjacency = (
            getattr(self, "adj", None)
            or getattr(self, "_adj", None)
            or getattr(self, "succ", None)
            or getattr(self, "_succ", None)
        )
        if adjacency is None:
            raise AttributeError("Could not recover adjacency dictionary from simulator pickle.")
        return [(left, right) for left, neighbors in adjacency.items() for right in neighbors.keys()]

    def get_cpds(self, variable: str | None = None):
        cpds = list(getattr(self, "cpds", ()))
        if variable is None:
            return cpds
        for cpd in cpds:
            if getattr(cpd, "variable", None) == variable:
                return cpd
        raise KeyError(f"No CPD found for variable {variable!r}.")


class _StubTabularCPD(_GenericPickleObject):
    pass


class _ModelUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module == "pgmpy.models.BayesianModel" and name == "BayesianModel":
            return _StubBayesianModel
        if module == "pgmpy.factors.discrete.CPD" and name == "TabularCPD":
            return _StubTabularCPD
        if module.startswith("numpy"):
            return super().find_class(module, name)
        return type(name, (_GenericPickleObject,), {})


def load_simulator_model(model_path: Path) -> _StubBayesianModel:
    with model_path.open("rb") as handle:
        model = _ModelUnpickler(handle).load()

    model_nodes = set(model.nodes())
    canonical_nodes = set(CANONICAL_VARIABLE_NAMES)
    if model_nodes != canonical_nodes:
        missing = sorted(canonical_nodes - model_nodes)
        extra = sorted(model_nodes - canonical_nodes)
        raise ValueError(
            "Simulator node set does not match canonical variable names. "
            f"Missing={missing[:5]} Extra={extra[:5]}"
        )
    if len(model.get_cpds()) != len(CANONICAL_VARIABLE_NAMES):
        raise ValueError("Simulator pickle does not contain one CPD per canonical variable.")
    return model


def load_simulator_edges(model_path: Path) -> List[Tuple[str, str]]:
    model = load_simulator_model(model_path)
    return list(model.edges())


def load_metadata(path: Path) -> Dict[str, object]:
    payload = json.loads(path.read_text())
    required_keys = {"sample_size", "level", "seed", "observed_indices"}
    missing = required_keys - set(payload)
    if missing:
        raise ValueError(f"Metadata file {path} is missing keys: {sorted(missing)}")
    return payload


def find_metadata_for_selection(
    data_dir: Path,
    levels: Sequence[int],
    sample_sizes: Sequence[int] | None = None,
    seeds: Sequence[int] | None = None,
    datasets: Sequence[str] | None = None,
) -> List[Path]:
    selected = set(levels)
    selected_sample_sizes = set(sample_sizes) if sample_sizes is not None else None
    selected_seeds = set(seeds) if seeds is not None else None
    selected_datasets = set(datasets) if datasets is not None else None
    metadata_paths = []
    for path in sorted(data_dir.glob("*.json")):
        metadata = load_metadata(path)
        if int(metadata["level"]) not in selected:
            continue
        if selected_sample_sizes is not None and int(metadata["sample_size"]) not in selected_sample_sizes:
            continue
        if selected_seeds is not None and int(metadata["seed"]) not in selected_seeds:
            continue
        if (
            selected_datasets is not None
            and path.name not in selected_datasets
            and path.stem not in selected_datasets
        ):
            continue
        metadata_paths.append(path)
    if not metadata_paths:
        raise FileNotFoundError(
            f"No metadata files found in {data_dir} for levels {sorted(selected)}, "
            f"sample_sizes={sorted(selected_sample_sizes) if selected_sample_sizes is not None else 'all'}, "
            f"seeds={sorted(selected_seeds) if selected_seeds is not None else 'all'}, "
            f"datasets={sorted(selected_datasets) if selected_datasets is not None else 'all'}"
        )
    return metadata_paths


def names_from_indices(indices: Sequence[int]) -> List[str]:
    return [CANONICAL_VARIABLE_NAMES[index] for index in indices]


def clinical_layer(name: str) -> str:
    """Return the simulator's clinical layer for a canonical variable."""
    try:
        index = CANONICAL_VARIABLE_NAMES.index(name)
    except ValueError as error:
        raise ValueError(f"Unknown neuropathic variable: {name!r}") from error
    if index < PATHOPHYSIOLOGICAL_END:
        return "pathophysiological"
    if index < RADICULOPATHY_END:
        return "radiculopathy"
    return "symptom"


def variable_description(name: str) -> str:
    """Expand structural abbreviations without adding graph-role information."""
    if name.startswith("DLS "):
        return f"discoligamentous injury at spinal level {name.removeprefix('DLS ')}"
    side, _, remainder = name.partition(" ")
    side_name = {"L": "left", "R": "right"}.get(side)
    base = remainder if side_name is not None else name
    if base.endswith(" Radikulopati"):
        prefix = f"{side_name} " if side_name else ""
        return f"{prefix}{base.removesuffix(' Radikulopati')} radiculopathy"
    translated = _BASE_TERM_TRANSLATIONS.get(base)
    if translated is None:
        return name
    prefix = f"{side_name} " if side_name else ""
    return f"{prefix}{translated}"


def select_ancestrally_closed_module(
    edges: Iterable[Tuple[str, str]], module_size: int
) -> List[str]:
    """Select a deterministic connected, ancestrally closed clinical module."""
    canonical = tuple(CANONICAL_VARIABLE_NAMES)
    if not 1 <= module_size <= len(canonical):
        raise ValueError(
            f"module_size must be between 1 and {len(canonical)}; got {module_size}."
        )
    edge_list = list(edges)
    node_set = set(canonical)
    unknown = sorted({node for edge in edge_list for node in edge} - node_set)
    if unknown:
        raise ValueError(f"Edges contain unknown neuropathic variables: {unknown[:5]}")
    parents = {node: set() for node in canonical}
    for parent, child in edge_list:
        parents[child].add(parent)
    canonical_index = {node: index for index, node in enumerate(canonical)}
    seeds = (
        *canonical[RADICULOPATHY_END:],
        *canonical[PATHOPHYSIOLOGICAL_END:RADICULOPATHY_END],
        *canonical[:PATHOPHYSIOLOGICAL_END],
    )
    candidates: list[tuple[tuple[object, ...], List[str]]] = []
    for seed in seeds:
        selected = _ancestor_closure(seed, parents)
        if len(selected) > module_size:
            continue
        while len(selected) < module_size:
            frontier = [
                node
                for node in canonical
                if node not in selected
                and parents[node]
                and parents[node].issubset(selected)
                and bool(parents[node] & selected)
            ]
            if not frontier:
                break
            present_layers = {clinical_layer(node) for node in selected}
            frontier.sort(
                key=lambda node: (
                    0
                    if clinical_layer(node) not in present_layers
                    else {"radiculopathy": 1, "symptom": 2, "pathophysiological": 3}[
                        clinical_layer(node)
                    ],
                    canonical_index[node],
                )
            )
            selected.add(frontier[0])
        if len(selected) != module_size or any(
            not parents[node].issubset(selected) for node in selected
        ):
            continue
        ordered = [node for node in canonical if node in selected]
        layer_counts = {
            layer: sum(clinical_layer(node) == layer for node in ordered)
            for layer in ("pathophysiological", "radiculopathy", "symptom")
        }
        internal_edges = sum(
            parent in selected and child in selected for parent, child in edge_list
        )
        score = (
            -sum(count > 0 for count in layer_counts.values()),
            -min(layer_counts.values()),
            -internal_edges,
            tuple(canonical_index[node] for node in ordered),
        )
        candidates.append((score, ordered))
    if not candidates:
        raise ValueError(
            f"Could not construct an ancestrally closed module of size {module_size}."
        )
    return min(candidates, key=lambda item: item[0])[1]


def observed_module_nodes(module_nodes: Sequence[str], level: int) -> List[str]:
    from cd_llm.data.generation.neuropathic import observed_indices_for_level

    observed = set(names_from_indices(observed_indices_for_level(level)))
    return [node for node in module_nodes if node in observed]


def _ancestor_closure(seed: str, parents: Dict[str, set[str]]) -> set[str]:
    closure = {seed}
    stack = [seed]
    while stack:
        node = stack.pop()
        for parent in parents[node]:
            if parent not in closure:
                closure.add(parent)
                stack.append(parent)
    return closure
