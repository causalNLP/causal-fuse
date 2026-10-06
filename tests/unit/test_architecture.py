from __future__ import annotations

import ast
import unittest
from pathlib import Path

from cd_llm.experiments.common.artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    artifact_schema_version,
    require_current_cache,
)
from cd_llm.paths import CODE_ROOT


LEGACY_ROOTS = {"baselines", "benchmark", "command", "datasets", "method", "metric"}


class ImportBoundaryTests(unittest.TestCase):
    def test_canonical_package_does_not_import_legacy_namespaces(self) -> None:
        violations: list[str] = []
        source_root = CODE_ROOT / "src" / "cd_llm"
        for path in source_root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    roots = {alias.name.split(".", 1)[0] for alias in node.names}
                elif isinstance(node, ast.ImportFrom) and node.module:
                    roots = {node.module.split(".", 1)[0]}
                else:
                    continue
                for root in roots & LEGACY_ROOTS:
                    violations.append(f"{path.relative_to(CODE_ROOT)} imports {root}")
        self.assertEqual(violations, [])

    def test_canonical_package_has_no_sys_path_bootstrap(self) -> None:
        offenders = [
            path.relative_to(CODE_ROOT).as_posix()
            for path in (CODE_ROOT / "src" / "cd_llm").rglob("*.py")
            if "sys.path.insert" in path.read_text(encoding="utf-8")
        ]
        self.assertEqual(offenders, [])

    def test_legacy_wrapper_namespaces_are_absent(self) -> None:
        for namespace in sorted(LEGACY_ROOTS):
            self.assertFalse((CODE_ROOT / namespace).exists(), namespace)
            self.assertFalse((CODE_ROOT / "src" / namespace).exists(), namespace)


class ArtifactSchemaTests(unittest.TestCase):
    def test_legacy_results_are_detected_but_not_resumable(self) -> None:
        legacy = {"rows": []}
        self.assertEqual(artifact_schema_version(legacy), 1)
        with self.assertRaisesRegex(ValueError, "cannot be resumed"):
            require_current_cache(legacy)

    def test_current_cache_schema_is_accepted(self) -> None:
        require_current_cache({"artifact_schema_version": ARTIFACT_SCHEMA_VERSION})


if __name__ == "__main__":
    unittest.main()
