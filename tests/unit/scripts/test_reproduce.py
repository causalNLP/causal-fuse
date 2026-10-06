from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "reproduce.py"
SPEC = importlib.util.spec_from_file_location("reproduce", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
reproduce = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reproduce
SPEC.loader.exec_module(reproduce)


class FakeProcess:
    def __init__(self, return_code: int):
        self.stdout = io.StringIO("")
        self.return_code = return_code

    def wait(self) -> int:
        return self.return_code


class ReproductionWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = {
            "OPENROUTER_API_KEY": "private-key-value",
            "QWEN_BASE_URL": "http://127.0.0.1:8000/v1",
            "QWEN_CHECKPOINT": "/models/qwen",
            "VLLM_EXECUTABLE": "/venv/bin/vllm",
        }

    def plan(self, name: str):
        return reproduce.WORKFLOWS[name]("test-run", self.env, True)

    def test_all_workflows_have_the_expected_complete_command_count(self) -> None:
        expected = {
            "synthetic-claim-pools": 183,
            "neuropathic-claim-fuse-qwen": 49,
            "neuropathic-claim-fuse-terra": 25,
            "neuropathic-gpc": 45,
            "neuropathic-causal-order-qwen": 23,
            "neuropathic-baselines": 11,
            "alarm-claim-fuse": 55,
            "alarm-baselines": 144,
            "export-main-results": 3,
            "export-cost-results": 1,
        }
        self.assertEqual(expected, {name: len(self.plan(name)) for name in reproduce.WORKFLOWS})

    def test_pilot_gates_precede_remaining_blocks(self) -> None:
        qwen = [command.label for command in self.plan("neuropathic-claim-fuse-qwen")]
        self.assertLess(qwen.index("qwen-block-1"), qwen.index("qwen-validate-pilot"))
        self.assertLess(qwen.index("qwen-validate-pilot"), qwen.index("qwen-block-2"))
        self.assertLess(qwen.index("cap-block-1"), qwen.index("cap-validate-pilot"))
        self.assertLess(qwen.index("cap-validate-pilot"), qwen.index("cap-block-2"))

        terra = [command.label for command in self.plan("neuropathic-claim-fuse-terra")]
        self.assertLess(terra.index("block-1"), terra.index("validate-pilot"))
        self.assertLess(terra.index("validate-pilot"), terra.index("block-2"))

        alarm = [command.label for command in self.plan("alarm-claim-fuse")]
        self.assertLess(alarm.index("stated-block-1"), alarm.index("stated-validate-pilot"))
        self.assertLess(alarm.index("stated-validate-pilot"), alarm.index("stated-block-2"))
        self.assertLess(alarm.index("consistency-block-1"), alarm.index("consistency-validate-pilot"))
        self.assertLess(alarm.index("consistency-validate-pilot"), alarm.index("consistency-block-2"))

    def test_task_ranges_are_complete(self) -> None:
        synthetic = [command.label for command in self.plan("synthetic-claim-pools")]
        self.assertEqual([f"compute-{index}" for index in range(180)], synthetic[1:181])
        gpc = [command.label for command in self.plan("neuropathic-gpc")]
        self.assertEqual([f"infer-{index}" for index in range(40)], gpc[4:44])
        alarm = [command.label for command in self.plan("alarm-baselines")]
        for method in ("bfs", "long", "order", "scp"):
            self.assertEqual(20, sum(label.startswith(f"collect-{method}-") for label in alarm))

    def test_dry_run_does_not_print_secrets_or_create_logs(self) -> None:
        commands = self.plan("neuropathic-claim-fuse-qwen")
        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            with patch.object(reproduce, "WORKSPACE_ROOT", Path(directory)), redirect_stdout(output):
                reproduce.run_commands(commands, workflow="neuropathic-claim-fuse-qwen", run_id="test-run", dry_run=True, base_env=self.env)
            self.assertNotIn(self.env["OPENROUTER_API_KEY"], output.getvalue())
            self.assertFalse((Path(directory) / "out").exists())

    def test_failure_stops_the_workflow(self) -> None:
        commands = [
            reproduce.Command("first", ("first",)),
            reproduce.Command("second", ("second",)),
            reproduce.Command("third", ("third",)),
        ]
        with tempfile.TemporaryDirectory() as directory:
            processes = [FakeProcess(0), FakeProcess(7)]
            with patch.object(reproduce, "WORKSPACE_ROOT", Path(directory)), patch.object(
                reproduce, "CODE_ROOT", Path(directory)
            ), patch.object(reproduce.subprocess, "Popen", side_effect=processes) as popen:
                with self.assertRaisesRegex(subprocess.CalledProcessError, "returned non-zero exit status 7"):
                    reproduce.run_commands(commands, workflow="test", run_id="test-run", dry_run=False, base_env={})
            self.assertEqual(2, popen.call_count)
            self.assertEqual(str(Path(directory) / "src"), popen.call_args_list[0].kwargs["env"]["PYTHONPATH"])


if __name__ == "__main__":
    unittest.main()
