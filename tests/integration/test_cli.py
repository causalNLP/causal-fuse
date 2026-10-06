from __future__ import annotations

import unittest

from cd_llm.cli.main import COMMANDS, main


class CLITests(unittest.TestCase):
    def test_command_groups_are_registered(self) -> None:
        self.assertEqual(set(COMMANDS), {"data", "experiment"})
        self.assertIn("claim-fuse-smc", COMMANDS["experiment"])
        self.assertIn("claim-fuse-smc-eig", COMMANDS["experiment"])
        self.assertIn("synthetic-claim-robustness", COMMANDS["experiment"])
        self.assertIn("synthetic-pairwise-expert-benchmark", COMMANDS["experiment"])
        self.assertIn("latent-mag-validation", COMMANDS["experiment"])
        self.assertIn("neuropathic-terra-smc", COMMANDS["experiment"])
        self.assertIn("neuropathic-causal-order-terra", COMMANDS["experiment"])
        self.assertIn("alarm-claim-pools", COMMANDS["experiment"])
        self.assertIn("alarm-matched-baselines", COMMANDS["experiment"])
        self.assertIn("generate-synthetic", COMMANDS["data"])

    def test_nested_help_reaches_the_underlying_command(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            main(["data", "generate-synthetic", "--help"])
        self.assertEqual(raised.exception.code, 0)

    def test_claim_fuse_smc_help_reaches_the_sidecar(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            main(["experiment", "claim-fuse-smc", "--help"])
        self.assertEqual(raised.exception.code, 0)

    def test_claim_fuse_smc_eig_help_reaches_the_driver(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            main(["experiment", "claim-fuse-smc-eig", "--help"])
        self.assertEqual(raised.exception.code, 0)

    def test_synthetic_claim_robustness_help_reaches_the_driver(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            main(["experiment", "synthetic-claim-robustness", "--help"])
        self.assertEqual(raised.exception.code, 0)

    def test_latent_mag_validation_help_reaches_the_driver(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            main(["experiment", "latent-mag-validation", "--help"])
        self.assertEqual(raised.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
