"""Resume API caching and automatically submit the gated offline experiment."""
import argparse
import asyncio
from contextlib import redirect_stderr, redirect_stdout
import subprocess
import sys

from cd_llm.experiments.common.artifacts import run_id, write_json
from cd_llm.experiments import neuropathic_new_prompt_cache as cache
from cd_llm.paths import WORKSPACE_ROOT
from run_neuropathic_ancestry_prompt_api import Tee


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--stage", choices=("prepare", "pilot", "fill", "audit", "run"), default="run")
    args = parser.parse_args()
    stamp = run_id(args.run_id)
    root, logs = (WORKSPACE_ROOT / folder / stamp / cache.STUDY for folder in ("results", "out"))
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / "run.log").open("a") as log, redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
        try:
            cache.prepare(root)
            if args.stage in ("pilot", "fill", "run") and not (root / "cache_audit.json").exists():
                asyncio.run(cache.fill(root, pilot_only=args.stage == "pilot"))
            if args.stage in ("audit", "run"):
                cache.audit_cache(root)
            if args.stage == "run":
                if not list(root.glob("submission_pilot_*.json")):
                    result = subprocess.run([sys.executable, str(WORKSPACE_ROOT / "code/scripts/submit_neuropathic_new_prompt_smc.py"),
                        "--run-id", stamp, "--stage", "pilot"], check=True, text=True, capture_output=True)
                    print(result.stdout, flush=True)
                write_json(root / "status.json", dict(stage="inference-submitted", trajectories=120, checkpoints=480))
        except Exception as exc:
            write_json(root / "runner_failure.json", dict(stage=args.stage, error=str(exc)))
            print(f"Run stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
            raise SystemExit(1) from None


if __name__ == "__main__":
    main()
