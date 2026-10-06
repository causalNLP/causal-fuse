"""Run or resume the API study and save command logs beside its results."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import sys

from cd_llm.experiments.common.artifacts import run_id
from cd_llm.experiments.neuropathic_ancestry_prompt_api import STUDY, main as experiment
from cd_llm.paths import WORKSPACE_ROOT


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--stage", choices=("prepare", "pilot", "fill", "report", "run"), default="run")
    args = parser.parse_args()
    stamp = run_id(args.run_id)
    logs = WORKSPACE_ROOT / "out" / stamp / STUDY
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / "run.log").open("a") as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            try:
                experiment([args.stage, "--run-id", stamp])
            except Exception as exc:
                print(f"Run stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
                raise SystemExit(1) from None


if __name__ == "__main__":
    main()
