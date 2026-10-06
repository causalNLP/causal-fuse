"""Registered recovery of interrupted ALARM inference and provider failures."""
import argparse
from dataclasses import replace
import io
import json
from pathlib import Path
import time
import urllib.error

from cd_llm.experiments import alarm_llm_baselines as base
from cd_llm.experiments import alarm_llm_paced as paced
from cd_llm.experiments import alarm_llm_repair as repair
from cd_llm.experiments.common.artifacts import read_json, write_json, sha256


def check_error_envelope(response):
    """Some upstream failures arrive as HTTP 200 with an error object."""
    error = response.get("error")
    if isinstance(error, dict):
        try:
            code = int(error.get("code", 502))
        except (TypeError, ValueError):
            code = 502
        raise urllib.error.HTTPError("https://openrouter.ai/api/v1/chat/completions",
            code, str(error.get("message", "provider error")), {},
            io.BytesIO(json.dumps(response).encode()))
    return response


class RecoveryTransport(paced.PacedTransport):
    def send(self, request):
        return check_error_envelope(super().send(request))

    def query(self, request):
        for attempt in range(4):
            try:
                return super().query(request)
            except RuntimeError as error:
                if getattr(error.__cause__, "code", None) not in (429, 500, 502, 503, 504) or attempt == 3:
                    raise
                time.sleep(60)


def valid_edges(payload, names):
    """Select only registered pairs for validating a model's identifier repair."""
    from cd_llm.baselines.guess2graph2026.llm_expert import parse_skeleton_response
    if not isinstance(payload, dict) or not isinstance(payload.get("edges"), list):
        raise ValueError("identifier repair requires a complete edges object")
    allowed = set(names)
    kept = []
    for edge in payload["edges"]:
        if not isinstance(edge, list) or len(edge) != 2 or not all(isinstance(x, str) for x in edge):
            raise ValueError("identifier repair requires two identifier strings")
        if all(x in allowed for x in edge):
            kept.append(edge)
    return parse_skeleton_response({"edges": kept}, names)


class IdentifierRepair(repair.GPCRepair):
    """One additional model repair, verified to preserve every valid old edge."""
    def query(self, request):
        from cd_llm.baselines.guess2graph2026.llm_expert import parse_skeleton_response
        original = self.transport
        history = []
        class Recording:
            def query(self, req):
                answer = original.query(req)
                history.append((req, answer))
                return answer
        self.transport = Recording()
        try:
            return super().query(request)
        except ValueError:
            if not history:
                raise
            previous, (response, source) = history[-1]
            choice = response["choices"][0]
            if choice["finish_reason"] != "stop":
                raise
            text = choice["message"]["content"]
            expected = valid_edges(json.loads(text), self.names)
            messages = [*previous["messages"], {"role": "assistant", "content": text},
                {"role": "user", "content": "Your JSON still contains an unregistered identifier. "
                 "Delete all pairs containing identifiers outside the allowed list below. "
                 "Keep every other pair exactly unchanged. Do not rename any identifier or infer any new edge. "
                 "Return only the corrected JSON object. Allowed identifiers are " + ", ".join(self.names) + "."}]
            fixed, fixed_source = original.query(dict(previous, messages=messages, max_tokens=12288))
            result = fixed["choices"][0]
            if result["finish_reason"] != "stop":
                raise ValueError("unfinished identifier correction")
            actual = parse_skeleton_response(json.loads(result["message"]["content"]), self.names)
            if actual != expected:
                raise ValueError("identifier correction changed valid biological claims")
            folder = original.root / "identifier_repairs"
            write_json(folder / (base.digest(request) + ".json"), dict(
                previous_response=source, corrected_response=fixed_source,
                preserved_edges=actual, rule="model removes unregistered pairs; all registered pairs must remain unchanged"))
            return fixed, fixed_source
        finally:
            self.transport = original


def install(directory):
    repair.install(directory)
    paced.INTERVAL = 4.0
    base.Transport = RecoveryTransport
    repair.GPCRepair = IdentifierRepair
    from cd_llm.baselines.causal_order2025 import local_vllm_backend as native
    original_init = native.LocalVLLMTransport.__init__
    def init(self, cache, config=None, **kwargs):
        return original_init(self, cache, replace(config, concurrency=8) if config else config, **kwargs)
    native.LocalVLLMTransport.__init__ = init
    save = base.save_prediction
    def annotated_save(*args, **kwargs):
        return save(*args, recovery_manifest_sha256=sha256(directory / "recovery_manifest.json"), **kwargs)
    base.save_prediction = annotated_save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("register", "run", "order"))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task", type=int, choices=range(20))
    args = parser.parse_args()
    directory = base.root(args.run_id)
    path = directory / "recovery_manifest.json"
    if args.command == "register":
        write_json(path, dict(created=time.time(), original_manifest_sha256=sha256(directory / "manifest.json"),
            implementation_sha256={str(Path(__file__)): sha256(Path(__file__))},
            dispatch_interval_seconds=4, order_concurrency=8,
            repairs=["retry error envelopes without accepting them as model responses",
                "bounded additional retries of transient failures with unchanged effective requests",
                "one further gPC identifier-only model repair, requiring exact preservation of valid pairs"],
            unchanged=["datasets", "generation seeds", "provider", "inference rules", "query budgets", "USD 10 cap"]), exclusive=True)
        return
    for filename, expected in read_json(path)["implementation_sha256"].items():
        assert sha256(Path(filename)) == expected
    install(directory)
    base.run(args.run_id, args.task, order=args.command == "order")


if __name__ == "__main__":
    main()
