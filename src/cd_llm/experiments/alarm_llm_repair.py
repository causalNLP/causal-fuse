"""Registered technical repairs for the frozen ALARM run; inference is unchanged."""
from __future__ import annotations

import argparse
import asyncio
import copy
from dataclasses import replace
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import time

from cd_llm.experiments import alarm_llm_baselines as base
from cd_llm.experiments import alarm_llm_transport as network
from cd_llm.experiments.common.artifacts import read_json, write_json, sha256


class ResilientTransport(network.Transport):
    def query(self, request):
        # Provider throttling changes scheduling, never the question or seed.
        for attempt in range(5):
            try:
                return super().query(request)
            except RuntimeError as error:
                cause=error.__cause__
                if getattr(cause,"code",None) not in (429,500,502,503,504) or attempt==4:
                    raise
                time.sleep(min(60,15*(attempt+1)))


class GPCRepair:
    def __init__(self, transport, names):
        self.transport,self.names=transport,tuple(names)

    def query(self, request):
        from cd_llm.baselines.guess2graph2026.llm_expert import parse_skeleton_response
        response,path=self.transport.query(request)
        def validate(payload):
            choice=payload["choices"][0]
            if choice["finish_reason"]!="stop":
                raise ValueError("unfinished response")
            return parse_skeleton_response(json.loads(choice["message"]["content"]),self.names)
        try:
            validate(response)
            return response,path
        except (ValueError,KeyError,TypeError):
            messages=[*request["messages"],dict(role="assistant",content=response["choices"][0]["message"].get("content") or ""),
                dict(role="user",content=("Repair only the JSON format and variable identifiers in your preceding answer. "
                    "Return the same skeleton as a JSON object with an edges list of two-element lists. "
                    "Allowed variable identifiers are "+", ".join(self.names)+". "
                    "Remove any edge containing an identifier outside this list. Do not add replacement edges, "
                    "new relationships, explanations or markdown. Use exactly the supplied variable identifiers."))]
            repaired,repair_path=self.transport.query(dict(request,messages=messages))
            validate(repaired)
            return repaired,repair_path


def register(directory):
    path=directory/"repair_manifest.json"
    if path.exists():
        verify(directory)
        return
    write_json(path,dict(original_manifest_sha256=sha256(directory/"manifest.json"),
        implementation_sha256={str(Path(__file__)):sha256(Path(__file__))},
        repairs=["one gPC format repair retaining assessment and removing unregistered identifiers",
            "larger local HTTP accept queue and bounded transport retries",
            "reject unfinished triplet answer blocks while retaining raw API responses"],
        model=network.MODEL,provider=network.PROVIDER,created=time.time()),exclusive=True)


def verify(directory):
    r=read_json(directory/"repair_manifest.json")
    assert r["original_manifest_sha256"]==sha256(directory/"manifest.json")
    for filename,expected in r["implementation_sha256"].items():
        assert sha256(Path(filename))==expected,"repair implementation changed"


def install(directory):
    verify(directory)
    # Do not edit the frozen adapters or their on-disk source hashes.
    ThreadingHTTPServer.request_queue_size=128
    original_proxy=base.proxy
    original_save=base.save_prediction
    base.Transport=ResilientTransport
    from contextlib import contextmanager
    @contextmanager
    def repaired_proxy(transport):
        class Checked:
            def query(self,request):
                if getattr(transport,"method",None)=="gpc":
                    response,path=GPCRepair(transport,base.metadata(directory)[1]).query(request)
                else:
                    response,path=transport.query(request)
                # A one-token A/B answer is intentionally length-limited.
                if request.get("max_tokens")!=1 and response["choices"][0].get("finish_reason")!="stop":
                    response=copy.deepcopy(response)
                    choice=response["choices"][0]
                    choice["message"]["content"]=(choice["message"].get("content") or "").replace("<Answer>","<UnfinishedAnswer>").replace("</Answer>","</UnfinishedAnswer>")
                return response,path
        with original_proxy(Checked()) as url:
            yield url
    base.proxy=repaired_proxy
    def save(*args,**kwargs):
        return original_save(*args,repair_manifest_sha256=sha256(directory/"repair_manifest.json"),**kwargs)
    base.save_prediction=save
    from cd_llm.baselines.causal_order2025 import local_vllm_backend as native
    original_init=native.LocalVLLMTransport.__init__
    def init(self,cache,config=None,**kwargs):
        original_init(self,cache,replace(config,max_attempts=5) if config else config,**kwargs)
    native.LocalVLLMTransport.__init__=init


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("register","run","order"))
    p.add_argument("--run-id",required=True)
    p.add_argument("--task",type=int,choices=range(20))
    args=p.parse_args()
    directory=base.root(args.run_id)
    if args.command=="register":
        register(directory)
    else:
        install(directory)
        base.run(args.run_id,args.task,order=args.command=="order")


if __name__=="__main__":
    main()
