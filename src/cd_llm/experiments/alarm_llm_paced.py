"""Resume native ALARM methods with globally paced API dispatches."""
import argparse
from pathlib import Path
import time

from cd_llm.experiments import alarm_llm_repair as repair
from cd_llm.experiments import alarm_llm_baselines as base
from cd_llm.experiments.common.artifacts import read_json,write_json,sha256

INTERVAL=2.0


class PacedTransport(repair.ResilientTransport):
    def send(self,request):
        # Every retry, not just each logical query, passes through this gate.
        with repair.network.file_lock(self.root/"locks/dispatch-rate.lock"):
            path=self.root/"last_dispatch.txt"
            previous=float(path.read_text()) if path.exists() else 0.
            delay=previous+INTERVAL-time.time()
            if delay>0:
                time.sleep(delay)
            path.write_text(str(time.time()))
        return super().send(request)

    def query(self,request):
        if request["messages"][-1]["content"].startswith("Repair only the JSON format"):
            request=dict(request,max_tokens=12288)
        return super().query(request)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("register","run","order"))
    p.add_argument("--run-id",required=True)
    p.add_argument("--task",type=int,choices=range(20))
    args=p.parse_args()
    directory=base.root(args.run_id)
    path=directory/"pacing_manifest.json"
    if args.command=="register":
        write_json(path,dict(dispatch_interval_seconds=INTERVAL,gpc_format_tokens=12288,
            created=time.time(),implementation_sha256={str(Path(__file__)):sha256(Path(__file__))},
            reason="Observed shared-provider 429 throttling under concurrent collection. Pace all physical retries as well as original dispatches. Preserve native effective requests except the registered gPC format completion allowance."),exclusive=True)
        return
    for filename,expected in read_json(path)["implementation_sha256"].items():
        assert sha256(Path(filename))==expected
    repair.install(directory)
    base.Transport=PacedTransport
    save=base.save_prediction
    def annotated_save(*args,**kwargs):
        return save(*args,pacing_manifest_sha256=sha256(path),**kwargs)
    base.save_prediction=annotated_save
    base.run(args.run_id,args.task,order=args.command=="order")


if __name__=="__main__":
    main()
