"""Measured native-budget baselines and saved SMC budget checkpoints."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def draw(root):
    with (root/"summary.csv").open() as stream: rows=list(csv.DictReader(stream))
    fig,axes=plt.subplots(1,2,figsize=(9.5,3.2),sharey=True)
    colors=dict(smc_group_i="#4677B4",smc_group_ii="#D08C36",smc_all="#288979")
    markers={10:"o",20:"s",50:"D",100:"^"}
    for ax,n in zip(axes,(200,1000)):
        for r in rows:
            if int(r["sample_size"])!=n or int(r["complete"])!=10: continue
            x=float(r["total_tokens_mean"])
            if x<=0: continue
            y=float(r["expected_distance_mean"])
            smc=r["method"].startswith("smc_")
            ax.errorbar(x,y,xerr=float(r["total_tokens_se"]),yerr=float(r["expected_distance_se"]),
                fmt=markers[int(r["budget"])] if smc else "X",color=colors.get(r["method"],"#555555"),
                markersize=5,capsize=2,elinewidth=.7,linestyle="none",alpha=.9)
            if not smc: ax.annotate(r["label"],(x,y),xytext=(4,4),textcoords="offset points",fontsize=6)
        ax.set(xscale="log",xlabel="Input + output tokens",title=f"n = {n}")
        ax.grid(alpha=.15)
        ax.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("ED ↓")
    handles=[Line2D([],[],marker="o",linestyle="none",color=c,label=l) for c,l in zip(colors.values(),("SMC local","SMC ancestry","SMC mixed"))]
    handles += [Line2D([],[],marker=m,linestyle="none",color="#555555",label=f"{b} queries") for b,m in markers.items()]
    fig.legend(handles=handles,loc="center left",bbox_to_anchor=(.83,.5),frameon=False,fontsize=8)
    fig.tight_layout(rect=(0,0,.84,1))
    fig.savefig(root/"alarm_matched_performance_cost.pdf",bbox_inches="tight")
    plt.close(fig)


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument("root",type=Path)
    draw(parser.parse_args().root)
