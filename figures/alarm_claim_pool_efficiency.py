"""ALARM claim-family recovery by attempted queries and recorded token cost."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator

ROOT = Path(__file__).resolve().parents[2]
POOLS = (('group_i','Local (Group I)','#397ca4'),('group_ii','Ancestry (Group II)','#D9822B'),('all','Mixed (All)','#117766'))
MARKERS = {10:'o',20:'s',50:'^',100:'D'}


def generate(stamp):
    root = ROOT/'results'/stamp/'alarm_claim_pools'
    completion = json.loads((root/'completion.json').read_text())
    assert completion['complete'] and completion['audit_passed'] and completion['checkpoints']==240
    digest = hashlib.sha256((root/'summary.csv').read_bytes()).hexdigest()
    assert completion['artifacts']['summary.csv']==digest
    with (root/'summary.csv').open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows)==30
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':8,'pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(2,2,figsize=(5.5,3.25),sharey=True,sharex='col')
    for i,n in enumerate((200,1000)):
        for group,label,color in POOLS:
            points = sorted((r for r in rows if int(r['sample_size'])==n and r['group']==group),key=lambda r:int(r['budget']))
            assert [int(r['budget']) for r in points]==[0,10,20,50,100]
            axes[i,0].plot([int(r['budget']) for r in points],[float(r['expected_distance_mean']) for r in points],color=color,lw=.8)
            for r in points:
                b=int(r['budget'])
                if b==0:
                    continue
                for j,x in ((0,b),(1,float(r['total_tokens_mean']))):
                    axes[i,j].errorbar(x,float(r['expected_distance_mean']),yerr=float(r['expected_distance_se']),
                        xerr=float(r['total_tokens_se']) if j else None,fmt=MARKERS[b],linestyle='none',
                        color=color,markersize=3.5,lw=.65,capsize=1.5)
        init = next(r for r in rows if int(r['sample_size'])==n and r['group']=='all' and int(r['budget'])==0)
        axes[i,0].errorbar(0,float(init['expected_distance_mean']),yerr=float(init['expected_distance_se']),
                          fmt='x',color='#69737c',markersize=4,lw=.65,capsize=1.5)
        for j in (0,1):
            ax=axes[i,j]
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.grid(axis='y',alpha=.18,lw=.5)
            ax.tick_params(labelsize=7,length=3)
            if j==0:
                ax.set_ylabel(f'$n={n}$\nED')
                ax.set_xticks([0,20,50,100])
            else:
                ax.set_xscale('log')
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x/1000:g}k'))
                ax.minorticks_off()
            if i==0:
                ax.set_title('Query expenditure' if j==0 else 'Token expenditure',fontsize=8.5)
            if i==1:
                ax.set_xlabel('Attempted queries' if j==0 else 'Input + output tokens (log)',fontsize=7.5)
    pool_handles=[Patch(facecolor=c,label=l) for _,l,c in POOLS]
    pool_handles.append(Line2D([],[],marker='x',color='#69737c',ls='none',label='PC+GES (0 queries)',markersize=4))
    fig.legend(handles=pool_handles,loc='upper left',bbox_to_anchor=(.79,.87),
        title='Claim pool\n(confidence cap 0.8)',title_fontsize=7,fontsize=6.6,frameon=False,borderaxespad=0)
    fig.legend(handles=[Line2D([],[],color='#4b535b',marker=m,ls='none',markersize=3.5,label=str(b)) for b,m in MARKERS.items()],
        loc='upper left',bbox_to_anchor=(.79,.48),title='Attempted queries',title_fontsize=6.8,fontsize=6.8,frameon=False,borderaxespad=0)
    fig.subplots_adjust(left=.115,right=.77,bottom=.17,top=.92,wspace=.22,hspace=.24)
    fig.savefig(root/'alarm_claim_pool_efficiency.pdf',bbox_inches='tight',pad_inches=.025)
    fig.savefig(root/'preview.png',dpi=200,bbox_inches='tight',pad_inches=.025)
    plt.close(fig)
    (root/'figure_provenance.json').write_text(json.dumps(dict(summary_sha256=digest,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),smc_points=24,
        figure_sha256=hashlib.sha256((root/'alarm_claim_pool_efficiency.pdf').read_bytes()).hexdigest()),indent=2)+'\n')
    print(root/'alarm_claim_pool_efficiency.pdf')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id',required=True)
    generate(parser.parse_args().run_id)
