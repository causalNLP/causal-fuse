"""Plot paired Qwen recovery--runtime and recovery--token figures."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedFormatter, FixedLocator, MaxNLocator, NullLocator

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = '20260921T220659Z'

POOLS = (
    ('group_i', 'Local (Group I)', '#397ca4'),
    ('group_ii', 'Ancestry (Group II)', '#D9822B'),
    ('all', 'All', '#117766'),
)
RUNTIME_QUERY_COLORS = {
    10: '#73C476',
    20: '#41AB5D',
    50: '#238B45',
    100: '#006D2C',
}
RUNTIME_BASELINE_COLOR = '#747C84'
RUNTIME_BASELINE_ERROR_COLOR = '#929AA2'
RUNTIME_BASELINES = (
    ('gPC-Guess', 'gPC-Guess', 'o'),
    ('PC+Long-Srisk', 'PC+Long', 'o'),
    ('BFS+Pearson', 'BFS+Pearson', 'o'),
    ('SCP Pattern 2', 'SCP', 'o'),
    ('Causal Order+PC', 'Causal\nOrder+PC', 'o'),
)
RUNTIME_BASELINE_LABEL_OFFSETS = {
    'gPC-Guess': (-4, 7, 'right', 'bottom'),
    'PC+Long-Srisk': (4, -7, 'left', 'top'),
    'BFS+Pearson': (5, 7, 'left', 'bottom'),
    'SCP Pattern 2': (-4, -8, 'right', 'top'),
    'Causal Order+PC': (-2, 8, 'right', 'bottom'),
}
SAMPLE_SIZES = (200, 1000)
STRATEGIES = ('stated', 'consistency')
BUDGETS = (10, 20, 50, 100)

ED_LIMITS = (9., 28.)
RUNTIME_ED_LIMITS = (9., 27.)
RUNTIME_LIMITS = (4.5, 120_000.)
TOKEN_LIMITS = (650., 2.4e7)
RUNTIME_TICKS = (
    (10., '10s'),
    (60., '1m'),
    (600., '10m'),
    (3600., '1h'),
    (43_200., '12h'),
    (86_400., '24h'),
)
TOKEN_TICKS = (
    (1e3, '1k'),
    (1e4, '10k'),
    (1e5, '100k'),
    (1e6, '1M'),
    (1e7, '10M'),
)

CLAIM_FUSE_MARKER_SIZE = 6.1
BASELINE_MARKER_SIZE = 5.4
BASELINE_GRAY = '#69737c'
ANNOTATION_GRAY = '#4b535b'
GRID_GRAY = '#c8cdd1'
BASELINE_LABELS = {
    'SCP Pattern 2': 'SCP',
    'Causal Order+PC': 'Causal\nOrder+PC',
    'PC+Long-Srisk': 'PC+Long',
    'BFS+Pearson': 'BFS+Pearson',
    'gPC-Guess': 'gPC-Guess',
}
BASELINE_OFFSETS = {
    'gPC-Guess': (0, 8, 'center', 'bottom', False),
    'PC+Long-Srisk': (3, -8, 'left', 'top', False),
    'BFS+Pearson': (8, 9, 'left', 'bottom', True),
    'Causal Order+PC': (0, 11, 'center', 'bottom', False),
    'SCP Pattern 2': (0, -10, 'center', 'top', False),
}


def generate(stamp=DEFAULT_RUN):
    root = ROOT / 'results' / stamp / 'neuropathic_performance_cost'
    protocol = json.loads((root / 'protocol.json').read_text())
    assert (protocol.get('claim_pool_protocol_locked') or protocol.get('matched_protocol')
            or protocol.get('table2_group_i_endpoints_reproduced'))
    cap = protocol.get('answer_confidence_cap', .99)
    rows = list(csv.DictReader((root / 'summary.csv').open()))
    validate_rows(rows)

    runtime_pdf = ROOT / 'paper/figs/neuropathic_performance_runtime.pdf'
    runtime_png = ROOT / 'paper/figs/neuropathic_performance_runtime.png'
    runtime_svg = ROOT / 'paper/figs/neuropathic_performance_runtime.svg'
    token_pdf = ROOT / 'paper/figs/neuropathic_performance_tokens.pdf'
    token_png = ROOT / 'paper/figs/neuropathic_performance_tokens.png'
    render_runtime_figure(rows, runtime_pdf, runtime_png, runtime_svg)
    render_figure(rows, 'tokens', token_pdf, token_png)
    write_metadata(rows, root, cap)
    print(runtime_pdf, flush=True)
    print(token_pdf, flush=True)


def validate_rows(rows):
    """Check that the paired figures receive the complete, unchanged summary."""
    assert len(rows) == 58
    for n in SAMPLE_SIZES:
        baselines = [row for row in rows
                     if row['strategy'] == 'native' and int(row['sample_size']) == n]
        assert sorted(row['method'] for row in baselines) == sorted(BASELINE_LABELS)
        for strategy in STRATEGIES:
            selected = [row for row in rows if row['method'] == 'SMC-Active'
                        and row['strategy'] == strategy and int(row['sample_size']) == n]
            assert len(selected) == len(POOLS) * len(BUDGETS)
            for group, _label, _color in POOLS:
                assert sorted(int(row['budget']) for row in selected if row['group'] == group) == list(BUDGETS)


def style_panel(ax, xlim, bottom, left):
    """Apply the shared panel geometry and ED treatment."""
    ax.set_xscale('log')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ED_LIMITS)
    ax.minorticks_off()
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.grid(which='major', color=GRID_GRAY, alpha=.34, lw=.7, zorder=0)
    ax.tick_params(axis='both', labelsize=9.2, length=3.5, width=.75,
                   labelbottom=bottom, labelleft=left)


def format_token_axis(ax, bottom, left):
    style_panel(ax, TOKEN_LIMITS, bottom, left)
    ax.xaxis.set_major_locator(FixedLocator([value for value, _label in TOKEN_TICKS]))
    ax.set_xticklabels([label for _value, label in TOKEN_TICKS])


def plot_claim_fuse(ax, row, color, x_key, xerr_key, show_xerr):
    """Plot one Claim-Fuse mean and its selected unchanged SE bars."""
    x = float(row[x_key])
    y = float(row['expected_distance_mean'])
    ax.scatter(x, y, s=CLAIM_FUSE_MARKER_SIZE ** 2, marker='o', facecolor=color,
               edgecolor='white', linewidth=.75, zorder=3)
    xerr = float(row[xerr_key]) if show_xerr else None
    ax.errorbar(x, y, xerr=xerr, yerr=float(row['expected_distance_se']),
                fmt='none', ecolor=color, elinewidth=1.05, capsize=2.4,
                capthick=.95, zorder=4)


def annotate_baselines(ax, row, x_key):
    """Direct-label a native baseline without changing its data coordinates."""
    method = row['method']
    x = float(row[x_key])
    y = float(row['expected_distance_mean'])
    dx, dy, ha, va, leader = BASELINE_OFFSETS[method]
    arrowprops = None
    if leader:
        arrowprops = dict(arrowstyle='-', lw=.7, color=ANNOTATION_GRAY,
                          shrinkA=2, shrinkB=4)
    ax.annotate(BASELINE_LABELS[method], xy=(x, y), xytext=(dx, dy),
                textcoords='offset points', ha=ha, va=va, fontsize=8.2,
                linespacing=.92, color=ANNOTATION_GRAY, arrowprops=arrowprops,
                zorder=5, annotation_clip=True)


def plot_baselines(ax, rows, x_key, xerr_key, marker, show_xerr):
    """Plot fixed-size gray native baselines with selected unchanged SE bars."""
    for row in rows:
        x = float(row[x_key])
        y = float(row['expected_distance_mean'])
        ax.scatter(x, y, s=BASELINE_MARKER_SIZE ** 2, marker=marker,
                   facecolor=BASELINE_GRAY, edgecolor='white', linewidth=.6, zorder=3)
        xerr = float(row[xerr_key]) if show_xerr else None
        ax.errorbar(x, y, xerr=xerr, yerr=float(row['expected_distance_se']),
                    fmt='none', ecolor=BASELINE_GRAY, elinewidth=.95, capsize=2.3,
                    capthick=.9, zorder=4)
        annotate_baselines(ax, row, x_key)


def add_claim_pool_legend(fig):
    handles = [Patch(facecolor=color, edgecolor='none', label=label)
               for _group, label, color in POOLS]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.55, .012), ncol=3,
               title='Claim pool', title_fontsize=9.6, fontsize=9.2, frameon=False,
               borderaxespad=0, handlelength=1., handletextpad=.45, columnspacing=1.35)


def add_preference_cue(ax):
    ax.text(.035, .055, r'$\leftarrow$ lower cost' + '\n' + r'$\downarrow$ lower ED',
            transform=ax.transAxes, ha='left', va='bottom', fontsize=7.5,
            color=ANNOTATION_GRAY, linespacing=1.05, zorder=5)


def runtime_panel_rows(rows, n, strategy):
    """Return the exact saved All checkpoints and native rows for one panel."""
    claim = sorted((row for row in rows if row['method'] == 'SMC-Active'
                    and row['group'] == 'all' and row['strategy'] == strategy
                    and int(row['sample_size']) == n), key=lambda row: int(row['budget']))
    native = sorted((row for row in rows if row['strategy'] == 'native'
                     and int(row['sample_size']) == n), key=lambda row: row['method'])
    assert [int(row['budget']) for row in claim] == list(BUDGETS)
    assert sorted(row['method'] for row in native) == sorted(
        method for method, _label, _marker in RUNTIME_BASELINES)
    return claim, native


def render_runtime_figure(rows, output_pdf, output_png, output_svg=None):
    """Render Figure 3 from saved means and one-SE values without moving points."""
    plt.rcParams.update({
        'font.family': 'DejaVu Serif',
        'font.size': 9,
        'pdf.fonttype': 42,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.0), sharex=True, sharey=True)
    fig.subplots_adjust(left=.11, right=.985, bottom=.215, top=.91,
                        wspace=.17, hspace=.23)
    baseline_labels = {method: label for method, label, _marker in RUNTIME_BASELINES}
    for i, n in enumerate(SAMPLE_SIZES):
        for j, strategy in enumerate(STRATEGIES):
            ax = axes[i, j]
            claim, native = runtime_panel_rows(rows, n, strategy)
            for row in claim + native:
                x = float(row['approx_runtime_seconds_mean'])
                y = float(row['expected_distance_mean'])
                yerr = float(row['expected_distance_se'])
                if not (RUNTIME_LIMITS[0] < x < RUNTIME_LIMITS[1]
                        and RUNTIME_ED_LIMITS[0] < y - yerr
                        and y + yerr < RUNTIME_ED_LIMITS[1]):
                    raise ValueError('runtime figure limits would clip a point or ED error bar')

            for row in native:
                x = float(row['approx_runtime_seconds_mean'])
                y = float(row['expected_distance_mean'])
                ax.errorbar(x, y, yerr=float(row['expected_distance_se']), fmt='none',
                            ecolor=RUNTIME_BASELINE_ERROR_COLOR, elinewidth=.8,
                            capsize=2, capthick=.8, zorder=2)
                ax.plot(x, y, linestyle='none', marker='o', markersize=4.8,
                        markerfacecolor=RUNTIME_BASELINE_COLOR, markeredgecolor='white',
                        markeredgewidth=.4, zorder=2.5)
                if i == 0:
                    dx, dy, ha, va = RUNTIME_BASELINE_LABEL_OFFSETS[row['method']]
                    ax.annotate(baseline_labels[row['method']], (x, y), xytext=(dx, dy),
                                textcoords='offset points', ha=ha, va=va,
                                fontsize=7.6, color='#505960', zorder=6)

            for row in claim:
                x = float(row['approx_runtime_seconds_mean'])
                y = float(row['expected_distance_mean'])
                color = RUNTIME_QUERY_COLORS[int(row['budget'])]
                ax.errorbar(x, y, yerr=float(row['expected_distance_se']), fmt='none',
                            ecolor=color, elinewidth=.85, capsize=2,
                            capthick=.8, zorder=4)
                ax.plot(x, y, linestyle='none', marker='o', markersize=5.7,
                        markerfacecolor=color, markeredgecolor='white',
                        markeredgewidth=.5, zorder=5)

            ax.set_xscale('log')
            ax.set_xlim(*RUNTIME_LIMITS)
            ax.set_ylim(*RUNTIME_ED_LIMITS)
            ax.set_yticks((10, 15, 20, 25))
            ax.xaxis.set_major_locator(FixedLocator([value for value, _label in RUNTIME_TICKS]))
            ax.xaxis.set_major_formatter(FixedFormatter([label for _value, label in RUNTIME_TICKS]))
            for tick_text, (_value, label) in zip(ax.get_xticklabels(), RUNTIME_TICKS):
                if label == '12h':
                    tick_text.set_ha('right')
                elif label == '24h':
                    tick_text.set_ha('left')
            ax.xaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_minor_locator(NullLocator())
            ax.set_axisbelow(True)
            ax.grid(axis='y', which='major', color='#D9DDE1', linewidth=.55,
                    alpha=.55, zorder=0)
            for side in ('left', 'bottom'):
                ax.spines[side].set_color('#4B535A')
                ax.spines[side].set_linewidth(.8)
            ax.tick_params(axis='both', which='major', colors='#4B535A',
                           labelsize=8, length=2.8, width=.7,
                           labelbottom=i == 1, labelleft=j == 0)
            if i == 0:
                title = 'Stated confidence' if j == 0 else '8-response consistency'
                ax.set_title(title, fontsize=10.2, pad=5)

    fig.supylabel('ED', x=.012, fontsize=9.5)
    fig.supxlabel('Runtime', x=.55, y=.105, fontsize=9.5)
    for i, n in enumerate(SAMPLE_SIZES):
        box = axes[i, 0].get_position()
        fig.text(.065, (box.y0 + box.y1) / 2, rf'$n={n}$',
                 ha='center', va='center', fontsize=9.3)

    query_handles = [Line2D([], [], linestyle='none', marker='none', label='CLAIM-FUSE:')]
    query_handles += [Line2D([], [], linestyle='none', marker='o', markersize=5.7,
                             markerfacecolor=RUNTIME_QUERY_COLORS[budget],
                             markeredgecolor='white', markeredgewidth=.5,
                             label=str(budget) if budget != 100 else '100 queries')
                      for budget in BUDGETS]
    fig.legend(handles=query_handles, loc='lower center', bbox_to_anchor=(.55, .045),
               ncol=5, fontsize=8, frameon=False, borderaxespad=0,
               handlelength=.8, handletextpad=.25, columnspacing=.9)

    fig.savefig(output_pdf, bbox_inches='tight', pad_inches=.025)
    if output_svg is not None:
        fig.savefig(output_svg, bbox_inches='tight', pad_inches=.025)
    fig.savefig(output_png, dpi=400, bbox_inches='tight', pad_inches=.025)
    plt.close(fig)


def render_figure(rows, cost, output_pdf, output_png):
    """Render the paired figure, retaining Figure 4's existing styling."""
    if cost == 'runtime':
        return render_runtime_figure(rows, output_pdf, output_png)
    if cost != 'tokens':
        raise ValueError(f'unknown cost axis: {cost}')

    plt.rcParams.update({
        'font.family': 'DejaVu Serif',
        'font.size': 10,
        'pdf.fonttype': 42,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    fig = plt.figure(figsize=(7.2, 5.25))
    grid = fig.add_gridspec(2, 2, left=.105, right=.985, bottom=.205, top=.94,
                            wspace=.20, hspace=.24)
    axes = {}
    master = None
    pool_colors = {group: color for group, _label, color in POOLS}
    for i, n in enumerate(SAMPLE_SIZES):
        for j, strategy in enumerate(STRATEGIES):
            ax = fig.add_subplot(grid[i, j], sharex=master, sharey=master)
            axes[i, j] = ax
            if master is None:
                master = ax
            selected = [row for row in rows if row['method'] == 'SMC-Active'
                        and row['strategy'] == strategy and int(row['sample_size']) == n]
            for row in selected:
                plot_claim_fuse(ax, row, pool_colors[row['group']],
                                'total_tokens_mean', 'total_tokens_se', True)
            baselines = [row for row in rows
                         if row['strategy'] == 'native' and int(row['sample_size']) == n]
            plot_baselines(ax, baselines, 'total_tokens_mean', 'total_tokens_se', 'D', True)
            format_token_axis(ax, bottom=i == 1, left=j == 0)
            if i == 0:
                title = 'Stated Confidence' if j == 0 else 'Consistency (8 Responses)'
                ax.set_title(title, fontsize=11.2, pad=7)

    add_preference_cue(axes[1, 0])
    add_claim_pool_legend(fig)
    fig.supylabel('ED', x=.012, y=.57, fontsize=11.3)
    fig.supxlabel('Input + output tokens (log scale)', x=.55, y=.105, fontsize=11.3)
    row_label_x = axes[0, 0].get_position().x0 - .058
    for i, n in enumerate(SAMPLE_SIZES):
        panel_box = axes[i, 0].get_position()
        fig.text(row_label_x, (panel_box.y0 + panel_box.y1) / 2, rf'$n = {n}$',
                 rotation=90, ha='center', va='center', fontsize=10.3)

    fig.savefig(output_pdf, dpi=300, bbox_inches='tight', pad_inches=.02)
    fig.savefig(output_png, dpi=400, bbox_inches='tight', pad_inches=.02)
    plt.close(fig)


def write_metadata(rows, root, cap):
    plotted = []
    for n in SAMPLE_SIZES:
        for strategy in STRATEGIES:
            for row in rows:
                if (row['method'] == 'SMC-Active' and row['strategy'] == strategy
                        and int(row['sample_size']) == n):
                    plotted.append(dict(panel=[n, strategy], **row))
    (root / 'plotted_smc_points.json').write_text(json.dumps(plotted, indent=2) + '\n')
    lines = [
        '# Qwen recovery, approximate runtime, and recorded token expenditure',
        '',
        f'Qwen3.5-9B with response-confidence cap {cap:g}; ten paired data seeds. Figure 3 shows independent All Claim-Fuse checkpoints on a genuine logarithmic runtime axis, using progressively darker greens for larger attempted-query budgets and vertical ED error bars; native baselines are directly labeled in the top row. Figure 4 plots all claim pools against input plus output tokens. Both use fixed-size markers; cost is encoded only on the horizontal axis.',
        '',
        '| n | Confidence | Pool | Queries | Mean ED (SE) | Mean runtime, s | Mean tokens |',
        '|---:|---|---|---:|---:|---:|---:|',
    ]
    for row in rows:
        pool = row['group'] if row['method'] == 'SMC-Active' else row['method']
        budget = row['budget'] if row['method'] == 'SMC-Active' else 'native'
        lines.append(
            f"| {row['sample_size']} | {row['strategy']} | {pool} | {budget} | "
            f"{float(row['expected_distance_mean']):.2f} ({float(row['expected_distance_se']):.2f}) | "
            f"{float(row['approx_runtime_seconds_mean']):,.1f} | "
            f"{float(row['total_tokens_mean']):,.1f} |"
        )
    lines += [
        '',
        'No new model calls. Protocol and source hashes are in protocol.json; per-dataset totals and the Claim-Fuse response ledger are supplied separately.',
        'The Claim-Fuse points share initialization, acquisition, and confidence settings across claim pools. Baselines retain their native prompting and acquisition rules.',
    ]
    (root / 'report.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', default=DEFAULT_RUN)
    generate(parser.parse_args().run_id)
