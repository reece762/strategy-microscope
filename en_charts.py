#!/usr/bin/env python3
"""英文選股影片的兩張圖。"""
import json
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 155

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')

BG, TX, TX2, GRID = '#0a0e14', '#c9d1d9', '#8b949e', '#2a3138'
UP, DN, BLUE, GOLD, GRAY = '#3fb950', '#f85149', '#58a6ff', '#f0b429', '#6e7681'


def style(ax, title, ylab='', xlab=''):
    ax.set_title(title, color=TX, fontsize=10.5, pad=8, loc='left')
    ax.tick_params(colors=TX2, labelsize=8.5, length=3)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    if ylab:
        ax.set_ylabel(ylab, color=TX2, fontsize=8.5)
    if xlab:
        ax.set_xlabel(xlab, color=TX2, fontsize=8.5)
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, lw=0.5, alpha=0.45)


def save(fig, name):
    fig.patch.set_facecolor(BG)
    fig.savefig(os.path.join(OUT, name), facecolor=BG, bbox_inches='tight')
    plt.close(fig)
    return name


def chart_main(D):
    """左：逐年超額（正負各半）；右：12 年平均 vs 標準差。"""
    Y = pd.DataFrame([r for r in D['yearly'] if r['k'] > 0]).reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 3.6),
                             gridspec_kw={'width_ratios': [1.55, 1]})
    fig.patch.set_facecolor(BG)

    ax = axes[0]
    ex = (Y['ret'] - Y['rand_mean']) * 100
    yrs = [r[:7] for r in Y['asof']]
    colors = [UP if v > 0 else DN for v in ex]
    ax.bar(range(len(ex)), ex, color=colors, alpha=.85)
    ax.axhline(0, color=GRAY, lw=.9)
    ax.axhline(ex.mean(), color=GOLD, lw=1.5, ls='--',
               label=f'平均超額 {ex.mean():+.2f}pp')
    for i, v in enumerate(ex):
        ax.text(i, v + (0.8 if v >= 0 else -1.8), f'{v:+.0f}', ha='center',
                color=TX2, fontsize=7)
    ax.set_xticks(range(len(yrs)))
    ax.set_xticklabels(yrs, rotation=45, ha='right', fontsize=7.5)
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=TX, fontsize=8)
    style(ax, '逐年超額（篩選組合 − 隨機同檔數）：正負各半，平均貼在零',
          ylab='超額報酬 pp')

    ax = axes[1]
    s = D['screen_all']
    labels = ['篩選組合', '全體等權', '隨機同檔數']
    vals = [s['mean_ret'] * 100, s['mean_univ'] * 100, s['mean_rand'] * 100]
    bars = ax.bar(labels, vals, color=[BLUE, GRAY, GRAY], alpha=.9, width=.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f'{v:+.2f}%',
                ha='center', color=TX, fontsize=9)
    ax.set_ylim(0, max(vals) * 1.28)
    ax.errorbar([0], [vals[0]], yerr=[s['mean_ret'] * 0 + 11.89 / np.sqrt(len(Y))],
                fmt='none', ecolor=GOLD, capsize=6, lw=1.5)
    ax.text(0, vals[0] + 2.0, f"超額 vs 隨機\n{s['excess_vs_rand']*100:+.2f}pp (t=0.07)",
            ha='center', color=GOLD, fontsize=8, fontweight='bold')
    style(ax, '12 年平均：三者幾乎相同', ylab='年化報酬 %')
    fig.tight_layout()
    return save(fig, 'en_screen.png')


def chart_filters(D):
    """各條篩選器的超額貢獻——只有營收成長那條明顯。"""
    pf = D['per_filter']
    names = {'f1_px50': '5年漲幅>50%', 'f2_pe': 'PE 10~50', 'f4_peg': 'PEG<2',
             'f5_eps': 'EPS>0', 'f6_revg': '營收5年CAGR>20%', 'f7_nig': '淨利5年CAGR>20%',
             'f8_margin': '淨利率>10%', 'f9_ocf': '現金流>0（替代）'}
    order = ['f6_revg', 'f7_nig', 'f4_peg', 'f5_eps', 'f9_ocf', 'f1_px50', 'f8_margin', 'f2_pe']
    ex = [pf[k]['excess'] * 100 for k in order]
    beat = [f"{pf[k]['beat']}/{pf[k]['n_years']}" for k in order]

    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    fig.patch.set_facecolor(BG)
    y = np.arange(len(order))
    ax.barh(y, ex, color=[UP if v > 1 else (GRAY if abs(v) < 1 else DN) for v in ex],
            alpha=.88, height=.62)
    ax.axvline(0, color=GRAY, lw=.9)
    for i, (v, b) in enumerate(zip(ex, beat)):
        ax.text(v + (0.15 if v >= 0 else -0.15), i, f'{v:+.2f}pp  ({b})',
                va='center', ha='left' if v >= 0 else 'right', color=TX, fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels([names[k] for k in order], fontsize=8.5)
    ax.set_xlim(min(ex) - 2.2, max(ex) + 2.6)
    ax.invert_yaxis()
    style(ax, '單獨使用各條篩選器：只有「營收成長」明顯有效（括號為勝過基準的年數）',
          xlab='相對全體等權的超額報酬 pp')
    fig.tight_layout()
    return save(fig, 'en_filters.png')


def main():
    D = json.load(open(os.path.join(OUT, 'en_screen.json')))
    charts = dict(main=chart_main(D), filters=chart_filters(D))
    D['charts'] = charts
    json.dump(D, open(os.path.join(OUT, 'en_all.json'), 'w'), ensure_ascii=False, indent=1,
              default=str)
    print('已寫 en_all.json')
    s = D['screen_all']
    print(f"  篩選 {s['mean_ret']*100:+.2f}%  全體 {s['mean_univ']*100:+.2f}%  "
          f"隨機 {s['mean_rand']*100:+.2f}%  超額(vs隨機) {s['excess_vs_rand']*100:+.2f}pp")
    print('  charts:', charts)


if __name__ == '__main__':
    main()
