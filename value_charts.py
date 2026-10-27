#!/usr/bin/env python3
"""兩個價值型選股方法的圖。"""
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
UP, DN, BLUE, GOLD, GRAY, PUR = '#3fb950', '#f85149', '#58a6ff', '#f0b429', '#6e7681', '#bc8cff'


def style(ax, title, ylab='', xlab=''):
    ax.set_title(title, color=TX, fontsize=10, pad=8, loc='left')
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


def chart_fragility(D):
    """左：兩方法逐年超額；右：剔除最極端一年後的崩塌。"""
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.7),
                             gridspec_kw={'width_ratios': [1.75, 1]})
    fig.patch.set_facecolor(BG)

    ax = axes[0]
    for key, color, label, off in (('everything_money', BLUE, 'Everything Money 8 支柱', -0.2),
                                   ('phil_town', GOLD, 'Phil Town Big Five', 0.2)):
        y = pd.DataFrame(D[key]['yearly'])
        ex = (y['ret'] - y['rand']) * 100
        xs = [int(r[:4]) + off for r in y['asof']]
        ax.bar(xs, ex, width=.36, color=color, alpha=.85, label=label)
    ax.axhline(0, color=GRAY, lw=.9)
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=TX, fontsize=8, loc='upper right')
    ax.annotate('2014：只選 2 檔\n+51.9pp → 全靠這一年', xy=(2014, 45), xytext=(2016.2, 33),
                color=DN, fontsize=7.5,
                arrowprops=dict(arrowstyle='->', color=DN, lw=1.2))
    style(ax, '逐年超額（策略 − 隨機同檔數）', ylab='超額報酬 pp',
          xlab='再平衡年度')

    ax = axes[1]
    s = D['everything_money']
    y = pd.DataFrame(s['yearly'])
    ex = (y['ret'] - y['rand']).to_numpy()
    i = int(np.argmax(np.abs(ex)))
    rest = np.delete(ex, i)
    labels = ['全部 12 年', '剔除 2014', '只算選出\n≥10 檔的 10 年']
    big = y[y['k'] >= 10]
    e2 = (big['ret'] - big['rand']).to_numpy()
    vals = [ex.mean(), rest.mean(), e2.mean()]
    bars = ax.bar(labels, [v * 100 for v in vals], color=[BLUE, DN, DN], alpha=.9, width=.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 100 + 0.25,
                f'{v*100:+.2f}pp', ha='center', color=TX, fontsize=9, fontweight='bold')
    ax.axhline(0, color=GRAY, lw=.9)
    ax.set_ylim(-0.5, 5.4)
    style(ax, 'Everything Money：優勢僅來自單一年份', ylab='平均超額 pp')
    fig.tight_layout()
    return save(fig, 'val_fragility.png')


def chart_roic(D):
    """各條單獨使用：ROIC 是兩套方法唯一共同為正的一條。"""
    pf = D['pt_flags']; ef = D['em_flags']
    pt_names = {'pt1_roic': 'ROIC≥10%', 'pt2_sales': '營收成長≥10%',
                'pt3_eps': 'EPS成長≥10%', 'pt4_equity': '權益成長≥10%',
                'pt5_fcf': '現金流成長≥10%'}
    em_names = {'em1_pe5': '5年PE<22.5', 'em2_roic5': '5年ROIC>9%',
                'em3_rev': '營收成長', 'em4_ni': '淨利成長',
                'em5_shares': '股數未增', 'em6_debt': '負債/現金流<5',
                'em7_ocf': '現金流成長', 'em8_mcap': '市值/現金流<20'}

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 3.5))
    fig.patch.set_facecolor(BG)

    for ax, flags, names, title, hl in (
        (axes[0], pf, pt_names, 'Phil Town（門檻皆 10%）', 'pt1_roic'),
        (axes[1], ef, em_names, 'Everything Money（各柱門檻不同）', 'em2_roic5'),
    ):
        ks = [k for k in names if k in flags and 'note' not in flags[k]]
        vals = [flags[k]['excess'] * 100 for k in ks]
        order = np.argsort(vals)[::-1]
        ks = [ks[i] for i in order]
        vals = [vals[i] for i in order]
        yy = np.arange(len(ks))
        colors = [GOLD if k == hl else (UP if v > 0.5 else (GRAY if abs(v) < 0.5 else DN))
                  for k, v in zip(ks, vals)]
        ax.barh(yy, vals, color=colors, alpha=.9, height=.62)
        ax.axvline(0, color=GRAY, lw=.9)
        for i, (k, v) in enumerate(zip(ks, vals)):
            b = flags[k]
            ax.text(v + (0.12 if v >= 0 else -0.12), i,
                    f'{v:+.2f}pp {b["beat"]}/{b["n_years"]}',
                    va='center', ha='left' if v >= 0 else 'right',
                    color=TX if k == hl else TX2, fontsize=8,
                    fontweight='bold' if k == hl else 'normal')
        ax.set_yticks(yy)
        ax.set_yticklabels([names[k] for k in ks], fontsize=8.5)
        lo, hi = min(vals), max(vals)
        ax.set_xlim(lo - 1.6, hi + 1.7)
        ax.invert_yaxis()
        style(ax, title, xlab='超額報酬 pp（後為勝過基準年數）')

    axes[0].text(0.99, 0.03, '金柱＝ROIC', transform=axes[0].transAxes,
                 color=GOLD, fontsize=7.5, ha='right')
    fig.suptitle('兩套方法唯一共同為正的一條：ROIC（投入資本報酬率）',
                 color=TX, fontsize=10.5, x=0.02, ha='left', y=1.03)
    fig.tight_layout()
    return save(fig, 'val_roic.png')


def main():
    D = json.load(open(os.path.join(OUT, 'value_screen.json')))
    charts = dict(fragility=chart_fragility(D), roic=chart_roic(D))
    D['charts'] = charts
    json.dump(D, open(os.path.join(OUT, 'value_all.json'), 'w'),
              ensure_ascii=False, indent=1, default=str)
    print('已寫 value_all.json')
    print('  charts:', charts)


if __name__ == '__main__':
    main()
