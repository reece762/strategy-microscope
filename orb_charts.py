#!/usr/bin/env python3
"""Charts for the ORB card (out/*.png)."""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 155

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
BG, TX, TX2, GRID = '#0a0e14', '#c9d1d9', '#8b949e', '#21262d'
UP, DN, BLUE, GOLD = '#3fb950', '#f85149', '#58a6ff', '#f0b429'


def style(ax, title, sub=''):
    ax.set_facecolor(BG)
    ax.set_title(title, color=TX, fontsize=12, pad=16 if sub else 8, loc='left')
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, color=TX2, fontsize=9, va='bottom')
    ax.tick_params(colors=TX2, labelsize=9, length=3)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def chart_grid():
    g = pd.read_csv(os.path.join(OUT, 'orb_grid.csv'))
    fig, ax = plt.subplots(figsize=(9.6, 5.0), facecolor=BG)
    bins = np.arange(-0.24, 0.06, 0.01)
    ax.hist(g.mean_r, bins=bins, color=BLUE, alpha=0.75, edgecolor=BG)
    ax.axvline(0, color=TX2, lw=1.2, ls='--')
    ax.axvline(g.mean_r.median(), color=GOLD, lw=1.4)
    best = g.loc[g.mean_r.idxmax()]
    ax.scatter([best.mean_r], [2.5], color=UP, zorder=5, s=45)
    ax.text(-0.235, ax.get_ylim()[1]*0.92,
            f'最佳：{best.label}  {best.mean_r:+.3f}R（t={best.t:.2f}）',
            color=UP, fontsize=8.8, va='top')
    ax.text(g.mean_r.median() - 0.004, ax.get_ylim()[1]*0.92,
            f'中位數 {g.mean_r.median():+.3f}R', color=GOLD, fontsize=9, ha='right')
    style(ax, 'ORB 的 160 種參數組合，全部集中在零附近',
          f'{len(g)} 組＝5 個標的 × 4 種區間長度 × 2 種停損 × 4 種目標；'
          f'正的只有 {int((g.mean_r>0).sum())} 組，且沒有一組 t > 1')
    ax.set_xlabel('每筆交易平均 R（扣除 1 bp／邊成本）', color=TX2, fontsize=9)
    ax.set_ylabel('參數組合數', color=TX2, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'orb_grid_hist.png'), facecolor=BG)
    plt.close(fig)


def chart_filters():
    f = pd.read_csv(os.path.join(OUT, 'orb_filters.csv'))
    f = f[f.condition != '全部'].copy()
    f = f.sort_values('daily_mean_r')
    fig, ax = plt.subplots(figsize=(9.8, 5.6), facecolor=BG)
    y = np.arange(len(f))
    colors = [UP if v > 0 else DN for v in f.daily_mean_r]
    ax.barh(y, f.daily_mean_r, color=colors, alpha=0.85, height=0.62)
    ax.axvline(0, color=TX2, lw=1.0, ls='--')
    for i, r in enumerate(f.itertuples()):
        x = r.daily_mean_r
        ax.text(x + (0.004 if x >= 0 else -0.004), i,
                f'{x:+.3f}  (n={r.n_trades}, t={r.t:.1f})',
                va='center', ha='left' if x >= 0 else 'right',
                color=TX2, fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(f.condition, color=TX, fontsize=9)
    style(ax, '影片說的「濾網」大多沒有改善，唯一偏正的是突破量',
          '15 分鐘開盤區間、全區間停損、2R 目標；數值為每日等權平均 R')
    ax.set_xlabel('每日平均 R（正＝有優勢）', color=TX2, fontsize=9)
    ax.set_xlim(-0.24, 0.19)
    ax.margins(y=0.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'orb_filters.png'), facecolor=BG)
    plt.close(fig)


def chart_expansion():
    e = pd.read_csv(os.path.join(OUT, 'orb_expansion.csv'))
    pats = [('inside', '前一日內包日'), ('nr4', '前一日 NR4'), ('nr7', '前一日 NR7')]
    fig, ax = plt.subplots(figsize=(8.6, 4.8), facecolor=BG)
    x = np.arange(len(pats)); w = 0.34
    a = [e[e.pattern == p].next_mean.mean() for p, _ in pats]
    b = [e[e.pattern == p].other_mean.mean() for p, _ in pats]
    ax.bar(x - w/2, a, w, color=GOLD, alpha=0.9, label='前一日出現該形態')
    ax.bar(x + w/2, b, w, color=BLUE, alpha=0.65, label='其他日子')
    for i in range(len(pats)):
        ax.text(x[i]-w/2, a[i]+0.015, f'{a[i]:.2f}', ha='center', color=TX, fontsize=9)
        ax.text(x[i]+w/2, b[i]+0.015, f'{b[i]:.2f}', ha='center', color=TX, fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([n for _, n in pats], color=TX, fontsize=10)
    ax.set_ylim(0, 1.15)
    style(ax, '「波動收縮後會擴張」只對內包日成立，NR4／NR7 反而更縮',
          '縱軸＝次日日內振幅 ÷ 14 日平均真實振幅（1.0＝與平常一樣大）；5 個標的平均')
    ax.set_ylabel('次日振幅／ATR14', color=TX2, fontsize=9)
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=TX, fontsize=8.5, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'orb_expansion.png'), facecolor=BG)
    plt.close(fig)


def chart_equity():
    frames = []
    for sym in ['QQQ', 'SPY', 'IWM', 'NVDA', 'AAPL']:
        p = os.path.join(OUT, f'orb_trades_{sym}.csv')
        if os.path.exists(p):
            d = pd.read_csv(p); d['sym'] = sym; frames.append(d)
    alltr = pd.concat(frames, ignore_index=True)
    port = alltr.groupby('date').r.mean().sort_index()
    fig, ax = plt.subplots(figsize=(9.6, 4.6), facecolor=BG)
    ax.plot(pd.to_datetime(port.index), port.cumsum(), color=BLUE, lw=1.8,
            label='五標的等權組合')
    for sym, c in [('QQQ', GOLD), ('SPY', UP)]:
        s = alltr[alltr.sym == sym].groupby('date').r.mean().sort_index()
        ax.plot(pd.to_datetime(s.index), s.cumsum(), color=c, lw=1.1, alpha=0.8, label=sym)
    ax.axhline(0, color=TX2, lw=0.9, ls='--')
    style(ax, 'ORB 累積 R：兩年下來繞著零上下，沒有累積優勢',
          '每筆風險固定 1R；已扣 1 bp／邊成本；每日等權')
    ax.set_ylabel('累積 R', color=TX2, fontsize=9)
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=TX, fontsize=8.5, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'orb_equity.png'), facecolor=BG)
    plt.close(fig)


if __name__ == '__main__':
    chart_grid()
    chart_filters()
    chart_expansion()
    chart_equity()
    print('charts done')
