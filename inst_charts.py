#!/usr/bin/env python3
"""13F 機構持倉分析的兩張圖（給策略顯微鏡用）。"""
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
SRC = '/root/.openclaw/workspace/backtest_videos/institutional/panel_full.csv'

BG, TX, TX2, GRID = '#0a0e14', '#c9d1d9', '#8b949e', '#2a3138'
UP, DN, BLUE, GOLD, GRAY = '#3fb950', '#f85149', '#58a6ff', '#f0b429', '#6e7681'


def style(ax, title, ylab=''):
    ax.set_title(title, color=TX, fontsize=10, pad=8, loc='left')
    ax.tick_params(colors=TX2, labelsize=8.5, length=3)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    if ylab:
        ax.set_ylabel(ylab, color=TX2, fontsize=8.5)
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, lw=0.5, alpha=0.45)


def save(fig, name):
    fig.patch.set_facecolor(BG)
    fig.savefig(os.path.join(OUT, name), facecolor=BG, bbox_inches='tight')
    plt.close(fig)
    return name


def main():
    p = pd.read_csv(SRC, parse_dates=['period'])
    col = 'ba_200_1e8'
    rows = []
    for per, g in p.dropna(subset=[col]).groupby('period'):
        g = g.copy()
        g['r'] = g[col].rank(method='first')
        g['q'] = pd.qcut(g.r, 5, labels=False)
        m = g.groupby('q').mkt.mean()
        rows.append(dict(period=per, spread=m.iloc[-1] - m.iloc[0], top=m.iloc[-1], n=len(g)))
    s = pd.DataFrame(rows).sort_values('period')
    s['roll4'] = s.spread.rolling(4).mean()

    # 1) 每季多空差距
    fig, ax = plt.subplots(figsize=(11.4, 3.6))
    fig.patch.set_facecolor(BG)
    colors = [BLUE if d < pd.Timestamp('2020-01-01') else UP for d in s.period]
    ax.bar(s.period, s.spread * 100, width=60, color=colors, alpha=.75)
    ax.plot(s.period, s.roll4 * 100, color=GOLD, lw=1.8, label='4 季移動平均')
    ax.axhline(0, color=GRAY, lw=.9)
    ax.axvline(pd.Timestamp('2020-01-01'), color=TX2, lw=.9, ls='--')
    ax.text(pd.Timestamp('2019-10-01'), ax.get_ylim()[1] * .92, '樣本內 2013–2019',
            ha='right', color=BLUE, fontsize=8)
    ax.text(pd.Timestamp('2020-04-01'), ax.get_ylim()[1] * .92, '樣本外 2020–2026',
            ha='left', color=UP, fontsize=8)
    style(ax, '每季「主動型經理人加碼家數」最高兩成減最低兩成的報酬差距', '每季差距（%）')
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=TX, fontsize=8, loc='lower right')
    save(fig, 'n12_13f_spread.png')

    # 2) 累積：只做多最高兩成 vs 大盤
    cum_top = (1 + s.top).cumprod()
    s['year'] = s.period.dt.year
    fig, ax = plt.subplots(figsize=(11.4, 3.4))
    fig.patch.set_facecolor(BG)
    ax.plot(s.period, cum_top, color=UP, lw=2.0, label='最高兩成（只做多，扣大盤後累積）')
    ax.axhline(1, color=GRAY, lw=.9)
    ax.axvline(pd.Timestamp('2020-01-01'), color=TX2, lw=.9, ls='--')
    style(ax, '只做多最高兩成：51 季累積（每季 +1.5% 左右，未扣成本）', '累積倍數')
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=TX, fontsize=8, loc='upper left')
    save(fig, 'n12_13f_cum.png')

    json.dump({'charts': {'13f_spread': 'n12_13f_spread.png',
                          '13f_cum': 'n12_13f_cum.png'}},
              open(os.path.join(OUT, 'inst_all.json'), 'w'))
    print('done; mean spread', round(s.spread.mean() * 100, 2), '%')


if __name__ == '__main__':
    main()
