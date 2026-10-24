#!/usr/bin/env python3
"""老魯兩張可回測卡片的圖。

兩張都不是「可持有策略」，所以沒有淨值曲線可畫；圖要回答各自的問題：
  1. 蝶式：單次交易的損益分布 —— 勝率 24% 長什麼樣子、賠的那一半賠多少
  2. 九方圖：九方圖價位 vs 同距離隨機鄰近價位的觸及率 —— 有差嗎
"""
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
BASE = '/root/.openclaw/workspace/backtest_videos/data_daily'
LAOLU = '/root/.openclaw/workspace/laolu'

BG = '#0a0e14'
TX = '#c9d1d9'
TX2 = '#8b949e'
GRID = '#8b949e'
UP = '#3fb950'
DN = '#f85149'
BLUE = '#58a6ff'
GOLD = '#f0b429'
GRAY = '#6e7681'


def _style(ax, title, ylab=''):
    ax.set_title(title, color=TX, fontsize=11, pad=8, loc='left')
    ax.tick_params(colors=TX2, labelsize=9, length=3)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(alpha=.13, color=GRID, lw=.4, axis='y')
    if ylab:
        ax.set_ylabel(ylab, color=TX2, fontsize=9)


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f'lao_{name}.png')
    fig.savefig(path, format='png', facecolor=BG, bbox_inches='tight', pad_inches=.15)
    plt.close(fig)
    return path


def load(sym):
    d = pd.read_csv(f'{BASE}/{sym}.csv')
    d['date'] = pd.to_datetime(d['date'], format='%m/%d/%Y')
    return d.sort_values('date').reset_index(drop=True)


# ------------------------------------------------------------ 1. 蝶式損益分布
def bfly_chart():
    """他影片的數字：中心 425、寬 11 點、成本 6.15、15 日到期。
    把每一段 15 日報酬縮放到中心價，算到期損益。"""
    center, w, cost, h = 425.0, 11.0, 6.15, 15
    c = load('SPY')['close'].values
    r = c[h:] / c[:-h] - 1.0
    S = center * (1 + r)
    pnl = (np.maximum(S - (center - w), 0) - 2 * np.maximum(S - center, 0)
           + np.maximum(S - (center + w), 0) - cost)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9),
                            gridspec_kw={'width_ratios': [1.35, 1]})

    # 左：損益分布
    ax = axes[0]
    bins = np.linspace(pnl.min(), pnl.max(), 60)
    ax.hist(pnl[pnl <= 0], bins=bins, color=DN, alpha=.85, label='虧損')
    ax.hist(pnl[pnl > 0], bins=bins, color=UP, alpha=.85, label='獲利')
    ax.axvline(0, color=TX, lw=1, ls='--', alpha=.7)
    ax.axvline(pnl.mean(), color=GOLD, lw=1.8)
    ax.annotate(f'平均 {pnl.mean():+.2f}', xy=(pnl.mean(), ax.get_ylim()[1] * .72),
                xytext=(6, 0), textcoords='offset points', color=GOLD,
                fontsize=9, fontweight='bold')
    _style(ax, '多頭蝶式：2,497 段 15 日報酬的單次損益', '段數')
    ax.set_xlabel('每股損益（美元）· 成本 6.15', color=TX2, fontsize=9)
    ax.legend(frameon=False, labelcolor=TX, fontsize=8.5, loc='upper left')

    win = (pnl > 0).mean()
    ax.text(.98, .93, f'勝率 {win*100:.1f}%', transform=ax.transAxes, ha='right',
            color=TX, fontsize=10, fontweight='bold')

    # 右：多頭 vs 空頭
    ax = axes[1]
    labels = ['多頭蝶式\n（他說是買彩票）', '空頭蝶式\n（他推薦的方向）']
    wins = [win * 100, (1 - win) * 100]
    means = [pnl.mean(), -pnl.mean()]
    x = np.arange(2)
    ax.bar(x - .17, wins, .32, color=[DN, UP], alpha=.9, label='勝率 %')
    ax.bar(x + .17, [m * 10 for m in means], .32, color=[GRAY, GOLD],
           alpha=.9, label='平均損益 ×10')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=TX, fontsize=8.5)
    for i, (wv, mv) in enumerate(zip(wins, means)):
        ax.text(i - .17, wv + 1.5, f'{wv:.1f}%', ha='center', color=TX,
                fontsize=9, fontweight='bold')
        ax.text(i + .17, mv * 10 + (1.5 if mv > 0 else -3.5), f'{mv:+.2f}',
                ha='center', color=TX, fontsize=9, fontweight='bold')
    ax.axhline(0, color=TX2, lw=.8)
    _style(ax, '同一組價差翻面後：勝率與期望值同時翻正')
    ax.legend(frameon=False, labelcolor=TX, fontsize=8, loc='upper left')

    return _save(fig, 'bfly')


# ------------------------------------------------------------ 2. 九方圖觸及率
def gann_chart():
    d = json.load(open(f'{LAOLU}/laolu_calc.json'))
    g, rev = d['gann'], d['gann_reversal']
    syms = [s for s in ('SPY', 'QQQ', 'IWM') if s in g]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9))

    # 左：觸及率
    ax = axes[0]
    x = np.arange(len(syms))
    gv = [g[s]['gann_touch_rate'] * 100 for s in syms]
    cv = [g[s]['jitter_control_touch_rate'] * 100 for s in syms]
    ax.bar(x - .17, gv, .32, color=BLUE, alpha=.9, label='九方圖價位')
    ax.bar(x + .17, cv, .32, color=GRAY, alpha=.9, label='同距離隨機鄰近價位')
    for i, (a, b) in enumerate(zip(gv, cv)):
        ax.text(i, max(a, b) + 1.2, f'差 {a-b:+.2f}pp', ha='center',
                color=TX2, fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s}\n{g[s]["events_levels"]} 個價位' for s in syms],
                       color=TX, fontsize=8.5)
    ax.set_ylim(0, max(gv + cv) * 1.28)
    _style(ax, '未來 60 根內觸及率：兩組幾乎重疊', '%')
    ax.legend(frameon=False, labelcolor=TX, fontsize=8.5, loc='upper right')

    # 右：觸及後 20 根的反轉報酬＋t 值
    ax = axes[1]
    gv2 = [rev[s]['gann_reversal_mean_pct'] for s in syms]
    cv2 = [rev[s]['jitter_reversal_mean_pct'] for s in syms]
    ax.bar(x - .17, gv2, .32, color=BLUE, alpha=.9, label='九方圖價位')
    ax.bar(x + .17, cv2, .32, color=GRAY, alpha=.9, label='隨機鄰近價位')
    ax.axhline(0, color=TX2, lw=.8)
    for i, s in enumerate(syms):
        ax.text(i, max(gv2[i], cv2[i]) + .05 if max(gv2[i], cv2[i]) > 0
                else min(gv2[i], cv2[i]) - .12,
                f't={rev[s]["t"]:+.2f}', ha='center', color=GOLD,
                fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(syms, color=TX, fontsize=9)
    _style(ax, '觸及後 20 根的反轉報酬（t 值全部接近零）', '%')
    ax.legend(frameon=False, labelcolor=TX, fontsize=8.5, loc='lower left')

    return _save(fig, 'gann')


if __name__ == '__main__':
    keys = {}
    keys['bfly'] = bfly_chart()
    keys['gann'] = gann_chart()
    out = {f'lao_{k}': os.path.basename(v) for k, v in keys.items()}
    json.dump(out, open(os.path.join(OUT, 'lao_charts.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))
