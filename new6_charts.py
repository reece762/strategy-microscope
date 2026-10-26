#!/usr/bin/env python3
"""本輪三張卡的圖（修正版）。

修正原因（見 n6_final.py 與 ablation.py 的說明）：
1. 第一版用「每根 K 線符合條件就算一筆」的交易，平行通道 221,186 筆裡
   56% 是持倉期重複。改用去重後的樣本。
2. 第一版把「通道下軌 vs 跌破均線後站回」當成消融對照組——那不是
   同一策略拿掉一個條件，而是換了另一個訊號，導致「通道選得比較好」
   這個結論方向錯誤。改用真正的安慰劑組（均線支撐）與隨機組。
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
ABL = '/root/.openclaw/workspace/reports/_agent_pattern/ablation.json'

BG = '#0a0e14'
TX = '#c9d1d9'
TX2 = '#8b949e'
GRID = '#2a3138'
UP = '#3fb950'
DN = '#f85149'
BLUE = '#58a6ff'
GOLD = '#f0b429'
GRAY = '#6e7681'


def _style(ax, title, ylab='', xlab=''):
    ax.set_title(title, color=TX, fontsize=11, pad=8, loc='left')
    ax.tick_params(colors=TX2, labelsize=9, length=3)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    if ylab:
        ax.set_ylabel(ylab, color=TX2, fontsize=9)
    if xlab:
        ax.set_xlabel(xlab, color=TX2, fontsize=9)
    ax.set_facecolor(BG)
    ax.grid(True, color=GRID, lw=0.5, alpha=0.5)


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.patch.set_facecolor(BG)
    fig.savefig(p, facecolor=BG, bbox_inches='tight')
    plt.close(fig)
    return name


def chart_trendline(D):
    tl = D['trendline']
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    fig.patch.set_facecolor(BG)

    labs = [f'趨勢線突破\n（去重 {tl["dedup"]["n"]:,} 筆）',
            f'突破 60 日新高\n（去重 {tl["control"]["n"]:,} 筆）']
    raw = [tl['dedup']['avg'], tl['control']['avg']]
    exc = [tl['dedup']['excess'], tl['control']['excess']]
    x = np.arange(2)
    w = 0.36
    b1 = axes[0].bar(x - w / 2, np.array(raw) * 100, w, label='原始報酬', color=BLUE, alpha=.85)
    b2 = axes[0].bar(x + w / 2, np.array(exc) * 100, w, label='扣掉市場後', color=DN, alpha=.85)
    for bars in (b1, b2):
        for r in bars:
            h = r.get_height()
            axes[0].text(r.get_x() + r.get_width() / 2, h + (0.02 if h >= 0 else -0.09),
                         f'{h:+.2f}%', ha='center', color=TX, fontsize=9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labs, fontsize=8.5)
    axes[0].axhline(0, color=GRAY, lw=.8)
    axes[0].legend(facecolor=BG, edgecolor=GRID, labelcolor=TX, fontsize=8, loc='upper right')
    _style(axes[0], '平均每筆報酬：兩組扣掉市場後都轉負', ylab='%')

    by = pd.DataFrame(tl['by_year'])
    axes[1].bar(by['year'], by['ex'] * 100,
                color=[UP if v > 0 else DN for v in by['ex']], alpha=.85, width=.72)
    axes[1].axhline(0, color=GRAY, lw=.8)
    _style(axes[1], f'逐年超額：{tl["years_pos"]}/{tl["years_n"]} 年為正', ylab='超額 %')
    axes[1].set_xticks(list(by['year'])[::2])
    axes[1].tick_params(axis='x', labelsize=8)

    fig.tight_layout()
    return save(fig, 'tl_ab.png')


def chart_channel(D, A):
    ch = D['channel']
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    fig.patch.set_facecolor(BG)

    # 左：真正的消融——通道 vs 均線安慰劑 vs 隨機
    names = ['A 觸及通道下軌',
             'B 觸及 20 日均線\n（安慰劑）',
             'C 隨機同日']
    keys = ['channel', 'ma20', 'random']
    exc = [A[k]['dedup']['excess'] for k in keys]
    tst = [A[k]['dedup']['t'] for k in keys]
    cols = [DN, GOLD, GRAY]
    bars = axes[0].bar(names, np.array(exc) * 100, color=cols, alpha=.85, width=.6)
    for i, (r, t_) in enumerate(zip(bars, tst)):
        h = r.get_height()
        axes[0].text(i, h - 0.028, f'{h:+.2f}%\nt={t_:+.1f}', ha='center',
                     color=TX, fontsize=8.5, va='top')
    axes[0].axhline(0, color=GRAY, lw=.8)
    axes[0].set_ylim(min(exc) * 100 * 1.35, 0.05)
    axes[0].tick_params(axis='x', labelsize=8.5)
    _style(axes[0], '消融：通道幾何與均線支撐幾乎相同', ylab='超額 %')

    # 右：出場原因
    rs = ch['dedup']['reasons']
    names2 = {'ma20': '跌破 20 日均線', 'stop': '停損 3%',
              'target': '到達通道上軌', 'time': '期滿'}
    items = sorted(rs.items(), key=lambda kv: -kv[1])
    labs = [names2.get(k, k) for k, _ in items]
    vals = [v * 100 for _, v in items]
    cols2 = [names2.get(k) == '到達通道上軌' and UP or (k == 'stop' and DN or BLUE)
             for k, _ in items]
    axes[1].barh(labs, vals, color=cols2, alpha=.85)
    for i, v in enumerate(vals):
        axes[1].text(v + 1, i, f'{v:.1f}%', va='center', color=TX, fontsize=9)
    _style(axes[1], f'出場原因：只有 {rs.get("target",0)*100:.1f}% 走到上軌', xlab='佔比 %')

    fig.tight_layout()
    return save(fig, 'ch_ab.png')


def chart_panic(P):
    hold, base = P['hold'], P['base']
    hs = sorted(int(k) for k in hold)
    fig, ax = plt.subplots(figsize=(8.2, 3.7))
    fig.patch.set_facecolor(BG)
    x = np.arange(len(hs))
    w = .36
    vp = [hold[str(h)]['mean'] * 100 for h in hs]
    vb = [base[str(h)] * 100 for h in hs]
    b1 = ax.bar(x - w / 2, vp, w, label='恐慌日隔天買進', color=DN, alpha=.85)
    b2 = ax.bar(x + w / 2, vb, w, label='隨便哪天買進（無條件）', color=GRAY, alpha=.85)
    for bars in (b1, b2):
        for r in bars:
            h = r.get_height()
            ax.text(r.get_x() + r.get_width() / 2, h + (.08 if h >= 0 else -.2),
                    f'{h:+.2f}%', ha='center', color=TX, fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f'持有 {h} 日' for h in hs], fontsize=9)
    ax.axhline(0, color=GRAY, lw=.8)
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=TX, fontsize=8.5)
    _style(ax, f'恐慌日買進：四個持有期全部不如「隨便哪天買」（{P["events"]} 天）',
           ylab='報酬 %')
    fig.tight_layout()
    return save(fig, 'pn_hold.png')


def main():
    D = json.load(open(os.path.join(OUT, 'n6_final.json')))
    A = json.load(open(ABL))
    P = json.load(open(os.path.join(OUT, 'panic_keep.json')))

    f1 = chart_trendline(D)
    f2 = chart_channel(D, A)
    f3 = chart_panic(P)

    res = dict(trendline=D['trendline'], channel=D['channel'],
               ablation={k: v['dedup'] for k, v in A.items()},
               panic=P, charts=dict(tl=f1, ch=f2, pn=f3))
    json.dump(res, open(os.path.join(OUT, 'new6_cards.json'), 'w'),
              ensure_ascii=False, indent=1)

    print('已寫 out/new6_cards.json')
    tl, ch = D['trendline'], D['channel']
    print(f'  趨勢線 去重 n={tl["dedup"]["n"]:,} 超額 {tl["dedup"]["excess"]*100:+.3f}% t={tl["dedup"]["t"]:+.2f}')
    print(f'  通道   去重 n={ch["dedup"]["n"]:,} 超額 {ch["dedup"]["excess"]*100:+.3f}% t={ch["dedup"]["t"]:+.2f}')
    for k in ('channel', 'ma20', 'random'):
        a = A[k]['dedup']
        print(f'  消融 {k:8s} n={a["n"]:>7,} 超額 {a["excess"]*100:+.3f}% t={a["t"]:+.2f}')


if __name__ == '__main__':
    main()
