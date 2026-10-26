#!/usr/bin/env python3
"""第三批三張圖。"""
import json
import os

import matplotlib
import numpy as np

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


def chart_turtle(T):
    """左：策略 vs 對齊基準；右：他的「砍虧損」主張，確實成立。"""
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.5))
    fig.patch.set_facecolor(BG)

    labels = ['20 進 / 10 出', '55 進 / 20 出']
    vals = [T['20_10'], T['55_20']]
    x = np.arange(2)
    w = 0.34

    ax = axes[0]
    strat = [v['trim'] * 100 for v in vals]
    base = [v['base']['trim'] * 100 for v in vals]
    ax.bar(x - w / 2, strat, w, label='唐奇安策略', color=BLUE, alpha=.9)
    ax.bar(x + w / 2, base, w, label='對齊基準（任意日買、持有同天數）', color=GRAY, alpha=.9)
    for i, (s, b) in enumerate(zip(strat, base)):
        ax.text(i - w / 2, s + 0.08, f'{s:+.2f}%', ha='center', color=TX, fontsize=8)
        ax.text(i + w / 2, b + 0.08, f'{b:+.2f}%', ha='center', color=TX, fontsize=8)
        ax.annotate('', xy=(i + w / 2, b), xytext=(i - w / 2, s),
                    arrowprops=dict(arrowstyle='->', color=DN, lw=1.2))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n持有 {v['bars']} 天" for l, v in zip(labels, vals)], fontsize=8.5)
    ax.axhline(0, color=GRAY, lw=.8)
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=TX, fontsize=7.5, loc='upper left')
    style(ax, '突破進場沒有加分：都低於「隨便買也一樣」', ylab='截尾平均報酬 %')

    # 右：他的風險管理主張
    ax = axes[1]
    o = T['20_10']
    things = ['勝率', '賺賠比', '跌逾20%\n的機率']
    strat_v = [o['win'] * 100, o['payoff'] * 10, o['big_loss'] * 100]
    base_v = [o['base']['pos'] * 100, 11.9, 0.0]     # 對照的賠率 1.19×10
    xx = np.arange(3)
    ax.bar(xx - w / 2, strat_v, w, label='唐奇安 20/10', color=UP, alpha=.9)
    ax.bar(xx + w / 2, base_v, w, label='隨機持有對照', color=GRAY, alpha=.9)
    ax.set_xticks(xx)
    ax.set_xticklabels(things, fontsize=8.5)
    ax.set_yscale('log')
    for i, (s, b) in enumerate(zip(strat_v, base_v)):
        ax.text(i - w / 2, s * 1.15, f'{s:.1f}', ha='center', color=TX, fontsize=7.5)
        ax.text(i + w / 2, max(b, 0.03) * 1.15, f'{b:.1f}', ha='center', color=TX2, fontsize=7.5)
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=TX, fontsize=7.5)
    style(ax, '但他的「砍虧損」主張是真的：賺賠比高、大跌率極低', ylab='數值（對數軸；賠率×10）')
    fig.tight_layout()
    return save(fig, 'b3_turtle.png')


def chart_breadth(B):
    """兩個指數給出相反的訊號——這才是重點。"""
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.6), sharey=True)
    fig.patch.set_facecolor(BG)
    hs = ['10', '20', '60']

    for ax, name in zip(axes, ('SPY', 'QQQ')):
        r = B[name]
        d = [r['div'][h]['trim'] * 100 if h in r['div'] else np.nan for h in hs]
        s = [r['sync'][h]['trim'] * 100 if h in r['sync'] else np.nan for h in hs]
        x = np.arange(len(hs))
        w = 0.34
        ax.bar(x - w / 2, d, w, label='背馳後', color=DN, alpha=.9)
        ax.bar(x + w / 2, s, w, label='同步創高後', color=UP, alpha=.9)
        for i, h in enumerate(hs):
            if h in r['diff']:
                dv = r['diff'][h]
                sig = not (dv['ci'][0] < 0 < dv['ci'][1])
                mark = '★' if sig else ''
                ax.text(i, max(d[i], s[i]) + 0.12,
                        f"{dv['diff']*100:+.2f}{mark}", ha='center',
                        color=GOLD if sig else TX2, fontsize=8.5,
                        fontweight='bold' if sig else 'normal')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{h} 日' for h in hs], fontsize=9)
        ax.axhline(0, color=GRAY, lw=.8)
        ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=TX, fontsize=7.5, loc='upper left')
        style(ax, f'{name}', ylab='報酬 %' if name == 'SPY' else '')

    axes[0].text(0.02, 0.03, '★ = 信賴區間不含零', transform=axes[0].transAxes,
                 color=GOLD, fontsize=7.5)
    fig.suptitle('同一個訊號，SPY 與 QQQ 給出相反的答案（60 日 +1.08 對 +0.18）',
                 color=TX, fontsize=10.5, x=0.02, ha='left', y=1.02)
    fig.tight_layout()
    return save(fig, 'b3_breadth.png')


def chart_ep1(E, ET):
    """EP1 vs 基準，含信賴區間。"""
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    fig.patch.set_facecolor(BG)

    hs = ['20', '60']
    ep = [ET[h]['ep1'] * 100 for h in hs]
    ba = [ET[h]['base'] * 100 for h in hs]
    dif = [ET[h]['diff'] * 100 for h in hs]
    lo = [ET[h]['ci'][0] * 100 for h in hs]
    hi = [ET[h]['ci'][1] * 100 for h in hs]

    x = np.arange(len(hs))
    w = 0.3
    ax.bar(x - w / 2, ep, w, label='EP1 量價突破後', color=UP, alpha=.9)
    ax.bar(x + w / 2, ba, w, label='同期上升趨勢基準', color=GRAY, alpha=.9)
    for i, h in enumerate(hs):
        ax.text(i - w / 2, ep[i] + 0.1, f'{ep[i]:+.2f}%', ha='center', color=TX, fontsize=8.5)
        ax.text(i + w / 2, ba[i] + 0.1, f'{ba[i]:+.2f}%', ha='center', color=TX2, fontsize=8.5)
        ax.text(i, max(ep[i], ba[i]) + 0.55,
                f"差 +{dif[i]:.2f}pp\nt={ET[h]['t']:+.2f}", ha='center',
                color=GOLD, fontsize=8.5, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{h} 日（訊號日 {ET[h]["n_sig_days"]:,} 天）' for h in hs], fontsize=9)
    ax.axhline(0, color=GRAY, lw=.8)
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=TX, fontsize=8)
    style(ax, 'EP1 是全批唯一有正優勢的：量價確認後突破，60 日多 1.04 個百分點', ylab='報酬 %')
    fig.tight_layout()
    return save(fig, 'b3_ep1.png')


def main():
    T = json.load(open(os.path.join(OUT, 'turtle_fair.json')))
    B = json.load(open(os.path.join(OUT, 'breadth_fix.json')))
    E = json.load(open(os.path.join(OUT, 'batch3b.json')))['ep1']
    ET = json.load(open(os.path.join(OUT, 'ep1_t.json')))

    charts = dict(turtle=chart_turtle(T), breadth=chart_breadth(B), ep1=chart_ep1(E, ET))
    out = dict(turtle=T, breadth=B, ep1=E, ep1_t=ET, charts=charts)
    json.dump(out, open(os.path.join(OUT, 'batch3_all.json'), 'w'),
              ensure_ascii=False, indent=1)
    print('已寫 batch3_all.json')
    print('  charts:', charts)


if __name__ == '__main__':
    main()
