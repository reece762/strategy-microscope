#!/usr/bin/env python3
"""J Law VCP 那張卡的圖。

兩個問題要回答：
  1. 表面上很強（年化 37.6%）——但扣掉市場之後剩多少？
  2. VCP 型態本身有貢獻嗎，還是「順勢突破」就夠了？

所以左圖畫「原始 vs 超額」，右圖畫 VCP 與對照組的分布對比。
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
VCP = '/root/.openclaw/workspace/reports/_vcp'

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


def chart_decomp():
    r = json.load(open(f'{VCP}/vcp_result.json'))
    c = json.load(open(f'{VCP}/ctrl_result.json'))
    v = pd.read_csv(f'{VCP}/vcp_trades.csv')
    ct = pd.read_csv(f'{VCP}/ctrl_trades.csv')
    ex = pd.read_csv(f'{VCP}/vcp_excess.csv')

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7))
    fig.patch.set_facecolor(BG)

    # ---- 左：原始 vs 超額（兩組）
    ax = axes[0]
    labs = ['VCP\n（他的三招）', '對照組\n（只留趨勢+突破）']
    ce = pd.read_csv(f'{VCP}/ctrl_excess.csv')
    raw = [v.ret.mean(), ct.ret.mean()]
    exc = [ex.ex.mean(), ce.ex.mean()]
    x = np.arange(2)
    w = 0.34
    b1 = ax.bar(x - w / 2, np.array(raw) * 100, w, label='原始報酬',
                color=BLUE, alpha=0.85)
    b2 = ax.bar(x + w / 2, np.array(exc) * 100, w, label='扣掉市場後（超額）',
                color=DN, alpha=0.85)
    for bars in (b1, b2):
        for r_ in bars:
            h = r_.get_height()
            ax.text(r_.get_x() + r_.get_width() / 2, h + 0.15,
                    f'{h:+.2f}%', ha='center', color=TX, fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=8.5)
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=TX, fontsize=8, loc='upper right')
    _style(ax, '平均每筆報酬：扣掉市場之後幾乎歸零', ylab='%')

    # ---- 右：報酬分布（VCP vs 對照）
    ax = axes[1]
    bins = np.linspace(-0.12, 0.45, 45)
    ax.hist(ct.ret.clip(-0.12, 0.45), bins=bins, color=GRAY, alpha=0.55,
            label=f'對照組（{len(ct):,} 筆）', density=True)
    ax.hist(v.ret.clip(-0.12, 0.45), bins=bins, color=BLUE, alpha=0.60,
            label=f'VCP（{len(v):,} 筆）', density=True)
    ax.axvline(0, color=GRAY, lw=0.8, ls='--')
    ax.axvline(-0.07, color=DN, lw=1.2, ls=':')
    ax.text(-0.07, ax.get_ylim()[1] * 0.92, ' 停損 −7%', color=DN, fontsize=8)
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=TX, fontsize=8)
    _style(ax, '報酬分布：兩組的形狀幾乎一樣', ylab='密度', xlab='單筆報酬')
    ax.xaxis.set_major_formatter(lambda v_, _: f'{v_*100:.0f}%')

    fig.tight_layout()
    return save(fig, 'vcp_decomp.png')


def chart_year():
    ex = pd.read_csv(f'{VCP}/vcp_excess.csv')
    g = ex.groupby('year')['ex'].mean()
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    fig.patch.set_facecolor(BG)
    cols = [UP if v > 0 else DN for v in g.values]
    ax.bar(g.index, g.values * 100, color=cols, alpha=0.85, width=0.72)
    ax.axhline(0, color=GRAY, lw=0.8)
    pos = int((g > 0).sum())
    _style(ax, f'逐年超額報酬：{len(g)} 年中有 {pos} 年為正，正負都很劇烈',
           ylab='超額 %', xlab='')
    ax.set_xticks(list(g.index)[::2])
    ax.tick_params(axis='x', labelsize=8)
    fig.tight_layout()
    return save(fig, 'vcp_year.png')


def main():
    f1 = chart_decomp()
    f2 = chart_year()

    # 把卡片要用的數字集中寫進一份 JSON，卡片模組只讀這裡
    r = json.load(open(f'{VCP}/vcp_result.json'))
    c = json.load(open(f'{VCP}/ctrl_result.json'))
    sw = json.load(open(f'{VCP}/vcp_sweep.json'))
    v = pd.read_csv(f'{VCP}/vcp_trades.csv')
    ct = pd.read_csv(f'{VCP}/ctrl_trades.csv')
    ex = pd.read_csv(f'{VCP}/vcp_excess.csv')
    ce = pd.read_csv(f'{VCP}/ctrl_excess.csv')

    def nwt(x, lag=5):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        n = len(x)
        mu = x.mean()
        ee = x - mu
        s = ee @ ee / n
        for L in range(1, lag + 1):
            s += 2 * (1 - L / (lag + 1)) * (ee[L:] @ ee[:-L]) / n
        return float(mu / np.sqrt(s / n))

    res = dict(
        n=len(v), win=float((v.ret > 0).mean()), avg=float(v.ret.mean()),
        med=float(v.ret.median()), stop=float((v.reason == 'stop').mean()),
        hold=float(v.hold.mean()),
        avg_win=float(v.ret[v.ret > 0].mean()),
        avg_loss=float(v.ret[v.ret <= 0].mean()),
        excess=float(ex.ex.mean()), excess_t=nwt(ex.ex),
        ctrl_n=len(ct), ctrl_win=float((ct.ret > 0).mean()),
        ctrl_avg=float(ct.ret.mean()), ctrl_excess=float(ce.ex.mean()),
        ctrl_excess_t=nwt(ce.ex),
        big_win=float((v.ret > 0.20).mean()),
        ctrl_big_win=float((ct.ret > 0.20).mean()),
        trim_top1=float(v.ret.sort_values(ascending=False).head(int(len(v) * 0.01)).sum()
                        / v.ret.sum()),
        trim_top5=float(v.ret.sort_values(ascending=False).head(int(len(v) * 0.05)).sum()
                        / v.ret.sum()),
        trim_avg=float(v.ret[v.ret < v.ret.quantile(0.95)].mean()),
        ctrl_trim_avg=float(ct.ret[ct.ret < ct.ret.quantile(0.95)].mean()),
        strat_ann=float(r['bench']['strat_ann']),
        mkt_ann=float(r['bench']['mkt_ann']),
        years_pos=int((ex.groupby('year')['ex'].mean() > 0).sum()),
        years_n=int(ex.year.nunique()),
        sweep=[dict(v=x['variant'], n=x['n'], excess=x['excess'], t=x['t']) for x in sw],
        charts=dict(decomp=f1, year=f2),
    )
    json.dump(res, open(f'{OUT}/vcp_card.json', 'w'), ensure_ascii=False, indent=1)
    print('已寫 out/vcp_card.json')
    print(f"  VCP 原始 {res['avg']*100:+.2f}% 超額 {res['excess']*100:+.2f}% t={res['excess_t']:+.2f}")
    print(f"  對照 原始 {res['ctrl_avg']*100:+.2f}% 超額 {res['ctrl_excess']*100:+.2f}% t={res['ctrl_excess_t']:+.2f}")
    print(f"  剔除最好5%: VCP {res['trim_avg']*100:+.2f}% vs 對照 {res['ctrl_trim_avg']*100:+.2f}%")
    print(f"  逐年超額為正 {res['years_pos']}/{res['years_n']}")


if __name__ == '__main__':
    main()
