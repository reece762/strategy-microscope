#!/usr/bin/env python3
"""第二梯（三支）的圖表。

1. 馬克維茨：三種配置的年化/回撤/波動對照，以及權重隨時間的變化
2. 比特幣定投：累計投入 vs 持幣市值（四種方案）
3. BTC/ETH 輪動：倍數對照 + 三種參數
"""
import json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 155

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
BG = '#0a0e14'
TX = '#c9d1d9'
TX2 = '#8b949e'
GRID = '#8b949e'
UP = '#3fb950'
DN = '#f85149'
BLUE = '#58a6ff'
GOLD = '#f0b429'
PURPLE = '#a371f7'
CYAN = '#39c5cf'


def _style(ax, title, ylab=''):
    ax.set_title(title, color=TX, fontsize=11, pad=8, loc='left')
    ax.tick_params(colors=TX2, labelsize=9, length=3)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(alpha=.13, color=GRID, lw=.4, axis='y')
    if ylab:
        ax.set_ylabel(ylab, color=TX2, fontsize=9)


def _save(fig, name):
    path = os.path.join(OUT, f't2_{name}.png')
    fig.savefig(path, format='png', facecolor=BG, bbox_inches='tight', pad_inches=.15)
    plt.close(fig)
    return path


# ------------------------------------------------- 1. 馬克維茨
def markowitz_chart(d):
    mk = d['markowitz']
    order = [('max_sharpe', '最大夏普', BLUE),
             ('min_var', '最小變異', CYAN),
             ('equal', '等權', PURPLE),
             ('spy_tlt_60_40', '股六債四', GOLD),
             ('spy_bh', '只買 SPY', TX2)]
    labels, cagr, mdd, vol, cols = [], [], [], [], []
    for key, lbl, col in order:
        v = mk.get(key)
        if not v:
            continue
        labels.append(lbl)
        cagr.append(v['cagr'] * 100)
        mdd.append(abs(v['mdd']) * 100)
        vol.append(v['vol'] * 100)
        cols.append(col)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.0, 3.2))
    x = np.arange(len(labels))
    w = .36
    b1 = a1.bar(x - w / 2, cagr, w, color=cols, alpha=.9, label='年化報酬')
    b2 = a1.bar(x + w / 2, mdd, w, color=cols, alpha=.35, label='最大回撤（絕對值）')
    for bars in (b1, b2):
        for b in bars:
            a1.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f'{b.get_height():.1f}%', ha='center', va='bottom',
                    color=TX2, fontsize=7.4)
    a1.set_xticks(x)
    a1.set_xticklabels(labels, color=TX2, fontsize=8, rotation=18, ha='right')
    a1.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:.0f}%'))
    a1.set_ylim(0, 42)
    _style(a1, '報酬與回撤：最大夏普贏過等權，但輸給只買 SPY', '')
    a1.legend(fontsize=8, facecolor='none', edgecolor='none', labelcolor=TX2, loc='upper right')

    # 波動 vs 報酬（散點，看效率）
    a2.scatter(vol, cagr, s=70, c=cols, zorder=3)
    for lbl, v_, c_ in zip(labels, vol, cagr):
        a2.annotate(lbl, (v_, c_), textcoords='offset points', xytext=(6, 4),
                    color=TX2, fontsize=8)
    a2.set_xlabel('年化波動（風險）', color=TX2, fontsize=9)
    a2.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:.0f}%'))
    a2.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:.0f}%'))
    _style(a2, '風險—報酬平面：SPY 在右上，最小變異在左下', '年化報酬')
    return _save(fig, 'markowitz')


def weights_chart(d):
    wlog = d.get('weights', {}).get('max_sharpe') or []
    if len(wlog) < 6:
        return None
    dates = pd.to_datetime([r['date'] for r in wlog])
    syms = list(wlog[0]['w'].keys())
    mat = np.array([[r['w'][s] for s in syms] for r in wlog]).T

    fig, ax = plt.subplots(figsize=(9.0, 3.0))
    cols = [BLUE, UP, GOLD, PURPLE, CYAN, DN][:len(syms)]
    ax.stackplot(dates, mat, labels=syms, colors=cols, alpha=.85)
    ax.set_ylim(0, 1)
    ax.set_xlim(dates[0], dates[-1])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v*100:.0f}%'))
    _style(ax, '最大夏普組合的權重演變：幾乎長期滿倉股票，債券只在少數時期出現',
           '權重')
    ax.legend(fontsize=8, facecolor='none', edgecolor='none', labelcolor=TX2,
              loc='upper center', ncol=len(syms))
    return _save(fig, 'weights')


# ------------------------------------------------- 2. 定投
def dca_chart(d):
    """把四種方案的「累計投入」與「持幣市值」畫出來。
    這是這支影片最容易誤讀的地方：他的『翻四倍』是市值÷投入，
    不是年化報酬，所以圖上要同時畫兩條線。"""
    dca = d['dca']
    # 從 tier2 重新算曲線太重，這裡用摘要數字做長條圖 + 說明
    order = [('plain_DCA', '普通定投\n每週四 1000', TX2),
             ('MA250_X10_Y0.1_buyonly', '均線定投\nMA250, 10%, 0.1', BLUE),
             ('MA600_X3_Y0.45_buyonly', '改參數\nMA600, 3%, 0.45', GOLD),
             ('MA600_X3_Y0.45_sell', '允許賣出\nMA600, 3%, 0.45', UP)]
    labels, inv, val, cols = [], [], [], []
    for key, lbl, col in order:
        v = dca.get(key)
        if not v:
            continue
        labels.append(lbl)
        inv.append(v['invested'])
        val.append(v['value'])
        cols.append(col)

    fig, ax = plt.subplots(figsize=(8.6, 3.2))
    x = np.arange(len(labels))
    w = .34
    b1 = ax.bar(x - w / 2, inv, w, color=TX2, alpha=.45, label='累計投入')
    b2 = ax.bar(x + w / 2, val, w, color=cols, alpha=.9, label='期末持幣市值')
    for b in b1:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h, f'{h/1000:.0f}k',
                ha='center', va='bottom', color=TX2, fontsize=7.4)
    for b in b2:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h, f'{h/1000:.0f}k',
                ha='center', va='bottom', color=TX2, fontsize=7.4)
    ax.axhline(0, color=TX2, lw=.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=TX2, fontsize=8)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v/1000:.0f}k'))
    _style(ax, '比特幣定投四種方案（2017-12 起，美金）', '金額')
    ax.legend(fontsize=8, facecolor='none', edgecolor='none', labelcolor=TX2, loc='upper left')
    return _save(fig, 'dca')


# --------------------------------------------- 3. BTC/ETH 輪動
def btceth_chart(d):
    rot = d.get('rotation') or {}
    order = [('N10_plain', 'N=10'), ('N20_plain', 'N=20'), ('N60_plain', 'N=60'),
             ('N20_cash_when_both_down', 'N=20 + 雙跌空倉'),
             ('btc_bh', '只持 BTC'), ('eth_bh', '只持 ETH')]
    labels, mult, mdd, cols = [], [], [], []
    for key, lbl in order:
        v = rot.get(key)
        if not v:
            continue
        labels.append(lbl)
        mult.append(v['multiple'])
        mdd.append(abs(v['mdd']) * 100)
        cols.append(TX2 if 'bh' in key else (GOLD if 'cash' in key else BLUE))

    fig, ax = plt.subplots(figsize=(8.6, 3.2))
    x = np.arange(len(labels))
    bars = ax.bar(x, mult, .55, color=cols, alpha=.9)
    for b, m, dd in zip(bars, mult, mdd):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f'{m:.0f}x\n-{dd:.0f}%', ha='center', va='bottom', color=TX2, fontsize=7.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=TX2, fontsize=8.4)
    ax.set_ylim(0, max(mult) * 1.28)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:.0f}x'))
    _style(ax, 'BTC/ETH 輪動：這次輪動真的贏了，但回撤仍達八成', '淨值倍數')
    ax.text(0.99, 0.96, '每根柱下方為最大回撤', transform=ax.transAxes,
            ha='right', va='top', color=TX2, fontsize=7.5)
    return _save(fig, 'btceth')


def main():
    d = json.load(open(os.path.join(OUT, 'tier2.json')))
    out = {}
    for name, fn in (('markowitz', markowitz_chart), ('weights', weights_chart),
                     ('dca', dca_chart), ('btceth', btceth_chart)):
        try:
            p = fn(d)
            out[name] = p
            print(f'{name:10s} -> {os.path.basename(p) if p else "SKIP"}')
        except Exception as e:
            print(f'{name:10s} -> ERROR {type(e).__name__}: {e}')
    json.dump({k: os.path.basename(v) for k, v in out.items() if v},
              open(os.path.join(OUT, 't2_charts.json'), 'w'))


if __name__ == '__main__':
    main()
