#!/usr/bin/env python3
"""「阿尔法量化价格行为」三張卡片的圖表。

每張圖只回答一個問題：
  1. bug：五票制的方向票分布 —— 為什麼「看多」幾乎不出現
  2. 支撐阻力：十個波動分位裡，守住率差額正負亂跳（雜訊的形狀）
  3. 六策略：策略 vs 同曝險不擇時 vs 買進持有
"""
import json
import os

import matplotlib
import numpy as np

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
    path = os.path.join(OUT, f'al_{name}.png')
    fig.savefig(path, format='png', facecolor=BG, bbox_inches='tight', pad_inches=.15)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- 1. bug
def votes_chart(d):
    """原樣 vs 修正：看多／中性／看空 的比例。"""
    syms = ['SPY', 'QQQ', 'BTC']
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), sharey=True)

    for ax, lbl, name in ((axes[0], 'orig', '原樣（趨勢棒票永遠 −1）'),
                          (axes[1], 'fix', '把該票修好之後')):
        long_ = [d['pa'][s][lbl]['pct_long'] * 100 for s in syms]
        neut = [d['pa'][s][lbl]['pct_neutral'] * 100 for s in syms]
        short = [d['pa'][s][lbl]['pct_short'] * 100 for s in syms]
        x = np.arange(len(syms))
        ax.bar(x, long_, .58, color=UP, label='看多（總分 ≥ 3）')
        ax.bar(x, neut, .58, bottom=long_, color=GRAY, label='中性（−2 ~ +2）')
        ax.bar(x, short, .58, bottom=np.array(long_) + np.array(neut),
               color=DN, label='看空（總分 ≤ −3）')
        for i, (a, b) in enumerate(zip(long_, short)):
            if a > 3:
                ax.text(i, a / 2, f'{a:.1f}%', ha='center', va='center',
                        color='#0a0e14', fontsize=8.5, fontweight='bold')
            if b > 8:
                ax.text(i, 100 - b / 2, f'{b:.1f}%', ha='center', va='center',
                        color='#0a0e14', fontsize=8.5, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(syms, color=TX2)
        ax.set_ylim(0, 100)
        _style(ax, name, '交易日佔比' if ax is axes[0] else '')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:.0f}%'))

    axes[0].legend(facecolor=BG, edgecolor='#30363d', labelcolor=TX2,
                   fontsize=8, loc='lower left', framealpha=.9)
    fig.suptitle('五票制引擎：兩個分支都回傳 −1，讓「看多」幾乎不可能出現',
                 color=TX, fontsize=11.5, y=1.03, x=.02, ha='left')
    return _save(fig, 'votes')


def entries_chart(d):
    """原樣的逐年進場次數：十年只有十幾次。"""
    fig, ax = plt.subplots(figsize=(10, 3.2))
    syms = ['SPY', 'QQQ', 'BTC']
    cols = [BLUE, GOLD, UP]
    w = .27
    for j, (s, c) in enumerate(zip(syms, cols)):
        by = d['pa'][s]['orig']['entries_by_year']
        years = sorted(int(y) for y in by)
        vals = [by[str(y)] for y in years]
        label = s
        if s == 'BTC':
            yy = [y for y in years if 2017 <= y <= 2026]
            vv = [by[str(y)] for y in yy]
            ax.bar(np.array(yy) - 8 + j * w, vv, w, color=c, label=label)
        else:
            ax.bar(np.array(years) - w + j * w, vals, w, color=c, label=label)

    tot = {s: sum(d['pa'][s]['orig']['entries_by_year'].values()) for s in syms}
    ax.set_xlabel('年份', color=TX2, fontsize=9)
    _style(ax, '原樣引擎的進場次數：SPY 十年共 %d 次、QQQ %d 次'
           % (tot['SPY'], tot['QQQ']), '進場次數')
    ax.legend(facecolor=BG, edgecolor='#30363d', labelcolor=TX2, fontsize=8.5,
              ncol=3, loc='upper left')
    return _save(fig, 'entries')


# ------------------------------------------------------- 2. 支撐阻力
def sr_chart(d):
    """十個波動分位：支撐守住率 vs 同波動對照。正負亂跳＝雜訊。"""
    syms = ['SPY', 'QQQ', 'IWM']
    fig, axes = plt.subplots(3, 1, figsize=(10.4, 8.2), sharex=True)

    for ax, s in zip(axes, syms):
        dec = d['sr'][s]['deciles']
        x = np.arange(len(dec))
        sup = [dd['sup_hold'] * 100 if dd['sup_hold'] is not None else np.nan for dd in dec]
        ctl = [dd['ctrl_hold_dn'] * 100 for dd in dec]
        diff = [(dd['sup_hold'] - dd['ctrl_hold_dn']) * 100
                if dd['sup_hold'] is not None else np.nan for dd in dec]
        ax.bar(x - .19, sup, .34, color=BLUE, label='支撐守住率')
        ax.bar(x + .19, ctl, .34, color=GRAY, label='同波動分位對照')
        for i, v in enumerate(diff):
            if np.isnan(v):
                continue
            ax.text(i, max(sup[i] if not np.isnan(sup[i]) else 0, ctl[i]) + 2.5,
                    f'{v:+.0f}', ha='center', fontsize=7.5,
                    color=UP if v > 0 else DN)
        labels = [f"{dd['q']}\n(n={dd['sup_n']})" for dd in dec]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, color=TX2, fontsize=7.5)
        ax.set_ylim(0, 105)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:.0f}%'))
        pos = d['sr'][s]['sup_pos']
        tot = len([x for x in diff if not np.isnan(x)])
        _style(ax, f'{s}：{pos}/{tot} 個分位為正，加權 {d["sr"][s]["d_sup"]*100:+.1f}pp',
               '守住率' if ax is axes[0] else '')
    axes[0].legend(facecolor=BG, edgecolor='#30363d', labelcolor=TX2, fontsize=8,
                   loc='upper right', ncol=2)
    fig.suptitle('支撐守住率：數字是「支撐 − 同波動對照」的差額（百分點）',
                 color=TX, fontsize=11.5, y=.995, x=.01, ha='left')
    fig.tight_layout()
    return _save(fig, 'sr')


# ------------------------------------------------------- 3. 六策略
def six_chart(d):
    """策略 vs 同曝險不擇時 vs 買進持有（SPY）。"""
    rows = d['six']['SPY']
    names = [k for k in rows if not k.startswith('_')]
    strat = [rows[k]['strat_cagr'] * 100 for k in names]
    em = [rows[k]['em_cagr'] * 100 for k in names]
    bh = rows[names[0]]['bh_cagr'] * 100

    fig, ax = plt.subplots(figsize=(10.4, 4.0))
    x = np.arange(len(names))
    w = .26
    ax.bar(x - w, strat, w, color=BLUE, label='策略')
    ax.bar(x, em, w, color=GRAY, label='同曝險、不擇時')
    ax.axhline(bh, color=GOLD, ls='--', lw=1.4, label=f'買進持有 {bh:.2f}%')

    for i, (a, b, k) in enumerate(zip(strat, em, names)):
        ax.text(i - w, a + (1.2 if a >= 0 else -1.2), f'{a:.1f}', ha='center',
                fontsize=7.5, color=TX2, va='bottom' if a >= 0 else 'top')
        t = rows[k]['excess_t']
        ax.text(i, min(min(a, b), 0) - 3.2, f't={t:+.1f}', ha='center',
                fontsize=7.5, color=UP if abs(t) < 2 else DN)

    ax.axhline(0, color='#30363d', lw=.8)
    short = ['均線交叉', 'MACD', 'RSI均值', 'RSI超買賣', '唐奇安', '布林回歸']
    ax.set_xticks(x)
    ax.set_xticklabels(short, color=TX2, fontsize=9)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:.0f}%'))
    ax.set_ylim(min(min(strat), min(em), 0) - 8, max(max(strat), bh) + 4)
    _style(ax, 'SPY：六個策略全輸買進持有；唐奇安連「同曝險」都顯著輸（t=−2.6）',
           '年化報酬')
    ax.legend(facecolor=BG, edgecolor='#30363d', labelcolor=TX2, fontsize=8.5,
              loc='upper right')
    return _save(fig, 'six')


def six_t_chart(d):
    """t 值：是否顯著贏過同曝險基準。四個市場分組。"""
    fig, ax = plt.subplots(figsize=(10.4, 3.6))
    syms = ['SPY', 'QQQ', 'BTC']
    names = [k for k in d['six']['SPY'] if not k.startswith('_')]
    w = .25
    cols = {'SPY': BLUE, 'QQQ': GOLD, 'BTC': UP}
    x = np.arange(len(names))
    for j, s in enumerate(syms):
        ts = [d['six'][s][k]['excess_t'] for k in names]
        ax.bar(x - w + j * w, ts, w, color=cols[s], label=s)
    ax.axhline(0, color='#30363d', lw=.8)
    for lv, c in ((2, DN), (-2, DN)):
        ax.axhline(lv, color=c, ls=':', lw=1, alpha=.55)
        ax.axhline(-lv if lv > 0 else -lv, color=c, ls=':', lw=1, alpha=.55)
    ax.text(len(names) - .5, 2.15, '±2 大致為顯著界線', color=DN, fontsize=7.5,
            ha='right')
    short = ['均線交叉', 'MACD', 'RSI均值', 'RSI超買賣', '唐奇安', '布林回歸']
    ax.set_xticks(x)
    ax.set_xticklabels(short, color=TX2, fontsize=9)
    _style(ax, '扣掉「少持倉」的功勞之後：18 個組合只有 1 個達到顯著，而且是負的',
           't 值（策略日報酬 − 同曝險日報酬）')
    ax.legend(facecolor=BG, edgecolor='#30363d', labelcolor=TX2, fontsize=8.5,
              ncol=3, loc='lower left')
    return _save(fig, 'six_t')


def main():
    d = json.load(open(os.path.join(OUT, 'alpha.json')))
    out = {}
    for name, fn in (('votes', votes_chart), ('entries', entries_chart),
                     ('sr', sr_chart), ('six', six_chart), ('six_t', six_t_chart)):
        try:
            p = fn(d)
            out[name] = os.path.basename(p)
            print(f'{name:9s} -> {os.path.basename(p)}')
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f'{name:9s} -> ERROR {type(e).__name__}: {e}')
    json.dump(out, open(os.path.join(OUT, 'alpha_charts.json'), 'w'))
    print('wrote out/alpha_charts.json')


if __name__ == '__main__':
    main()
