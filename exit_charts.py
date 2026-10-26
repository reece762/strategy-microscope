#!/usr/bin/env python3
"""出場四（五）部曲的三張圖。

左中右三張各講一件事：
  A 賣出代價 vs 基準（見頂、破均線、破前低、波幅擴張、回檔）
  B VCP 收窄 vs V 形急漲——他的說法與資料相反
  C 大盤下跌日：強勢股 vs 弱勢股
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


def chart_signals(F, B):
    sig = [
        ('抛物線+不尋常下跌', 'abnorm'),
        ('首次跌破 20 日均線', 'brk20'),
        ('跌破已確認前波低點', 'brk_swing'),
        ('波幅擴張 1.5 倍', 'volexp'),
        ('延伸狀態', 'extended'),
        ('回檔 30~36%', 'dd30'),
    ]
    base20 = B['20']['trim']
    vals = []
    for lab, k in sig:
        b = F[k]['by_h'].get('20')
        vals.append((lab, b['trim_raw'] if b else np.nan, F[k]['n']))

    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    fig.patch.set_facecolor(BG)
    y = np.arange(len(vals))
    colors = [UP if v > base20 else DN for _, v, _ in vals]
    ax.barh(y, [v * 100 for _, v, _ in vals], color=colors, alpha=.85, height=.62)
    ax.axvline(base20 * 100, color=GOLD, lw=1.6, ls='--',
               label=f'同期上升趨勢日基準 {base20*100:+.2f}%')
    for i, (lab, v, n) in enumerate(vals):
        ax.text(v * 100 + (0.18 if v >= base20 else -0.18), i,
                f'{v*100:+.2f}%', va='center',
                ha='left' if v >= base20 else 'right', color=TX, fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels([f'{lab}\n(n={n:,})' for lab, _, n in vals], fontsize=8)
    ax.axvline(0, color=GRAY, lw=.8)
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=TX, fontsize=8, loc='lower right')
    style(ax, '出現訊號後 20 日：棒子超過金色虛線＝賣早了（紅色＝賣對了）', xlab='20 日報酬 %')
    ax.invert_yaxis()
    fig.tight_layout()
    return save(fig, 'ex_signals.png')


def chart_vcp(BL):
    v = BL['vcp']
    diff = BL['vcp_diff']
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.7))
    fig.patch.set_facecolor(BG)

    hs = [5, 10, 20, 60]
    a = [v['VCP收窄'][str(h)]['trim'] * 100 for h in hs]
    b = [v['V形急漲'][str(h)]['trim'] * 100 for h in hs]
    x = np.arange(len(hs))
    w = .36
    axes[0].bar(x - w / 2, a, w, label=f'VCP 收窄（n={v["VCP收窄"]["20"]["n"]:,}）',
                color=BLUE, alpha=.9)
    axes[0].bar(x + w / 2, b, w, label=f'V 形急漲（n={v["V形急漲"]["20"]["n"]:,}）',
                color=GOLD, alpha=.9)
    for i, (p, q) in enumerate(zip(a, b)):
        axes[0].text(i - w / 2, p + .15, f'{p:+.1f}', ha='center', color=TX, fontsize=7.5)
        axes[0].text(i + w / 2, q + .15, f'{q:+.1f}', ha='center', color=TX, fontsize=7.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'{h} 日' for h in hs], fontsize=9)
    axes[0].axhline(0, color=GRAY, lw=.8)
    axes[0].legend(facecolor=BG, edgecolor=GRID, labelcolor=TX, fontsize=7.5, loc='upper left')
    style(axes[0], '他到阻力位說「VCP 要抱、V 形要賣」', ylab='報酬 %')

    d = [diff[str(h)]['diff'] * 100 for h in hs]
    lo = [diff[str(h)]['ci'][0] * 100 for h in hs]
    hi = [diff[str(h)]['ci'][1] * 100 for h in hs]
    err = [np.array(d) - np.array(lo), np.array(hi) - np.array(d)]
    axes[1].bar(x, d, .5, color=DN, alpha=.85, yerr=err, capsize=4,
                error_kw=dict(ecolor=TX2, lw=1.2))
    for i, val in enumerate(d):
        axes[1].text(i, val - 0.5, f'{val:+.2f}', ha='center', va='top',
                     color=TX, fontsize=8.5)
    axes[1].axhline(0, color=GRAY, lw=.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'{h} 日' for h in hs], fontsize=9)
    style(axes[1], 'VCP − V形：全部為負（他的說法反了）', ylab='差值 %')

    fig.tight_layout()
    return save(fig, 'ex_vcp.png')


def chart_mktdown():
    p = os.path.join(OUT, 'exit_signals_fix.json')
    d = json.load(open(p))
    m = d['mktdown']['by_h']
    hs = [h for h in ('5', '10', '20', '60') if h in m]
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    fig.patch.set_facecolor(BG)
    x = np.arange(len(hs))
    w = .36
    s = [m[h]['strong'] * 100 for h in hs]
    wk = [m[h]['weak'] * 100 for h in hs]
    ax.bar(x - w / 2, s, w, label='強勢股（距高點 <15%）', color=BLUE, alpha=.9)
    ax.bar(x + w / 2, wk, w, label='弱勢股（距高點 ≥30%）', color=GOLD, alpha=.9)
    for i, (p1, p2) in enumerate(zip(s, wk)):
        ax.text(i - w / 2, p1 + (.08 if p1 >= 0 else -.25), f'{p1:+.2f}',
                ha='center', color=TX, fontsize=8)
        ax.text(i + w / 2, p2 + (.08 if p2 >= 0 else -.25), f'{p2:+.2f}',
                ha='center', color=TX, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{h} 日' for h in hs], fontsize=9)
    ax.axhline(0, color=GRAY, lw=.8)
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=TX, fontsize=8)
    style(ax, '大盤跌逾 1% 當天：之後反彈更多的是「已跌深的弱勢股」', ylab='報酬 %')
    fig.tight_layout()
    return save(fig, 'ex_mktdown.png')


def main():
    F = json.load(open(os.path.join(OUT, 'exit_signals_fix.json')))
    BL = json.load(open(os.path.join(OUT, 'exit_baseline.json')))

    charts = dict(sig=chart_signals(F, BL['baseline']),
                  vcp=chart_vcp(BL), mkt=chart_mktdown())

    res = dict(signals=F, baseline=BL['baseline'], vcp=BL['vcp'],
               vcp_diff=BL['vcp_diff'], mktdown=F['mktdown'], charts=charts)
    json.dump(res, open(os.path.join(OUT, 'exit_all.json'), 'w'),
              ensure_ascii=False, indent=1)

    print('已寫 exit_all.json')
    print(f"  基準 20日 {BL['baseline']['20']['trim']*100:+.2f}%")
    for k in ('abnorm', 'brk20', 'brk_swing', 'volexp', 'extended', 'dd30'):
        b = F[k]['by_h'].get('20')
        if b:
            print(f"  {F[k]['label']:<28} {b['trim_raw']*100:+.2f}%  "
                  f"相對基準 {(b['trim_raw']-BL['baseline']['20']['trim'])*100:+.2f}pp")
    print(f"  VCP − V形 20日 {BL['vcp_diff']['20']['diff']*100:+.2f}% "
          f"CI[{BL['vcp_diff']['20']['ci'][0]*100:+.2f},{BL['vcp_diff']['20']['ci'][1]*100:+.2f}]")


if __name__ == '__main__':
    main()
