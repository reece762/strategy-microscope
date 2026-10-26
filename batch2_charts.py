#!/usr/bin/env python3
"""第二批四支影片的三張圖。"""
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


def chart_ma(D):
    S, B = D['signals'], D['baseline']
    base20 = B['20']['trim']
    items = [('10MA 低量拉回', 'ma10'), ('20MA 低量拉回', 'ma20'),
             ('EMA10（同規則）', 'ema10'), ('Darvas 箱型突破', 'box_break')]
    vals = [(lab, S[k]['by_h']['20']['trim'], S[k]['n'], S[k]['by_h']['20']['t']) for lab, k in items]

    fig, ax = plt.subplots(figsize=(8.6, 3.5))
    fig.patch.set_facecolor(BG)
    y = np.arange(len(vals))
    ax.barh(y, [v * 100 for _, v, _, _ in vals],
            color=[UP if v > base20 else DN for _, v, _, _ in vals], alpha=.85, height=.6)
    ax.axvline(base20 * 100, color=GOLD, lw=1.6, ls='--',
               label=f'上升趨勢日基準 {base20*100:+.2f}%')
    for i, (lab, v, n, t) in enumerate(vals):
        ax.text(v * 100 + 0.05, i, f'{v*100:+.2f}%  (t={t:+.1f})', va='center',
                ha='left', color=TX, fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels([f'{lab}\n(n={n:,})' for lab, _, n, _ in vals], fontsize=8.2)
    ax.axvline(0, color=GRAY, lw=.8)
    ax.set_xlim(0, max(v for _, v, _, _ in vals) * 100 * 1.45)
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=TX, fontsize=8, loc='lower right')
    style(ax, '均線拉回與箱型突破：全部貼在基準線上，沒有優勢', xlab='之後 20 日報酬 %')
    ax.invert_yaxis()
    fig.tight_layout()
    return save(fig, 'b2_ma.png')


def chart_pullback(D):
    """拉回買入法：三根柱 + 差值信賴區間。"""
    S, B, DF = D['signals'], D['baseline'], D['pullback_diff']
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6))
    fig.patch.set_facecolor(BG)

    hs = [5, 10, 20, 60]
    dip = [S['dip_buy']['by_h'][str(h)]['trim'] * 100 for h in hs]
    rec = [S['recovery']['by_h'][str(h)]['trim'] * 100 for h in hs]
    base = [B[str(h)]['trim'] * 100 for h in hs]
    x = np.arange(len(hs))
    w = .27
    axes[0].bar(x - w, base, w, label='基準（上升趨勢日）', color=GRAY, alpha=.85)
    axes[0].bar(x, dip, w, label='拉回中買（他所反對）', color=DN, alpha=.9)
    axes[0].bar(x + w, rec, w, label='反轉買（他所主張）', color=UP, alpha=.9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f'{h} 日' for h in hs], fontsize=9)
    axes[0].axhline(0, color=GRAY, lw=.8)
    axes[0].legend(facecolor=BG, edgecolor=GRID, labelcolor=TX, fontsize=7.5)
    style(axes[0], '他說「拉回中不要買、要等反轉」', ylab='報酬 %')

    d = [DF[str(h)]['diff'] * 100 for h in hs]
    lo = [DF[str(h)]['ci'][0] * 100 for h in hs]
    hi = [DF[str(h)]['ci'][1] * 100 for h in hs]
    err = [np.array(d) - np.array(lo), np.array(hi) - np.array(d)]
    axes[1].bar(x, d, .5, color=BLUE, alpha=.85, yerr=err, capsize=4,
                error_kw=dict(ecolor=TX2, lw=1.2))
    for i, val in enumerate(d):
        axes[1].text(i, val + (0.12 if val >= 0 else -0.28), f'{val:+.2f}',
                     ha='center', color=TX, fontsize=8.5)
    axes[1].axhline(0, color=GRAY, lw=.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f'{h} 日' for h in hs], fontsize=9)
    style(axes[1], '反轉買 − 拉回中買（只有 60 日站得住）', ylab='差值 %')
    fig.tight_layout()
    return save(fig, 'b2_pullback.png')


def chart_bull(D):
    """牛市年齡：只有 2 個獨立事件達到 700 日，無法檢定。"""
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    fig.patch.set_facecolor(BG)

    rows = [('SPY 最長', 689, '<400 日'), ('QQQ 最長', 826, '<400 日')]
    ax.barh([0, 1], [689, 826], color=[BLUE, GOLD], alpha=.85, height=.5)
    ax.axvline(700, color=DN, lw=1.8, ls='--', label='他說的門檻 700 日')
    ax.axvline(800, color=DN, lw=1.8, ls=':', label='他說的門檻 800 日')
    for i, (lab, v, _) in enumerate(rows):
        ax.text(v - 20, i, f'{v} 日約 {v/30.4:.0f} 個月', va='center', ha='right',
                color='#0a0e14', fontsize=9, fontweight='bold')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['SPY\n2001-2026', 'QQQ\n1999-2026'], fontsize=9)
    ax.set_xlim(0, 950)
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=TX, fontsize=8, loc='lower right')
    style(ax, '26 年裡只有 2 次牛市活到 700 日曆日以上', xlab='最長牛齡（日曆日）')
    fig.tight_layout()
    return save(fig, 'b2_bullage.png')


def main():
    D = json.load(open(os.path.join(OUT, 'batch2.json')))
    BA = json.load(open(os.path.join(OUT, 'bull_age_cal.json')))

    charts = dict(ma=chart_ma(D), pullback=chart_pullback(D), bull=chart_bull(D))
    D['charts'] = charts
    D['bull_age_cal'] = BA
    json.dump(D, open(os.path.join(OUT, 'batch2_all.json'), 'w'), ensure_ascii=False, indent=1)

    print('已寫 batch2_all.json')
    print('  基準 20日', f"{D['baseline']['20']['trim']*100:+.2f}%")
    for k in ('ma10', 'ma20', 'ema10', 'dip_buy', 'recovery', 'box_break'):
        b = D['signals'][k]['by_h']['20']
        print(f"  {D['signals'][k]['label']:<34}{b['trim']*100:+.2f}%  "
              f"vs基準 {(b['trim']-D['baseline']['20']['trim'])*100:+.2f}pp")
    print('  SMA/EMA 20日差:', f"{D['sma_ema']['20']['diff']*100:+.3f}pp",
          f"重疊 {D['sma_ema']['20']['overlap']*100:.1f}%")


if __name__ == '__main__':
    main()
