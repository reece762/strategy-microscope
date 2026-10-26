#!/usr/bin/env python3
"""本輪三張新卡的圖：趨勢線、平行通道、恐慌進場。

三張圖都想回答同一個問題：**這個訊號本身有沒有超額報酬？**
所以每張圖都把「原始報酬」與「扣掉市場之後」並排，
右側再放對照組或無條件基準。
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
LINES = '/root/.openclaw/workspace/reports/_agent_lines'
PATT = '/root/.openclaw/workspace/reports/_agent_pattern'

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


def two_bars(ax, labels, raw, exc, title, ylab='%'):
    x = np.arange(len(labels))
    w = 0.36
    b1 = ax.bar(x - w / 2, np.array(raw) * 100, w, label='原始報酬',
                color=BLUE, alpha=0.85)
    b2 = ax.bar(x + w / 2, np.array(exc) * 100, w, label='扣掉市場後',
                color=DN, alpha=0.85)
    for bars in (b1, b2):
        for r in bars:
            h = r.get_height()
            ax.text(r.get_x() + r.get_width() / 2, h + (0.02 if h >= 0 else -0.06),
                    f'{h:+.2f}%', ha='center', color=TX, fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=TX, fontsize=8,
              loc='upper right')
    _style(ax, title, ylab=ylab)


def chart_trendline():
    r = json.load(open(f'{LINES}/trendline_result.json'))
    t = pd.read_csv(f'{LINES}/trendline_trades.csv')
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    fig.patch.set_facecolor(BG)

    two_bars(axes[0],
             ['趨勢線突破\n（他的四準則）', '對照組\n（突破 60 日新高）'],
             [r['stats']['avg'], r['control']['avg']],
             [r['stats']['excess'], r['control']['excess']],
             f'平均每筆報酬：兩組扣掉市場後都轉負（n={r["stats"]["n"]:,} / '
             f'{r["control"]["n"]:,}）')

    # 右：逐年超額
    yr = pd.DataFrame(r['by_year'])
    axes[1].bar(yr['year'], yr['ex'] * 100,
                color=[UP if v > 0 else DN for v in yr['ex']], alpha=0.85, width=0.72)
    axes[1].axhline(0, color=GRAY, lw=0.8)
    _style(axes[1], f'逐年超額報酬：{r["years_pos"]}/{r["years_n"]} 年為正',
           ylab='超額 %')
    axes[1].set_xticks(list(yr['year'])[::2])
    axes[1].tick_params(axis='x', labelsize=8)

    fig.tight_layout()
    return save(fig, 'tl_ab.png')


def chart_channel():
    r = json.load(open(f'{PATT}/channel_panic_result.json'))
    c = pd.read_csv(f'{PATT}/channel_trades.csv')
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    fig.patch.set_facecolor(BG)

    two_bars(axes[0],
             ['平行通道\n（低量回下軌）', '對照組\n（回站 20 日均線）'],
             [r['channel']['avg'], r['channel_control']['avg']],
             [r['channel']['excess'], r['channel_control']['excess']],
             f'平均每筆報酬（n={r["channel"]["n"]:,} / '
             f'{r["channel_control"]["n"]:,}）')

    # 右：出場原因分布
    vc = c['reason'].value_counts()
    names = {'ma20': '跌破 20 日均線', 'stop': '停損 3%',
             'target': '到達上軌', 'time': '期滿'}
    labs = [names.get(k, k) for k in vc.index]
    axes[1].barh(labs, vc.values / len(c) * 100,
                 color=[BLUE, DN, UP, GRAY][:len(vc)], alpha=0.85)
    for i, v in enumerate(vc.values / len(c) * 100):
        axes[1].text(v + 0.8, i, f'{v:.1f}%', va='center', color=TX, fontsize=9)
    _style(axes[1], '出場原因：只有 6% 真的走到上軌', xlab='佔比 %')

    fig.tight_layout()
    return save(fig, 'ch_ab.png')


def chart_panic():
    r = json.load(open(f'{PATT}/channel_panic_result.json'))['panic']
    hold = r['hold']
    base = r['base']
    hs = sorted(int(k) for k in hold)
    fig, ax = plt.subplots(figsize=(8.2, 3.7))
    fig.patch.set_facecolor(BG)

    x = np.arange(len(hs))
    w = 0.36
    v_p = [hold[str(h)]['mean'] * 100 for h in hs]
    v_b = [base[str(h)] * 100 for h in hs]
    b1 = ax.bar(x - w / 2, v_p, w, label='恐慌日隔天買進', color=DN, alpha=0.85)
    b2 = ax.bar(x + w / 2, v_b, w, label='隨便哪天買進（無條件）',
                color=GRAY, alpha=0.85)
    for bars in (b1, b2):
        for rr in bars:
            h = rr.get_height()
            ax.text(rr.get_x() + rr.get_width() / 2, h + (0.08 if h >= 0 else -0.2),
                    f'{h:+.2f}%', ha='center', color=TX, fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f'持有 {h} 日' for h in hs], fontsize=9)
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=TX, fontsize=8.5)
    _style(ax, f'恐慌日買進：四個持有期全部不如「隨便哪天買」'
               f'（共 {r["events"]} 天）', ylab='報酬 %')
    fig.tight_layout()
    return save(fig, 'pn_hold.png')


def main():
    f1 = chart_trendline()
    f2 = chart_channel()
    f3 = chart_panic()

    tl = json.load(open(f'{LINES}/trendline_result.json'))
    cp = json.load(open(f'{PATT}/channel_panic_result.json'))
    tl_t = pd.read_csv(f'{LINES}/trendline_trades.csv')

    res = dict(
        trendline=dict(
            n=tl['stats']['n'], win=tl['stats']['win'],
            avg=tl['stats']['avg'], excess=tl['stats']['excess'],
            excess_t=tl['stats']['excess_t'],
            trimmed=tl['stats']['trimmed'],
            stop_rate=tl['stats']['stop_rate'], hold=tl['stats']['hold'],
            ctrl_n=tl['control']['n'], ctrl_win=tl['control']['win'],
            ctrl_avg=tl['control']['avg'], ctrl_excess=tl['control']['excess'],
            ctrl_excess_t=tl['control']['excess_t'],
            years_pos=tl['years_pos'], years_n=tl['years_n'],
            touch_med=float(tl_t['touches'].median()),
            span_med=float(tl_t['span'].median()),
        ),
        channel=dict(
            n=cp['channel']['n'], win=cp['channel']['win'],
            avg=cp['channel']['avg'], excess=cp['channel']['excess'],
            excess_t=cp['channel']['excess_t'],
            trimmed=cp['channel']['trimmed'],
            hold=cp['channel']['hold'],
            ctrl_n=cp['channel_control']['n'],
            ctrl_win=cp['channel_control']['win'],
            ctrl_avg=cp['channel_control']['avg'],
            ctrl_excess=cp['channel_control']['excess'],
            ctrl_excess_t=cp['channel_control']['excess_t'],
        ),
        panic=dict(events=cp['panic']['events'], hold=cp['panic']['hold'],
                   base=cp['panic']['base']),
        charts=dict(tl=f1, ch=f2, pn=f3),
    )
    json.dump(res, open(f'{OUT}/new6_cards.json', 'w'),
              ensure_ascii=False, indent=1)
    print('已寫 out/new6_cards.json')
    print(f"  趨勢線 n={res['trendline']['n']:,} 超額 {res['trendline']['excess']*100:+.2f}% "
          f"t={res['trendline']['excess_t']:+.2f}")
    print(f"  通道   n={res['channel']['n']:,} 超額 {res['channel']['excess']*100:+.2f}% "
          f"t={res['channel']['excess_t']:+.2f}")
    print(f"  恐慌   {res['panic']['events']} 天")


if __name__ == '__main__':
    main()
