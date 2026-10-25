#!/usr/bin/env python3
"""RTA《交易之路》第二期五張卡片的圖，以及其中一張的淨值曲線。

五個問題各自的圖：
  1. rta_ma      MA120 用「拐頭」還是「破線」？加同曝險基準線
  2. rta_ma60    MA60  vs  MA120：報酬、回撤、部位翻轉次數
  3. rta_retrace 次要運動回撤幅度分布（看 1/3~2/3 到底佔多少）
  4. rta_vol     日線波動幅度分布（0.5% 還是 5%）
  5. rta_duration 主要趨勢持續多久（多久算「一個趨勢」）

輸出：out/rta_*.png、out/rta_charts.json、out/rta_curves.json
"""
import json
import os
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 155

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, 'out')
EPOCH = pd.Timestamp('1970-01-01')

import rta as R  # noqa: E402
from btc import load_btc, perf  # noqa: E402

BG = '#0a0e14'
TX = '#c9d1d9'
TX2 = '#8b949e'
GRID = '#30363d'
UP = '#3fb950'
DN = '#f85149'
BLUE = '#58a6ff'
GOLD = '#f0b429'
GRAY = '#6e7681'
PURPLE = '#bc8cff'


def _style(ax, title, ylab='', xlab=''):
    ax.set_title(title, color=TX, fontsize=10.5, pad=8, loc='left')
    ax.tick_params(colors=TX2, labelsize=8.5, length=3)
    ax.set_facecolor(BG)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    if ylab:
        ax.set_ylabel(ylab, color=TX2, fontsize=8.5)
    if xlab:
        ax.set_xlabel(xlab, color=TX2, fontsize=8.5)
    ax.grid(True, color=GRID, alpha=.45, lw=.6)


def _save(fig, name):
    path = os.path.join(OUT, name)
    fig.patch.set_facecolor(BG)
    fig.savefig(path, facecolor=BG, bbox_inches='tight', pad_inches=.22)
    plt.close(fig)
    return name


def _log_curve(ax, idx, vals, color, label, lw=1.5):
    ax.plot(idx, vals, color=color, lw=lw, label=label)


# ---------------------------------------------------------------- 1 均線確認
def chart_ma(d, variants):
    close = d['close']
    ret = close.pct_change().fillna(0)
    fig, ax = plt.subplots(figsize=(9.6, 3.9))

    specs = [('破線進出', DN, '破線進出（收盤在線上方做多、下方做空）'),
             ('拐頭進出', GOLD, '拐頭進出（院長主張的方式）')]
    curves = {}
    for name, col, lab in specs:
        eq, _, _ = R.run_position(close, variants[name])
        curves[name] = eq
        ax.plot(eq.index, eq.values, color=col, lw=1.6, label=lab)

    bh = (1 + ret).cumprod()
    ax.plot(bh.index, bh.values, color=BLUE, lw=1.3, alpha=.75, label='買進持有')

    # 同曝險基準：買進持有 × 該策略的平均曝險
    for name, col in (('破線進出', DN), ('拐頭進出', GOLD)):
        exp = variants[name].shift(1).fillna(0).abs().mean()
        em = (1 + ret * exp).cumprod()
        ax.plot(em.index, em.values, color=col, lw=1.0, ls=':', alpha=.8,
                label=f'同曝險基準（{name}，{exp*100:.0f}% 曝險）')

    ax.set_yscale('log')
    _style(ax, 'MA120 的「拐頭」訊號，表現不如最笨的「破線」', '淨值（倍，對數軸）')
    ax.legend(facecolor='#11161d', edgecolor=GRID, labelcolor=TX2, fontsize=7.4, loc='upper left')
    return _save(fig, 'rta_ma.png'), curves, bh


# ---------------------------------------------------------------- 2 MA60 vs 120
def chart_ma60(d):
    close = d['close']
    res = {}
    for n in (60, 120):
        ma, above, _, _ = R.ma_signals(d, n)
        pos = pd.Series(np.where(above, 1.0, -1.0), index=d.index)
        pos[ma.isna()] = 0.0
        eq, _, _ = R.run_position(close, pos)
        res[n] = dict(perf=perf(eq),
                      flips=int((pos.diff().abs() > 0).sum()))

    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.2))
    labels = ['MA60', 'MA120']
    keys = [60, 120]
    colors = [GRAY, GOLD]
    for ax, (field, title, fmt) in zip(axes, [
            ('cagr', '年化報酬', '{:.1f}%'),
            ('mdd', '最大回撤', '{:.1f}%')]):
        vals = [res[k]['perf'][field] * 100 for k in keys]
        bars = ax.bar(labels, vals, color=colors, width=.55)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2,
                    v + (1.5 if v >= 0 else -3.5), fmt.format(v),
                    ha='center', color=TX, fontsize=8.5)
        _style(ax, title)
    vals = [res[k]['flips'] for k in keys]
    bars = axes[2].bar(labels, vals, color=colors, width=.55)
    for b, v in zip(bars, vals):
        axes[2].text(b.get_x() + b.get_width() / 2, v + 4, str(v),
                     ha='center', color=TX, fontsize=8.5)
    _style(axes[2], '部位翻轉次數（9 年）')
    fig.suptitle('MA60 報酬較高，但回撤更大、假訊號也更多', color=TX,
                 fontsize=10.5, x=.01, ha='left', y=1.04)
    return _save(fig, 'rta_ma60.png'), res


# ---------------------------------------------------------------- 3 回撤分布
def chart_retrace(d):
    close = d['close']
    v = close.values
    th = 0.20
    piv = R.zigzag(close, th)
    out, over = [], 0
    for k in range(2, len(piv)):
        a, b, c = piv[k - 2], piv[k - 1], piv[k]
        imp = abs(v[b] - v[a])
        if imp / abs(v[a]) < 0.20:
            continue
        r = abs(v[c] - v[b]) / imp
        if r > 1.0:
            over += 1
            continue
        out.append(r)
    s = pd.Series(out)

    fig, ax = plt.subplots(figsize=(9.6, 3.8))
    ax.hist(s, bins=np.arange(0, 1.05, 0.05), color=PURPLE, alpha=.75,
            edgecolor=BG, lw=.5)
    for x, lab, col in ((1 / 3, '1/3', GOLD), (2 / 3, '2/3', GOLD)):
        ax.axvline(x, color=col, lw=1.2, ls='--')
        ax.text(x, ax.get_ylim()[1] * .92, lab, color=col, fontsize=8.5,
                ha='center')
    ax.axvline(.236, color=BLUE, lw=1.0, ls=':')
    ax.text(.236, ax.get_ylim()[1] * .78, '0.236', color=BLUE, fontsize=8,
            ha='center')
    _style(ax, f'次要運動回撤幅度：只有 {(s.between(1/3, 2/3)).mean()*100:.0f}% 落在 1/3~2/3 之間',
           '次數', '回撤 ÷ 前一波推進')
    ax.text(.98, .95, f'樣本 {len(s)} 段\n另有 {over} 段回撤超過起點（屬趨勢反轉，已排除）',
            transform=ax.transAxes, ha='right', va='top', color=TX2, fontsize=7.6)
    return _save(fig, 'rta_retrace.png'), s, over


# ---------------------------------------------------------------- 4 波動幅度
def chart_vol(d):
    av = R.atr_pct(d, 14) * 100
    rng3 = ((d['high'].rolling(3).max() / d['low'].rolling(3).min() - 1) * 100).dropna()

    fig, ax = plt.subplots(figsize=(9.6, 3.6))
    ax.hist(av, bins=60, color=BLUE, alpha=.7, edgecolor=BG, lw=.4,
            label='日線 ATR(14) ÷ 收盤價')
    med = av.median()
    ax.axvline(med, color=GOLD, lw=1.5)
    ax.text(med, ax.get_ylim()[1] * .9, f' 中位數 {med:.2f}%', color=GOLD, fontsize=8.5)
    ax.axvline(0.5, color=DN, lw=1.5, ls='--')
    ax.text(0.5, ax.get_ylim()[1] * .72, ' 課程寫的 0.5%', color=DN, fontsize=8.5)
    _style(ax, '「短暫趨勢波動 0.5%」與資料差了一個數量級', '天數', '波動幅度（%）')
    ax.set_xlim(0, 15)
    ax.legend(facecolor='#11161d', edgecolor=GRID, labelcolor=TX2, fontsize=7.6)
    return _save(fig, 'rta_vol.png'), av, rng3


# ---------------------------------------------------------------- 5 趨勢持續
def chart_duration(d):
    close = d['close']
    ths = [0.20, 0.25, 0.30, 0.40]
    stats = {}
    fig, ax = plt.subplots(figsize=(9.6, 3.7))
    width = 0.2
    for i, th in enumerate(ths):
        piv = R.zigzag(close, th)
        days = [(d.index[piv[k + 1]] - d.index[piv[k]]).days
                for k in range(len(piv) - 1)]
        s = pd.Series(days)
        stats[th] = dict(n=len(s), median=float(s.median()),
                         q75=float(s.quantile(.75)), max=int(s.max()),
                         over_q=float((s > 91).mean()),
                         over_y=float((s > 365).mean()))
        ax.bar([x + (i - 1.5) * width for x in range(1)],
               [s.median()], width=width,
               color=[GRAY, BLUE, GOLD, PURPLE][i],
               label=f'反轉門檻 {int(th*100)}%（中位 {s.median():.0f} 天）')
    ax.axhline(91.3, color=GOLD, lw=1.2, ls='--')
    ax.text(0.02, 95, '一季（91 天）', color=GOLD, fontsize=8)
    ax.axhline(365, color=DN, lw=1.2, ls=':')
    ax.text(0.02, 370, '一年（365 天）', color=DN, fontsize=8)
    ax.set_xticks([])
    ax.set_xlim(-1, 1)
    _style(ax, '主要趨勢的中位長度是 48~90 天，不是「一年多」', '天數')
    ax.legend(facecolor='#11161d', edgecolor=GRID, labelcolor=TX2, fontsize=7.6,
              loc='upper left')
    return _save(fig, 'rta_duration.png'), stats


# ---------------------------------------------------------------- 淨值曲線
def curves_json(d, variants, src_curves, bh):
    close = d['close']
    df = pd.DataFrame({
        's1': src_curves['破線進出'],
        's2': src_curves['拐頭進出'],
        'b': bh,
    }).dropna()
    df = df / df.iloc[0]
    idx = list(df.index)
    card = dict(
        id='rta-ma120',
        label='MA120：破線 vs 拐頭 vs 買進持有',
        ppy=365,
        d=[(pd.Timestamp(x).normalize() - EPOCH).days for x in idx],
        series=[
            dict(id='po', label='破線進出', color=DN, ppy=365,
                 v=[round(float(v), 4) for v in df['s1']]),
            dict(id='turn', label='拐頭進出', color=GOLD, ppy=365,
                 v=[round(float(v), 4) for v in df['s2']]),
            dict(id='bh', label='買進持有', color=BLUE, ppy=365,
                 v=[round(float(v), 4) for v in df['b']]),
        ],
        start=str(idx[0].date()), end=str(idx[-1].date()),
    )
    return {'rta-ma120': card}


def main():
    d = load_btc()
    variants = R.build_variants(d)
    os.makedirs(OUT, exist_ok=True)

    p1, curves, bh = chart_ma(d, variants)
    p2, ma_res = chart_ma60(d)
    p3, retr, over = chart_retrace(d)
    p4, av, rng3 = chart_vol(d)
    p5, dur = chart_duration(d)

    charts = {'rta_ma': p1, 'rta_ma60': p2, 'rta_retrace': p3,
              'rta_vol': p4, 'rta_duration': p5}
    with open(os.path.join(OUT, 'rta_charts.json'), 'w', encoding='utf-8') as f:
        json.dump(charts, f, ensure_ascii=False, indent=1)

    with open(os.path.join(OUT, 'rta_curves.json'), 'w', encoding='utf-8') as f:
        json.dump(curves_json(d, variants, curves, bh), f, ensure_ascii=False)

    # 把摘要數字也寫出來，方便卡片引用（不手寫，避免過期）
    summary = dict(
        ma60={
            'ma60': {'cagr': ma_res[60]['perf']['cagr'], 'mdd': ma_res[60]['perf']['mdd'],
                     'flips': ma_res[60]['flips']},
            'ma120': {'cagr': ma_res[120]['perf']['cagr'], 'mdd': ma_res[120]['perf']['mdd'],
                      'flips': ma_res[120]['flips']},
        },
        retrace={'n': int(len(retr)), 'over': int(over),
                 'median': float(retr.median()),
                 'in_range': float(retr.between(1 / 3, 2 / 3).mean()),
                 'le_236': float((retr <= 0.236).mean()),
                 'gt_2_3': float((retr > 2 / 3).mean())},
        vol={'atr_median': float(av.median()), 'atr_mean': float(av.mean()),
             'rng3_median': float(rng3.median())},
        duration={str(k): v for k, v in dur.items()},
    )
    with open(os.path.join(OUT, 'rta_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)

    print('圖已輸出：', ', '.join(charts.values()))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
