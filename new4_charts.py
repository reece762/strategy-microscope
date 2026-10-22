#!/usr/bin/env python3
"""四支新影片（＋永久投資組合）的圖表。

每張圖只回答一個問題：
  1. 燭型形態：並池後「訊號後 20 日超額報酬」與 t 值（正＝符合宣稱方向）
  2. 均線用法：同上
  3. 放量：同上（並列出高位／低位）
  4. 槓鈴：兩條防禦腿的淨值曲線對照（短債 vs 長債）
  5. 永久投資組合：兩個版本淨值曲線 vs 單獨持有 SPY
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 155

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')   # 本檔已在 xbx/ 內，不要再接一層
BG = '#0a0e14'
TX = '#c9d1d9'
TX2 = '#8b949e'
GRID = '#8b949e'
UP = '#3fb950'
DN = '#f85149'
BLUE = '#58a6ff'
GOLD = '#f0b429'
PURPLE = '#bc8cff'


def _style(ax, title, ylab=''):
    ax.set_title(title, color=TX, fontsize=10.5, pad=8, loc='left')
    ax.tick_params(colors=TX2, labelsize=9, length=3)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(alpha=.13, color=GRID, lw=.4, axis='y')
    if ylab:
        ax.set_ylabel(ylab, color=TX2, fontsize=9)


def _save(fig, name):
    path = os.path.join(OUT, f'{name}.png')
    fig.savefig(path, format='png', facecolor=BG, bbox_inches='tight', pad_inches=.15)
    plt.close(fig)
    return path


def _excess_chart(rows, title, fname, min_n=30):
    """橫條圖：超額報酬，標註 t 值與樣本數。"""
    rows = [r for r in rows if r.get('n20', 0) >= min_n and r.get('ex20') is not None]
    if not rows:
        return None
    rows = sorted(rows, key=lambda r: r['ex20'])
    names = [r['name'] for r in rows]
    vals = [r['ex20'] * 100 for r in rows]
    ts = [r.get('t20') or 0 for r in rows]
    ns = [r.get('n_non20') or r.get('n20') for r in rows]

    fig, ax = plt.subplots(figsize=(7.9, max(2.4, .42 * len(rows) + 1.1)))
    y = np.arange(len(rows))
    cols = [UP if v > 0 else DN for v in vals]
    ax.barh(y, vals, color=cols, alpha=.85, height=.62)
    for i, (v, t, n) in enumerate(zip(vals, ts, ns)):
        off = .06 if v >= 0 else -.06
        ax.text(v + off, i, f'{v:+.2f}%  t={t:+.2f}  n={n}',
                va='center', ha='left' if v >= 0 else 'right',
                color=TX2, fontsize=8)
    ax.axvline(0, color=TX2, lw=.8)
    ax.set_yticks(y)
    ax.set_yticklabels(names, color=TX2, fontsize=9)
    lo, hi = min(vals), max(vals)
    pad = max(.15, (hi - lo) * .35)
    ax.set_xlim(lo - pad, hi + pad)
    ax.xaxis.set_major_formatter(lambda v, _: f'{v:+.0f}%')
    _style(ax, title, '訊號後 20 日超額報酬（已扣同期無條件平均）')
    return _save(fig, fname)


def kline_chart(rows):
    return _excess_chart(rows, '燭型形態：20 日超額報酬（並池 5 檔 ETF＋BTC）',
                         'n4_kline')


def ma_chart(rows):
    return _excess_chart(rows, '均線用法：20 日超額報酬（並池）', 'n4_ma')


def volume_chart(rows):
    return _excess_chart(rows, '放量：20 日超額報酬（並池）', 'n4_volume')


def barbell_chart():
    p = os.path.join(OUT, 'barbell2.json')
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    fig, ax = plt.subplots(figsize=(7.9, 3.3))
    picks = [
        ('cash_10', '90% 短債 + 10% 小型股', BLUE),
        ('cash_30', '70% 短債 + 30% 小型股', GOLD),
        ('tlt_10', '90% 長債 TLT + 10% 小型股', DN),
        ('bh_IWM', '單獨持有小型股 IWM', TX2),
    ]
    for key, lbl, col in picks:
        if key not in d:
            continue
        idx = [np.datetime64(x) for x in d[key]['idx']]
        ax.plot(idx, d[key]['series'], color=col, lw=1.5,
                label=f"{lbl}（年化 {d[key]['cagr']*100:.2f}%）")
    ax.set_yscale('log')
    ax.legend(fontsize=8, facecolor='none', edgecolor='none', labelcolor=TX2, loc='upper left')
    _style(ax, '槓鈴策略：防禦腿用短債 vs 長債，差別決定生死', '淨值（倍，對數軸）')
    return _save(fig, 'n4_barbell')


def perm_chart():
    p = os.path.join(OUT, 'perm.json')
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    fig, ax = plt.subplots(figsize=(7.9, 3.3))
    colors = {'spy': BLUE, 'qqq': GOLD}
    for k, col in colors.items():
        v = d['variants'].get(k)
        if not v:
            continue
        idx = [np.datetime64(x) for x in v['idx']]
        ax.plot(idx, v['series'], color=col, lw=1.5,
                label=f"{v['label']}（年化 {v['cagr']*100:.2f}%，回撤 {v['mdd']*100:.1f}%）")
    ax.legend(fontsize=8, facecolor='none', edgecolor='none', labelcolor=TX2, loc='upper left')
    _style(ax, '永久投資組合：四類資產各 25%，35%／15% 再平衡', '淨值（倍）')
    return _save(fig, 'n4_perm')


def main():
    d = json.load(open(os.path.join(OUT, 'pooled_t.json')))
    out = {}
    for name, fn, arg in (('kline', kline_chart, d['kline']),
                          ('ma', ma_chart, d['ma']),
                          ('volume', volume_chart, d['volume'])):
        try:
            p = fn(arg)
            out[name] = os.path.basename(p) if p else None
            print(f'{name:9s} -> {out[name] or "SKIP"}')
        except Exception as e:
            print(f'{name:9s} -> ERROR {type(e).__name__}: {e}')
    for name, fn in (('barbell', barbell_chart), ('perm', perm_chart)):
        try:
            p = fn()
            out[name] = os.path.basename(p) if p else None
            print(f'{name:9s} -> {out[name] or "SKIP"}')
        except Exception as e:
            print(f'{name:9s} -> ERROR {type(e).__name__}: {e}')
    json.dump({k: v for k, v in out.items() if v},
              open(os.path.join(OUT, 'n4_charts.json'), 'w'))


if __name__ == '__main__':
    main()
