#!/usr/bin/env python3
"""第一梯（六支影片）的圖表。

每張圖只回答一個問題，避免把多種口徑塞進同一張圖：
  1. MACD 背離：訊號後 20 日「超額」報酬（已扣掉同期無條件平均）
  2. 量比：訊號次數與後續報酬（樣本極少，圖上要看得出來）
  3. 跳空：回補天數分布
  4. KDJ：金叉後 20 日超額 + 勝率對照無條件基準
  5. 月份效應：各月平均報酬 vs 全樣本平均
  6. 幣圈日內：每小時成交量佔比（北京時間）
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


def _style(ax, title, ylab=''):
    ax.set_title(title, color=TX, fontsize=11, pad=8, loc='left')
    ax.tick_params(colors=TX2, labelsize=9, length=3)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(alpha=.13, color=GRID, lw=.4, axis='y')
    if ylab:
        ax.set_ylabel(ylab, color=TX2, fontsize=9)


def _save(fig, name):
    path = os.path.join(OUT, f't1_{name}.png')
    fig.savefig(path, format='png', facecolor=BG, bbox_inches='tight', pad_inches=.15)
    plt.close(fig)
    return path


def pct(v, _=None):
    return f'{v:+.1f}%'


# ---------------------------------------------------------- 1. MACD
def macd_chart(d):
    rows = []
    for mk, v in d['macd'].items():
        for key, lbl in (('top', '頂背離'), ('bottom', '底背離')):
            s = v[key]
            if s.get('r20_excess_dir') is None:
                continue
            rows.append(dict(market=mk.replace('納斯達克綜合指數', '納斯達克'),
                             kind=lbl, excess=s['r20_excess_dir'] * 100,
                             t=s.get('r20_excess_t', 0), n=s['n']))
    if not rows:
        return None
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.8, 3.2))
    mkts = list(dict.fromkeys(df['market']))
    x = np.arange(len(mkts))
    w = 0.36
    for i, (kind, col) in enumerate((('頂背離', DN), ('底背離', UP))):
        sub = df[df['kind'] == kind].set_index('market').reindex(mkts)
        vals = sub['excess'].fillna(0).values
        bars = ax.bar(x + (i - .5) * w, vals, w, color=col, alpha=.85, label=f'{kind}後 20 日')
        for b, v, t in zip(bars, vals, sub['t'].fillna(0).values):
            va = 'bottom' if v >= 0 else 'top'
            ax.text(b.get_x() + b.get_width() / 2, v, f'{v:+.2f}%\nt={t:+.2f}',
                    ha='center', va=va, color=TX2, fontsize=7.5)
    ax.axhline(0, color=TX2, lw=.8)
    ax.set_xticks(x); ax.set_xticklabels(mkts, color=TX2, fontsize=9)
    ax.yaxis.set_major_formatter(FuncFormatter(pct))
    _style(ax, 'MACD 背離：訊號後 20 日報酬「減去同期無條件平均」（正＝符合宣稱方向）',
           '超額報酬（百分點）')
    ax.legend(fontsize=8.5, facecolor='none', edgecolor='none', labelcolor=TX2, loc='best')
    return _save(fig, 'macd')


# ------------------------------------------------------- 2. 量比
def vr_chart(d):
    rows = []
    for mk in ('SPY', 'QQQ'):
        v = d['vr'].get(mk)
        if not v:
            continue
        rows.append(dict(market=mk, n=v['n_buy'], h1=(v['buy_h1_mean'] or 0) * 100))
    if not rows:
        return None
    df = pd.DataFrame(rows)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 2.8))
    a1.bar(df['market'], df['n'], color=BLUE, alpha=.85, width=.5)
    for i, (m, n) in enumerate(zip(df['market'], df['n'])):
        a1.text(i, n, f'{int(n)} 次', ha='center', va='bottom', color=TX2, fontsize=8.5)
    _style(a1, '量比 2.5–5 買訊次數（10 年）', '次數')
    a2.bar(df['market'], df['h1'], color=[UP if v > 0 else DN for v in df['h1']], alpha=.85, width=.5)
    a2.axhline(0, color=TX2, lw=.8)
    for i, v in enumerate(df['h1']):
        a2.text(i, v, f'{v:+.2f}%', ha='center',
                va='bottom' if v >= 0 else 'top', color=TX2, fontsize=8.5)
    _style(a2, '買訊後 1 日平均報酬', '報酬')
    a2.yaxis.set_major_formatter(FuncFormatter(pct))
    return _save(fig, 'vr')


# ------------------------------------------------------- 3. 跳空
def gap_chart(d):
    rows = []
    for mk in ('SPY', 'QQQ'):
        v = d['gap'].get(mk)
        if not v or not v.get('n'):
            continue
        rows.append(dict(market=mk, fill=v['fill_rate'] * 100, w10=(v['pct_within_10d'] or 0) * 100,
                         med=v['median_days'] or 0, n=v['n']))
    if not rows:
        return None
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    x = np.arange(len(df))
    w = .34
    b1 = ax.bar(x - w / 2, df['fill'], w, color=BLUE, alpha=.85, label='最終回補率')
    b2 = ax.bar(x + w / 2, df['w10'], w, color=GOLD, alpha=.85, label='10 日內回補率')
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f'{b.get_height():.1f}%',
                    ha='center', va='bottom', color=TX2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['market']}\n（{r['n']} 次，中位 {r['med']:.0f} 天）" for _, r in df.iterrows()],
                       color=TX2, fontsize=9)
    ax.set_ylim(0, 112)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:.0f}%'))
    _style(ax, '跳空回補：幾乎都會補，但四成要等超過 10 天', '比率')
    ax.legend(fontsize=8.5, facecolor='none', edgecolor='none', labelcolor=TX2, loc='lower right')
    return _save(fig, 'gap')


# --------------------------------------------------------- 4. KDJ
def kdj_chart(d):
    rows = []
    for mk, v in d['kdj'].items():
        if not isinstance(v, dict) or 'gold' not in v:
            continue
        g = v['gold']
        if g.get('h20_hit') is None:
            continue
        base = v['base'].get(20, {}).get('hit_up')
        rows.append(dict(market=mk, hit=g['h20_hit'] * 100,
                         base=(base or 0) * 100,
                         excess=(g.get('h20_excess') or 0) * 100,
                         n=g['n']))
    if not rows:
        return None
    df = pd.DataFrame(rows)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.8, 2.9))
    x = np.arange(len(df)); w = .34
    a1.bar(x - w / 2, df['hit'], w, color=BLUE, alpha=.85, label='金叉後 20 日上漲率')
    a1.bar(x + w / 2, df['base'], w, color=TX2, alpha=.45, label='同期無條件上漲率')
    a1.set_xticks(x); a1.set_xticklabels(df['market'], color=TX2, fontsize=9)
    a1.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:.0f}%'))
    a1.set_ylim(0, 100)
    _style(a1, '勝率幾乎等於「什麼都不做」', '比率')
    a1.legend(fontsize=8, facecolor='none', edgecolor='none', labelcolor=TX2, loc='lower right')

    a2.bar(x, df['excess'], .5,
           color=[UP if v > 0 else DN for v in df['excess']], alpha=.85)
    a2.axhline(0, color=TX2, lw=.8)
    for i, v in enumerate(df['excess']):
        a2.text(i, v, f'{v:+.2f}%', ha='center',
                va='bottom' if v >= 0 else 'top', color=TX2, fontsize=8.5)
    a2.set_xticks(x); a2.set_xticklabels(df['market'], color=TX2, fontsize=9)
    _style(a2, '扣掉基準後的超額報酬', '超額')
    a2.yaxis.set_major_formatter(FuncFormatter(pct))
    return _save(fig, 'kdj')


# ------------------------------------------------------ 5. 月份效應
def month_chart(d):
    names = [k for k, v in d['month'].items() if 'months' in v]
    if not names:
        return None
    fig, ax = plt.subplots(figsize=(8.0, 3.0))
    months = list(range(1, 13))
    w = .8 / max(len(names), 1)
    cols = [BLUE, UP, GOLD, '#a371f7']
    for i, mk in enumerate(names):
        v = d['month'][mk]
        vals = [(v['months'].get(m, {}).get('mean', np.nan) * 100) for m in months]
        ax.bar([m + (i - len(names) / 2 + .5) * w for m in months], vals, w,
               color=cols[i % len(cols)], alpha=.85,
               label=f"{mk.replace('納斯達克綜合指數（55 年）', '納斯達克 55 年')}")
    ax.axhline(0, color=TX2, lw=.8)
    ax.set_xticks(months)
    ax.set_xticklabels([f'{m}月' for m in months], color=TX2, fontsize=8.5)
    _style(ax, '各月平均報酬：「五窮六絕七翻身」在美股與幣圈都不成立', '平均月報酬')
    ax.yaxis.set_major_formatter(FuncFormatter(pct))
    ax.legend(fontsize=8, facecolor='none', edgecolor='none', labelcolor=TX2,
              loc='upper left', ncol=len(names))
    return _save(fig, 'month')


# ------------------------------------------------------ 6. 幣圈日內
def intraday_chart(d):
    names = [k for k, v in d['intraday'].items() if v.get('n')]
    if not names:
        return None
    fig, ax = plt.subplots(figsize=(8.0, 2.9))
    cols = [GOLD, BLUE]
    for i, mk in enumerate(names):
        v = d['intraday'][mk]
        prof = {p['hour']: p['vol_share'] for p in v['profile']}
        vals = [prof.get(h, 0) * 100 for h in range(24)]
        ax.plot(range(24), vals, lw=1.7, color=cols[i % 2], label=f'{mk}（{v["n"]:,} 小時）',
                marker='o', ms=2.6)
    ax.axhline(100 / 24, color=TX2, ls='--', lw=.9)
    ax.text(23.4, 100 / 24, '均勻分布 4.17%', color=TX2, fontsize=7.5, va='bottom', ha='right')
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f'{h:02d}' for h in range(0, 24, 2)], color=TX2, fontsize=8.5)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:.1f}%'))
    _style(ax, '比特幣每小時成交量佔比（北京時間）：高峰在 20:00–01:00', '該小時佔全天比重')
    ax.legend(fontsize=8.5, facecolor='none', edgecolor='none', labelcolor=TX2, loc='upper left')
    return _save(fig, 'intraday')



# ------------------------------------------------------ 7. 大小盤輪動
def rotation_chart(d):
    rot = d.get('rotation') or {}
    rows = [(k, v) for k, v in rot.items() if isinstance(v, dict) and 'cagr' in v]
    if not rows:
        return None
    # 只畫三個 N 的「原始版」與「雙跌空倉版」，再加兩個買進持有參照
    order = [('N10_plain', 'N=10 輪動'), ('N20_plain', 'N=20 輪動'), ('N60_plain', 'N=60 輪動'),
             ('N20_cash_when_both_down', 'N=20 + 雙跌空倉'),
             ('big_bh', 'SPY 買進持有'), ('small_bh', 'IWM 買進持有')]
    labels, vals, mdd, cols = [], [], [], []
    for key, lbl in order:
        v = rot.get(key)
        if not v:
            continue
        labels.append(lbl)
        vals.append(v['cagr'] * 100)
        mdd.append(v['mdd'] * 100)
        cols.append(GOLD if 'bh' not in key else TX2)
    fig, ax = plt.subplots(figsize=(8.0, 3.1))
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, .55, color=cols, alpha=.9)
    for b, v, m in zip(bars, vals, mdd):
        ax.text(b.get_x() + b.get_width() / 2, v, f'{v:.1f}%\n回撤 {m:.0f}%',
                ha='center', va='bottom' if v >= 0 else 'top', color=TX2, fontsize=7.6)
    ax.axhline(0, color=TX2, lw=.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=TX2, fontsize=8, rotation=18, ha='right')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:.0f}%'))
    _style(ax, '大小盤輪動 vs 直接持有：六種參數全部輸給只買 SPY（2016–2026）',
           '年化報酬')
    return _save(fig, 'rotation')


def main():
    d = json.load(open(os.path.join(OUT, 'tier1.json')))
    out = {}
    for name, fn in (('macd', macd_chart), ('vr', vr_chart), ('gap', gap_chart),
                     ('kdj', kdj_chart), ('month', month_chart), ('intraday', intraday_chart),
                     ('rotation', rotation_chart)):
        try:
            p = fn(d)
            out[name] = p
            print(f'{name:9s} -> {os.path.basename(p) if p else "SKIP"}')
        except Exception as e:
            print(f'{name:9s} -> ERROR {type(e).__name__}: {e}')
    json.dump({k: os.path.basename(v) for k, v in out.items() if v},
              open(os.path.join(OUT, 't1_charts.json'), 'w'))


if __name__ == '__main__':
    main()
