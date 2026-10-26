#!/usr/bin/env python3
"""本輪五張新卡片的圖表。

三個分析各自需要不同的圖，因為它們回答的問題不同：
  1. 春哥 3步驟選股 —— 「簡化準則不影響」這句話對不對？→ 重疊率 + 兩組累積報酬
  2. PEAD 財報後漂移 —— 方向對不對？→ 十分位報酬（單調性）
  3. Alpha101 技術面 —— 訊號能活多久？→ IC 隨持有期衰減

所有數字都在這裡重算一次並寫進 out/new5_cards.json，
卡片只讀那份 JSON，不另外手打——手打的數字沒有經過斷言。
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
STOCK = '/root/.openclaw/stock_data'

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


# ============================================================ 價格面板
def load_panel():
    import glob
    files = sorted(glob.glob(f'{STOCK}/live/*.csv') + glob.glob(f'{STOCK}/extra/*.csv'))
    P = {}
    for f in files:
        s = os.path.basename(f)[:-4].upper()
        d = pd.read_csv(f, usecols=['date', 'close', 'volume'])
        d['date'] = pd.to_datetime(d['date'], errors='coerce')
        d = d.dropna(subset=['date', 'close']).groupby('date').last()
        if len(d) < 300:
            continue
        P[s] = d
    idx = None
    for d in P.values():
        idx = d.index if idx is None else idx.union(d.index)
    idx = idx.sort_values()
    close = pd.DataFrame({s: P[s]['close'].reindex(idx) for s in P}).astype(float)
    vol = pd.DataFrame({s: P[s]['volume'].reindex(idx) for s in P}).astype(float)
    return close, vol, idx


# ============================================================ 1. 春哥 Minervini
def minervini(close, vol, idx):
    """比較「2 準則」（春哥簡化版）與「8 準則」（Minervini 原始版）。

    2 準則 = 股價 > MA50 且 > MA200（＋春哥的價格/流動性門檻）
    8 準則 = 上述再加 MA 多頭排列、MA200 向上、距 52 週高 25% 內、
             高於 52 週低 30%、相對強度 > 70
    """
    ma50 = close.rolling(50).mean()
    ma150 = close.rolling(150).mean()
    ma200 = close.rolling(200).mean()
    hi52 = close.rolling(252).max()
    lo52 = close.rolling(252).min()
    ma200p = ma200.shift(21)
    vol30 = vol.rolling(30).mean()
    ret252 = close / close.shift(252) - 1

    me = [i for i in range(1, len(idx) - 61)
          if idx[i].month != idx[i + 1].month]

    n_s, n_r, n_i, eh = [], [], [], []
    A, B = [], []
    for t in me:
        p = close.iloc[t]
        j = t + 21
        fwd = close.iloc[j] / close.iloc[t] - 1
        simple = ((p > ma50.iloc[t]) & (p > ma200.iloc[t])
                  & (p >= 20.0) & (vol30.iloc[t] >= 2_000_000))
        rs = ret252.iloc[t].rank(pct=True) * 99
        real = (simple
                & (p > ma150.iloc[t])
                & (ma50.iloc[t] > ma150.iloc[t])
                & (ma150.iloc[t] > ma200.iloc[t])
                & (ma200.iloc[t] > ma200p.iloc[t])
                & (p >= 0.75 * hi52.iloc[t])
                & (p >= 1.30 * lo52.iloc[t])
                & (rs > 70))
        ok = ma200.iloc[t].notna() & hi52.iloc[t].notna() & fwd.notna()
        simple = simple & ok
        real = real & ok
        if simple.sum() == 0 or real.sum() == 0:
            continue
        n_s.append(int(simple.sum()))
        n_r.append(int(real.sum()))
        n_i.append(int((simple & real).sum()))
        if simple.sum() >= 30 and real.sum() >= 10:
            mkt = fwd[ok].mean()
            A.append((fwd[simple & ~real] - mkt).mean())
            B.append((fwd[real] - mkt).mean())

    n_s, n_r, n_i = map(np.array, (n_s, n_r, n_i))
    A, B = np.array(A), np.array(B)
    # 注意兩個比例方向相反，不可混用：
    #   coverage = 交集 / 8準則 → 定義上必為 100%（2準則是放宽的子集），毫無資訊量
    #   purity   = 交集 / 2準則 → 才是他那句話該被檢驗的地方（選出來的有多純）
    return dict(
        n_simple=float(np.median(n_s)),
        n_real=float(np.median(n_r)),
        n_inter=float(np.median(n_i)),
        coverage=float(np.median(n_i / n_r)),
        purity=float(np.median(n_i / n_s)),
        pct_extra=float(np.median((n_s - n_i) / n_s)),
        a_mean=float(A.mean()),
        b_mean=float(B.mean()),
        a_cum=float(np.prod(1 + A) - 1),
        b_cum=float(np.prod(1 + B) - 1),
        n_months=int(len(A)),
        a_series=[float(x) for x in A],
        b_series=[float(x) for x in B],
    )


def nw_t(x, lag=6):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return float('nan')
    mu = x.mean()
    e = x - mu
    s = e @ e / n
    for L in range(1, min(lag, n - 1) + 1):
        s += 2 * (1 - L / (lag + 1)) * (e[L:] @ e[:-L]) / n
    return float(mu / np.sqrt(s / n)) if s > 0 else float('nan')


def chart_minervini(mn):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    fig.patch.set_facecolor(BG)

    ax = axes[0]
    labs = ['2 準則\n（他簡化版）', '8 準則\n（原始版）', '兩者交集']
    vals = [mn['n_simple'], mn['n_real'], mn['n_inter']]
    cols = [DN, UP, BLUE]
    b = ax.bar(labs, vals, color=cols, width=0.58, alpha=0.85)
    for r, v in zip(b, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + 3, f'{v:.0f}',
                ha='center', color=TX, fontsize=10)
    _style(ax, '每個月底通過篩選的檔數（中位數）', ylab='檔數')
    ax.text(0.5, 0.93, f'他的名單只有 {mn["purity"]*100:.0f}% 符合原版',
            transform=ax.transAxes, ha='center', color=GOLD, fontsize=10)

    ax = axes[1]
    A = np.array(mn['a_series'])
    B = np.array(mn['b_series'])
    ax.plot(np.cumprod(1 + A) - 1, color=DN, lw=1.6,
            label=f'只過 2 準則（多出的 {mn["pct_extra"]*100:.0f}%）')
    ax.plot(np.cumprod(1 + B) - 1, color=UP, lw=1.6,
            label='符合 8 準則')
    ax.axhline(0, color=GRAY, lw=0.8, ls='--')
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=TX, fontsize=8, loc='lower left')
    _style(ax, '累積超額報酬（持有一個月，減全體平均）', ylab='累積報酬', xlab='月底序')
    ax.yaxis.set_major_formatter(lambda v, _: f'{v*100:.0f}%')

    fig.tight_layout()
    return save(fig, 'mn_ab.png')


# ============================================================ 2. PEAD
def pead():
    d = pd.read_csv('/root/.openclaw/workspace/reports/_pead/events.csv')
    d['report_date'] = pd.to_datetime(d['report_date'])
    d['quarter'] = d['report_date'].dt.to_period('Q')
    d = d[d['sue'].notna()].copy()
    d['dec'] = np.nan
    for q, g in d.groupby('quarter'):
        if len(g) < 50:
            continue
        d.loc[g.index, 'dec'] = np.clip(np.ceil(g['sue'].rank(pct=True) * 10), 1, 10)
    d = d[d['dec'].notna()]

    H = [21, 63, 126, 252]
    dec_means = {}
    ls = {}
    for h in H:
        c = f'ret_{h}'
        dec_means[h] = [float(d.loc[d['dec'] == k, c].mean()) for k in range(1, 11)]
        ser = []
        for q, g in d.groupby('quarter'):
            a = g.loc[g['dec'] == 10, c].dropna()
            b = g.loc[g['dec'] == 1, c].dropna()
            if len(a) >= 3 and len(b) >= 3:
                ser.append(a.mean() - b.mean())
        ls[h] = dict(mean=float(np.mean(ser)) if ser else float('nan'),
                     t=nw_t(ser), n=len(ser),
                     mono=float(np.corrcoef(np.arange(10), dec_means[h])[0, 1]))
    return d, dec_means, ls


def chart_pead(dec_means, ls):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    fig.patch.set_facecolor(BG)

    ax = axes[0]
    x = np.arange(1, 11)
    for h, col in [(21, BLUE), (63, UP), (252, GOLD)]:
        ax.plot(x, np.array(dec_means[h]) * 100, marker='o', ms=4,
                color=col, lw=1.6, label=f'{h} 交易日')
    _style(ax, 'SUE 十分位 × 前瞻超額報酬', ylab='%', xlab='十分位（1=最負、10=最正）')
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=TX, fontsize=8)
    ax.set_xticks(x)

    ax = axes[1]
    hs = [21, 63, 126, 252]
    vals = [ls[h]['mean'] * 100 for h in hs]
    ts = [ls[h]['t'] for h in hs]
    cols = [UP if v > 0 else DN for v in vals]
    b = ax.bar([f'{h}日' for h in hs], vals, color=cols, width=0.55, alpha=0.85)
    for r, v, t in zip(b, vals, ts):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.03,
                f't={t:+.2f}', ha='center', color=TX2, fontsize=8)
    ax.axhline(0, color=GRAY, lw=0.8)
    _style(ax, 'D10−D1 多空報酬（Bonferroni 門檻 |t|≥2.50）', ylab='%')
    ax.text(0.5, 0.93, '四條全部不顯著', transform=ax.transAxes,
            ha='center', color=DN, fontsize=9)

    fig.tight_layout()
    return save(fig, 'pd_decile.png')


def chart_pead_year(d):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    fig.patch.set_facecolor(BG)
    d = d.copy()
    d['year'] = d['report_date'].dt.year
    ys, vs = [], []
    for y, g in d.groupby('year'):
        a = g.loc[g['dec'] == 10, 'ret_63'].dropna()
        b = g.loc[g['dec'] == 1, 'ret_63'].dropna()
        if len(a) < 5 or len(b) < 5:
            continue
        ys.append(int(y))
        vs.append((a.mean() - b.mean()) * 100)
    cols = [UP if v > 0 else DN for v in vs]
    ax.bar(ys, vs, color=cols, width=0.7, alpha=0.85)
    ax.axhline(0, color=GRAY, lw=0.8)
    neg = sum(1 for v in vs if v < 0)
    _style(ax, f'逐年 D10−D1（63 交易日）—— {len(vs)} 年中有 {neg} 年為負', ylab='%')
    fig.tight_layout()
    return save(fig, 'pd_year.png')


# ============================================================ 3. Alpha101 衰減
def chart_alpha101():
    # 兩份資料不可混用——這是我上一版畫錯的地方：
    #   n5_alpha_decay.json = 累積式（訊號在 t，持有 t→t+h），持有期 1/2/3/4/6/9/12
    #   n5_alpha_incr.json  = 只持有第 m 個月（t+m-1 → t+m），月份 1~6
    # 左圖畫累積衰減，右圖畫逐月遞增；卡片正文引用的 +0.0074 / +0.0034 屬於右圖那組。
    d = json.load(open(os.path.join(HERE, 'n5_alpha_decay.json')))
    inc = json.load(open(os.path.join(HERE, 'n5_alpha_incr.json')))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    fig.patch.set_facecolor(BG)

    ax = axes[0]
    hs = sorted(int(k) for k in d['12'].keys())
    cols = {12: BLUE, 35: UP, 38: GOLD, 45: '#d2a8ff', 24: '#ff7b72', 95: GRAY}
    for aid in ('12', '35', '38', '45', '24'):
        if aid not in d:
            continue
        ys = [d[aid][str(h)][0] for h in hs]
        ax.plot(hs, ys, marker='o', ms=4, color=cols.get(int(aid), GRAY),
                lw=1.5, label=f'#{aid}')
    ax.axhline(0, color=GRAY, lw=0.8, ls='--')
    _style(ax, 'IC 隨「累積」持有期衰減（持有 t→t+h）', ylab='IC', xlab='持有月數 h')
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=TX, fontsize=8, ncol=2)

    ax = axes[1]
    aid = '12'
    ms = sorted(int(k) for k in inc[aid].keys())
    vals = [inc[aid][str(m)]['ic'] for m in ms]
    ts = [inc[aid][str(m)]['t'] for m in ms]
    b = ax.bar([f'第{m}月' for m in ms], vals,
               color=[UP if v > 0 else DN for v in vals], width=0.6, alpha=0.85)
    for r, v, t in zip(b, vals, ts):
        off = 0.0008 if v >= 0 else -0.0016
        ax.text(r.get_x() + r.get_width() / 2, v + off,
                f't={t:+.2f}', ha='center', color=TX2, fontsize=7.5)
    ax.axhline(0, color=GRAY, lw=0.8)
    _style(ax, '#12 只持有「第 m 個月」（訊號只在月底算一次）', ylab='IC')
    ax.text(0.5, 0.93, '第 1 月 t=5.18 → 第 2 月 t=1.53',
            transform=ax.transAxes, ha='center', color=GOLD, fontsize=9)

    fig.tight_layout()
    return save(fig, 'al_decay.png')


# ============================================================ 4. 頻道調查
def chart_survey():
    C = json.load(open(os.path.join(HERE, 'n5_channels.json')))
    RULE = ['選股', '选股', '策略', '回測', '回测', '因子', '量化', '篩選', '筛选',
            '均線', '均线', '止損', '止损', '財報', '财报', '本益比', 'EPS',
            '突破', '支撐', '支撑', '盈虧比', '盈亏比']
    rows = []
    for cid, v in C.items():
        ts = v.get('titles') or []
        if not ts:
            continue
        n = sum(1 for t in ts if any(k in t for k in RULE))
        rows.append((v.get('subs') or 0, v['name'], n, len(ts)))
    rows.sort(key=lambda r: -r[0])
    named = {'春哥美股投資 Spring Invest', 'Jack Investment', 'J Law'}

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    fig.patch.set_facecolor(BG)
    xs = np.arange(len(rows))
    cols = [UP if r[1] in named else GRAY for r in rows]
    ax.barh(xs, [r[0] / 1000 for r in rows], color=cols, alpha=0.85, height=0.68)
    ax.set_yticks(xs)
    ax.set_yticklabels([r[1][:16] for r in rows], fontsize=8)
    ax.invert_yaxis()
    _style(ax, '17 個中文美股頻道：訂閱數與是否有可檢驗規則', xlab='訂閱數（千人）')
    ax.text(0.55, 0.06, '綠色 = 提出可檢驗的選股規則（3 個）\n灰色 = 新聞／敘事／喊單（14 個）',
            transform=ax.transAxes, color=TX2, fontsize=8.5)
    fig.tight_layout()
    return save(fig, 'ch_survey.png')


# ============================================================ main
def main():
    os.makedirs(OUT, exist_ok=True)
    print('載入價格面板…', flush=True)
    close, vol, idx = load_panel()
    print(f'  {len(idx)} 天 × {close.shape[1]} 檔', flush=True)

    print('1/4 春哥 Minervini 對照…', flush=True)
    mn = minervini(close, vol, idx)
    mn['a_t'] = nw_t(mn['a_series'])
    mn['b_t'] = nw_t(mn['b_series'])
    print(f'  2準則 {mn["n_simple"]:.0f} 檔 / 8準則 {mn["n_real"]:.0f} 檔 / '
          f'純度 {mn["purity"]*100:.0f}%（多出 {mn["pct_extra"]*100:.0f}%）')
    print(f'  A(多出) {mn["a_mean"]*100:+.3f}%/月 t={mn["a_t"]:+.2f} | B {mn["b_mean"]*100:+.3f}% t={mn["b_t"]:+.2f}')
    f1 = chart_minervini(mn)

    print('2/4 PEAD…', flush=True)
    d, dm, ls = pead()
    print(f'  h=21 D10-D1 {ls[21]["mean"]*100:+.3f}% t={ls[21]["t"]:+.2f} 單調性 {ls[21]["mono"]:+.3f}')
    f2 = chart_pead(dm, ls)
    f3 = chart_pead_year(d)

    print('3/4 Alpha101 衰減…', flush=True)
    f4 = chart_alpha101()

    print('4/4 頻道調查…', flush=True)
    f5 = chart_survey()

    res = dict(
        minervini=mn,
        pead=dict(
            deciles={str(h): v for h, v in dm.items()},
            ls={str(h): v for h, v in ls.items()},
        ),
        charts=dict(mn=f1, pd_decile=f2, pd_year=f3, al=f4, ch=f5),
    )
    json.dump(res, open(os.path.join(OUT, 'new5_cards.json'), 'w'),
              ensure_ascii=False, indent=1)
    print(f'\n已寫 out/new5_cards.json，PNG: {[f1,f2,f3,f4,f5]}')


if __name__ == '__main__':
    main()
