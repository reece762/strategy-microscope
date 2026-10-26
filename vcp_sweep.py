#!/usr/bin/env python3
"""參數穩健性掃描：換幾個合理設定，看「VCP 沒有超額報酬」這個結論會不會翻。

我對 VCP 的形式化做了幾個主觀選擇（分形窗口、最後一段上限、量能倍數、
停損幅度）。如果結論只在一組設定下成立，那是我挑出來的；如果換設定
都指向同一件事，那才是資料在說話。
"""
import json
import os
import sys

import numpy as np
import pandas as pd

import vcp_test as V

OUT = '/root/.openclaw/workspace/reports/_vcp'
os.makedirs(OUT, exist_ok=True)


def market_index(stocks):
    px = pd.DataFrame({s: d['close'] for s, d in stocks.items()}).sort_index()
    # 注意順序：必須先 mean（跳過 NaN）再 fillna。
    # 若寫成 pct_change().fillna(0).mean(axis=1)，早期年份只有少數股票有資料，
    # 大量 NaN 被填成 0 會把市場日報酬稀釋掉，基準變低、超額報酬被高估。
    # （實測兩種寫法讓同一批交易的超額報酬從 +0.01% 變成 +0.50%。）
    mkt = px.pct_change().mean(axis=1).fillna(0)
    return (1 + mkt).cumprod()


def excess(df, cum):
    a = pd.to_datetime(df['entry_date'])
    b = pd.to_datetime(df['exit_date'])
    s = cum.reindex(a, method='ffill').to_numpy()
    e = cum.reindex(b, method='ffill').to_numpy()
    return df['ret'].to_numpy() - (e / s - 1)


def nwt(x, lag=5):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return float('nan')
    mu = x.mean()
    ee = x - mu
    s = ee @ ee / n
    for L in range(1, lag + 1):
        s += 2 * (1 - L / (lag + 1)) * (ee[L:] @ ee[:-L]) / n
    return float(mu / np.sqrt(s / n))


VARIANTS = [
    ('基準（我的預設）', dict(PIVOT_W=5, LAST_MAX=0.10, VOL_MULT=1.2, STOP_PCT=0.07, MIN_SEGS=2)),
    ('最後收縮放寬到 15%', dict(PIVOT_W=5, LAST_MAX=0.15, VOL_MULT=1.2, STOP_PCT=0.07, MIN_SEGS=2)),
    ('分形窗口改 3 根', dict(PIVOT_W=3, LAST_MAX=0.10, VOL_MULT=1.2, STOP_PCT=0.07, MIN_SEGS=2)),
    ('停損放寬到 10%', dict(PIVOT_W=5, LAST_MAX=0.10, VOL_MULT=1.2, STOP_PCT=0.10, MIN_SEGS=2)),
    ('只要 3 段以上收縮', dict(PIVOT_W=5, LAST_MAX=0.10, VOL_MULT=1.2, STOP_PCT=0.07, MIN_SEGS=3)),
    ('不要求量能配合', dict(PIVOT_W=5, LAST_MAX=0.10, VOL_MULT=0.0, STOP_PCT=0.07, MIN_SEGS=2)),
]


def main():
    print('載入…', flush=True)
    S = V.load()
    print(f'  {len(S)} 檔', flush=True)
    cum = market_index(S)

    rows = []
    for name, cfg in VARIANTS:
        for k, v in cfg.items():
            setattr(V, k, v)
        t = V.backtest(S)
        if t.empty:
            print(f'  {name}: 無訊號')
            continue
        ex = excess(t, cum)
        rows.append(dict(
            variant=name, n=len(t),
            win=float((t['ret'] > 0).mean()),
            avg=float(t['ret'].mean()),
            mkt=float((t['ret'].mean() - ex.mean())),
            excess=float(ex.mean()), t=nwt(ex),
            stop=float((t['reason'] == 'stop').mean()),
            trim=float(t[t['ret'] < t['ret'].quantile(0.95)]['ret'].mean()),
        ))
        r = rows[-1]
        print(f'  {name:<20} n={r["n"]:>5,} 勝率{r["win"]*100:>5.1f}% '
              f'原始{r["avg"]*100:>+6.2f}% 超額{r["excess"]*100:>+6.2f}% t={r["t"]:>+5.2f}')

    json.dump(rows, open(f'{OUT}/vcp_sweep.json', 'w'), ensure_ascii=False, indent=1)

    print('\n=== 結論檢查：所有變體的超額報酬是否都接近零 ===')
    ok = all(abs(r['excess']) < 0.01 for r in rows)
    print(f'  {len(rows)} 個變體，全部 |超額| < 1%：{"是" if ok else "否"}')
    for r in rows:
        flag = '✓' if (r['t'] is not None and abs(r['t']) < 2) else '✗'
        print(f'  {flag} {r["variant"]:<20} 超額 {r["excess"]*100:+6.2f}%  t={r["t"]:+5.2f}')


if __name__ == '__main__':
    main()
