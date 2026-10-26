#!/usr/bin/env python3
"""修正超額報酬的基準算法（精簡版）。

## 我原本錯在哪

我用「等權市場指數的複利報酬」當基準：
    excess = 個股報酬 − (指數期末 / 指數期初 − 1)

問題是**波動拖累（volatility drag）不對稱**。
單一檔股票 20 日的複利報酬被它自己的高波動拖下來；
指數（616 檔平均）波動低得多，拖得少。
所以同一段期間，指數的複利報酬會**系統性地高於**「隨機挑一檔股票的期望報酬」。

實測（4,000 次隨機抽樣，20 日）：
    平均個股 20 日報酬 − 指數複利 20 日報酬 = −0.186%（t = −6.8）
    持有 5 日 −0.015%、10 日 −0.081%、20 日 −0.155%、40 日 −0.242%

**這個偏誤的大小，剛好等於我先前報告的那些「超額 −0.02% ~ −0.19%」。**

最直接的證據：**隨機進場的超額報酬應該恰好為零**，
但我用舊基準算出來是 −0.24%（t = −4.57）。純隨機策略不可能顯著賠錢。

## 正確的基準

對每一筆交易，取「同一天進場、同樣持有 h 日」的**橫斷面平均報酬**
（所有當天有價格的股票，各自的 h 日報酬取算術平均）。
這樣隨機挑一檔股票的期望超額恰好是 0，符合定義。
"""
import glob
import json
import os

import numpy as np
import pandas as pd

STOCK = '/root/.openclaw/stock_data'
OUT = '/root/.openclaw/workspace/backtest_videos/xbx/out'
LINES = '/root/.openclaw/workspace/reports/_agent_lines'
PATT = '/root/.openclaw/workspace/reports/_agent_pattern'

MAXH = 61


def load_prices():
    cols = {}
    for f in sorted(glob.glob(f'{STOCK}/live/*.csv') + glob.glob(f'{STOCK}/extra/*.csv')):
        try:
            d = pd.read_csv(f, usecols=lambda c: c.lower() in ('date', 'close'))
        except Exception:
            continue
        d.columns = [c.lower() for c in d.columns]
        if 'date' not in d.columns or 'close' not in d.columns or len(d) < 300:
            continue
        d['date'] = pd.to_datetime(d['date'])
        cols[os.path.basename(f).replace('.csv', '')] = (
            d.dropna(subset=['close']).set_index('date')['close'])
    P = pd.concat(cols, axis=1, sort=True).sort_index()
    return P.where(P > 0)


def build_bench(P):
    """bench[h] = 每個進場日的橫斷面平均 h 日前瞻報酬（float array, 對齊 P.index）。"""
    A = P.to_numpy(float)
    n = len(A)
    bench = np.full((MAXH + 1, n), np.nan)
    for h in range(1, MAXH + 1):
        fwd = np.full(n, np.nan)
        num = A[:n - h]
        den = A[h:]
        with np.errstate(invalid='ignore', divide='ignore'):
            r = den / num - 1.0
        cnt = np.sum(~np.isnan(r), axis=1)
        ssum = np.nansum(r, axis=1)
        good = cnt >= 30
        fwd[:n - h][good] = ssum[good] / cnt[good]
        bench[h] = fwd
    return bench


def dedup(df):
    keep = []
    for _, g in df.sort_values('entry_date').groupby('symbol'):
        last = None
        for i, r in g.iterrows():
            if last is None or r.entry_date >= last:
                keep.append(i)
                last = r.exit_date
            elif r.exit_date > last:
                last = r.exit_date
    return df.loc[keep]


def nwt(ex, lag=10):
    n = len(ex)
    mu = ex.mean()
    e = ex - mu
    s = (e ** 2).mean()
    for l in range(1, lag + 1):
        s += 2 * (1 - l / (lag + 1)) * (e[l:] * e[:-l]).mean()
    return mu / np.sqrt(max(s, 1e-18) / n)


def main():
    P = load_prices()
    idx = P.index
    pos = pd.Series(np.arange(len(idx)), index=idx)
    print(f'母體 {P.shape[1]} 檔 × {len(idx):,} 日', flush=True)
    bench = build_bench(P)
    print('基準表完成', flush=True)

    def apply_bench(df):
        df = dedup(df)
        h = df['hold'].astype(int).clip(1, MAXH).to_numpy()
        p0 = pos.reindex(pd.DatetimeIndex(df.entry_date)).to_numpy()
        bm = np.full(len(df), np.nan)
        ok = ~np.isnan(p0)
        for hh in np.unique(h[ok]):
            m = ok & (h == hh)
            bm[m] = bench[int(hh)][p0[m].astype(int)]
        df = df.assign(bm=bm)
        df = df.dropna(subset=['bm'])
        df['excess'] = df['ret'].to_numpy() - df['bm'].to_numpy()
        return df

    def summ(df, label):
        ex = df.excess.to_numpy()
        q = np.quantile(ex, .95)
        return dict(label=label, n=int(len(ex)),
                    win=float(df.ret.gt(0).mean()), avg=float(df.ret.mean()),
                    bench=float(df.bm.mean()), excess=float(ex.mean()),
                    t=float(nwt(ex)), trimmed=float(ex[ex <= q].mean()),
                    hold=float(df['hold'].mean()))

    jobs = [
        ('trendline', f'{LINES}/trendline_trades.csv', '趨勢線突破'),
        ('trend_ctrl', f'{LINES}/trendline_control.csv', '突破 20 日新高'),
        ('channel', f'{PATT}/channel_trades.csv', '平行通道'),
        ('chan_ctrl', f'{PATT}/channel_control.csv', '通道對照（跌破MA20後站回）'),
    ]
    res = {}
    for key, path, label in jobs:
        df = pd.read_csv(path)
        df['entry_date'] = pd.to_datetime(df.entry_date)
        df['exit_date'] = pd.to_datetime(df.exit_date)
        df = apply_bench(df)
        res[key] = summ(df, label)
        s = res[key]
        print(f"  {s['label']:24s} n={s['n']:>7,} 原始 {s['avg']*100:+.3f}% "
              f"基準 {s['bench']*100:+.3f}% 超額 {s['excess']*100:+.3f}% "
              f"t={s['t']:+.2f} 剔除5% {s['trimmed']*100:+.3f}%", flush=True)
        json.dump(res, open(f'{OUT}/fixed_bench.json', 'w'), ensure_ascii=False, indent=1)

    # 健檢：隨機進場的超額應為 0
    rng = np.random.default_rng(20261027)
    A = P.to_numpy(float)
    lo, hi = 60, len(idx) - 70
    rows = []
    for _ in range(6000):
        i = int(rng.integers(lo, hi))
        h = int(rng.choice([5, 10, 20, 40]))
        j = i + h
        ok = ~np.isnan(A[i]) & ~np.isnan(A[j])
        if ok.sum() < 30:
            continue
        c = rng.choice(np.where(ok)[0])
        rows.append((i, h, float(A[j, c] / A[i, c] - 1)))
    rd = pd.DataFrame(rows, columns=['i', 'h', 'ret'])
    bm = np.array([bench[int(h)][int(i)] for i, h in zip(rd.i, rd.h)])
    ex = rd.ret.to_numpy() - bm
    ex = ex[~np.isnan(ex)]
    print(f"\n  【健檢】隨機進場 n={len(ex):,} 超額 {ex.mean()*100:+.4f}% "
          f"t={nwt(ex):+.2f}  ← 應接近 0", flush=True)
    res['sanity_random'] = dict(n=int(len(ex)), excess=float(ex.mean()), t=float(nwt(ex)))

    json.dump(res, open(f'{OUT}/fixed_bench.json', 'w'), ensure_ascii=False, indent=1)
    print('已寫 fixed_bench.json')


if __name__ == '__main__':
    main()
