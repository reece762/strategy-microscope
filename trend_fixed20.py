#!/usr/bin/env python3
"""三組進場條件，固定持有 20 日（無停損、無均線出場），純比「選股」能力。

## 為什麼要做這個

我先前用「3% 停損 + 跌破 20 日均線」出場，得到：
  趨勢線 −0.025%、突破 20 日新高 −0.156%、隨機 −0.094%
   → 趨勢線顯著優於 20 日新高（差值 +0.13pp，CI 不含零）

但注意三組的平均持有天數是 10.1、12.1、6.5 日。
**超額報酬是「個股報酬 − 同期市場報酬」，持有越久要扣的市場報酬越多。**
所以「持有 12 天的那組超額更差」可能純粹是因為它抱得久，
不是因為它的選股能力差。這是一個致命的比較陷阱。

獨立實作改用固定持有 20 日（三組完全一樣），得到：
  趨勢線 −0.150%、20 日新高 −0.128%、隨機 +0.014%
   → 趨勢線與 20 日新高無差異、與隨機也無差異

兩邊結論相反。這裡用我的資料、固定持有 20 日重跑一次，
把「出場規則」這個變數完全拿掉，只剩「進場條件」在比。
"""
import glob
import json
import os

import numpy as np
import pandas as pd

STOCK = '/root/.openclaw/stock_data'
LINES = '/root/.openclaw/workspace/reports/_agent_lines'
OUT = '/root/.openclaw/workspace/backtest_videos/xbx/out'

HOLD = 20
COOL = 10
SEED = 20261027


def load():
    out = {}
    for f in sorted(glob.glob(f'{STOCK}/live/*.csv') + glob.glob(f'{STOCK}/extra/*.csv')):
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        d.columns = [c.lower() for c in d.columns]
        if not {'date', 'close'}.issubset(d.columns) or len(d) < 300:
            continue
        d['date'] = pd.to_datetime(d['date'])
        out[os.path.basename(f).replace('.csv', '')] = (
            d.dropna(subset=['close']).sort_values('date').reset_index(drop=True))
    return out


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


def build(stocks, mode, counts=None, rng=None):
    rows = []
    for s, df in stocks.items():
        n = len(df)
        if n < HOLD + 80:
            continue
        close = df['close'].to_numpy(float)
        if mode == 'high20':
            hi20 = pd.Series(close).rolling(20).max().shift(1).to_numpy()
        want = None if counts is None else counts.get(s, 0)
        if mode == 'random' and not want:
            continue
        last = -999
        picked = 0
        for k in range(60, n - HOLD - 1):
            if k - last < COOL:
                continue
            if mode == 'high20':
                if not (close[k] > hi20[k]):
                    continue
            elif mode == 'random':
                # 每 5 天檢查一次，符合冷卻後才抽
                if rng.random() > 0.02:
                    continue
            rows.append(dict(symbol=s, entry_date=df['date'].iloc[k],
                             exit_date=df['date'].iloc[k + HOLD],
                             ret=float(close[k + HOLD] / close[k] - 1),
                             hold=HOLD))
            last = k
            picked += 1
            if want is not None and picked >= want:
                break
    return pd.DataFrame(rows)


def main():
    stocks = load()
    P = pd.concat([d.set_index('date')['close'] for d in stocks.values()],
                  axis=1, sort=True).sort_index()
    nav = (1 + P.pct_change().mean(axis=1).fillna(0)).cumprod()

    t = pd.read_csv(f'{LINES}/trendline_trades.csv')
    t['entry_date'] = pd.to_datetime(t.entry_date)
    t['exit_date'] = pd.to_datetime(t.exit_date)
    t = dedup(t)

    # 趨勢線改成固定持有 20 日（原本出場日不同，重算報酬）
    tn = []
    for _, r in t.iterrows():
        s = r['symbol']
        if s not in stocks:
            continue
        ser = stocks[s].set_index('date')['close']
        i = ser.index.searchsorted(r['entry_date'])
        if i >= len(ser) or i + HOLD >= len(ser):
            continue
        tn.append(dict(symbol=s, entry_date=ser.index[i], exit_date=ser.index[i + HOLD],
                       ret=float(ser.iloc[i + HOLD] / ser.iloc[i] - 1), hold=HOLD))
    tn = dedup(pd.DataFrame(tn))

    counts = tn.groupby('symbol').size().to_dict()
    rng = np.random.default_rng(SEED)

    hi = dedup(build(stocks, 'high20'))
    rd = dedup(build(stocks, 'random', counts, rng))

    def exc(df):
        m0 = nav.reindex(df.entry_date).ffill().values
        m1 = nav.reindex(df.exit_date).ffill().values
        return df.ret.values - (m1 / m0 - 1)

    def summ(df, label):
        e = exc(df)
        q = np.quantile(e, .95)
        return dict(label=label, n=int(len(e)), win=float(df.ret.gt(0).mean()),
                    avg=float(df.ret.mean()), excess=float(e.mean()), t=float(nwt(e)),
                    trimmed=float(e[e <= q].mean()), hold=float(df.hold.mean()))

    a, b, c = exc(tn), exc(hi), exc(rd)

    def boot(x, y, reps=2000):
        d = [rng.choice(x, len(x), True).mean() - rng.choice(y, len(y), True).mean()
             for _ in range(reps)]
        return [float(np.quantile(d, .025)), float(np.quantile(d, .975))]

    res = dict(trendline=summ(tn, '趨勢線突破'), high20=summ(hi, '突破 20 日新高'),
               random=summ(rd, '隨機進場'),
               diff_vs_high=dict(mean=float(a.mean() - b.mean()), ci=boot(a, b)),
               diff_vs_random=dict(mean=float(a.mean() - c.mean()), ci=boot(a, c)))
    json.dump(res, open(f'{OUT}/trend_fixed20.json', 'w'), ensure_ascii=False, indent=1)

    print(f'固定持有 {HOLD} 日，三組同一出場規則')
    for k in ('trendline', 'high20', 'random'):
        s = res[k]
        print(f"  {s['label']:14s} n={s['n']:>7,} 勝率 {s['win']*100:5.1f}% "
              f"原始 {s['avg']*100:+.3f}% 超額 {s['excess']*100:+.3f}% t={s['t']:+.2f} "
              f"剔除5% {s['trimmed']*100:+.3f}%")
    for k in ('diff_vs_high', 'diff_vs_random'):
        d = res[k]
        print(f"  {k:16s} {d['mean']*100:+.4f}%  CI [{d['ci'][0]*100:+.3f}, {d['ci'][1]*100:+.3f}] "
              f"{'含零（無差異）' if d['ci'][0] < 0 < d['ci'][1] else '不含零'}")


if __name__ == '__main__':
    main()
