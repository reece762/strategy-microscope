#!/usr/bin/env python3
"""趨勢線的兩項消融：對手是「20 日新高」與「隨機進場」。

## 為什麼要自己重做

趨勢線子代理報了兩組很關鍵的對照（趨勢線 vs 20 日新高：無差異；
趨勢線 vs 隨機進場：更差），但它的出場規則是「固定持有 20 日」，
與春哥原話（停損 3%、跌破 20 日均線出場）不同。
兩者混用會讓「誰比誰差」變成規則差異，不是訊號差異。

所以這裡全部用自己的出場規則（停損 3%、跌破 20 日均線、最長 60 日），
只換「進場條件」，三個版本才能互相比較：
  T 趨勢線突破（春哥 + J Law 的規則）
  H 突破 20 日新高（拿掉畫線這件事）
  R 隨機進場（同樣的出場規則，日期隨機）
"""
import glob
import json
import os

import numpy as np
import pandas as pd

STOCK = '/root/.openclaw/stock_data'
OUT = '/root/.openclaw/workspace/backtest_videos/xbx/out'

STOP = 0.03
MAXHOLD = 60
COOL = 10
W_CONFIRM = 5
SEED = 20261027


def load():
    files = sorted(glob.glob(f'{STOCK}/live/*.csv') + glob.glob(f'{STOCK}/extra/*.csv'))
    out = {}
    for f in files:
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        d.columns = [c.lower() for c in d.columns]
        need = {'date', 'high', 'low', 'close', 'volume'}
        if not need.issubset(d.columns) or len(d) < 300:
            continue
        d['date'] = pd.to_datetime(d['date'])
        out[os.path.basename(f).replace('.csv', '')] = d.sort_values('date').reset_index(drop=True)
    return out


def exits(df, k):
    """從第 k 天收盤進場，回傳 (出場索引, 原因, 出場價)。"""
    close = df['close'].to_numpy(float)
    low = df['low'].to_numpy(float)
    ma20 = pd.Series(close).rolling(20).mean().to_numpy()
    n = len(close)
    entry = close[k]
    stop = entry * (1 - STOP)
    for q in range(k + 1, min(k + MAXHOLD + 1, n)):
        if low[q] <= stop:
            return q, 'stop', stop
        if close[q] < ma20[q]:
            return q, 'ma20', close[q]
    ej = min(k + MAXHOLD, n - 1)
    return ej, 'time', close[ej]


def trade_row(s, df, k, ej, reason, px):
    return dict(symbol=s, entry_date=df['date'].iloc[k], exit_date=df['date'].iloc[ej],
                ret=float(px / df['close'].iloc[k] - 1), reason=reason, hold=int(ej - k))


def run_high(stocks):
    trades = []
    for s, df in stocks.items():
        close = df['close'].to_numpy(float)
        n = len(close)
        hi20 = pd.Series(close).rolling(20).max().shift(1).to_numpy()
        last = -999
        for k in range(60, n - 1):
            if k - last < COOL:
                continue
            if not (close[k] > hi20[k]):
                continue
            ej, reason, px = exits(df, k)
            trades.append(trade_row(s, df, k, ej, reason, px))
            last = k
    return pd.DataFrame(trades)


def run_random(stocks, counts, rng):
    """每檔抽與趨勢線相同筆數的隨機進場日，套同樣的出場規則。"""
    trades = []
    for s, df in stocks.items():
        nt = counts.get(s, 0)
        if nt <= 0:
            continue
        n = len(df)
        lo, hi_ = 60, n - 2
        if hi_ <= lo:
            continue
        cand = rng.choice(np.arange(lo, hi_), size=min(nt * 3, hi_ - lo), replace=False)
        cand = np.sort(cand)
        taken = []
        for k in cand:
            if taken and k - taken[-1] < COOL:
                continue
            ej, reason, px = exits(df, int(k))
            trades.append(trade_row(s, df, int(k), ej, reason, px))
            taken.append(int(k))
            if len(taken) >= nt:
                break
    return pd.DataFrame(trades)


def market_cum(stocks):
    closes = [df.set_index('date')['close'] for df in stocks.values()]
    P = pd.concat(closes, axis=1, sort=True).sort_index()
    return (1 + P.pct_change().mean(axis=1).fillna(0)).cumprod()


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


def summarize(df, mcum, label, reps=None):
    m0 = mcum.reindex(df.entry_date).ffill().values
    m1 = mcum.reindex(df.exit_date).ffill().values
    ex = df.ret.values - (m1 / m0 - 1)
    q = np.quantile(ex, 0.95)
    tr = ex[ex <= q]
    out = dict(label=label, n=int(len(ex)), win=float(df.ret.gt(0).mean()),
               avg=float(df.ret.mean()), excess=float(ex.mean()), t=float(nwt(ex)),
               trimmed=float(tr.mean()), hold=float(df.hold.mean()),
               stop_rate=float((df.reason == 'stop').mean()))
    if reps is not None:
        out['ci'] = [float(np.quantile(reps, .025)), float(np.quantile(reps, .975))]
    return out


def boot_diff(a, b, rng, reps=2000):
    """兩組超額報酬差值（a − b）的拔靴信賴區間。"""
    d = []
    for _ in range(reps):
        xa = rng.choice(a, len(a), replace=True).mean()
        xb = rng.choice(b, len(b), replace=True).mean()
        d.append(xa - xb)
    return np.array(d)


def main():
    stocks = load()
    mcum = market_cum(stocks)
    rng = np.random.default_rng(SEED)
    print(f'母體 {len(stocks)} 檔', flush=True)

    # 趨勢線：沿用既有交易檔（已是同一套出場規則）
    t = pd.read_csv('/root/.openclaw/workspace/reports/_agent_lines/trendline_trades.csv')
    t['entry_date'] = pd.to_datetime(t.entry_date)
    t['exit_date'] = pd.to_datetime(t.exit_date)
    t = dedup(t)
    counts = t.groupby('symbol').size().to_dict()

    hi = dedup(run_high(stocks))
    print(f'20 日新高 {len(hi):,} 筆', flush=True)

    rd = dedup(run_random(stocks, counts, rng))
    print(f'隨機進場 {len(rd):,} 筆', flush=True)

    def exc(df):
        m0 = mcum.reindex(df.entry_date).ffill().values
        m1 = mcum.reindex(df.exit_date).ffill().values
        return df.ret.values - (m1 / m0 - 1)

    a, b, c = exc(t), exc(hi), exc(rd)

    res = dict(
        trendline=summarize(t, mcum, '趨勢線突破'),
        high20=summarize(hi, mcum, '突破 20 日新高'),
        random=summarize(rd, mcum, '隨機進場'),
    )
    d_th = boot_diff(a, b, rng)
    d_tr = boot_diff(a, c, rng)
    res['diff_trend_vs_high'] = dict(mean=float(a.mean() - b.mean()),
                                     ci=[float(np.quantile(d_th, .025)),
                                         float(np.quantile(d_th, .975))])
    res['diff_trend_vs_random'] = dict(mean=float(a.mean() - c.mean()),
                                       ci=[float(np.quantile(d_tr, .025)),
                                           float(np.quantile(d_tr, .975))])

    json.dump(res, open(f'{OUT}/trend_ablation.json', 'w'), ensure_ascii=False, indent=1)

    for k in ('trendline', 'high20', 'random'):
        s = res[k]
        print(f"  {s['label']:14s} n={s['n']:>7,} 勝率 {s['win']*100:5.1f}% "
              f"原始 {s['avg']*100:+.3f}% 超額 {s['excess']*100:+.3f}% t={s['t']:+.2f} "
              f"剔除5% {s['trimmed']*100:+.3f}% 持有 {s['hold']:.1f}日")
    for k in ('diff_trend_vs_high', 'diff_trend_vs_random'):
        d = res[k]
        print(f"  {k}: {d['mean']*100:+.4f}%  CI [{d['ci'][0]*100:+.3f}%, {d['ci'][1]*100:+.3f}%]"
              f"  {'含零（無差異）' if d['ci'][0] < 0 < d['ci'][1] else '不含零（有差異）'}")
    print('已寫 trend_ablation.json')


if __name__ == '__main__':
    main()
