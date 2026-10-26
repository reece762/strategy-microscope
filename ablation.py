#!/usr/bin/env python3
"""平行通道的消融測試：到底「通道幾何」本身有沒有貢獻？

為什麼要做這個：
第一版卡片的對照組是「跌破 20 日均線後站回 + 低量」——那不是
「同一策略拿掉通道條件」，而是另一個完全不同的訊號，勝率 24.9%。
用它來證明「通道選得比較好」是不成立的比較。

正確的消融必須只改一件事：把「價格觸及通道下軌」換成別的支撐定義，
其餘（低量、收盤在 20 日均線上方、停損 3%、跌破均線出場、最長 60 日）完全相同。

三個變體：
  A 通道下軌   ：low 距通道下軌 ≤2%（原策略）
  B 均線支撐   ：low 距 20 日均線 ≤2%（安慰劑：支撐是真的，但沒有通道）
  C 隨機同日   ：同一天隨機抽與 A 相同數量的股票（純頻率對照）
所有變體都套同樣的 10 日冷卻，避免同一段行情被切成多筆。
"""
import glob
import json
import os

import numpy as np
import pandas as pd

STOCK = '/root/.openclaw/stock_data'
OUT = '/root/.openclaw/workspace/reports/_agent_pattern'

TOL = 0.02
COOL = 10
STOP = 0.03
MAXHOLD = 60
W_CONFIRM = 5          # 分形右側確認根數：第 k 天只能用到 i <= k-5 的擺動點
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
        need = {'date', 'open', 'high', 'low', 'close', 'volume'}
        if not need.issubset(d.columns) or len(d) < 300:
            continue
        d['date'] = pd.to_datetime(d['date'])
        d = d.sort_values('date').reset_index(drop=True)
        out[os.path.basename(f).replace('.csv', '')] = d
    return out


def swings_low(low, w=5):
    """左 5 右 5 的分形低點；右側需 5 根確認，防前視。"""
    n = len(low)
    piv = np.zeros(n, dtype=bool)
    for i in range(w, n - w):
        if low[i] == low[i - w:i + w + 1].min():
            piv[i] = True
    return piv


def run(stocks, mode, rng=None):
    trades = []
    for s, df in stocks.items():
        n = len(df)
        if n < 120:
            continue
        close = df['close'].to_numpy(float)
        low = df['low'].to_numpy(float)
        vol = df['volume'].to_numpy(float)
        ma20 = pd.Series(close).rolling(20).mean().to_numpy()
        vol20 = pd.Series(vol).rolling(20).mean().to_numpy()
        piv = swings_low(low) if mode == 'channel' else None

        last = -999
        for k in range(80, n - 1):
            if k - last < COOL:
                continue
            if not (ma20[k] == ma20[k] and close[k] > ma20[k]):
                continue
            if not (vol20[k] > 0 and vol[k] < vol20[k]):
                continue

            if mode == 'channel':
                # 最近兩個已確認擺動低點連成下軌，並用區間高點平移出通道寬度
                # 關鍵：piv[i] 成立需要看到 i+5 根之後的 K 線，
                # 所以在第 k 天只能用 i <= k-5 的擺動點。
                # （第一版寫 range(k-1,...) 等於提前 4 天知道未來，
                #   把超額報酬灌成 +0.23%、t=+8.74 的假象。）
                pv = [i for i in range(k - W_CONFIRM, max(59, k - 200), -1) if piv[i]]
                if len(pv) < 2:
                    continue
                i2, i1 = pv[0], pv[1]
                if i1 == i2:
                    continue
                slope = (low[i2] - low[i1]) / (i2 - i1)
                rail = low[i2] + slope * (k - i2)
                if rail <= 0:
                    continue
                if abs(low[k] - rail) / rail > TOL:
                    continue
            elif mode == 'ma20':
                if abs(low[k] - ma20[k]) / ma20[k] > TOL:
                    continue
            elif mode != 'random':
                raise ValueError(mode)

            entry = close[k]
            stop = entry * (1 - STOP)
            ej = reason = px = None
            for q in range(k + 1, min(k + MAXHOLD + 1, n)):
                if low[q] <= stop:
                    ej, reason, px = q, 'stop', stop
                    break
                if close[q] < ma20[q]:
                    ej, reason, px = q, 'ma20', close[q]
                    break
            if ej is None:
                ej = min(k + MAXHOLD, n - 1)
                reason, px = 'time', close[ej]

            trades.append(dict(symbol=s, entry_date=df['date'].iloc[k],
                               exit_date=df['date'].iloc[ej],
                               ret=float(px / entry - 1), reason=reason,
                               hold=int(ej - k)))
            last = k
    t = pd.DataFrame(trades)
    if mode == 'random' and len(t):
        t = t.sample(n=min(len(t), rng.integers(2000, 20000)), random_state=SEED)
    return t


def market_cum(stocks):
    closes = []
    for s, df in stocks.items():
        closes.append(df.set_index('date')['close'])
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


def summarize(df, mcum, label):
    m0 = mcum.reindex(df.entry_date).ffill().values
    m1 = mcum.reindex(df.exit_date).ffill().values
    ex = df.ret.values - (m1 / m0 - 1)
    q = np.quantile(ex, 0.95)
    tr = ex[ex <= q]
    vc = df.reason.value_counts(normalize=True)
    return dict(label=label, n=int(len(ex)),
                win=float(df.ret.gt(0).mean()),
                avg=float(df.ret.mean()),
                excess=float(ex.mean()),
                t=float(nwt(ex)),
                trimmed=float(tr.mean()),
                stop_rate=float(vc.get('stop', 0.0)),
                ma20_rate=float(vc.get('ma20', 0.0)),
                hold=float(df.hold.mean()))


def main():
    stocks = load()
    print(f'母體 {len(stocks)} 檔', flush=True)
    mcum = market_cum(stocks)
    rng = np.random.default_rng(SEED)

    res = {}
    for mode, label in (('channel', 'A 通道下軌'), ('ma20', 'B 均線支撐（安慰劑）'),
                        ('random', 'C 隨機同日')):
        t = run(stocks, mode, rng)
        if not len(t):
            continue
        d = dedup(t)
        raw = summarize(t, mcum, label + '／原始')
        dd = summarize(d, mcum, label + '／去重')
        res[mode] = dict(raw=raw, dedup=dd, overlap=float(1 - len(d) / len(t)))
        print(f"{label:18s} 原始 n={raw['n']:>7,} 勝率 {raw['win']*100:5.1f}% "
              f"超額 {raw['excess']*100:+.4f}% t={raw['t']:+.2f}", flush=True)
        print(f"{'':18s} 去重 n={dd['n']:>7,} 勝率 {dd['win']*100:5.1f}% "
              f"超額 {dd['excess']*100:+.4f}% t={dd['t']:+.2f} "
              f"停損 {dd['stop_rate']*100:.1f}% 剔除5% {dd['trimmed']*100:+.3f}%", flush=True)

    json.dump(res, open(f'{OUT}/ablation.json', 'w'), ensure_ascii=False, indent=1)
    print('已寫 ablation.json')


if __name__ == '__main__':
    main()
