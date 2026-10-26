#!/usr/bin/env python3
"""海龜／唐奇安：補上「持有期對齊」的對照組。

## 為什麼要補

第一版的隨機對照組固定持有 25 天，卻拿去跟
「20 進/10 出（實際平均持有 27 天）」與「55 進/20 出（實際平均持有 50 天）」比。
**持有期不同，報酬本來就不能直接比**——50 天的 +2.63% 跟 25 天的 +1.90%
不是「策略贏或輸」，只是「抱得久所以漲得多」。

正確對照：**同一批股票、同一段期間，任意一天買進、持有相同天數。**

同時補上「扣交易成本」的版本：每次來回 0.2%（保守估計，含滑價）。
43,949 筆交易在 611 檔股票上，平均每檔每年不到 4 筆，成本影響有限但要算。
"""
import glob
import json
import os

import numpy as np
import pandas as pd

STOCK = '/root/.openclaw/stock_data'
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
COST = 0.002   # 來回 0.2%


def trimmed(x, p=0.01):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 50:
        return float(np.median(x)) if len(x) else np.nan
    lo, hi = np.quantile(x, [p, 1 - p])
    return float(x[(x >= lo) & (x <= hi)].mean())


def load_stocks():
    out = {}
    for f in sorted(glob.glob(f'{STOCK}/live/*.csv') + glob.glob(f'{STOCK}/extra/*.csv')):
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        d.columns = [c.lower() for c in d.columns]
        if not {'date', 'open', 'high', 'low', 'close', 'volume'}.issubset(d.columns):
            continue
        if len(d) < 400:
            continue
        d['date'] = pd.to_datetime(d['date'])
        out[os.path.basename(f).replace('.csv', '')] = \
            d.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
    return out


def turtle_sim(stocks, n_enter, n_exit):
    trades = []
    for sym, d in stocks.items():
        close = d['close'].to_numpy(float)
        high = d['high'].to_numpy(float)
        low = d['low'].to_numpy(float)
        n = len(close)
        if n < 300:
            continue
        hi_e = pd.Series(high).rolling(n_enter).max().shift(1).to_numpy()
        lo_x = pd.Series(low).rolling(n_exit).min().shift(1).to_numpy()
        pos = 0.0
        entry_i = -1
        for i in range(max(n_enter, n_exit) + 1, n):
            if pos == 0:
                if not np.isnan(hi_e[i]) and close[i] > hi_e[i]:
                    pos = close[i]; entry_i = i
            else:
                if not np.isnan(lo_x[i]) and close[i] < lo_x[i]:
                    trades.append(dict(ret=float(close[i]) / pos - 1,
                                       bars=int(i - entry_i)))
                    pos = 0.0
        if pos > 0:
            trades.append(dict(ret=float(close[-1]) / pos - 1,
                               bars=int(n - 1 - entry_i)))
    return pd.DataFrame(trades)


def horizon_baseline(stocks, bars):
    """任意一天買進、持有 bars 天的報酬（全樣本，截尾）。"""
    vals = []
    for sym, d in stocks.items():
        c = d['close'].to_numpy(float)
        n = len(c)
        if n < bars + 80:
            continue
        f = np.full(n, np.nan)
        f[:-bars] = c[bars:] / c[:-bars] - 1
        v = f[max(55, bars):]           # 避開前段（均線暖身）
        vals.append(v[~np.isnan(v)])
    v = np.concatenate(vals)
    return dict(n=int(len(v)), trim=trimmed(v), mean=float(v.mean()),
                median=float(np.median(v)), pos=float((v > 0).mean()),
                big_loss=float((v <= -0.20).mean()))


def main():
    stocks = load_stocks()
    print(f'母體 {len(stocks)} 檔', flush=True)

    out = {}
    for n_enter, n_exit in ((20, 10), (55, 20)):
        t = turtle_sim(stocks, n_enter, n_exit)
        r = t['ret'].to_numpy()
        bars = int(round(t['bars'].mean()))
        base = horizon_baseline(stocks, bars)

        wins = r[r > 0]
        losses = r[r <= 0]
        rec = dict(n_enter=n_enter, n_exit=n_exit, n=int(len(r)),
                   mean=float(r.mean()), median=float(np.median(r)),
                   trim=trimmed(r), win=float((r > 0).mean()),
                   avg_win=float(wins.mean()), avg_loss=float(losses.mean()),
                   payoff=float(wins.mean() / abs(losses.mean())),
                   big_loss=float((r <= -0.20).mean()),
                   bars=int(bars), bars_median=float(t['bars'].median()),
                   # 扣成本
                   mean_net=float(r.mean() - COST),
                   trim_net=trimmed(r) - COST,
                   # 對齊持有期的基準
                   base=base,
                   edge_vs_base=float(trimmed(r) - base['trim']),
                   edge_net_vs_base=float(trimmed(r) - COST - base['trim']),
                   # 每筆平均賺賠的絕對值（衡量「賺得多還是輸得多」）
                   profit_factor=float(wins.sum() / abs(losses.sum())) if len(losses) else np.nan)
        out[f'{n_enter}_{n_exit}'] = rec

        print(f"\n── 唐奇安 {n_enter} 進 / {n_exit} 出 ──")
        print(f"  交易 {rec['n']:,} 筆，平均持有 {rec['bars']} 個交易日（中位 {rec['bars_median']:.0f}）")
        print(f"  策略：平均 {rec['mean']*100:+.2f}%  截尾 {rec['trim']*100:+.2f}%  "
              f"勝率 {rec['win']*100:.1f}%  賺賠比 {rec['payoff']:.2f}  "
              f"獲利因子 {rec['profit_factor']:.2f}")
        print(f"  對齊基準（任意日買進、持有 {rec['bars']} 天）："
              f"截尾 {base['trim']*100:+.2f}%  勝率 {base['pos']*100:.1f}%")
        print(f"  ★ 優勢：{rec['edge_vs_base']*100:+.2f}pp   扣成本後 {rec['edge_net_vs_base']*100:+.2f}pp")

    json.dump(out, open(f'{OUT}/turtle_fair.json', 'w'), ensure_ascii=False, indent=1)
    print('\n已寫 turtle_fair.json')


if __name__ == '__main__':
    main()
