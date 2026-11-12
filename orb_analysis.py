#!/usr/bin/env python3
"""ORB: full test battery for the Bilibili video BV1b74i6DE2V.

Outputs (out/):
  orb_grid.csv          every symbol x variant, mean R / t / win / PF
  orb_filters.csv       video's context filters on the canonical variant
  orb_trades_*.csv      trade-level data for charts
  orb_results.json      headline numbers used by orb_cards.py / orb_charts.py
"""
import json
import os

import numpy as np
import pandas as pd

import orb_test as o

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
SYMS = ['QQQ', 'SPY', 'IWM', 'NVDA', 'AAPL']
COST = 1.0  # bps per side; ~1 cent on a $600 ETF is 1.7bp, half-spread is less


def tstat(x):
    x = pd.Series(x).dropna().astype(float)
    if len(x) < 3 or x.std(ddof=1) == 0:
        return 0.0
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))


def pooled_daily(trades):
    """Equal-weight one decision per day: average R across symbols traded."""
    if trades is None or len(trades) == 0:
        return pd.Series(dtype=float)
    return trades.groupby('date').r.mean().sort_index()


def summarize(trades, label, cost_note=''):
    r = trades.r
    s = o.stats(trades, label)
    s['cost_note'] = cost_note
    if len(trades):
        port = pooled_daily(trades)
        s['pool_t'] = tstat(port)
        s['pool_mean_r'] = float(port.mean())
        s['n_days'] = int(len(port))
    return s


def grid():
    rows = []
    for sym in SYMS:
        for orm in [5, 15, 30, 60]:
            for sm in ['full', 'atr']:
                for tR in [1.0, 2.0, 3.0, 10.0]:
                    tr = o.backtest(sym, or_min=orm, trig='close', stop_mode=sm,
                                    target_R=tR, cost_bps=COST)
                    s = o.stats(tr, f'{sym}|or{orm}|{sm}|t{tR:g}')
                    s.update(sym=sym, or_min=orm, stop=sm, target=tR)
                    rows.append(s)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'orb_grid.csv'), index=False)
    return df


def canonical_trades():
    tr = {}
    for sym in SYMS:
        tr[sym] = o.backtest(sym, or_min=15, trig='close', stop_mode='full',
                             target_R=2.0, cost_bps=COST)
        tr[sym].to_csv(os.path.join(OUT, f'orb_trades_{sym}.csv'), index=False)
    return tr


def filter_table(tr):
    alltr = pd.concat(tr.values(), ignore_index=True)
    conds = {
        '全部': alltr.index == alltr.index,
        '前一日內包日': alltr.prev_inside,
        '前一日 NR4': alltr.prev_nr4,
        '前一日 NR7': alltr.prev_nr7,
        '前一日收縮（內包或NR4/NR7）': alltr.prev_inside | alltr.prev_nr4 | alltr.prev_nr7,
        '區間寬度最窄 1/3': alltr.width_atr <= alltr.width_atr.quantile(1/3),
        '區間寬度中間 1/3': (alltr.width_atr > alltr.width_atr.quantile(1/3)) &
                            (alltr.width_atr <= alltr.width_atr.quantile(2/3)),
        '區間寬度最寬 1/3': alltr.width_atr > alltr.width_atr.quantile(2/3),
        '突破量 > 開盤區間均量 1.5 倍': alltr.vol_ratio > 1.5,
        '突破量 < 開盤區間均量 0.8 倍': alltr.vol_ratio < 0.8,
        '收盤價在 20 日均線上方': alltr.above_ma20,
        '收盤價在 20 日均線下方': ~alltr.above_ma20,
        '做多': alltr.side == 1,
        '做空': alltr.side == -1,
    }
    rows = []
    for name, mask in conds.items():
        s = alltr[mask]
        if len(s) == 0:
            continue
        per = s.groupby('date').r.mean()
        rows.append(dict(condition=name, n_trades=int(len(s)), n_days=int(len(per)),
                         mean_r=float(s.r.mean()), daily_mean_r=float(per.mean()),
                         t=tstat(per), win=float((s.r > 0).mean()),
                         pf=float(s.r[s.r > 0].sum() / -s.r[s.r < 0].sum())
                         if (s.r < 0).any() else float('inf')))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'orb_filters.csv'), index=False)
    return df


def expansion_table():
    rows = []
    for sym in SYMS:
        df = o.load(sym)
        d = o.daily_frame(df)
        d['rng_atr'] = d['range'] / d['atr14']
        d['next_rng'] = d['rng_atr'].shift(-1)
        for col in ['inside', 'nr4', 'nr7']:
            a = d[d[col] == True].next_rng.dropna()
            b = d[d[col] == False].next_rng.dropna()
            rows.append(dict(sym=sym, pattern=col, n=int(len(a)),
                             next_mean=float(a.mean()), next_median=float(a.median()),
                             other_mean=float(b.mean()), other_median=float(b.median())))
    return pd.DataFrame(rows)


def year_table(tr):
    alltr = pd.concat(tr.values(), ignore_index=True)
    alltr['year'] = alltr.date.str[:4]
    rows = []
    for y, g in alltr.groupby('year'):
        per = g.groupby('date').r.mean()
        rows.append(dict(year=y, n_trades=int(len(g)), n_days=int(len(per)),
                         mean_r=float(g.r.mean()), daily_mean_r=float(per.mean()),
                         t=tstat(per), win=float((g.r > 0).mean())))
    return pd.DataFrame(rows)


def main():
    g = grid()
    tr = canonical_trades()
    ft = filter_table(tr)
    et = expansion_table()
    yt = year_table(tr)

    alltr = pd.concat(tr.values(), ignore_index=True)
    port = alltr.groupby('date').r.mean()
    head = dict(
        period=f'{alltr.date.min()} .. {alltr.date.max()}',
        symbols=SYMS, cost_bps_per_side=COST,
        n_trades=int(len(alltr)), n_days=int(port.size),
        mean_r=float(alltr.r.mean()), daily_mean_r=float(port.mean()),
        t=float(tstat(port)), win=float((alltr.r > 0).mean()),
        pf=float(alltr.r[alltr.r > 0].sum() / -alltr.r[alltr.r < 0].sum()),
        sum_r=float(alltr.r.sum()),
        per_symbol={s: dict(n=int(len(t)), mean_r=float(t.r.mean()),
                            win=float((t.r > 0).mean()),
                            sum_r=float(t.r.sum())) for s, t in tr.items()},
        grid_best=dict(g.sort_values('mean_r', ascending=False).iloc[0][
            ['label', 'mean_r', 'n', 't', 'win']]),
        grid_median_mean_r=float(g.mean_r.median()),
        grid_positive=int((g.mean_r > 0).sum()),
        grid_total=int(len(g)),
    )
    json.dump(head, open(os.path.join(OUT, 'orb_results.json'), 'w'),
              indent=1, default=str)

    print('=== HEAD ===')
    print(json.dumps(head, indent=1, default=str))
    print('\n=== GRID (sorted by mean R) ===')
    print(g.sort_values('mean_r', ascending=False)[
        ['label', 'n', 'mean_r', 'win', 'pf', 't']].round(3).head(15).to_string(index=False))
    print('\n=== FILTERS ===')
    print(ft.round(3).to_string(index=False))
    print('\n=== EXPANSION ===')
    print(et.round(3).to_string(index=False))
    print('\n=== YEARS ===')
    print(yt.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
