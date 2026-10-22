#!/usr/bin/env python3
"""Extra series the cards need for their compact P&L charts.

1. Weekday-only equity: hold the index only on Mondays (or only Thursdays),
   cash otherwise. This is the visual form of his 周内效应 claim, and it shows
   the 1971-1999 vs 2000-2026 split without a table.
2. Avoid-the-window equity: what happens if you actually act on his escape
   windows (flat from Dec 10 for 5 sessions, else long). Tests whether the
   warning is tradeable, not just statistically visible.
"""
import json, os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
OUT = os.path.join(HERE, 'out')


def load_fred(sid):
    df = pd.read_csv(os.path.join(DATA, f'{sid}.csv'))
    df.columns = ['date', 'close']
    df['date'] = pd.to_datetime(df['date'])
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    return df.dropna().set_index('date').sort_index()


def weekday_only(df, wd, cost_bp=1.0):
    r = df['close'].pct_change().fillna(0)
    pos = pd.Series((r.index.dayofweek == wd).astype(float), index=r.index)
    turn = pos.diff().abs().fillna(0)
    return (1 + pos * r - turn * cost_bp / 1e4).cumprod()


def avoid_windows(df, mmdd='12-10', hold=5, cost_bp=1.0):
    """Flat for `hold` sessions starting at the anchor date each year."""
    r = df['close'].pct_change().fillna(0)
    pos = pd.Series(1.0, index=r.index)
    idx = r.index
    for yr in range(idx[0].year, idx[-1].year + 1):
        anchor = pd.Timestamp(f'{yr}-{mmdd}')
        after = idx[idx >= anchor]
        if len(after) == 0:
            continue
        i0 = idx.get_loc(after[0])
        pos.iloc[i0:i0 + hold] = 0.0
    turn = pos.diff().abs().fillna(0)
    eq = (1 + pos * r - turn * cost_bp / 1e4).cumprod()
    return eq, pos


def stats(eq, per_year=252):
    eq = pd.Series(eq).dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    return dict(cagr=float(eq.iloc[-1] ** (1 / yrs) - 1),
                multiple=float(eq.iloc[-1]),
                mdd=float((eq.values / np.maximum.accumulate(eq.values) - 1).min()))


def pack(eq):
    return dict(idx=[d.strftime('%Y-%m-%d') for d in eq.index],
                v=[float(x) for x in eq.values])


if __name__ == '__main__':
    nas = load_fred('NASDAQCOM')
    out = {}

    # --- weekday-only, split by era ---
    for era, sub in [('early', nas[nas.index < '2000-01-01']),
                     ('late', nas[nas.index >= '2000-01-01']),
                     ('all', nas)]:
        mon = weekday_only(sub, 0)
        thu = weekday_only(sub, 3)
        bh = sub['close'] / sub['close'].iloc[0]
        out[f'wdonly_{era}'] = dict(
            mon=pack(mon / mon.iloc[0]), thu=pack(thu / thu.iloc[0]),
            bh=pack(bh),
            mon_stats=stats(mon / mon.iloc[0]), thu_stats=stats(thu / thu.iloc[0]))
        print(f"{era:5s} {sub.index[0].date()}..{sub.index[-1].date()}  "
              f"只做週一 {out[f'wdonly_{era}']['mon_stats']['multiple']:.2f}x  "
              f"只做週四 {out[f'wdonly_{era}']['thu_stats']['multiple']:.2f}x  "
              f"買進持有 {bh.iloc[-1]:.2f}x")

    # --- avoid the Dec 10 window ---
    eq, pos = avoid_windows(nas)
    bh = nas['close'] / nas['close'].iloc[0]
    out['avoid_dec'] = dict(eq=pack(eq), bh=pack(bh),
                            stats=stats(eq), bh_stats=stats(bh), exposure=float(pos.mean()))
    print(f"\n避開 12/10 後 5 日：年化 {stats(eq)['cagr']*100:.2f}%  "
          f"買進持有 {stats(bh)['cagr']*100:.2f}%  持倉 {pos.mean()*100:.1f}%")

    json.dump(out, open(os.path.join(OUT, 'extra.json'), 'w'), indent=1)
    print('written out/extra.json')
