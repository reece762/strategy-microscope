#!/usr/bin/env python3
"""ORB (Opening Range Breakout) backtest on 1-minute US ETF bars.

Rules taken from the Bilibili video BV1b74i6DE2V (transcript in subs/):
  - opening range = first 5/15/30/60 minutes of the regular session (09:30 ET)
  - entry on the first bar that CLOSES outside the range (next bar open),
    plus a wick/stop-order variant for comparison
  - stop = opposite side of range ("full"), half range, or k*ATR
  - target = k * risk (R multiples), otherwise exit at the session close
  - context filters the video claims matter: prior-day volatility contraction
    (inside day, NR4, NR7), opening-range width, breakout-bar volume,
    trend alignment with the prior close / 20-day average.

No look-ahead: signals use completed bars only; entries fill at the next bar's
open. If a bar contains both stop and target, the stop is assumed first
(conservative). Costs are charged per side in basis points.
"""
import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
DATA = os.path.join(OUT, 'orb1m')
os.makedirs(OUT, exist_ok=True)


def load(sym):
    path = os.path.join(DATA, f'{sym}.csv')
    df = pd.read_csv(path)
    df['dt'] = pd.to_datetime(df['time_key'])
    df = df[(df.dt.dt.time >= pd.Timestamp('09:31').time()) &
            (df.dt.dt.time <= pd.Timestamp('16:00').time())]
    df = df.sort_values('dt').drop_duplicates('dt')
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    df['date'] = df.dt.dt.date
    return df.reset_index(drop=True)


def daily_frame(df):
    g = df.groupby('date')
    d = pd.DataFrame({
        'open': g['open'].first(),
        'high': g['high'].max(),
        'low': g['low'].min(),
        'close': g['close'].last(),
        'n': g.size(),
    })
    d['range'] = d.high - d.low
    prev_close = d.close.shift(1)
    tr = pd.concat([d.high - d.low, (d.high - prev_close).abs(),
                    (d.low - prev_close).abs()], axis=1).max(axis=1)
    d['atr14'] = tr.rolling(14).mean()
    d['prev_close'] = prev_close
    d['prev_high'] = d.high.shift(1)
    d['prev_low'] = d.low.shift(1)
    d['inside'] = (d.high < d.prev_high) & (d.low > d.prev_low)
    d['inside2'] = d.inside & d.inside.shift(1).fillna(False).astype(bool)
    d['nr4'] = d['range'] <= d['range'].rolling(4).min()
    d['nr7'] = d['range'] <= d['range'].rolling(7).min()
    d['ma20'] = d.close.rolling(20).mean()
    # The video's filter is "the PREVIOUS session showed contraction, so TODAY
    # expands". Keep both same-day and prior-day flags; only the prior-day ones
    # test the claim.
    for c in ['inside', 'inside2', 'nr4', 'nr7']:
        d['prev_' + c] = d[c].shift(1)
    return d


def simulate_day(day_df, meta, or_min=15, trig='close', stop_mode='full',
                 target_R=2.0, cost_bps=2.0, atr_k=2.0, min_bars_after=10):
    """Return a dict for one session, or None if no completed trade."""
    n_or = or_min
    if len(day_df) < n_or + 3:
        return None
    orb = day_df.iloc[:n_or]
    hi, lo = orb.high.max(), orb.low.min()
    width = hi - lo
    if width <= 0:
        return None
    rest = day_df.iloc[n_or:].reset_index(drop=True)
    if len(rest) < 2:
        return None
    # numpy views: per-bar .iloc on a DataFrame is ~50x slower and this loop
    # runs over every session for every variant.
    r_open = rest.open.to_numpy()
    r_high = rest.high.to_numpy()
    r_low = rest.low.to_numpy()
    r_close = rest.close.to_numpy()
    r_vol = rest.volume.to_numpy()
    r_dt = rest.dt.to_numpy()
    n_rest = len(rest)

    # find first breakout
    side = 0
    sig_i = None
    if trig == 'close':
        for i in range(n_rest):
            c = r_close[i]
            if c > hi:
                side, sig_i = 1, i
                break
            if c < lo:
                side, sig_i = -1, i
                break
        if side == 0:
            return None
        if sig_i + 1 >= n_rest:
            return None
        entry = r_open[sig_i + 1]
        sig_time = r_dt[sig_i]
    elif trig == 'close2':
        # first close outside, then require the NEXT bar to close beyond the
        # first breakout bar's extreme; enter on the bar after that.
        for i in range(n_rest):
            c = r_close[i]
            if c > hi:
                side, sig_i = 1, i
                break
            if c < lo:
                side, sig_i = -1, i
                break
        if side == 0 or sig_i + 2 >= n_rest:
            return None
        first_ext = r_high[sig_i] if side == 1 else r_low[sig_i]
        ok = r_close[sig_i + 1] > first_ext if side == 1 else r_close[sig_i + 1] < first_ext
        if not ok:
            return None
        sig_i = sig_i + 1
        entry = r_open[sig_i + 1]
        sig_time = r_dt[sig_i]
    elif trig == 'wick':
        for i in range(n_rest):
            h, l = r_high[i], r_low[i]
            if h > hi:
                side, sig_i, entry = 1, i, max(r_open[i], hi)
                break
            if l < lo:
                side, sig_i, entry = -1, i, min(r_open[i], lo)
                break
        if side == 0:
            return None
        sig_time = r_dt[sig_i]
        if sig_i + 1 >= n_rest:
            return None
    elif trig == 'retest':
        # first close breakout, then limit entry at the broken boundary
        for i in range(n_rest):
            c = r_close[i]
            if c > hi:
                side, sig_i = 1, i
                break
            if c < lo:
                side, sig_i = -1, i
                break
        if side == 0:
            return None
        boundary = hi if side == 1 else lo
        filled = None
        for j in range(sig_i + 1, n_rest):
            if side == 1 and r_low[j] <= boundary:
                filled = j
                break
            if side == -1 and r_high[j] >= boundary:
                filled = j
                break
        if filled is None:
            return None
        sig_time = r_dt[filled]
        entry = boundary
        sig_i = filled
    else:
        raise ValueError(trig)

    if stop_mode == 'full':
        stop = lo if side == 1 else hi
    elif stop_mode == 'half':
        stop = (hi + lo) / 2.0
    elif stop_mode == 'atr':
        stop = entry - side * atr_k * meta['atr14']
    elif stop_mode == 'mix':
        struct = lo if side == 1 else hi
        stop = entry - side * atr_k * meta['atr14']
        stop = min(stop, struct) if side == 1 else max(stop, struct)
    else:
        raise ValueError(stop_mode)

    risk = side * (entry - stop)
    if risk <= 0:
        return None
    target = entry + side * target_R * risk

    # walk forward from the bar after entry
    start = sig_i + 1
    exit_px, exit_reason, exit_time = None, None, None
    for j in range(start, n_rest):
        h, l = r_high[j], r_low[j]
        if side == 1:
            if l <= stop:
                exit_px, exit_reason = stop, 'stop'
            elif h >= target:
                exit_px, exit_reason = target, 'target'
        else:
            if h >= stop:
                exit_px, exit_reason = stop, 'stop'
            elif l <= target:
                exit_px, exit_reason = target, 'target'
        if exit_px is not None:
            exit_time = r_dt[j]
            break
    if exit_px is None:
        exit_px, exit_reason, exit_time = r_close[-1], 'close', r_dt[-1]

    gross = side * (exit_px - entry) / entry
    net = gross - 2 * cost_bps / 1e4
    r = net * entry / risk  # risk in price terms == fraction risk

    # breakout-bar volume vs opening-range average volume
    vol_sig = float(r_vol[sig_i])
    vol_or = orb.volume.mean()
    return dict(
        date=str(meta.name), side=int(side), entry=float(entry), stop=float(stop),
        target=float(target), exit=float(exit_px), reason=exit_reason,
        entry_time=str(sig_time), exit_time=str(exit_time),
        width=float(width), width_atr=float(width / meta['atr14']),
        or_pct=float(width / entry * 100), risk_pct=float(risk / entry * 100),
        r=float(r), net_ret_pct=float(net * 100),
        vol_ratio=float(vol_sig / vol_or) if vol_or > 0 else np.nan,
        inside=bool(meta['inside']) if not pd.isna(meta['inside']) else False,
        inside2=bool(meta['inside2']) if not pd.isna(meta['inside2']) else False,
        nr4=bool(meta['nr4']) if not pd.isna(meta['nr4']) else False,
        nr7=bool(meta['nr7']) if not pd.isna(meta['nr7']) else False,
        prev_inside=bool(meta['prev_inside']) if not pd.isna(meta['prev_inside']) else False,
        prev_inside2=bool(meta['prev_inside2']) if not pd.isna(meta['prev_inside2']) else False,
        prev_nr4=bool(meta['prev_nr4']) if not pd.isna(meta['prev_nr4']) else False,
        prev_nr7=bool(meta['prev_nr7']) if not pd.isna(meta['prev_nr7']) else False,
        above_ma20=bool(meta['close'] > meta['ma20']) if not pd.isna(meta['ma20']) else False,
        gap=float(meta['open'] / meta['prev_close'] - 1) if meta['prev_close'] else np.nan,
    )


_CACHE = {}


def _prepared(sym):
    if sym not in _CACHE:
        df = load(sym)
        _CACHE[sym] = (df, daily_frame(df))
    return _CACHE[sym]


def backtest(sym, **kw):
    df, d = _prepared(sym)
    rows = []
    for date, day_df in df.groupby('date'):
        if date not in d.index:
            continue
        meta = d.loc[date]
        if pd.isna(meta['atr14']) or meta['n'] < 30:
            continue
        r = simulate_day(day_df, meta, **kw)
        if r:
            rows.append(r)
    return pd.DataFrame(rows)


def simulate_fakeout(day_df, meta, or_min=15, target_mode='opposite',
                     cost_bps=1.0, min_r=1.0):
    """SMC 'liquidity sweep then reclaim' model from the video.

    A bar pokes beyond the opening range but closes back inside -> trade the
    reversion. Stop goes just beyond the fakeout wick; the target is the
    opposite range boundary (skipped if that is less than min_r away).
    """
    n_or = or_min
    if len(day_df) < n_or + 3:
        return None
    orb = day_df.iloc[:n_or]
    hi, lo = orb.high.max(), orb.low.min()
    width = hi - lo
    if width <= 0:
        return None
    rest = day_df.iloc[n_or:].reset_index(drop=True)
    r_open = rest.open.to_numpy(); r_high = rest.high.to_numpy()
    r_low = rest.low.to_numpy(); r_close = rest.close.to_numpy()
    r_dt = rest.dt.to_numpy(); n_rest = len(rest)
    tick = max(0.01, float(rest.close.iloc[0]) * 1e-5)

    for i in range(n_rest - 1):
        side = 0
        if r_high[i] > hi and r_close[i] < hi:
            side, stop = -1, r_high[i] + tick
        elif r_low[i] < lo and r_close[i] > lo:
            side, stop = 1, r_low[i] - tick
        if side == 0:
            continue
        entry = r_open[i + 1]
        risk = side * (entry - stop)
        if risk <= 0:
            return None
        target_opp = lo if side == -1 else hi
        if target_mode == 'opposite':
            target = target_opp
            if side * (target - entry) / risk < min_r:
                return None
        else:
            target = entry + side * float(target_mode) * risk
        exit_px = exit_reason = exit_time = None
        for j in range(i + 1, n_rest):
            if side == -1:
                if r_high[j] >= stop:
                    exit_px, exit_reason = stop, 'stop'
                elif r_low[j] <= target:
                    exit_px, exit_reason = target, 'target'
            else:
                if r_low[j] <= stop:
                    exit_px, exit_reason = stop, 'stop'
                elif r_high[j] >= target:
                    exit_px, exit_reason = target, 'target'
            if exit_px is not None:
                exit_time = r_dt[j]
                break
        if exit_px is None:
            exit_px, exit_reason, exit_time = r_close[-1], 'close', r_dt[-1]
        net = side * (exit_px - entry) / entry - 2 * cost_bps / 1e4
        return dict(date=str(meta.name), side=int(side), entry=float(entry),
                    stop=float(stop), target=float(target), exit=float(exit_px),
                    reason=exit_reason, entry_time=str(r_dt[i + 1]),
                    exit_time=str(exit_time), width=float(width),
                    width_atr=float(width / meta['atr14']),
                    r=float(net * entry / risk),
                    net_ret_pct=float(net * 100),
                    prev_inside=bool(meta['prev_inside']) if not pd.isna(meta['prev_inside']) else False,
                    prev_nr4=bool(meta['prev_nr4']) if not pd.isna(meta['prev_nr4']) else False,
                    prev_nr7=bool(meta['prev_nr7']) if not pd.isna(meta['prev_nr7']) else False)
    return None


def backtest_fakeout(sym, **kw):
    df, d = _prepared(sym)
    rows = []
    for date, day_df in df.groupby('date'):
        if date not in d.index:
            continue
        meta = d.loc[date]
        if pd.isna(meta['atr14']) or meta['n'] < 30:
            continue
        r = simulate_fakeout(day_df, meta, **kw)
        if r:
            rows.append(r)
    return pd.DataFrame(rows)


def stats(trades, label=''):
    if trades is None or len(trades) == 0:
        return dict(label=label, n=0)
    r = trades.r
    win = (r > 0).mean()
    gross_win = r[r > 0].sum()
    gross_loss = -r[r < 0].sum()
    pf = gross_win / gross_loss if gross_loss > 0 else float('inf')
    t = r.mean() / (r.std(ddof=1) / np.sqrt(len(r))) if r.std(ddof=1) > 0 else 0.0
    return dict(label=label, n=int(len(r)), mean_r=float(r.mean()), sum_r=float(r.sum()),
                win=float(win), pf=float(pf), t=float(t),
                max_dd_r=float(_max_dd(r.cumsum())),
                avg_ret=float(trades.net_ret_pct.mean()))


def _max_dd(curve):
    peak = np.maximum.accumulate(curve.values)
    dd = curve.values - peak
    return dd.min() if len(dd) else 0.0


def main():
    variants = {}
    for sym in ['QQQ', 'SPY', 'IWM']:
        for or_min in [5, 15, 30, 60]:
            for stop_mode in ['full', 'half', 'atr']:
                for tR in [1.0, 2.0, 3.0]:
                    key = f'{sym}|or{or_min}|{stop_mode}|t{tR:g}'
                    tr = backtest(sym, or_min=or_min, trig='close', stop_mode=stop_mode,
                                  target_R=tR, cost_bps=2.0)
                    variants[key] = stats(tr, key)
                    tr.to_csv(os.path.join(OUT, f'orb_tr_{sym}_or{or_min}_{stop_mode}_t{tR:g}.csv'),
                              index=False)
    json.dump(variants, open(os.path.join(OUT, 'orb_variants.json'), 'w'), indent=1)
    rows = sorted(variants.values(), key=lambda x: -x.get('mean_r', -9))
    print(f"{'variant':34s} {'n':>4s} {'meanR':>7s} {'win':>6s} {'PF':>6s} {'t':>6s} {'maxDD':>7s}")
    for x in rows[:25]:
        print(f"{x['label']:34s} {x['n']:4d} {x['mean_r']:7.3f} {x['win']:6.1%} "
              f"{x['pf']:6.2f} {x['t']:6.2f} {x['max_dd_r']:7.1f}")


if __name__ == '__main__':
    main()
