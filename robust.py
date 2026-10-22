#!/usr/bin/env python3
"""Robustness for the ported strategies.

Two things the headline numbers cannot answer:

1. EXPOSURE MATCHING. "Skip Thursday" holds only 4/5 of the sessions (6/7 for
   crypto) and the MA rule holds ~41%. A strategy that is out of the market part
   of the time must be compared against the same average exposure, rebalanced
   daily, with no timing at all. Otherwise the comparison is just long-vs-flat.

2. SUB-PERIOD STABILITY for BTC. A single 9-year path that gains from deleting
   one weekday can be an artifact of a handful of crashes; splitting by year
   shows whether the edge is a persistent pattern or three memorable weeks.
"""
import glob, io, json, os, zipfile
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
DAILY = os.path.join(os.path.dirname(HERE), 'data_daily')
OUT = os.path.join(HERE, 'out')


def load_fred(sid):
    df = pd.read_csv(os.path.join(DATA, f'{sid}.csv'))
    df.columns = ['date', 'close']
    df['date'] = pd.to_datetime(df['date'])
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    return df.dropna().set_index('date').sort_index()


def load_etf(sym):
    df = pd.read_csv(os.path.join(DAILY, f'{sym}.csv'))
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
    return df.set_index('date').sort_index()


def load_btc():
    frames = []
    for f in sorted(glob.glob(os.path.join(DATA, 'BTCUSDT-1d-*.zip'))):
        with zipfile.ZipFile(f) as z:
            raw = z.read(z.namelist()[0]).decode()
        df = pd.read_csv(io.StringIO(raw), header=None).iloc[:, :6]
        df.columns = ['ts', 'open', 'high', 'low', 'close', 'volume']
        frames.append(df)
    d = pd.concat(frames, ignore_index=True)
    ts = pd.to_numeric(d['ts'], errors='coerce')
    ts = ts.where(ts < 1e14, ts / 1000.0)          # microsecond rows normalised
    d['date'] = pd.to_datetime(ts, unit='ms')
    for c in ['open', 'high', 'low', 'close', 'volume']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    return d.dropna(subset=['date', 'close']).drop_duplicates('date').set_index('date').sort_index()


def perf(eq, per_year=252):
    eq = pd.Series(eq).dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1)
    mdd = float((eq.values / np.maximum.accumulate(eq.values) - 1).min())
    r = eq.pct_change().dropna()
    sharpe = float(r.mean() / r.std() * np.sqrt(per_year)) if r.std() > 0 else 0.0
    return dict(cagr=cagr, mdd=mdd, sharpe=sharpe,
                calmar=float(cagr / abs(mdd)) if mdd else 0.0, multiple=float(eq.iloc[-1]))


def skip_pos(idx, skip_wd=3):
    return pd.Series((idx.dayofweek != skip_wd).astype(float), index=idx)


def regime_pos(d, ma_n=20):
    close = d['close']; ma = close.rolling(ma_n).mean()
    wd = close.index.dayofweek.to_numpy()
    c, m = close.to_numpy(), ma.to_numpy()
    pos = np.zeros(len(d)); prev = 0.0
    for i in range(len(d)):
        if i < ma_n + 1 or not np.isfinite(m[i - 1]):
            pos[i] = 0.0; prev = 0.0; continue
        bull = c[i - 1] > m[i - 1]
        k = wd[i]
        if bull:
            p = 1.0 if k in (0, 4) else (0.0 if k in (1, 3) else prev)
        else:
            p = 1.0 if k in (1, 2) else (0.0 if k in (0, 3) else prev)
        pos[i] = p; prev = p
    return pd.Series(pos, index=close.index)


def exposure_matched(d, pos, cost_bp):
    """Strategy return vs B&H at the same average exposure, both cost-charged."""
    r = d['close'].pct_change().fillna(0)
    turn_s = pos.diff().abs().fillna(pos.iloc[0])
    strat = (1 + pos * r - turn_s * cost_bp / 1e4).cumprod()

    L = float(pos.mean())
    # 100% index every day, scaled to L, daily rebalanced. No timing, so no
    # turnover beyond the rebalance drift, which we charge at the same rate.
    lev = pd.Series(L, index=r.index)
    turn_b = lev.diff().abs().fillna(0)
    em = (1 + lev * r - turn_b * cost_bp / 1e4).cumprod()
    return strat, em, L


def main():
    nas = load_fred('NASDAQCOM')
    spy = load_etf('SPY'); qqq = load_etf('QQQ'); btc = load_btc()

    markets = {'NASDAQ': (nas, 252), 'SPY': (spy, 252), 'QQQ': (qqq, 252), 'BTC': (btc, 365)}
    rows = []

    for name, (d, py) in markets.items():
        r = d['close'].pct_change().fillna(0)
        bh = (1 + r).cumprod()
        for sname, pos, cost in [
            ('跳過週四', skip_pos(d.index), 1.0 if name != 'BTC' else 5.0),
            ('均線×星期', regime_pos(d), 1.0 if name != 'BTC' else 5.0),
        ]:
            strat, em, L = exposure_matched(d, pos, cost)
            ps, pe, pb = perf(strat, py), perf(em, py), perf(bh, py)
            rows.append(dict(market=name, strategy=sname, exposure=L,
                             cagr=ps['cagr'], mdd=ps['mdd'], sharpe=ps['sharpe'],
                             calmar=ps['calmar'], multiple=ps['multiple'],
                             em_cagr=pe['cagr'], em_mdd=pe['mdd'], em_sharpe=pe['sharpe'],
                             em_calmar=pe['calmar'],
                             bh_cagr=pb['cagr'], bh_mdd=pb['mdd'],
                             bh_sharpe=pb['sharpe'], bh_calmar=pb['calmar'],
                             calmar_vs_em=ps['calmar'] - pe['calmar'],
                             sharpe_vs_em=ps['sharpe'] - pe['sharpe']))
            print(f"{name:7s} {sname:6s} 曝險{L*100:5.1f}% | 策略 {ps['cagr']*100:6.2f}% "
                  f"MDD{ps['mdd']*100:7.1f}% Sh{ps['sharpe']:4.2f} Cal{ps['calmar']:5.2f} || "
                  f"同曝險 {pe['cagr']*100:6.2f}% MDD{pe['mdd']*100:7.1f}% Sh{pe['sharpe']:4.2f} "
                  f"Cal{pe['calmar']:5.2f} || ΔCalmar {ps['calmar']-pe['calmar']:+.2f}",
                  flush=True)

    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'exposure_matched.csv'), index=False)

    # ---- BTC year-by-year: is the Thursday edge persistent? ----
    print('\n=== BTC 逐年：跳過週四 vs 買進持有 ===')
    r = btc['close'].pct_change().fillna(0)
    pos = skip_pos(btc.index)
    yr_rows = []
    for y, g in pos.groupby(pos.index.year):
        if len(g) < 200:
            continue
        rr = r.loc[g.index]
        s = (1 + g * rr).cumprod().iloc[-1] - 1
        b = (1 + rr).cumprod().iloc[-1] - 1
        yr_rows.append(dict(year=int(y), strat=float(s), bh=float(b), diff=float(s - b)))
        print(f'  {y}  策略 {s*100:+7.2f}%  買進持有 {b*100:+7.2f}%  差 {(s-b)*100:+6.2f}pp')
    win = sum(1 for x in yr_rows if x['diff'] > 0)
    print(f'  勝出年份 {win}/{len(yr_rows)}')
    pd.DataFrame(yr_rows).to_csv(os.path.join(OUT, 'btc_by_year.csv'), index=False)

    # worst single Thursdays: how concentrated is the damage?
    th = r[r.index.dayofweek == 3].sort_values()
    print(f"\n  週四共 {len(th)} 天，最差 5 天 {[round(x*100,1) for x in th.head(5)]}%")
    print(f"  剔除最差 5 個週四後，週四平均 {th.iloc[5:].mean()*100:+.4f}%")
    print(f'  週四平均 {th.mean()*100:+.4f}%  t={th.mean()/(th.std()/np.sqrt(len(th))):+.2f}')

    print('\nDONE robustness')


if __name__ == '__main__':
    main()
