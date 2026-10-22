#!/usr/bin/env python3
"""BTC/USDT daily: weekday effect, skip-Thursday, MA regime.

Crypto trades all seven days, so the weekday table is 7-wide and the regime rule
(which only names Mon-Fri outcomes) carries the previous state through weekends
instead of inventing a rule the video never stated.
"""
import glob, io, json, os, zipfile
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
OUT = os.path.join(HERE, 'out')
DAYS = {0: '週一', 1: '週二', 2: '週三', 3: '週四', 4: '週五', 5: '週六', 6: '週日'}


def load_btc():
    frames = []
    for f in sorted(glob.glob(os.path.join(DATA, 'BTCUSDT-1d-*.zip'))):
        with zipfile.ZipFile(f) as z:
            name = z.namelist()[0]
            raw = z.read(name).decode()
        df = pd.read_csv(io.StringIO(raw), header=None)
        # binance klines: open_time, open, high, low, close, volume, close_time, ...
        df = df.iloc[:, :6]
        df.columns = ['ts', 'open', 'high', 'low', 'close', 'volume']
        frames.append(df)
    d = pd.concat(frames, ignore_index=True)
    ts = pd.to_numeric(d['ts'], errors='coerce')
    # Binance switched to microsecond open_time partway through the archive, so
    # the unit has to be detected per row, not per file: 16-digit values are
    # microseconds, 13-digit are milliseconds. Dividing by 1000 once for the
    # long ones puts everything on the ms scale.
    ts = ts.where(ts < 1e14, ts / 1000.0)
    d['date'] = pd.to_datetime(ts, unit='ms')
    for c in ['open', 'high', 'low', 'close', 'volume']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d.dropna(subset=['date', 'close']).drop_duplicates('date').set_index('date').sort_index()
    return d


def perf(eq):
    eq = pd.Series(eq).dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1)
    mdd = float((eq.values / np.maximum.accumulate(eq.values) - 1).min())
    r = eq.pct_change().dropna()
    sharpe = float(r.mean() / r.std() * np.sqrt(365)) if r.std() > 0 else 0.0
    return dict(cagr=cagr, mdd=mdd, sharpe=sharpe,
                calmar=float(cagr / abs(mdd)) if mdd else 0.0,
                multiple=float(eq.iloc[-1]), years=yrs)


def weekday_rows(d, label):
    r = d['close'].pct_change().dropna()
    wd = r.index.dayofweek
    out = []
    for k in range(7):
        x = r[wd == k]
        if len(x) < 30:
            continue
        t = float(x.mean() / (x.std() / np.sqrt(len(x))))
        out.append(dict(label=label, wd=k, day=DAYS[k],
                        mean_pct=float(x.mean() * 100),
                        median_pct=float(x.median() * 100),
                        hit_pct=float((x > 0).mean() * 100),
                        n=int(len(x)), t_stat=t))
    return out


def skip_day(d, skip_wd=3, cost_bp=5.0):
    r = d['close'].pct_change().fillna(0)
    pos = pd.Series((r.index.dayofweek != skip_wd).astype(float), index=r.index)
    turn = pos.diff().abs().fillna(0)
    return (1 + pos * r - turn * cost_bp / 1e4).cumprod()


def regime(d, ma_n=20, cost_bp=5.0):
    close = d['close']
    ma = close.rolling(ma_n).mean()
    r = close.pct_change().fillna(0)
    wd = r.index.dayofweek.to_numpy()
    c, m = close.to_numpy(), ma.to_numpy()
    pos = np.zeros(len(d))
    prev = 0.0
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
    ps = pd.Series(pos, index=r.index)
    turn = ps.diff().abs().fillna(ps.iloc[0])
    return (1 + ps * r - turn * cost_bp / 1e4).cumprod(), ps


if __name__ == '__main__':
    d = load_btc()
    print(f'BTC/USDT {d.index[0].date()} .. {d.index[-1].date()}  {len(d)} 日')

    res = {'weekday': weekday_rows(d, 'BTC 2017-2026')}
    for r in res['weekday']:
        print(f"  {r['day']} 平均 {r['mean_pct']:+.4f}%  中位 {r['median_pct']:+.4f}%  "
              f"勝率 {r['hit_pct']:.1f}%  n={r['n']:4d}  t={r['t_stat']:+.2f}")

    bh = d['close'] / d['close'].iloc[0]
    sk = skip_day(d)
    rg, pos = regime(d)
    res['btc'] = dict(start=str(d.index[0].date()), end=str(d.index[-1].date()), n=len(d),
                      bh=perf(bh), skip=perf(sk), regime=perf(rg),
                      regime_exposure=float(pos.mean()),
                      regime_trades=int((pos.diff().abs() > 0).sum()))
    print()
    for k, lab in [('bh', '買進持有'), ('skip', '跳過週四'), ('regime', '均線×星期')]:
        p = res['btc'][k]
        print(f"  {lab:8s} 年化 {p['cagr']*100:7.2f}%  回撤 {p['mdd']*100:7.2f}%  "
              f"Sharpe {p['sharpe']:5.2f}  Calmar {p['calmar']:5.2f}  淨值 {p['multiple']:8.2f}x")
    print(f"  持倉比例 {res['btc']['regime_exposure']*100:.1f}%  換手 {res['btc']['regime_trades']}")

    prev = json.load(open(os.path.join(OUT, 'results.json')))
    prev['btc'] = res['btc']
    prev['btc_weekday'] = res['weekday']
    prev['curves']['BTC（2017 起）'] = dict(
        idx=[x.strftime('%Y-%m-%d') for x in d.index],
        bh=[float(x) for x in bh.values],
        skip=[float(x) for x in sk.values],
        regime=[float(x) for x in rg.values])
    json.dump(prev, open(os.path.join(OUT, 'results.json'), 'w'), ensure_ascii=False, indent=1)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(d.index, bh.values, lw=1.1, color='#8b96a3', label='買進持有')
    ax.plot(d.index, sk.values, lw=1.3, color='#3ba7d8', label='跳過週四')
    ax.plot(d.index, rg.values, lw=1.5, color='#e05a4f', label='均線×星期')
    ax.set_yscale('log'); ax.set_ylabel('淨值（對數）')
    ax.set_title('BTC/USDT 日線（2017 起）')
    ax.legend(fontsize=9); ax.grid(alpha=.3, which='both')
    plt.tight_layout(); plt.savefig(os.path.join(OUT, 'curve_btc.png')); plt.close()
    print('\nDONE')
