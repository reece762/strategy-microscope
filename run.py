#!/usr/bin/env python3
"""Backtest 邢不行's calendar strategies on US markets.

Ported from:
  BV1si4y177RU  A股周内效应 (day-of-week effect + skip-Thursday + MA regime)
  BV1638968EzE  A股四大逃生窗口 (calendar windows)

Data:
  NASDAQCOM  FRED, 1971-02-05 .. 2026-10-21 (price index, no dividends)
  SPY/QQQ    data_daily, 2018-01-02 .. 2025-12-31 (price return, no dividends)

Everything is price-return on both sides of a comparison, so the omission of
dividends does not bias the strategy-vs-benchmark result.
"""
import json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
DAILY = os.path.join(os.path.dirname(HERE), 'data_daily')
OUT = os.path.join(HERE, 'out')
os.makedirs(OUT, exist_ok=True)

DAYNAMES = {0: '週一', 1: '週二', 2: '週三', 3: '週四', 4: '週五'}


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


def perf(eq):
    eq = pd.Series(eq).dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1)
    mdd = float((eq.values / np.maximum.accumulate(eq.values) - 1).min())
    r = eq.pct_change().dropna()
    sharpe = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0
    return dict(cagr=cagr, mdd=mdd, sharpe=sharpe,
                calmar=float(cagr / abs(mdd)) if mdd else 0.0,
                multiple=float(eq.iloc[-1]), years=yrs)


# ---------------- Test A: weekday effect ----------------
def weekday_table(df, label):
    r = df['close'].pct_change().dropna()
    wd = r.index.dayofweek
    rows = []
    for d in range(5):
        x = r[wd == d]
        if len(x) < 30:
            continue
        t = float(x.mean() / (x.std() / np.sqrt(len(x))))
        rows.append(dict(label=label, wd=int(d), day=DAYNAMES[d],
                         mean_pct=float(x.mean() * 100),
                         median_pct=float(x.median() * 100),
                         hit_pct=float((x > 0).mean() * 100),
                         n=int(len(x)), t_stat=t))
    return rows


# ---------------- Test B: skip Thursday ----------------
def skip_thursday(df, cost_bp=1.0):
    """Hold every day except Thursday. Deterministic, no look-ahead."""
    r = df['close'].pct_change().fillna(0)
    pos = pd.Series((r.index.dayofweek != 3).astype(float), index=r.index)
    turnover = pos.diff().abs().fillna(0)
    ret = pos * r - turnover * cost_bp / 1e4
    return (1 + ret).cumprod()


# ---------------- Test C: MA20 regime x weekday ----------------
def regime_strategy(df, ma_n=20, cost_bp=1.0):
    """His 均线符合策略.

    Bull (yesterday's close > MA20): hold into Mon/Fri, flat into Tue/Thu,
    carry state on Wed. Bear: hold into Tue/Wed, flat into Mon/Thu, carry on Fri.
    Position for bar i uses only information available at close of bar i-1.
    """
    close = df['close']
    ma = close.rolling(ma_n).mean()
    r = close.pct_change().fillna(0)
    wd = r.index.dayofweek.to_numpy()
    c = close.to_numpy()
    m = ma.to_numpy()

    pos = np.zeros(len(df))
    prev = 0.0
    for i in range(len(df)):
        if i < ma_n + 1 or not np.isfinite(m[i - 1]):
            pos[i] = 0.0
            prev = 0.0
            continue
        bull = c[i - 1] > m[i - 1]
        d = wd[i]
        if bull:
            if d in (0, 4):
                p = 1.0
            elif d in (1, 3):
                p = 0.0
            else:
                p = prev
        else:
            if d in (1, 2):
                p = 1.0
            elif d in (0, 3):
                p = 0.0
            else:
                p = prev
        pos[i] = p
        prev = p

    ps = pd.Series(pos, index=r.index)
    turnover = ps.diff().abs().fillna(ps.iloc[0])
    ret = ps * r - turnover * cost_bp / 1e4
    return (1 + ret).cumprod(), ps


# ---------------- Test D: calendar escape windows ----------------
def calendar_windows(df, anchors=(('04-10', '4月10日'), ('08-31', '8月31日'),
                                  ('10-30', '10月30日'), ('12-10', '12月10日')),
                     horizons=(1, 5, 10, 20)):
    close = df['close']
    rows = []
    for mmdd, name in anchors:
        for h in horizons:
            rets = []
            for yr in range(close.index[0].year + 1, close.index[-1].year + 1):
                anchor = pd.Timestamp(f'{yr}-{mmdd}')
                fut = close.index[close.index >= anchor]
                if len(fut) == 0:
                    continue
                start = fut[0]
                i0 = close.index.get_loc(start)
                if i0 + h >= len(close):
                    continue
                rets.append(float(close.iloc[i0 + h] / close.iloc[i0] - 1))
            if len(rets) < 5:
                continue
            a = np.array(rets)
            t = float(a.mean() / (a.std(ddof=1) / np.sqrt(len(a))))
            rows.append(dict(window=name, horizon=h, mean_pct=float(a.mean() * 100),
                             hit_pct=float((a > 0).mean() * 100),
                             n=int(len(a)), t_stat=t))
    return rows


def main():
    nas = load_fred('NASDAQCOM')
    spy = load_etf('SPY')
    qqq = load_etf('QQQ')

    markets = {'納斯達克綜合指數（1971 起）': nas, 'SPY（2018 起）': spy, 'QQQ（2018 起）': qqq}

    wk = []
    wk += weekday_table(nas, '納斯達克 1971-2026')
    wk += weekday_table(nas[nas.index < '2000-01-01'], '納斯達克 1971-1999')
    wk += weekday_table(nas[nas.index >= '2000-01-01'], '納斯達克 2000-2026')
    wk += weekday_table(spy, 'SPY 2018-2025')
    wk += weekday_table(qqq, 'QQQ 2018-2025')

    res = {'weekday': wk, 'markets': {}}

    for name, df in markets.items():
        bh = df['close'] / df['close'].iloc[0]
        sk = skip_thursday(df)
        rg, pos = regime_strategy(df)
        res['markets'][name] = dict(
            start=str(df.index[0].date()), end=str(df.index[-1].date()),
            n=len(df),
            bh=perf(bh), skip=perf(sk), regime=perf(rg),
            regime_exposure=float(pos.mean()),
            regime_trades=int((pos.diff().abs() > 0).sum()),
        )
        res.setdefault('curves', {})[name] = dict(
            idx=[d.strftime('%Y-%m-%d') for d in df.index],
            bh=[float(x) for x in bh.values],
            skip=[float(x) for x in sk.values],
            regime=[float(x) for x in rg.values],
        )

    res['calendar_nasdaq'] = calendar_windows(nas)
    res['calendar_spy'] = calendar_windows(spy)

    with open(os.path.join(OUT, 'results.json'), 'w') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)

    # ---- print ----
    print('=== A. 星期效應（平均日報酬 %） ===')
    cur = None
    for r in wk:
        if r['label'] != cur:
            cur = r['label']
            print(f"\n  {cur}")
        print(f"    {r['day']}  平均 {r['mean_pct']:+.4f}%  中位 {r['median_pct']:+.4f}%  "
              f"勝率 {r['hit_pct']:.1f}%  n={r['n']:5d}  t={r['t_stat']:+.2f}")

    print('\n\n=== B/C. 策略績效 ===')
    for name, m in res['markets'].items():
        print(f"\n  {name}  ({m['start']} .. {m['end']}, {m['n']} 日)")
        for k, lab in [('bh', '買進持有'), ('skip', '跳過週四'), ('regime', '均線×星期')]:
            p = m[k]
            print(f"    {lab:8s} 年化 {p['cagr']*100:6.2f}%  回撤 {p['mdd']*100:7.2f}%  "
                  f"Sharpe {p['sharpe']:5.2f}  Calmar {p['calmar']:5.2f}  淨值 {p['multiple']:7.2f}x")
        print(f"    均線×星期的持倉比例 {m['regime_exposure']*100:.1f}%，換手 {m['regime_trades']} 次")

    print('\n\n=== D. 日曆窗口（納斯達克 1971-2026） ===')
    for r in res['calendar_nasdaq']:
        print(f"  {r['window']:>8s} +{r['horizon']:2d}日  平均 {r['mean_pct']:+.2f}%  "
              f"勝率 {r['hit_pct']:.0f}%  n={r['n']}  t={r['t_stat']:+.2f}")
    print('\n=== D. 日曆窗口（SPY 2018-2025） ===')
    for r in res['calendar_spy']:
        print(f"  {r['window']:>8s} +{r['horizon']:2d}日  平均 {r['mean_pct']:+.2f}%  "
              f"勝率 {r['hit_pct']:.0f}%  n={r['n']}  t={r['t_stat']:+.2f}")

    # ---- charts ----
    for name, cv in res['curves'].items():
        idx = pd.to_datetime(cv['idx'])
        fig, ax = plt.subplots(figsize=(10, 4.2))
        ax.plot(idx, cv['bh'], lw=1.1, color='#8b96a3', label='買進持有')
        ax.plot(idx, cv['skip'], lw=1.3, color='#3ba7d8', label='跳過週四')
        ax.plot(idx, cv['regime'], lw=1.5, color='#e05a4f', label='均線×星期')
        ax.set_yscale('log')
        ax.set_ylabel('淨值（對數）')
        ax.set_title(name)
        ax.legend(fontsize=9)
        ax.grid(alpha=.3, which='both')
        plt.tight_layout()
        fn = os.path.join(OUT, 'curve_' + {'納斯達克綜合指數（1971 起）': 'nasdaq',
                                           'SPY（2018 起）': 'spy',
                                           'QQQ（2018 起）': 'qqq'}[name] + '.png')
        plt.savefig(fn)
        plt.close()

    print('\nDONE')
    print('images:', [f for f in os.listdir(OUT) if f.endswith('.png')])


if __name__ == '__main__':
    main()
