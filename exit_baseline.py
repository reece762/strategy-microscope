#!/usr/bin/env python3
"""補兩個必要的對照：基準報酬，以及「阻力位前 VCP vs V 形急漲」。

## 為什麼要算基準

「訊號日之後 20 日報酬 +1.08%」單獨看沒有意義。
若同期「什麼都不做」的平均本來就是 +2%，那 +1.08% 代表賣出反而少賠——
也就是**賣對了**。所以每個訊號都必須減掉同期基準，才知道賣出是對是錯。

基準取「同一批股票、處於上升趨勢的所有日子」的 h 日前瞻報酬。
"""
import glob
import json
import os

import numpy as np
import pandas as pd

STOCK = '/root/.openclaw/stock_data'
OUT = '/root/.openclaw/workspace/backtest_videos/xbx/out'
W = 5
H = (5, 10, 20, 60)


def load():
    cols = {}
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
        cols[os.path.basename(f).replace('.csv', '')] = (
            d.dropna(subset=['close']).sort_values('date').reset_index(drop=True))
    return cols


def nwt(e, lag):
    e = np.asarray(e, float)
    e = e[~np.isnan(e)]
    n = len(e)
    if n < 40:
        return np.nan
    mu = e.mean()
    x = e - mu
    s = (x ** 2).mean()
    for l in range(1, min(lag, n - 1) + 1):
        s += 2 * (1 - l / (lag + 1)) * (x[l:] * x[:-l]).mean()
    return mu / np.sqrt(max(s, 1e-18) / n)


def trimmed(x, p=0.01):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 50:
        return float(np.median(x))
    lo, hi = np.quantile(x, [p, 1 - p])
    return float(x[(x >= lo) & (x <= hi)].mean())


def main():
    stocks = load()
    frames = []
    vcp_rows = []

    for sym, d in stocks.items():
        close = d['close'].to_numpy(float)
        high = d['high'].to_numpy(float)
        low = d['low'].to_numpy(float)
        vol = d['volume'].to_numpy(float)
        n = len(close)
        if n < 400:
            continue
        s = pd.Series(close)
        ma50 = s.rolling(50).mean().to_numpy()
        ma20 = s.rolling(20).mean().to_numpy()
        ma50_up = np.r_[np.full(10, np.nan), (ma50[10:] > ma50[:-10]).astype(float)]
        uptrend = (close > ma50) & (ma50_up == 1)

        df = pd.DataFrame({'symbol': sym, 'date': d['date'].to_numpy(), 'uptrend': uptrend})
        for h in H:
            f = np.full(n, np.nan)
            f[:-h] = close[h:] / close[:-h] - 1
            df[f'f{h}'] = f
        frames.append(df)

        # ---------- 阻力位：VCP vs V 形急漲 ----------
        # 已確認的擺動高點（右側 W 根）
        from numpy.lib.stride_tricks import sliding_window_view
        if n <= 2 * W + 1:
            continue
        winh = sliding_window_view(high, 2 * W + 1)
        is_ph = high[W:n - W] == winh.max(axis=1)
        piv_hi = np.full(n, np.nan)
        for i in np.arange(W, n - W)[is_ph]:
            if i + W < n:
                piv_hi[i + W] = high[i]
        prev_hi = pd.Series(piv_hi).ffill().to_numpy()

        # 兩週（10 日）區間寬度；以及更早的 10 日區間
        def rng(a, k):
            r = np.full(n, np.nan)
            for i in range(k, n):
                seg = a[i - k:i + 1]
                r[i] = (seg.max() - seg.min()) / seg.mean()
            return r
        r10 = rng(close, 10)
        r10_prev = np.r_[np.full(10, np.nan), r10[:-10]]

        # 接近前波高點（3% 內）
        near = (prev_hi > 0) & (np.abs(close / prev_hi - 1) <= 0.03)
        g20 = np.r_[np.full(20, np.nan), close[20:] / close[:-20] - 1]
        vshape = near & (g20 >= 0.40) & (r10 >= 0.10)          # V 形急漲、未整固
        vcp = near & (r10 <= 0.08) & (r10_prev >= 0.12) & (g20 >= 0.05)  # 收窄後到阻力
        vol_ok = vol >= pd.Series(vol).rolling(20).mean().to_numpy() * 1.2

        for i in np.where(vshape | vcp)[0]:
            if np.isnan(ma50[i]):
                continue
            rec = dict(symbol=sym, date=d['date'].iloc[i],
                       kind='V形急漲' if vshape[i] else 'VCP收窄',
                       uptrend=bool(uptrend[i]))
            for h in H:
                rec[f'f{h}'] = close[i + h] / close[i] - 1 if i + h < n else np.nan
            vcp_rows.append(rec)

    A = pd.concat(frames, ignore_index=True)
    print(f'股票日 {len(A):,}', flush=True)

    # ---------- 基準：上升趨勢日的平均前瞻報酬 ----------
    base = {}
    up = A[A['uptrend']]
    daily = {h: up.groupby('date')[f'f{h}'].mean() for h in H}
    for h in H:
        v = up[f'f{h}'].to_numpy()
        v = v[~np.isnan(v)]
        base[h] = dict(n=int(len(v)), trim=trimmed(v), median=float(np.median(v)),
                       pos=float((v > 0).mean()), big_loss=float((v <= -0.20).mean()),
                       t_daily=float(nwt(daily[h].to_numpy(), h)))

    print('\n=== 基準（上升趨勢日的平均前瞻報酬）===')
    for h in H:
        b = base[h]
        print(f"  {h:>2}日: 截尾 {b['trim']*100:+.2f}%  中位 {b['median']*100:+.2f}%  "
              f"上漲 {b['pos']*100:.1f}%  跌逾20% {b['big_loss']*100:.1f}%  t(日)={b['t_daily']:+.2f}")

    # ---------- VCP vs V 形 ----------
    V = pd.DataFrame(vcp_rows)
    vcp_res = {}
    print(f'\n=== 阻力位：VCP 收窄 vs V 形急漲（n={len(V):,}）===')
    for kind in ('VCP收窄', 'V形急漲'):
        sub = V[V.kind == kind]
        if len(sub) < 50:
            continue
        vcp_res[kind] = {}
        line = f'  {kind:<8} n={len(sub):>6,}'
        for h in H:
            v = sub[f'f{h}'].dropna().to_numpy()
            if len(v) < 50:
                continue
            dly = sub.dropna(subset=[f'f{h}']).groupby('date')[f'f{h}'].mean().to_numpy()
            vcp_res[kind][h] = dict(n=int(len(v)), trim=trimmed(v),
                                    median=float(np.median(v)),
                                    pos=float((v > 0).mean()),
                                    excess=float(trimmed(v) - base[h]['trim']),
                                    t_daily=float(nwt(dly, h)))
            line += f"  {h}日 {trimmed(v)*100:+6.2f}%"
        print(line)

    # 差值檢定（VCP − V形，以交易日為單位）
    vcp_diff = {}
    for h in H:
        a = V[(V.kind == 'VCP收窄')].dropna(subset=[f'f{h}']).groupby('date')[f'f{h}'].mean()
        b = V[(V.kind == 'V形急漲')].dropna(subset=[f'f{h}']).groupby('date')[f'f{h}'].mean()
        both = pd.concat([a.rename('a'), b.rename('b')], axis=1).dropna()
        if len(both) < 40:
            continue
        d = (both.a - both.b).to_numpy()
        rng = np.random.default_rng(7)
        boot = np.array([rng.choice(d, len(d), True).mean() for _ in range(1000)])
        vcp_diff[h] = dict(diff=float(d.mean()), t=float(nwt(d, h)),
                           ci=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                           n_days=int(len(both)))

    if vcp_diff:
        print('\n  VCP − V形 差值：')
        for h, v in vcp_diff.items():
            sig = '不含零' if not (v['ci'][0] < 0 < v['ci'][1]) else '含零'
            print(f"    {h:>2}日 {v['diff']*100:+.2f}%  CI[{v['ci'][0]*100:+.2f},{v['ci'][1]*100:+.2f}] "
                  f"t={v['t']:+.2f} {sig}  n_days={v['n_days']}")

    json.dump(dict(baseline=base, vcp=vcp_res, vcp_diff=vcp_diff),
              open(f'{OUT}/exit_baseline.json', 'w'), ensure_ascii=False, indent=1)
    print('\n已寫 exit_baseline.json')


if __name__ == '__main__':
    main()
