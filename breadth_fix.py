#!/usr/bin/env python3
"""市場寬度背馳：修正「篩選後才位移」的錯誤。

## 錯在哪裡

第一版寫成 `idx[mask].shift(-h)`。
**pandas 的 shift 是在「篩選後的序列」內位移，不是按日曆位移。**
若遮罩稀疏，`100 → 下一個符合遮罩的值` 可能跨了好幾個月。

實測：背馳日 469 天散落在 4,841 天裡，彼此平均相隔約 10 天，
**所以 shift(-60) 實際上是「往前 60 個背馳日」＝ 往前約 600 個交易日（近 3 年）**。
這就是為什麼會算出「背馳後 60 日 +28.63%、同步創高 +50.79%」——
**那根本不是 60 日的報酬，是三年的報酬。**

## 正確做法

**先在完整、連續、按日期排序的序列上算好前瞻報酬，再按遮罩選取。**
"""
import json
import os

import numpy as np
import pandas as pd
import glob

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
STOCK = '/root/.openclaw/stock_data'


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
        return float(np.median(x)) if len(x) else np.nan
    lo, hi = np.quantile(x, [p, 1 - p])
    return float(x[(x >= lo) & (x <= hi)].mean())


def build_breadth():
    frames = []
    for f in sorted(glob.glob(f'{STOCK}/live/*.csv') + glob.glob(f'{STOCK}/extra/*.csv')):
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        d.columns = [c.lower() for c in d.columns]
        if not {'date', 'close'}.issubset(d.columns) or len(d) < 400:
            continue
        d['date'] = pd.to_datetime(d['date'])
        d = d.dropna(subset=['close']).sort_values('date')
        c = d['close'].to_numpy(float)
        s = pd.Series(c)
        ma50 = s.rolling(50).mean().to_numpy()
        ma200 = s.rolling(200).mean().to_numpy()
        frames.append(pd.DataFrame({
            'date': d['date'].to_numpy(),
            'above50': (c > ma50) & ~np.isnan(ma50),
            'above200': (c > ma200) & ~np.isnan(ma200),
            'has50': ~np.isnan(ma50), 'has200': ~np.isnan(ma200),
        }))
    A = pd.concat(frames, ignore_index=True)
    g = A.groupby('date')
    pct50 = g.apply(lambda x: x.loc[x.has50, 'above50'].mean(), include_groups=False)
    pct200 = g.apply(lambda x: x.loc[x.has200, 'above200'].mean(), include_groups=False)
    b = pd.DataFrame({'pct50': pct50, 'pct200': pct200}).dropna()
    b = b[b.pct200 > 0]
    return b


def load_index(path):
    d = pd.read_csv(path)
    d.columns = [c.strip().lower() for c in d.columns]
    d = d.rename(columns={d.columns[0]: 'date'})
    d['date'] = pd.to_datetime(d['date'])
    d = d.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
    return d[['date', 'close']]


def main():
    print('建立市場寬度...', flush=True)
    b = build_breadth()
    print(f'  寬度 {len(b):,} 天', flush=True)

    out = {'methods': 'forward returns computed on the FULL series, then masked'}
    for name, path in (('SPY', 'data_yahoo/SPY_30y.csv'),
                       ('QQQ', 'data_yahoo/QQQ_30y.csv')):
        p = os.path.join(HERE, path)
        if not os.path.exists(p):
            continue
        ix = load_index(p).set_index('date')['close']
        j = b.join(ix.rename('idx'), how='inner').dropna().sort_index()
        if len(j) < 400:
            print(f'  {name} 對齊後只有 {len(j)} 天，跳過'); continue

        # ★★ 關鍵修正：先在完整序列上算前瞻報酬，再遮罩
        idx = j['idx']
        fwd = {}
        for h in (10, 20, 60):
            fwd[h] = (idx.shift(-h) / idx - 1).to_numpy()

        hi60 = (idx >= idx.rolling(60).max()).to_numpy()
        w20 = j['pct200'].rolling(20).mean()
        w200_hi = (w20 >= w20.rolling(60).max()).to_numpy()

        diverge = hi60 & (~w200_hi) & (w20 > 0).to_numpy()
        sync = hi60 & w200_hi

        rec = dict(n_days=int(len(j)), n_div=int(diverge.sum()), n_sync=int(sync.sum()),
                   pct200_min=float(j['pct200'].min()), pct200_max=float(j['pct200'].max()),
                   div={}, sync={}, diff={},
                   # 診斷：背馳日的平均間隔，用來證明舊版為何錯
                   div_gap_days=float(np.diff(np.where(diverge)[0]).mean()) if diverge.sum() > 1 else np.nan)

        for h in (10, 20, 60):
            a = fwd[h][diverge]
            a = a[~np.isnan(a)]
            bb = fwd[h][sync]
            bb = bb[~np.isnan(bb)]
            if len(a) >= 20:
                rec['div'][h] = dict(trim=trimmed(a), median=float(np.median(a)),
                                     pos=float((a > 0).mean()),
                                     big_loss=float((a <= -0.10).mean()),
                                     n=int(len(a)), t=float(nwt(a, h)))
            if len(bb) >= 20:
                rec['sync'][h] = dict(trim=trimmed(bb), median=float(np.median(bb)),
                                      pos=float((bb > 0).mean()),
                                      big_loss=float((bb <= -0.10).mean()),
                                      n=int(len(bb)), t=float(nwt(bb, h)))
            if len(a) >= 20 and len(bb) >= 20:
                rng = np.random.default_rng(31)
                boot = np.array([rng.choice(a, len(a), True).mean()
                                 - rng.choice(bb, len(bb), True).mean()
                                 for _ in range(1000)])
                rec['diff'][h] = dict(diff=float(a.mean() - bb.mean()),
                                      ci=[float(np.quantile(boot, .025)),
                                          float(np.quantile(boot, .975))],
                                      n_div=int(len(a)), n_sync=int(len(bb)))

        # 全期基準
        rec['all'] = {}
        for h in (10, 20, 60):
            v = fwd[h][~np.isnan(fwd[h])]
            rec['all'][h] = dict(trim=trimmed(v), pos=float((v > 0).mean()),
                                 median=float(np.median(v)), n=int(len(v)))

        out[name] = rec

        print(f"\n── {name}（{rec['n_days']:,} 天；寬度 200MA "
              f"{rec['pct200_min']*100:.0f}%~{rec['pct200_max']*100:.0f}%）──")
        print(f"   背馳 {rec['n_div']} 天 / 同步創高 {rec['n_sync']} 天"
              f"（背馳日平均相隔 {rec['div_gap_days']:.1f} 個交易日）")
        print(f"   {'持有期':<8}{'背馳後':>10}{'同步創高後':>12}{'全期基準':>10}{'差值':>10}{'CI':>20}")
        for h in (10, 20, 60):
            if h not in rec['diff']:
                continue
            dv = rec['diff'][h]
            print(f"   {h:>2}日{'':<4}{rec['div'][h]['trim']*100:>+9.2f}%"
                  f"{rec['sync'][h]['trim']*100:>+11.2f}%"
                  f"{rec['all'][h]['trim']*100:>+9.2f}%"
                  f"{dv['diff']*100:>+9.2f}%   [{dv['ci'][0]*100:+.2f},{dv['ci'][1]*100:+.2f}]")

    json.dump(out, open(f'{OUT}/breadth_fix.json', 'w'), ensure_ascii=False, indent=1)
    print('\n已寫 breadth_fix.json')


if __name__ == '__main__':
    main()
