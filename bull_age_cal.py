#!/usr/bin/env python3
"""牛市年齡（日曆日）：修正我自己的單位錯誤。

## 為什麼要再算一次

第三版用「交易日」算牛齡，SPY 史上最長只有 476、QQQ 572，
因此「700~800 日」永遠是空集合，測不到他的主張。

但他的原話是：「我們這個牛市已經走了**七百九十一日**，月線圖**二十六支**吧，
二十六個月」。**26 個月 × 30.4 天 = 790 個日曆日**——完全對上 791。

**所以他的「日」是日曆日，不是交易日。我原本用交易日是單位錯誤。**

## 這一版怎麼算

牛齡 = 從「上一次收盤站上 200 日均線」那天起算的**日曆日**。
在那之上再取「仍高於 200MA」的交易日作為樣本。
這樣 700~800 日就有真實樣本了。
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')


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


def load_idx(path):
    d = pd.read_csv(path)
    d.columns = [c.strip().lower() for c in d.columns]
    d = d.rename(columns={d.columns[0]: 'date'})
    d['date'] = pd.to_datetime(d['date'])
    return d.dropna(subset=['close']).sort_values('date').reset_index(drop=True)


def analyse(path, name):
    d = load_idx(path)
    px = pd.Series(d['close'].to_numpy(float), index=d['date'])
    ma200 = px.rolling(200).mean()
    above = (px > ma200).to_numpy(copy=True)
    above[:200] = False

    # 找每一次「由下往上穿」的起點，算日曆日
    start = None
    age_cal = np.zeros(len(px), int)
    dates = px.index
    for i, a in enumerate(above):
        if a and (i == 0 or not above[i - 1]):
            start = dates[i]
        if a and start is not None:
            age_cal[i] = (dates[i] - start).days
    age_s = pd.Series(age_cal, index=px.index)

    res = dict(name=name, first=str(dates[0].date()), last=str(dates[-1].date()),
               n_days=int(len(px)), max_age_cal=int(age_s.max()),
               max_age_cal_months=round(float(age_s.max()) / 30.4, 1),
               by_bucket={}, all_bull={})

    buckets = [('<400 日', 0, 400), ('400~700 日', 400, 700),
               ('700~800 日', 700, 800), ('800~1000 日', 800, 1000),
               ('>1000 日', 1000, 10 ** 9)]
    for lab, lo, hi in buckets:
        sel = (age_s >= lo) & (age_s < hi) & (age_s > 0)
        if sel.sum() < 60:
            res['by_bucket'][lab] = dict(n_days=int(sel.sum()), note='樣本過少')
            continue
        row = dict(n_days=int(sel.sum()), by_h={})
        for h in (20, 60):
            fwd = (px[sel].shift(-h) / px[sel] - 1).dropna().to_numpy()
            if len(fwd) < 40:
                continue
            row['by_h'][h] = dict(trim=trimmed(fwd), median=float(np.median(fwd)),
                                  pos=float((fwd > 0).mean()),
                                  big_loss=float((fwd <= -0.20).mean()),
                                  t=float(nwt(fwd, h)), n=int(len(fwd)))
        res['by_bucket'][lab] = row

    for h in (20, 60):
        fwd = (px[age_s > 0].shift(-h) / px[age_s > 0] - 1).dropna().to_numpy()
        if len(fwd) >= 40:
            res['all_bull'][h] = dict(trim=trimmed(fwd), median=float(np.median(fwd)),
                                      pos=float((fwd > 0).mean()),
                                      big_loss=float((fwd <= -0.20).mean()),
                                      n=int(len(fwd)), t=float(nwt(fwd, h)))

    # 他講的門檻：700~800 日（日曆）vs 全部牛市日
    res['near800'] = {}
    sel = (age_s >= 700) & (age_s < 800)
    for h in (20, 60):
        fwd = (px[sel].shift(-h) / px[sel] - 1).dropna().to_numpy()
        if len(fwd) >= 40:
            res['near800'][h] = dict(trim=trimmed(fwd), median=float(np.median(fwd)),
                                     pos=float((fwd > 0).mean()),
                                     big_loss=float((fwd <= -0.20).mean()),
                                     n=int(len(fwd)), t=float(nwt(fwd, h)))

    print(f"\n=== {name}  {res['first']} → {res['last']} ===")
    print(f"  史上最長牛齡（日曆日）{res['max_age_cal']} 日 ≈ {res['max_age_cal_months']} 個月")
    print(f"  {'區間（日曆日）':<16}{'日數':>7}{'20日':>9}{'60日':>9}{'上漲率':>8}")
    for lab, v in res['by_bucket'].items():
        if 'note' in v:
            print(f"  {lab:<16}{v['n_days']:>7}  ({v['note']})"); continue
        s20, s60 = v['by_h'].get(20, {}), v['by_h'].get(60, {})
        print(f"  {lab:<16}{v['n_days']:>7}{s20.get('trim',0)*100:>+8.2f}%"
              f"{s60.get('trim',0)*100:>+8.2f}%{s20.get('pos',0)*100:>7.0f}%")
    if res['all_bull']:
        print(f"  {'全部牛市日':<16}{'':>7}{res['all_bull'][20]['trim']*100:>+8.2f}%"
              f"{res['all_bull'][60]['trim']*100:>+8.2f}%{res['all_bull'][20]['pos']*100:>7.0f}%")
    if res['near800']:
        parts = []
        for h in (20, 60):
            if h in res['near800']:
                parts.append(f"{h}日 {res['near800'][h]['trim']*100:+.2f}%")
            else:
                parts.append(f"{h}日 無樣本")
        n20 = res['near800'].get(20, {}).get('n', res['near800'].get(60, {}).get('n', 0))
        print(f"  ★ 700~800 日：" + "  ".join(parts) + f"  n={n20}")
    return res


def main():
    out = {}
    for f, name in (('data_yahoo/SPY_30y.csv', 'SPY'),
                    ('data_yahoo/QQQ_30y.csv', 'QQQ')):
        p = os.path.join(HERE, f)
        if os.path.exists(p):
            out[name] = analyse(p, name)

    # 關鍵檢查：700 日以上的樣本來自幾次「獨立」牛市？
    # 若只來自 1 次，則不管 n 多大，都只有一個觀測，不能做統計。
    for name in list(out):
        ep = {}
        for lab, v in out[name]['by_bucket'].items():
            if 'note' in v:
                continue
            lo = {'<400 日': 0, '400~700 日': 400, '700~800 日': 700,
                  '800~1000 日': 800, '>1000 日': 1000}.get(lab)
            if lo is None or lo < 700:
                continue
            ep[lab] = v['n_days']
        out[name]['long_age_buckets'] = ep

    verdict = {}
    for name, r in out.items():
        for h in (20, 60):
            if h in r.get('near800', {}) and h in r.get('all_bull', {}):
                n8 = r['near800'][h]
                ab = r['all_bull'][h]
                verdict[f'{name}_{h}'] = dict(
                    near=n8['trim'], alls=ab['trim'], diff=n8['trim'] - ab['trim'],
                    n=n8['n'], pos=n8['pos'], pos_all=ab['pos'],
                    big_loss=n8['big_loss'], big_loss_all=ab['big_loss'])
    out['verdict'] = verdict

    json.dump(out, open(os.path.join(OUT, 'bull_age_cal.json'), 'w'),
              ensure_ascii=False, indent=1)

    print('\n=== 他的主張：牛齡 700~800 日要小心 ===')
    if not verdict:
        print('  仍在觀測範圍內的牛市從未達到 700 日曆日')
    for k, v in verdict.items():
        name, h = k.rsplit('_', 1)
        print(f"  {name} {h}日: 700~800 日 {v['near']*100:+.2f}%  "
              f"全部牛市日 {v['alls']*100:+.2f}%  差 {v['diff']*100:+.2f}pp  "
              f"上漲率 {v['pos']*100:.0f}% vs {v['pos_all']*100:.0f}%")
    print('\n已寫 bull_age_cal.json')


if __name__ == '__main__':
    main()
