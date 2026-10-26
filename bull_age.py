#!/usr/bin/env python3
"""牛市年齡：用真實指數（SPY/QQQ 30 年）重算。

## 為什麼要重算

第一版用「611 檔等權組合」當市場指數，算出來的史上最長牛齡只有 652 個交易日，
**所以他說的「700~800 日」在我的資料裡根本不存在**——那是資料限制，不是他錯。
等權組合會被個股輪動與退市拉平，牛熊循環比真實指數短很多。

改用 SPY（2001-10 起）與 QQQ（1999-03 起）真實指數重算，
才能測到他講的那個數字。

## 他的原話

「牛市月線圖的周期，七、八天……六、七百、或者八百左右」
「我們這個牛市已經走了七百九十一日，月線圖二十六支吧，二十六個月，二十幾個月的了」
「所以在那個周期上面，我們是比較 late 的現在」

意思是：牛齡 700~800 日之後要小心。這是一個可以用時間直接檢定的預測。
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
    # 第一個欄位是日期（無名），當索引
    datecol = d.columns[0]
    d = d.rename(columns={datecol: 'date'})
    d['date'] = pd.to_datetime(d['date'])
    d = d.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
    return d


def analyse(path, name):
    d = load_idx(path)
    px = pd.Series(d['close'].to_numpy(float), index=d['date'])
    ma200 = px.rolling(200).mean()
    above = (px > ma200)
    above[:200] = False

    age = np.zeros(len(px), int)
    run = 0
    for i, a in enumerate(above.to_numpy()):
        run = run + 1 if a else 0
        age[i] = run
    age_s = pd.Series(age, index=px.index)

    res = dict(name=name, first=str(px.index[0].date()), last=str(px.index[-1].date()),
               n_days=int(len(px)), max_age=int(age_s.max()), by_bucket={}, all_bull={})
    buckets = [('<400 日', 0, 400), ('400~700 日', 400, 700),
               ('700~800 日', 700, 800), ('800~900 日', 800, 900),
               ('>900 日', 900, 10 ** 9)]
    for lab, lo, hi in buckets:
        sel = (age_s >= lo) & (age_s < hi)
        if sel.sum() < 60:
            res['by_bucket'][lab] = dict(n_days=int(sel.sum()), note='樣本過少')
            continue
        row = dict(n_days=int(sel.sum()), by_h={})
        for h in (20, 60):
            fwd = (px[sel].shift(-h) / px[sel] - 1).dropna()
            v = fwd.to_numpy()
            if len(v) < 40:
                continue
            row['by_h'][h] = dict(trim=trimmed(v), median=float(np.median(v)),
                                  pos=float((v > 0).mean()),
                                  big_loss=float((v <= -0.20).mean()),
                                  t=float(nwt(v, h)), n=int(len(v)))
        res['by_bucket'][lab] = row

    for h in (20, 60):
        fwd = (px[age_s > 0].shift(-h) / px[age_s > 0] - 1).dropna()
        v = fwd.to_numpy()
        if len(v) >= 40:
            res['all_bull'][h] = dict(trim=trimmed(v), median=float(np.median(v)),
                                      pos=float((v > 0).mean()), n=int(len(v)),
                                      t=float(nwt(v, h)))

    # 他說的「接近 800 日要小心」：用 700~800 對比全部牛市日
    sel = (age_s >= 700) & (age_s < 800)
    res['near800'] = {}
    for h in (20, 60):
        fwd = (px[sel].shift(-h) / px[sel] - 1).dropna().to_numpy()
        if len(fwd) >= 40:
            res['near800'][h] = dict(trim=trimmed(fwd), median=float(np.median(fwd)),
                                     pos=float((fwd > 0).mean()), n=int(len(fwd)),
                                     t=float(nwt(fwd, h)))

    # 牛齡分佈：史上每次牛市跑了多久
    runs = []
    cur = 0
    for a in above.to_numpy():
        if a:
            cur += 1
        else:
            if cur > 0:
                runs.append(cur)
            cur = 0
    if cur > 0:
        runs.append(cur)
    res['bull_runs'] = [int(x) for x in runs]

    print(f"\n=== {name}  {res['first']} → {res['last']}  {res['n_days']:,} 日 ===")
    print(f"  史上最長牛齡 {res['max_age']} 交易日"
          f"（約 {res['max_age']/21:.0f} 個月）")
    print(f"  完整牛市次數 {len(runs)}，長度：{sorted(runs, reverse=True)[:8]}")
    print(f"  {'區間':<12}{'日數':>7}{'20日':>9}{'60日':>9}{'上漲率':>8}")
    for lab, v in res['by_bucket'].items():
        if 'note' in v:
            print(f"  {lab:<12}{v['n_days']:>7}  ({v['note']})"); continue
        s20 = v['by_h'].get(20, {})
        s60 = v['by_h'].get(60, {})
        print(f"  {lab:<12}{v['n_days']:>7}{s20.get('trim',0)*100:>+8.2f}%"
              f"{s60.get('trim',0)*100:>+8.2f}%{s20.get('pos',0)*100:>7.0f}%")
    if res['all_bull']:
        print(f"  {'全部牛市日':<12}{'':>7}{res['all_bull'][20]['trim']*100:>+8.2f}%"
              f"{res['all_bull'][60]['trim']*100:>+8.2f}%{res['all_bull'][20]['pos']*100:>7.0f}%")
    return res


def main():
    out = {}
    for f, name in (('data_yahoo/SPY_30y.csv', 'SPY'),
                    ('data_yahoo/QQQ_30y.csv', 'QQQ')):
        p = os.path.join(HERE, f)
        if not os.path.exists(p):
            print(f'缺檔 {f}'); continue
        out[name] = analyse(p, name)

    # 合併結論：他主張 700~800 日要小心，看是對是錯
    verdict = {}
    for name, r in out.items():
        if not r['near800'] or not r['all_bull']:
            continue
        for h in (20, 60):
            if h in r['near800'] and h in r['all_bull']:
                verdict[f'{name}_{h}'] = dict(
                    near=r['near800'][h]['trim'], alls=r['all_bull'][h]['trim'],
                    diff=r['near800'][h]['trim'] - r['all_bull'][h]['trim'],
                    n=r['near800'][h]['n'], pos=r['near800'][h]['pos'],
                    pos_all=r['all_bull'][h]['pos'])
    out['verdict'] = verdict

    json.dump(out, open(os.path.join(OUT, 'bull_age.json'), 'w'),
              ensure_ascii=False, indent=1)

    print('\n=== 他的主張：牛齡 700~800 日要小心（之後報酬應變差）===')
    for k, v in verdict.items():
        name, h = k.rsplit('_', 1)
        print(f"  {name} {h}日: 700~800 日 {v['near']*100:+.2f}%  "
              f"全部牛市日 {v['alls']*100:+.2f}%  差 {v['diff']*100:+.2f}pp  "
              f"上漲率 {v['pos']*100:.0f}% vs {v['pos_all']*100:.0f}%")
    print('\n已寫 bull_age.json')


if __name__ == '__main__':
    main()
