#!/usr/bin/env python3
"""把趨勢線與平行通道的結果統一成「去重後」的版本，供卡片使用。

為什麼要重算：
第一版直接把「每根 K 線只要符合條件就算一筆」的交易全丟進統計，
造成同一段持倉被切成好幾筆。實測平行通道 221,186 筆裡有 54.5% 的
相鄰交易持倉期重疊，總持倉佔比 65%——那已經接近「幾乎一直在場內」，
不是「選擇性進出」，t 值因此虛高。

去重規則：同檔股票同一時間只能有一筆部位。若新訊號出現時舊部位還沒出場，
不算新交易（但若新部位會出場得更晚，則延長舊部位的出場日）。
"""
import glob
import json
import os

import numpy as np
import pandas as pd

LINES = '/root/.openclaw/workspace/reports/_agent_lines'
PATT = '/root/.openclaw/workspace/reports/_agent_pattern'
OUT = '/root/.openclaw/workspace/backtest_videos/xbx/out'
STOCK = '/root/.openclaw/stock_data'


def market():
    px = []
    for f in sorted(glob.glob(f'{STOCK}/live/*.csv') + glob.glob(f'{STOCK}/extra/*.csv')):
        try:
            d = pd.read_csv(f, usecols=lambda c: c.lower() in ('date', 'close'))
        except Exception:
            continue
        d.columns = [c.lower() for c in d.columns]
        if 'date' not in d.columns or 'close' not in d.columns or len(d) < 300:
            continue
        d['date'] = pd.to_datetime(d['date'])
        px.append(d.set_index('date')['close'].rename(os.path.basename(f)))
    P = pd.concat(px, axis=1, sort=True).sort_index()
    return (1 + P.pct_change().mean(axis=1).fillna(0)).cumprod()


def dedup(df):
    keep = []
    for _, g in df.sort_values('entry_date').groupby('symbol'):
        last = None
        for i, r in g.iterrows():
            if last is None or r.entry_date >= last:
                keep.append(i)
                last = r.exit_date
            elif r.exit_date > last:
                last = r.exit_date
    return df.loc[keep]


def nwt(ex, lag=10):
    n = len(ex)
    mu = ex.mean()
    e = ex - mu
    s = (e ** 2).mean()
    for l in range(1, lag + 1):
        s += 2 * (1 - l / (lag + 1)) * (e[l:] * e[:-l]).mean()
    return mu / np.sqrt(max(s, 1e-18) / n)


def enrich(df, mcum):
    m0 = mcum.reindex(df.entry_date).ffill().values
    m1 = mcum.reindex(df.exit_date).ffill().values
    df = df.copy()
    df['excess'] = df.ret.values - (m1 / m0 - 1)
    df['year'] = pd.to_datetime(df.entry_date).dt.year
    return df


def summarize(df, label):
    ex = df.excess.values
    q = np.quantile(ex, 0.95)
    tr = ex[ex <= q]
    out = dict(
        label=label, n=int(len(ex)),
        win=float(df.ret.gt(0).mean()),
        avg=float(df.ret.mean()),
        excess=float(ex.mean()),
        t=float(nwt(ex)),
        trimmed=float(tr.mean()),
        trimmed_t=float(nwt(tr)),
        hold=float(df.hold.mean()),
    )
    if 'reason' in df.columns:
        vc = df.reason.value_counts(normalize=True)
        out['reasons'] = {k: float(v) for k, v in vc.items()}
        out['stop_rate'] = float(vc.get('stop', 0.0))
    return out


def main():
    mcum = market()

    # ---- 趨勢線
    t = pd.read_csv(f'{LINES}/trendline_trades.csv')
    t['entry_date'] = pd.to_datetime(t.entry_date)
    t['exit_date'] = pd.to_datetime(t.exit_date)
    t = enrich(t, mcum)
    td = enrich(dedup(t), mcum)

    tl_ctl = pd.read_csv(f'{LINES}/trendline_control.csv')
    tl_ctl['entry_date'] = pd.to_datetime(tl_ctl.entry_date)
    tl_ctl['exit_date'] = pd.to_datetime(tl_ctl.exit_date)
    ctl = enrich(dedup(enrich(tl_ctl, mcum)), mcum)

    # ---- 平行通道
    c = pd.read_csv(f'{PATT}/channel_trades.csv')
    c['entry_date'] = pd.to_datetime(c.entry_date)
    c['exit_date'] = pd.to_datetime(c.exit_date)
    c = enrich(c, mcum)
    cd = enrich(dedup(c), mcum)

    cc = pd.read_csv(f'{PATT}/channel_control.csv')
    cc['entry_date'] = pd.to_datetime(cc.entry_date)
    cc['exit_date'] = pd.to_datetime(cc.exit_date)
    ccd = enrich(dedup(enrich(cc, mcum)), mcum)

    def by_year(df):
        g = df.groupby('year').agg(n=('excess', 'size'), ex=('excess', 'mean'),
                                   raw=('ret', 'mean'))
        return [dict(year=int(k), n=int(v['n']), ex=float(v['ex']), raw=float(v['raw']))
                for k, v in g.iterrows()]

    trend = dict(raw=summarize(t, '原始'), dedup=summarize(td, '去重後'),
                 control=summarize(ctl, '對照組（去重）'),
                 by_year=by_year(td),
                 overlap_raw=float(1 - len(td) / len(t)),
                 years_pos=int(sum(1 for r in by_year(td) if r['ex'] > 0)),
                 years_n=len(by_year(td)))
    chan = dict(raw=summarize(c, '原始'), dedup=summarize(cd, '去重後'),
                control=summarize(ccd, '對照組（去重）'),
                by_year=by_year(cd),
                overlap_raw=float(1 - len(cd) / len(c)),
                years_pos=int(sum(1 for r in by_year(cd) if r['ex'] > 0)),
                years_n=len(by_year(cd)))

    res = dict(trendline=trend, channel=chan)
    json.dump(res, open(f'{OUT}/n6_final.json', 'w'), ensure_ascii=False, indent=1)

    for nm, d in (('趨勢線', trend), ('平行通道', chan)):
        print(f'=== {nm} ===')
        for k in ('raw', 'dedup', 'control'):
            s = d[k]
            print(f"  {s['label']:12s} n={s['n']:>7,}  勝率 {s['win']*100:5.1f}%  "
                  f"原始 {s['avg']*100:+.3f}%  超額 {s['excess']*100:+.4f}%  "
                  f"t={s['t']:+.2f}  剔除5% {s['trimmed']*100:+.3f}%")
        print(f"  持倉重疊比例 {d['overlap_raw']*100:.1f}%  "
              f"逐年超額為正 {d['years_pos']}/{d['years_n']}")
    print('已寫 out/n6_final.json')


if __name__ == '__main__':
    main()
