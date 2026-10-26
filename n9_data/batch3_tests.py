#!/usr/bin/env python3
"""第三批三支影片：海龜派、長期贏家（市場寬度背馳）、EP1（量價突破）。

## A. 海龜派／唐奇安通道（34W10kFBC0U）
他的原話（引述海龜法則）：
  「方法第一，價格升穿前二十日最高價就買入」
  「方法第二，價格升穿前五十五日最高價就買入」
這是全站最「硬」的規則——完全機械、無主觀判斷。
測法：20 日與 55 日高點突破（不含當日，避免自我參考），且處於上升趨勢。
另外測「只在趨勢過濾下突破」與「無過濾」的差別，因為海龜原則是純突破。

## B. 市場寬度背馳（YEhOJaQqhZQ）★ 這支最有價值
他的原話：
  「如果我們去看一些細節的話，包括就是大盤的 50 天比率還有 200 天的比率，
   背馳是越來越嚴重……那個分水嶺就在 8 月 10 日開始，連 200 天的比率都一起背馳，
   但是指數在不斷上升」
他的操作：「當我見到這種背馳信號，我及早做好一些防風措施，最簡單就是減低投資比重」
測法：
  - 每日算「站上 50MA 的股票比例」與「站上 200MA 的股票比例」（市場寬度）
  - 指數（等權）創 60 日新高，但寬度沒跟上 → 定義為「頂部背馳」
  - 之後市場報酬 vs 一般日子
**這是唯一一支他明確講了「怎麼看」而且我可以精確重建的。**

## C. EP1 量價突破（vpKJoVtic_Y，3.4 萬字）
他的原話：
  「強勁上升趨勢的話，理應就會見到在上升的時候是大成交量上升，
   在下跌的時候是相對低成交量拉回」
  「大陽燭出現之後它是伴隨大成交量上升，然後就低量拉回，
   所以我就部署了一個突破交易計劃，只要它帶量升穿這支燭身的高點」
測法：找「帶量大陽燭（漲 ≥5%、量 ≥2 倍均量）→ 之後 3~10 日低量拉回（量 < 均量）
     → 收盤突破該大陽燭的高點」的複合形態，看之後報酬。
"""
import glob
import json
import os

import numpy as np
import pandas as pd

STOCK = '/root/.openclaw/stock_data'
OUT = '/root/.openclaw/workspace/backtest_videos/xbx/out'
H = (5, 10, 20, 60)


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


def load():
    out = {}
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
        d = d.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
        out[os.path.basename(f).replace('.csv', '')] = d
    return out


def main():
    stocks = load()
    print(f'母體 {len(stocks)} 檔', flush=True)

    frames = []
    # 市場寬度用：每日記錄站上 50MA / 200MA 的股票數與總數
    breadth = {}

    for sym, d in stocks.items():
        close = d['close'].to_numpy(float)
        high = d['high'].to_numpy(float)
        low = d['low'].to_numpy(float)
        op = d['open'].to_numpy(float)
        vol = d['volume'].to_numpy(float)
        n = len(close)
        if n < 400:
            continue
        s = pd.Series(close)
        ma50 = s.rolling(50).mean().to_numpy()
        ma200 = s.rolling(200).mean().to_numpy()
        ma50_up = np.r_[np.full(10, np.nan), (ma50[10:] > ma50[:-10]).astype(float)]
        uptrend = (close > ma50) & (ma50_up == 1)

        prev_close = np.r_[np.nan, close[:-1]]
        day_ret = close / prev_close - 1
        v20 = pd.Series(vol).rolling(20).mean().to_numpy()

        # A. 唐奇安通道（不含當日）
        hi20 = pd.Series(high).rolling(20).max().shift(1).to_numpy()
        hi55 = pd.Series(high).rolling(55).max().shift(1).to_numpy()
        brk20 = close > hi20
        brk55 = close > hi55

        # C. EP1：帶量大陽燭 → 低量拉回 → 突破該燭高點
        big_up = (day_ret >= 0.05) & (vol >= 2 * v20)
        # 在過去 3~10 日內曾出現大陽燭，且當日為低量（< 均量）
        big_up_past = pd.Series(big_up.astype(float)).rolling(10).max().shift(1).to_numpy() > 0
        big_up_recent = pd.Series(big_up.astype(float)).rolling(10).sum().shift(1).to_numpy()
        low_vol_today = vol < v20
        # 該大陽燭的最高價（取最近 10 日內最高）
        big_candle_high = np.where(big_up, high, np.nan)
        bc_hi = pd.Series(big_candle_high).rolling(10).max().shift(1).to_numpy()
        # 突破該燭高點、低量、上升趨勢
        ep1_break = (close > bc_hi) & low_vol_today & uptrend & (big_up_recent >= 1)

        df = pd.DataFrame({
            'symbol': sym, 'date': d['date'].to_numpy(), 'uptrend': uptrend,
            'brk20': brk20, 'brk55': brk55,
            'brk20_up': brk20 & uptrend, 'brk55_up': brk55 & uptrend,
            'ep1_break': ep1_break,
            'day_ret': day_ret,
            'above50': (close > ma50) & ~np.isnan(ma50),
            'above200': (close > ma200) & ~np.isnan(ma200),
            'has50': ~np.isnan(ma50), 'has200': ~np.isnan(ma200),
        })
        for h in H:
            f = np.full(n, np.nan)
            f[:-h] = close[h:] / close[:-h] - 1
            df[f'f{h}'] = f
        frames.append(df)

    A = pd.concat(frames, ignore_index=True)
    print(f'股票日 {len(A):,}', flush=True)

    # ---------- 基準 ----------
    up = A[A.uptrend]
    base = {}
    for h in H:
        v = up[f'f{h}'].dropna().to_numpy()
        dly = up.groupby('date')[f'f{h}'].mean().to_numpy()
        base[h] = dict(trim=trimmed(v), median=float(np.median(v)),
                       pos=float((v > 0).mean()), big_loss=float((v <= -0.20).mean()),
                       t=float(nwt(dly, h)), n=int(len(v)))

    def stats(mask, label):
        out = {'label': label, 'n': int(mask.sum()), 'by_h': {}}
        if mask.sum() < 40:
            out['note'] = '樣本過少'
            return out
        for h in H:
            sub = A.loc[mask, ['date', f'f{h}']].dropna()
            if len(sub) < 40:
                continue
            dly = sub.groupby('date')[f'f{h}'].mean()
            v = sub[f'f{h}'].to_numpy()
            out['by_h'][h] = dict(n=int(len(v)), n_days=int(len(dly)),
                                  trim=trimmed(v), median=float(np.median(v)),
                                  pos=float((v > 0).mean()),
                                  big_loss=float((v <= -0.20).mean()),
                                  t=float(nwt(dly.to_numpy(), h)))
        return out

    res = {}
    res['brk20'] = stats(A.brk20, '唐奇安 20 日突破（海龜法一）')
    res['brk55'] = stats(A.brk55, '唐奇安 55 日突破（海龜法二）')
    res['brk20_up'] = stats(A.brk20_up, '20 日突破 + 上升趨勢過濾')
    res['brk55_up'] = stats(A.brk55_up, '55 日突破 + 上升趨勢過濾')
    res['ep1_break'] = stats(A.ep1_break, 'EP1 帶量大陽燭→低量拉回→突破')

    # ---------- B. 市場寬度背馳 ----------
    g = A.groupby('date')
    pct50 = g.apply(lambda x: x.loc[x.has50, 'above50'].mean(), include_groups=False)
    pct200 = g.apply(lambda x: x.loc[x.has200, 'above200'].mean(), include_groups=False)
    mkt_ret = g['day_ret'].mean()
    idx = (1 + mkt_ret.fillna(0)).cumprod()

    bh = pd.DataFrame({'pct50': pct50, 'pct200': pct200, 'idx': idx}).dropna()
    bh = bh[bh.pct200 > 0]
    # 指數創 60 日新高
    idx_hi60 = bh['idx'] >= bh['idx'].rolling(60).max()
    # 背馳定義：指數創新高，但寬度沒同步創新高
    #   寬度 20 日均值「未」創 60 日新高
    w50 = bh['pct50'].rolling(20).mean()
    w200 = bh['pct200'].rolling(20).mean()
    w50_hi = w50 >= w50.rolling(60).max()
    w200_hi = w200 >= w200.rolling(60).max()
    # 他說的：「8月10日開始連 200 天的比率都一起背馳，但是指數在不斷上升」
    diverge = idx_hi60 & (~w200_hi) & (w200 > 0)
    both_ok = idx_hi60 & w200_hi
    no_div = (~idx_hi60)

    wid = {}
    for lab, mask in (('頂部背馳（指數新高、200MA寬度未新高）', diverge),
                      ('同步創高（指數與寬度皆新高）', both_ok),
                      ('指數未創新高', no_div)):
        sel = bh[mask]
        if len(sel) < 30:
            wid[lab] = dict(n_days=int(len(sel)), note='樣本過少')
            continue
        row = dict(n_days=int(len(sel)), by_h={})
        for h in (10, 20, 60):
            fwd = (sel['idx'].shift(-h) / sel['idx'] - 1).dropna().to_numpy()
            if len(fwd) < 20:
                continue
            row['by_h'][h] = dict(trim=trimmed(fwd), median=float(np.median(fwd)),
                                  pos=float((fwd > 0).mean()),
                                  big_loss=float((fwd <= -0.10).mean()),
                                  n=int(len(fwd)), t=float(nwt(fwd, h)))
        wid[lab] = row

    # 背馳 vs 同步創高 的差值（每日一個觀測）
    div_diff = {}
    for h in (10, 20, 60):
        a = (bh.loc[diverge, 'idx'].shift(-h) / bh.loc[diverge, 'idx'] - 1).dropna()
        b = (bh.loc[both_ok, 'idx'].shift(-h) / bh.loc[both_ok, 'idx'] - 1).dropna()
        if len(a) < 20 or len(b) < 20:
            continue
        diff = float(a.mean() - b.mean())
        # 拔靴
        rng = np.random.default_rng(23)
        boot = np.array([rng.choice(a.to_numpy(), len(a), True).mean()
                         - rng.choice(b.to_numpy(), len(b), True).mean()
                         for _ in range(1000)])
        div_diff[h] = dict(diff=diff, ci=[float(np.quantile(boot, .025)),
                                          float(np.quantile(boot, .975))],
                           n_div=int(len(a)), n_sync=int(len(b)))

    out = dict(baseline=base, signals=res, breadth=wid, breadth_diff=div_diff,
               breadth_stats=dict(
                   n_days=int(len(bh)),
                   pct200_min=float(bh['pct200'].min()),
                   pct200_max=float(bh['pct200'].max()),
                   n_diverge=int(diverge.sum()), n_sync=int(both_ok.sum())))
    json.dump(out, open(f'{OUT}/batch3.json', 'w'), ensure_ascii=False, indent=1)

    print(f"\n{'訊號':<38}{'n':>9}{'20日':>9}{'60日':>9}{'vs基準20':>10}")
    print('-' * 78)
    for k, v in res.items():
        if 'note' in v:
            print(f"{v['label']:<38}{v['n']:>9}  ({v['note']})"); continue
        b20 = v['by_h'].get(20, {})
        b60 = v['by_h'].get(60, {})
        d20 = b20.get('trim', np.nan) - base[20]['trim']
        print(f"{v['label']:<38}{v['n']:>9,}{b20.get('trim',0)*100:>+8.2f}%"
              f"{b60.get('trim',0)*100:>+8.2f}%{d20*100:>+9.2f}pp")
    print(f"\n基準（上升趨勢日）  20日 {base[20]['trim']*100:+.2f}%  60日 {base[60]['trim']*100:+.2f}%")

    print('\n【市場寬度背馳】他的主張：背馳出現要減倉')
    for lab, v in wid.items():
        if 'note' in v:
            print(f"  {lab}: {v['n_days']} 天（{v['note']}）"); continue
        s20 = v['by_h'].get(20, {})
        s60 = v['by_h'].get(60, {})
        print(f"  {lab}")
        print(f"    n={v['n_days']:>4} 天  20日 {s20.get('trim',0)*100:+.2f}% "
              f"(上漲 {s20.get('pos',0)*100:.0f}%)  60日 {s60.get('trim',0)*100:+.2f}%")
    print('\n  背馳 − 同步創高：')
    for h, v in div_diff.items():
        sig = '不含零' if not (v['ci'][0] < 0 < v['ci'][1]) else '含零'
        print(f"    {h:>2}日 {v['diff']*100:+.2f}%  "
              f"CI[{v['ci'][0]*100:+.2f},{v['ci'][1]*100:+.2f}]  {sig}  "
              f"(背馳{v['n_div']}天 / 同步{v['n_sync']}天)")
    print(f"\n  寬度統計：200MA 比例 {bh['pct200'].min()*100:.1f}%~{bh['pct200'].max()*100:.1f}%")

    print('\n已寫 batch3.json')


if __name__ == '__main__':
    main()
