#!/usr/bin/env python3
"""第二批四支影片的檢驗：移動平均線、拉回買入法、避開股災、Darvas 箱型。

## 四組主張與測法

### A. 移動平均線（Q4Ersajf9qg，2 萬字，最長的一支）
他的核心主張：
  1. 「強勢股跌到 10 天線或 20 天線的時候，你不入場就可能不會再有機會入場」
  2. 均線是「區間」不是「死數字」（所以他同時開 SMA 與 EMA）
  3. **SMA 與 EMA「其實是不太會引致績效上有一個顯著的分別」**（他自己說的）
測法：
  - 上升趨勢中，股價觸及 10MA 後收在其上、且成交量低於 20 日均量 → 之後報酬
  - 同樣條件改用 20MA
  - 同一個訊號分別用 SMA10 與 EMA10 判定，比較兩者的報酬差（驗證他的第 3 點）

### B. 拉回買入法（1_QYg9V5xqc）
八條規則，最核心、也最反直覺的是第三條：
  **「拉回買入不是在下跌的時候買，而是在 pullback recovery（股價反轉重新向上）的時候才買」**
測法：
  - 上升趨勢中，先有一段拉回（3~10 日內下跌 5% 以上）
  - 「拉回中買」＝在下跌過程中買（當日下跌且跌破前一日低點）
  - 「反轉買」＝拉回後首次收復前一日高點（recovery）
  - 比較兩者之後 h 日報酬。**這是本批最乾淨的一組對照。**

### C. 避開股災（MoVU4tpLmIU）
他的主張（很具體）：
  「牛市月線圖的周期，七、八天……六、七百、或者八百左右」
  「我們這個牛市已經走了七百九十一日……所以在那個周期上面，我們是比較 late 的現在」
測法：
  - 建立全體等權市場指數，算「站上 200 日均線之後經過幾個交易日」
  - 按牛齡分組（<400、400~700、700~900、>900 日）看之後 20/60 日報酬
  - **這是他唯一給出「時間數字」的預測，可以直接檢定。**

### D. Darvas 箱型（wx2lkQbeebE）
達華斯的箱型理論：股票在箱內盤整，突破箱頂買進，跌回箱底停損。
測法：
  - 箱＝過去 15 日的高低位區間，且區間寬度 ≤15%
  - 突破＝收盤站上箱頂，且處於上升趨勢
  - 之後 h 日報酬 vs 基準

## 判定基準
一律對照「同期上升趨勢日」的基準報酬（20 日 +0.93%，60 日 +3.25%）。
均線與箱型的 t 值在交易日層級計算（每日一個觀測）並用 Newey-West。
存活者偏差未修正，方向一律為高估。
"""
import glob
import json
import os

import numpy as np
import pandas as pd

STOCK = '/root/.openclaw/stock_data'
OUT = '/root/.openclaw/workspace/backtest_videos/xbx/out'
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
        d = d.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
        cols[os.path.basename(f).replace('.csv', '')] = d
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
        return float(np.median(x)) if len(x) else np.nan
    lo, hi = np.quantile(x, [p, 1 - p])
    return float(x[(x >= lo) & (x <= hi)].mean())


def stats(mask, A, label):
    out = {'label': label, 'n': int(mask.sum()), 'by_h': {}}
    if mask.sum() < 40:
        out['note'] = '樣本過少'
        return out
    for h in H:
        sub = A.loc[mask, ['date', f'f{h}']].dropna()
        if len(sub) < 40:
            continue
        daily = sub.groupby('date')[f'f{h}'].mean()
        v = sub[f'f{h}'].to_numpy()
        out['by_h'][h] = dict(n=int(len(v)), n_days=int(len(daily)),
                              trim=trimmed(v), median=float(np.median(v)),
                              pos=float((v > 0).mean()),
                              big_loss=float((v <= -0.20).mean()),
                              t=float(nwt(daily.to_numpy(), h)))
    return out


def main():
    stocks = load()
    print(f'母體 {len(stocks)} 檔', flush=True)

    frames = []
    for sym, d in stocks.items():
        close = d['close'].to_numpy(float)
        high = d['high'].to_numpy(float)
        low = d['low'].to_numpy(float)
        vol = d['volume'].to_numpy(float)
        n = len(close)
        if n < 400:
            continue
        s = pd.Series(close)
        ma10 = s.rolling(10).mean().to_numpy()
        ma20 = s.rolling(20).mean().to_numpy()
        ma50 = s.rolling(50).mean().to_numpy()
        # EMA 版（驗證他「SMA 與 EMA 沒顯著差別」的說法）
        ema10 = s.ewm(span=10, adjust=False).mean().to_numpy()
        ema20 = s.ewm(span=20, adjust=False).mean().to_numpy()

        ma50_up = np.r_[np.full(10, np.nan), (ma50[10:] > ma50[:-10]).astype(float)]
        uptrend = (close > ma50) & (ma50_up == 1)
        v20 = pd.Series(vol).rolling(20).mean().to_numpy()
        lowvol = vol < v20

        prev_close = np.r_[np.nan, close[:-1]]
        prev_high = np.r_[np.nan, high[:-1]]
        prev_low = np.r_[np.nan, low[:-1]]
        day_ret = close / prev_close - 1

        # A1: 10MA 拉回（觸及 10MA 後收在其上、低量）
        touch10 = (low <= ma10) & (close > ma10)
        a1 = touch10 & lowvol & uptrend
        # A2: 20MA 拉回
        touch20 = (low <= ma20) & (close > ma20)
        a2 = touch20 & lowvol & uptrend
        # A3: 同一訊號用 EMA10 判定
        touch10e = (low <= ema10) & (close > ema10)
        a3 = touch10e & lowvol & uptrend

        # B: 拉回買入法
        # 拉回定義：過去 3~10 日內自 10 日高點回落 ≥5%
        hi10 = pd.Series(high).rolling(10).max().to_numpy()
        pullback = (close / hi10 - 1 <= -0.05)
        dip_buy = pullback & (day_ret < 0) & (close < prev_low) & uptrend
        recovery = pullback & (close > prev_high) & (day_ret > 0) & uptrend

        # D: Darvas 箱型突破
        box_hi = pd.Series(high).rolling(15).max().shift(1).to_numpy()
        box_lo = pd.Series(low).rolling(15).min().shift(1).to_numpy()
        tight = (box_hi - box_lo) / np.where(box_lo > 0, box_lo, np.nan) <= 0.15
        box_break = (close > box_hi) & tight & uptrend

        df = pd.DataFrame({
            'symbol': sym, 'date': d['date'].to_numpy(), 'uptrend': uptrend,
            'a1_ma10': a1, 'a2_ma20': a2, 'a3_ema10': a3,
            'dip_buy': dip_buy, 'recovery': recovery, 'box_break': box_break,
            'day_ret': day_ret,
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
                       pos=float((v > 0).mean()),
                       big_loss=float((v <= -0.20).mean()),
                       t=float(nwt(dly, h)), n=int(len(v)))

    res = {}
    res['ma10'] = stats(A.a1_ma10, A, '10MA 低量拉回（上升趨勢中）')
    res['ma20'] = stats(A.a2_ma20, A, '20MA 低量拉回（上升趨勢中）')
    res['ema10'] = stats(A.a3_ema10, A, 'EMA10 低量拉回（同一規則）')
    res['dip_buy'] = stats(A.dip_buy, A, '拉回中買（下跌當日跌破前低）')
    res['recovery'] = stats(A.recovery, A, '拉回後反轉買（收復前一日高點）')
    res['box_break'] = stats(A.box_break, A, 'Darvas 箱型突破（15日箱、寬度≤15%）')

    # ---------- 拉回買入法：反轉買 − 拉回中買（每日一個觀測） ----------
    diff_pb = {}
    for h in H:
        a = A.loc[A.recovery, ['date', f'f{h}']].dropna().groupby('date')[f'f{h}'].mean()
        b = A.loc[A.dip_buy, ['date', f'f{h}']].dropna().groupby('date')[f'f{h}'].mean()
        both = pd.concat([a.rename('a'), b.rename('b')], axis=1, sort=True).dropna()
        if len(both) < 40:
            continue
        dv = (both.a - both.b).to_numpy()
        rng = np.random.default_rng(11)
        boot = np.array([rng.choice(dv, len(dv), True).mean() for _ in range(1000)])
        diff_pb[h] = dict(diff=float(dv.mean()), t=float(nwt(dv, h)),
                          ci=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                          n_days=int(len(both)))

    # ---------- SMA vs EMA 差異（他的說法：沒顯著差別） ----------
    sma_ema = {}
    for h in (20, 60):
        a = A.loc[A.a1_ma10, f'f{h}'].dropna().to_numpy()
        b = A.loc[A.a3_ema10, f'f{h}'].dropna().to_numpy()
        sma_ema[h] = dict(sma=trimmed(a), ema=trimmed(b), n_sma=int(len(a)),
                          n_ema=int(len(b)),
                          diff=float(trimmed(a) - trimmed(b)),
                          overlap=float((A.a1_ma10 & A.a3_ema10).sum()
                                        / max((A.a1_ma10 | A.a3_ema10).sum(), 1)))

    # ---------- C: 牛市年齡 ----------
    mkt = A.groupby('date')['day_ret'].mean().sort_index()
    idx = (1 + mkt.fillna(0)).cumprod()
    ma200 = idx.rolling(200).mean()
    above = (idx > ma200).to_numpy()
    dates = idx.index.to_numpy()
    age = np.zeros(len(idx), int)
    run = 0
    for i, a in enumerate(above):
        run = run + 1 if a else 0
        age[i] = run
    age_s = pd.Series(age, index=idx.index)

    bull = {}
    buckets = [('<400 日', 0, 400), ('400~700 日', 400, 700),
               ('700~900 日', 700, 900), ('>900 日', 900, 10 ** 9)]
    for lab, lo, hi in buckets:
        sel = (age_s >= lo) & (age_s < hi)
        if sel.sum() < 60:
            continue
        sub = idx[sel]
        bull[lab] = dict(n_days=int(sel.sum()), by_h={})
        for h in (20, 60):
            fwd = sub.shift(-h) / sub - 1
            v = fwd.dropna().to_numpy()
            if len(v) < 40:
                continue
            bull[lab]['by_h'][h] = dict(trim=trimmed(v), median=float(np.median(v)),
                                        pos=float((v > 0).mean()),
                                        t=float(nwt(v, h)), n=int(len(v)))

    # 他說的「700~800 日左右要注意」
    sel = (age_s >= 700) & (age_s <= 800)
    near = {}
    for h in (20, 60):
        sub = idx[sel]
        fwd = (sub.shift(-h) / sub - 1).dropna().to_numpy()
        if len(fwd) >= 40:
            near[h] = dict(trim=trimmed(fwd), median=float(np.median(fwd)),
                           pos=float((fwd > 0).mean()), t=float(nwt(fwd, h)),
                           n=int(len(fwd)))

    # 全期牛市日（age>0）的市場前瞻報酬，作為牛市年齡的基準
    allbull = {}
    for h in (20, 60):
        sub = idx[age_s > 0]
        fwd = (sub.shift(-h) / sub - 1).dropna().to_numpy()
        if len(fwd) >= 40:
            allbull[h] = dict(trim=trimmed(fwd), pos=float((fwd > 0).mean()),
                              median=float(np.median(fwd)), n=int(len(fwd)))

    out = dict(baseline=base, signals=res, pullback_diff=diff_pb,
               sma_ema=sma_ema, bull_age=bull, bull_near=near,
               bull_all=allbull, age_max=int(age_s.max()))
    json.dump(out, open(f'{OUT}/batch2.json', 'w'), ensure_ascii=False, indent=1)

    # ---------- 印出 ----------
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

    print('\n【拉回買入法】反轉買 − 拉回中買：')
    for h, v in diff_pb.items():
        sig = '不含零' if not (v['ci'][0] < 0 < v['ci'][1]) else '含零'
        print(f"  {h:>2}日 {v['diff']*100:+.2f}%  CI[{v['ci'][0]*100:+.2f},{v['ci'][1]*100:+.2f}] "
              f"t={v['t']:+.2f} {sig}  n_days={v['n_days']}")

    print('\n【SMA vs EMA】他的說法：沒有顯著差別')
    for h, v in sma_ema.items():
        print(f"  {h}日: SMA {v['sma']*100:+.2f}%  EMA {v['ema']*100:+.2f}%  "
              f"差 {v['diff']*100:+.2f}pp  訊號重疊率 {v['overlap']*100:.1f}%")

    print('\n【牛市年齡】他說的 700~800 日：')
    for lab, v in bull.items():
        if not v['by_h']:
            continue
        s20 = v['by_h'].get(20, {})
        s60 = v['by_h'].get(60, {})
        print(f"  {lab:<12} n={v['n_days']:>5}  "
              f"20日 {s20.get('trim',0)*100:+.2f}% (上漲 {s20.get('pos',0)*100:.0f}%)  "
              f"60日 {s60.get('trim',0)*100:+.2f}%")
    if near:
        print(f"  700~800 日   20日 {near[20]['trim']*100:+.2f}%  "
              f"60日 {near[60]['trim']*100:+.2f}%  n={near[20]['n']}")
    if allbull:
        print(f"  全部牛市日   20日 {allbull[20]['trim']*100:+.2f}%  "
              f"60日 {allbull[60]['trim']*100:+.2f}%  n={allbull[20]['n']}")
    print(f"  史上最長牛齡 {out['age_max']} 交易日")

    print('\n已寫 batch2.json')


if __name__ == '__main__':
    main()
