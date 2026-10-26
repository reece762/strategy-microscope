#!/usr/bin/env python3
"""出場訊號的預測力：把 J Law 的出場系列寫成程式檢驗。

## 這一組跟前三張卡有什麼不同

前三張卡問「進場後賺不賺」，需要一個市場基準；
而基準一換，結論就從 −0.02% 變 +0.03%（見 1.19.0 的更正）。

**這一組問的是「同一批持股，出現訊號後是否比沒出現更差」——純粹的相對比較，
不需要任何基準。** 這是我特意選這個設計的原因：它不會被基準爭議污染。

## 五支影片、五組可檢驗的主張

1. 見頂（4UJvLrOYAMg）：
   「不尋常走勢」＝裂口＋大幅下跌且沒有收回，代表抛物線走勢出問題。
   可測：抛物線上漲中出現這種日子後，後續報酬是否變差。

2. 延伸（yIZOvhDgKRg）：
   **「延伸不是罪過」**，要看 context。45 度角可以一路持續。
   可測：延伸狀態下買進，後續報酬是否真的不差。（這是唯一「不該賣」的主張）

3. 轉弱（YEqkRFxN1IU）：
   從高點跌 30~36% 但 50MA 仍有支持 → 仍屬前半段（可接受）；
   跌超過 40% → 不正常；在 50/63 天線之下越久越不正常。
   可測：回檔深度分組的後續報酬。

4. 阻力位（_lUJIHcbZBc）：
   到前浪頂該不該賣，取決於「用什麼形式到達」：
   V 形急漲、無整固、高開低走 → 賣；
   VCP（波幅逐次收窄 15%→10%→7%、量縮後放量突破）→ **不該賣，要 go big**。
   可測：同一位置、兩種到達形式，後續報酬差多少。

5. 大市拖累（Myh-R2oald4）：
   大盤跌到半年新低時，強勢股被拖累是正常，要看結構有沒有壞。
   可測：大盤下跌日時，強勢股 vs 弱勢股的後續相對表現。

## 統計設計

每個訊號日 k 只用 k 及之前的資料判定（擺動點需右側 5 根確認），
報酬從 k 收盤起算 h 日。用 Newey-West（滯後 = h）處理重疊視窗造成的序列相關。
所有「差值」都給拔靴信賴區間，因為最終要判斷的是「有沒有差」而不是「差多少」。
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

STOCK = '/root/.openclaw/stock_data'
OUT = '/root/.openclaw/workspace/backtest_videos/xbx/out'
W = 5          # 擺動點確認根數
HORIZONS = (5, 10, 20, 60)
MAXHOLD = 120


def load():
    cols = {}
    for f in sorted(glob.glob(f'{STOCK}/live/*.csv') + glob.glob(f'{STOCK}/extra/*.csv')):
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        d.columns = [c.lower() for c in d.columns]
        need = {'date', 'open', 'high', 'low', 'close', 'volume'}
        if not need.issubset(d.columns) or len(d) < 400:
            continue
        d['date'] = pd.to_datetime(d['date'])
        d = d.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
        cols[os.path.basename(f).replace('.csv', '')] = d
    return cols


def prep(d):
    close = d['close'].to_numpy(float)
    high = d['high'].to_numpy(float)
    low = d['low'].to_numpy(float)
    op = d['open'].to_numpy(float)
    vol = d['volume'].to_numpy(float)
    n = len(close)

    s = pd.Series(close)
    ma20 = s.rolling(20).mean().to_numpy()
    ma50 = s.rolling(50).mean().to_numpy()

    tr = np.maximum(high - low, np.maximum(np.abs(high - np.r_[close[0], close[:-1]]),
                                           np.abs(low - np.r_[close[0], close[:-1]])))
    atr20 = pd.Series(tr).rolling(20).mean().to_numpy()
    atr60 = pd.Series(tr).rolling(60).mean().to_numpy()

    # 已確認的擺動低點：i 是低點、i+W 才可用
    from numpy.lib.stride_tricks import sliding_window_view
    piv_low = np.full(n, np.nan)
    if n > 2 * W + 1:
        win = sliding_window_view(low, 2 * W + 1)
        is_piv = low[W:n - W] == win.min(axis=1)
        idx = np.arange(W, n - W)[is_piv]
        piv_low[idx] = low[idx]
    # 可用時點 = 低點索引 + W
    piv_avail = np.full(n, np.nan)
    for i in np.where(~np.isnan(piv_low))[0]:
        j = i + W
        if j < n:
            piv_avail[j] = piv_low[i]
    last_piv = pd.Series(piv_avail).ffill().to_numpy()

    # 已確認的擺動高點（供阻力位判定）
    piv_hi = np.full(n, np.nan)
    if n > 2 * W + 1:
        winh = sliding_window_view(high, 2 * W + 1)
        is_ph = high[W:n - W] == winh.max(axis=1)
        idxh = np.arange(W, n - W)[is_ph]
        piv_hi[idxh] = high[idxh]
    piv_hi_avail = np.full(n, np.nan)
    for i in np.where(~np.isnan(piv_hi))[0]:
        j = i + W
        if j < n:
            piv_hi_avail[j] = piv_hi[i]

    prev_close = np.r_[np.nan, close[:-1]]
    day_ret = close / prev_close - 1
    gap = op / prev_close - 1

    return dict(close=close, high=high, low=low, op=op, vol=vol, n=n,
                ma20=ma20, ma50=ma50, atr20=atr20, atr60=atr60,
                last_piv=last_piv, piv_hi_avail=piv_hi_avail,
                day_ret=day_ret, gap=gap, dates=d['date'].to_numpy())


def fwd(close, h):
    out = np.full(len(close), np.nan)
    out[:-h] = close[h:] / close[:-h] - 1
    return out


def nwt(e, lag):
    e = e[~np.isnan(e)]
    n = len(e)
    if n < 30:
        return np.nan
    mu = e.mean()
    x = e - mu
    s = (x ** 2).mean()
    for l in range(1, min(lag, n - 1) + 1):
        s += 2 * (1 - l / (lag + 1)) * (x[l:] * x[:-l]).mean()
    return mu / np.sqrt(max(s, 1e-18) / n)


def main():
    stocks = load()
    print(f'母體 {len(stocks)} 檔', flush=True)

    # 逐檔算出所有日層級標記，最後併成一張大表
    rows = []
    for sym, d in stocks.items():
        p = prep(d)
        close, ma20, ma50 = p['close'], p['ma20'], p['ma50']
        n = p['n']
        ok = ~np.isnan(ma50) & ~np.isnan(ma20)
        if ok.sum() < 300:
            continue

        # 上升趨勢狀態：收盤在 50MA 上、且 50MA 本身在上升
        ma50_up = np.r_[np.full(10, np.nan), (ma50[10:] > ma50[:-10]).astype(float)]
        uptrend = (close > ma50) & (ma50_up == 1)

        # 抛物線：20 日漲幅 ≥ 25%
        g20 = np.r_[np.full(20, np.nan), close[20:] / close[:-20] - 1]
        parabolic = g20 >= 0.25

        # 訊號 1：不尋常下跌（裂口 ≤ −3% 或單日 ≤ −7%），且處於抛物線
        abnorm = ((p['gap'] <= -0.03) | (p['day_ret'] <= -0.07)) & parabolic

        # 訊號 2：首次跌破 20MA（前一日仍在之上）
        below20 = close < ma20
        first_break20 = below20 & ~np.r_[False, below20[:-1]] & uptrend

        # 訊號 3：跌破已確認的前波低點
        brk_swing = (close < p['last_piv']) & ~np.isnan(p['last_piv']) & uptrend

        # 訊號 4：波幅擴張（ATR20/ATR60 ≥ 1.5）且接近高點
        vol_exp = (p['atr20'] / np.where(p['atr60'] > 0, p['atr60'], np.nan) >= 1.5)
        near_high = close >= 0.85 * pd.Series(close).rolling(60).max().to_numpy()

        # 延伸：距 20MA ≥ 10% 且 60 日漲幅 ≥ 40%
        g60 = np.r_[np.full(60, np.nan), close[60:] / close[:-60] - 1]
        extended = (close / ma20 - 1 >= 0.10) & (g60 >= 0.40)

        # 回檔深度（從 60 日高點）
        hi60 = pd.Series(close).rolling(60).max().to_numpy()
        dd = close / hi60 - 1

        rec = dict(
            symbol=sym, dates=p['dates'], day_ret=p['day_ret'],
            uptrend=uptrend & ok, abnorm=abnorm & ok, brk20=first_break20,
            brk_swing=brk_swing & ok, volexp=vol_exp & near_high & ok,
            extended=extended & ok, dd=dd,
            ma50_ok=(close > ma50),
        )
        for h in HORIZONS:
            rec[f'f{h}'] = fwd(close, h)
        rows.append(rec)

    # 攤平成 DataFrame
    frames = []
    for r in rows:
        n = len(r['dates'])
        df = pd.DataFrame({
            'symbol': r['symbol'],
            'date': r['dates'],
            'day_ret': r['day_ret'],
            'uptrend': r['uptrend'],
            'abnorm': r['abnorm'],
            'brk20': r['brk20'],
            'brk_swing': r['brk_swing'],
            'volexp': r['volexp'],
            'extended': r['extended'],
            'dd': r['dd'],
        })
        for h in HORIZONS:
            df[f'f{h}'] = r[f'f{h}']
        frames.append(df)
    A = pd.concat(frames, ignore_index=True)
    print(f'總樣本 {len(A):,} 個股票日', flush=True)

    # 同期橫斷面基準（用來算超額，僅供參考；主要結論用組間差值，不需基準）
    for h in HORIZONS:
        bm = A.groupby('date')[f'f{h}'].transform('mean')
        A[f'x{h}'] = A[f'f{h}'] - bm

    res = {}

    def compare(mask, label, base_mask=None):
        base_mask = A['uptrend'] if base_mask is None else base_mask
        out = {'label': label, 'n': int(mask.sum()), 'n_base': int(base_mask.sum())}
        for h in HORIZONS:
            s = A.loc[mask, f'x{h}'].dropna()
            b = A.loc[base_mask, f'x{h}'].dropna()
            if len(s) < 30:
                continue
            diff = s.mean() - b.mean()
            # 拔靴差值
            rng = np.random.default_rng(42)
            ds = [rng.choice(s, len(s), True).mean() - rng.choice(b, len(b), True).mean()
                  for _ in range(400)]
            ds = np.array(ds)
            out[f'h{h}'] = dict(
                sig_excess=float(s.mean()), base_excess=float(b.mean()),
                diff=float(diff), t=float(nwt(s.to_numpy() - b.mean(), h)),
                ci=[float(np.quantile(ds, .025)), float(np.quantile(ds, .975))],
                sig_raw=float(A.loc[mask, f'f{h}'].mean()),
                base_raw=float(A.loc[base_mask, f'f{h}'].mean()),
            )
        return out

    # ---- 1. 見頂：抛物線中的不尋常下跌
    res['top_abnormal'] = compare(A['abnorm'], '抛物線中出現不尋常下跌')
    # ---- 2. 轉弱：回檔深度分組
    up = A['uptrend']
    res['dd_30_36'] = compare(up & A['dd'].between(-0.36, -0.30), '回檔 30~36%（50MA 仍支撐）')
    res['dd_40plus'] = compare(up & (A['dd'] <= -0.40), '回檔超過 40%')
    # ---- 3. 首破 20MA
    res['break20'] = compare(A['brk20'], '首次跌破 20 日均線')
    # ---- 4. 跌破前波低點
    res['break_swing'] = compare(A['brk_swing'], '跌破已確認的前波低點')
    # ---- 5. 波幅擴張
    res['vol_exp'] = compare(A['volexp'], '波幅擴張（ATR20/ATR60 ≥ 1.5）')
    # ---- 6. 延伸（不該賣的主張）
    res['extended'] = compare(A['extended'], '延伸狀態（距 20MA ≥10%、60日漲 ≥40%）')
    # ---- 7. 大盤下跌日：強勢 vs 弱勢
    mkt = A.groupby('date')['day_ret'].mean()
    mkt_down = A['date'].map(mkt) <= -0.01
    strong = A['date'].map(mkt).notna()
    res['mktdown_strong'] = compare(mkt_down & (A['dd'] > -0.15), '大盤跌 ≥1% 當日：強勢股',
                                   base_mask=mkt_down & (A['dd'] <= -0.30))

    json.dump(res, open(f'{OUT}/exit_signals.json', 'w'), ensure_ascii=False, indent=1)

    print(f"\n{'訊號':<34}{'n':>9} {'5日差':>8} {'10日差':>8} {'20日差':>8} {'60日差':>8}")
    for k, v in res.items():
        if 'h20' not in v:
            continue
        line = f"{v['label']:<34}{v['n']:>9,}"
        for h in HORIZONS:
            if f'h{h}' in v:
                line += f" {v[f'h{h}']['diff']*100:>+7.2f}%"
            else:
                line += f" {'—':>8}"
        print(line)
    print('\n（差值＝訊號日的後續報酬 − 同期上升趨勢日的後續報酬，負值代表訊號後確實較差）')
    print('已寫 exit_signals.json')


if __name__ == '__main__':
    main()
