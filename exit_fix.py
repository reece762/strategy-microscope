#!/usr/bin/env python3
"""出場訊號的第二輪：修掉兩個方法錯誤，並換成「賣出的代價」。

## 第一輪結果為什麼不能直接用

第一輪跑出來幾乎所有訊號組的後續報酬都「高於」基準，方向一致得可疑。
查了程式，有兩個實質錯誤：

### 錯誤一：稀有組的平均被離群值綁架
「回檔超過 40%」只有 134 筆，平均 60 日報酬 +51.88%。
134 筆可以拉出 51% 的平均，只能是有幾檔股票漲了好幾倍（暴跌後暴漲的迷因股）。
**用平均值描述這種分布是錯的**，而且它的 t 值只有 1.43，根本不該被引用。

### 錯誤二：交叉去均值後比較子集，t 值必然虛高
原本的作法是先把個股報酬減去「同日橫斷面平均」（去均值），
再比較「強勢股」與「弱勢股」兩組的平均。
問題是：**同一天裡，這兩組的去均值報酬加權後必然為零**——
它們是同一塊餅的兩半，差異有一部分是算術上被強制的，不是市場給的。
再加上幾十萬筆重疊視窗，t 值就膨脹到 −75 這種不可能的量級。

正確做法：**每個交易日只算一個觀測值**（當天強勢股平均 − 當天弱勢股平均），
再用交易日序列算 t 值。這樣 t 值才有意義。

## 這一輪改成什麼

1. 「賣出的代價」取代「報酬差」：
   直接問「如果在訊號日賣出、h 日後買回，相對於持有不動，會少賺多少」。
   這才是投資人真正的問題，也避開去均值的算術陷阱。
2. 稀有組用**截尾平均**（去掉最高與最低 1%），不用平均值。
3. 所有 t 值都在**交易日層級**計算（每天一個觀測），
   這樣重疊視窗的問題由 Newey-West 正確處理。

## 一個必須大聲說的限制

母體是「今天還存在的 616 檔股票」。
抛物線暴漲後崩盤下市的股票**不在母體裡**，
所以「抛物線中出現不尋常下跌之後還會漲」這個結果，
**被存活者偏差系統性高估**。
真實世界裡，同樣的訊號之後有相當比例是 -50% 到 -100%。
這個限制大到足以改變結論的解讀方向，必須寫進卡片。
"""
import glob
import json
import os
from functools import lru_cache

import numpy as np
import pandas as pd

STOCK = '/root/.openclaw/stock_data'
OUT = '/root/.openclaw/workspace/backtest_videos/xbx/out'
W = 5
HORIZONS = (5, 10, 20, 60)


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


def prep(d):
    close = d['close'].to_numpy(float)
    high = d['high'].to_numpy(float)
    low = d['low'].to_numpy(float)
    op = d['open'].to_numpy(float)
    n = len(close)
    s = pd.Series(close)
    ma20 = s.rolling(20).mean().to_numpy()
    ma50 = s.rolling(50).mean().to_numpy()

    from numpy.lib.stride_tricks import sliding_window_view
    piv_low_avail = np.full(n, np.nan)
    if n > 2 * W + 1:
        win = sliding_window_view(low, 2 * W + 1)
        is_piv = low[W:n - W] == win.min(axis=1)
        for i in np.arange(W, n - W)[is_piv]:
            if i + W < n:
                piv_low_avail[i + W] = low[i]
    last_piv = pd.Series(piv_low_avail).ffill().to_numpy()

    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - np.r_[close[0], close[:-1]]),
                               np.abs(low - np.r_[close[0], close[:-1]])))
    atr20 = pd.Series(tr).rolling(20).mean().to_numpy()
    atr60 = pd.Series(tr).rolling(60).mean().to_numpy()
    prev = np.r_[np.nan, close[:-1]]
    return dict(close=close, ma20=ma20, ma50=ma50, atr20=atr20, atr60=atr60,
                last_piv=last_piv, n=n, dates=d['date'].to_numpy(),
                day_ret=close / prev - 1, gap=op / prev - 1)


def fwd(close, h):
    out = np.full(len(close), np.nan)
    out[:-h] = close[h:] / close[:-h] - 1
    return out


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
    print(f'母體 {len(stocks)} 檔', flush=True)

    frames = []
    for sym, d in stocks.items():
        p = prep(d)
        close, ma20, ma50, n = p['close'], p['ma20'], p['ma50'], p['n']
        if n < 400:
            continue
        ma50_up = np.r_[np.full(10, np.nan), (ma50[10:] > ma50[:-10]).astype(float)]
        uptrend = (close > ma50) & (ma50_up == 1)
        g20 = np.r_[np.full(20, np.nan), close[20:] / close[:-20] - 1]
        g60 = np.r_[np.full(60, np.nan), close[60:] / close[:-60] - 1]
        hi60 = pd.Series(close).rolling(60).max().to_numpy()
        dd = close / hi60 - 1
        below20 = close < ma20

        df = pd.DataFrame({
            'symbol': sym, 'date': p['dates'], 'day_ret': p['day_ret'],
            'uptrend': uptrend,
            'abnorm': (((p['gap'] <= -0.03) | (p['day_ret'] <= -0.07)) & (g20 >= 0.25)),
            'brk20': below20 & ~np.r_[False, below20[:-1]] & uptrend,
            'brk_swing': (close < p['last_piv']) & ~np.isnan(p['last_piv']) & uptrend,
            'volexp': (p['atr20'] / np.where(p['atr60'] > 0, p['atr60'], np.nan) >= 1.5)
                      & (close >= 0.85 * hi60),
            'extended': ((close / ma20 - 1 >= 0.10) & (g60 >= 0.40)),
            'dd': dd,
        })
        for h in HORIZONS:
            df[f'f{h}'] = fwd(close, h)
        frames.append(df)
    A = pd.concat(frames, ignore_index=True)
    print(f'股票日 {len(A):,}', flush=True)

    mkt = A.groupby('date')['day_ret'].mean()

    def report(mask, label, lag_h=None):
        """賣出代價：訊號日賣出、h 日後買回，相對持有不動的損失（正數＝少賺）。"""
        out = {'label': label, 'n': int(mask.sum()), 'by_h': {}}
        if mask.sum() < 40:
            out['note'] = '樣本過少'
            return out
        for h in HORIZONS:
            v = A.loc[mask, f'f{h}'].to_numpy()
            v = v[~np.isnan(v)]
            if len(v) < 40:
                continue
            # 以交易日為單位聚合（若同日多檔，取當日訊號股平均）
            sub = A.loc[mask, ['date', f'f{h}']].dropna()
            daily = sub.groupby('date')[f'f{h}'].mean()
            out['by_h'][h] = dict(
                n=int(len(v)),
                n_days=int(len(daily)),
                mean_raw=float(v.mean()),
                trim_raw=trimmed(v),
                median_raw=float(np.median(v)),
                t_daily=float(nwt(daily.to_numpy(), h)),
                pos_rate=float((v > 0).mean()),
                big_loss=float((v <= -0.20).mean()),
            )
        return out

    res = {}
    up = A['uptrend']
    res['abnorm'] = report(A['abnorm'], '抛物線中出現不尋常下跌')
    res['brk20'] = report(A['brk20'], '首次跌破 20 日均線')
    res['brk_swing'] = report(A['brk_swing'], '跌破已確認的前波低點')
    res['volexp'] = report(A['volexp'], '波幅擴張（ATR20/ATR60 ≥ 1.5）')
    res['extended'] = report(A['extended'], '延伸狀態（距 20MA ≥10%、60 日漲 ≥40%）')
    res['dd30'] = report(up & A['dd'].between(-0.36, -0.30), '上升趨勢中回檔 30~36%')
    res['dd40'] = report(up & (A['dd'] <= -0.40), '上升趨勢中回檔超過 40%')

    # 大盤下跌日：強勢 vs 弱勢（每天一個觀測，避免去均值的算術陷阱）
    md = A['date'].map(mkt) <= -0.01
    for h in HORIZONS:
        strong = A.loc[md & (A['dd'] > -0.15)].groupby('date')[f'f{h}'].mean()
        weak = A.loc[md & (A['dd'] <= -0.30)].groupby('date')[f'f{h}'].mean()
        both = pd.concat([strong.rename('s'), weak.rename('w')], axis=1).dropna()
        if h == HORIZONS[0]:
            res['mktdown'] = {'label': '大盤跌 ≥1% 當日：強勢股 − 弱勢股', 'by_h': {}}
        if len(both) >= 40:
            d = (both.s - both.w).to_numpy()
            res['mktdown']['by_h'][h] = dict(
                n_days=int(len(both)),
                strong=float(both.s.mean()), weak=float(both.w.mean()),
                diff=float(d.mean()), t_daily=float(nwt(d, h)))

    json.dump(res, open(f'{OUT}/exit_signals_fix.json', 'w'), ensure_ascii=False, indent=1)

    print(f"\n{'訊號（賣出代價：正數＝賣早了，少賺）':<36}{'n':>8}{'5日':>9}{'20日':>9}{'60日':>9}")
    print('-' * 76)
    for k in ('abnorm', 'brk20', 'brk_swing', 'volexp', 'extended', 'dd30', 'dd40'):
        v = res[k]
        if 'note' in v:
            print(f"{v['label']:<36}{v['n']:>8}  ({v['note']})")
            continue
        line = f"{v['label']:<36}{v['n']:>8,}"
        for h in (5, 20, 60):
            b = v['by_h'].get(h)
            line += f" {b['trim_raw']*100:>+8.2f}%" if b else f" {'—':>9}"
        print(line)

    print('\n細部（20 日）：')
    for k in ('abnorm', 'brk20', 'brk_swing', 'volexp', 'extended', 'dd30', 'dd40'):
        v = res[k]
        b = v['by_h'].get(20)
        if not b:
            continue
        print(f"  {v['label']:<32} 截尾 {b['trim_raw']*100:+.2f}%  中位 {b['median_raw']*100:+.2f}%  "
              f"上漲機率 {b['pos_rate']*100:.1f}%  t(日)={b['t_daily']:+.2f}  "
              f"跌逾20%機率 {b['big_loss']*100:.1f}%")

    m = res['mktdown']['by_h'].get(20)
    if m:
        print(f"\n  大盤跌日 強−弱（20日）: 強 {m['strong']*100:+.2f}%  弱 {m['weak']*100:+.2f}%  "
              f"差 {m['diff']*100:+.2f}%  t(日)={m['t_daily']:+.2f}  n_days={m['n_days']}")
    print('\n已寫 exit_signals_fix.json')


if __name__ == '__main__':
    main()
