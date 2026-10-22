#!/usr/bin/env python3
"""驗證 B 站 UP 主「阿尔法量化价格行为」開源碼裡的策略。

來源：github.com/will2085991040-ui/density-sr
- aiquant/backtest/strategies.py  六個回測策略（一字不改照抄）
- aiquant/engine/signals.py       五票制 PA 訊號引擎

目的：不看影片標題，直接把他寫的規則跑在自己的資料上。
"""
import glob
import io
import os
import sys
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DAILY = os.path.join(os.path.dirname(HERE), 'data_daily')
DATA = os.path.join(HERE, 'data')


def load_etf(sym):
    df = pd.read_csv(os.path.join(DAILY, f'{sym}.csv'))
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
    for c in ['open', 'high', 'low', 'close', 'volume']:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df.set_index('date').sort_index().dropna(subset=['close'])


def load_btc():
    frames = []
    for f in sorted(glob.glob(os.path.join(DATA, 'BTCUSDT-1d-*.zip'))):
        with zipfile.ZipFile(f) as z:
            raw = z.read(z.namelist()[0]).decode()
        d = pd.read_csv(io.StringIO(raw), header=None).iloc[:, :6]
        d.columns = ['ts', 'open', 'high', 'low', 'close', 'volume']
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    ts = pd.to_numeric(d['ts'], errors='coerce')
    ts = ts.where(ts < 1e14, ts / 1000.0)
    d['date'] = pd.to_datetime(ts, unit='ms').dt.normalize()
    for c in ['open', 'high', 'low', 'close', 'volume']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    return d.dropna(subset=['close']).drop_duplicates('date') \
            .set_index('date').sort_index()


# ---------------------------------------------------------------- 指標
def sma(s, n):
    return s.rolling(n).mean()


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def macd_series(s):
    dif = ema(s, 12) - ema(s, 26)
    dea = ema(dif, 9)
    return dif, dea, (dif - dea) * 2


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def bollinger(s, n=20, k=2.0):
    mid = s.rolling(n).mean()
    sd = s.rolling(n).std(ddof=0)
    return mid + k * sd, mid, mid - k * sd


def atr(df, n=14):
    h, l, c = df['high'], df['low'], df['close']
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


# ------------------------------------------- 他的六個策略（原樣照抄）
def s_ma_cross(df, fast=5, slow=20):
    return (sma(df['close'], fast) > sma(df['close'], slow)).astype(int)


def s_macd(df):
    dif, dea, _ = macd_series(df['close'])
    return (dif > dea).astype(int)


def s_rsi_mean(df, period=14):
    return (rsi(df['close'], period) < 50).astype(int)


def s_rsi_band(df, period=14, lo=30, hi=70):
    r = rsi(df['close'], period)
    pos = np.zeros(len(df))
    hold = 0
    for i in range(len(df)):
        v = r.iloc[i]
        if np.isnan(v):
            pos[i] = hold
        elif v < lo:
            hold = 1
        elif v > hi:
            hold = 0
        pos[i] = hold
    return pd.Series(pos, index=df.index)


def s_breakout(df, n=20):
    hh = df['high'].rolling(n).max().shift(1)
    return (df['close'] > hh).astype(int)


def s_boll_rev(df, window=20, k=2.0):
    up, mid, lo = bollinger(df['close'], window, k)
    c = df['close']
    pos = np.zeros(len(df))
    hold = 0
    for i in range(len(df)):
        if np.isnan(mid.iloc[i]):
            pos[i] = hold
        else:
            if c.iloc[i] < lo.iloc[i]:
                hold = 1
            elif c.iloc[i] > up.iloc[i]:
                hold = 0
        pos[i] = hold
    return pd.Series(pos, index=df.index)


STRATS = {
    '双均线交叉(5/20)': s_ma_cross,
    'MACD金叉': s_macd,
    'RSI均值回归(<50持有)': s_rsi_mean,
    'RSI超买超卖(30/70)': s_rsi_band,
    '唐奇安突破(20)': s_breakout,
    '布林回归(20,2)': s_boll_rev,
}


# ---------------------------------------------------------------- 回測
def backtest(df, fn, cost):
    """照他的設計：訊號用當日收盤，部位從隔日生效（pos 前移一格）。"""
    c = df['close'].values
    pos = fn(df).values
    ret = np.diff(c) / c[:-1]
    held = pos[:-1]
    r = np.where(held > 0, ret, 0.0)
    turn = np.abs(np.diff(np.concatenate([[0], held])))
    strat = r - turn * cost
    eq = np.cumprod(1 + strat)
    n = len(strat)
    yrs = max(n / 252, 1e-9)
    cagr = eq[-1] ** (1 / yrs) - 1 if eq[-1] > 0 else -1
    mdd = float((eq / np.maximum.accumulate(eq) - 1).min())
    vol = float(strat.std(ddof=1) * np.sqrt(252))
    sharpe = float(strat.mean() / strat.std(ddof=1) * np.sqrt(252)) if strat.std() > 0 else 0
    trades = int((np.diff(np.concatenate([[0], (held > 0).astype(int)])) > 0).sum())
    expo = float((held > 0).mean())
    return dict(cagr=cagr, mdd=mdd, vol=vol, sharpe=sharpe, trades=trades,
                expo=expo, eq=eq, n=n)


def buyhold(df):
    c = df['close'].values
    ret = np.diff(c) / c[:-1]
    eq = np.cumprod(1 + ret)
    yrs = max(len(ret) / 252, 1e-9)
    return dict(cagr=eq[-1] ** (1 / yrs) - 1,
                mdd=float((eq / np.maximum.accumulate(eq) - 1).min()),
                vol=float(ret.std(ddof=1) * np.sqrt(252)),
                sharpe=float(ret.mean() / ret.std(ddof=1) * np.sqrt(252)),
                n=len(ret))


# --------------------------------------- 他的五票制 PA 引擎（照抄，含原 bug）
def votes_series(df, fix_trend_bar=False):
    """逐日重算他 signals.py 的 _votes()，回傳方向票總和 s。"""
    c = df['close']
    e20 = ema(c, 20)
    h, l = df['high'], df['low']
    o = df['open']
    s_ema = np.sign(e20 - e20.shift(10))
    ma20 = c.rolling(20).mean()
    s_ma = np.sign(c - ma20)
    hh = h > h.rolling(24).max().shift(1)
    ll = l < l.rolling(24).min().shift(1)
    s_hh = np.where(hh, 1, np.where(ll, -1, 0))
    upr = (c > o).rolling(10).mean()
    if fix_trend_bar:
        s_bar = np.where(upr >= 0.6, 1, np.where(upr <= 0.4, -1, 0))
    else:
        # 原文：v = -1 if upr >= 0.6 else (-1 if upr <= 0.4 else 0)  ← 兩支都 -1
        s_bar = np.where((upr >= 0.6) | (upr <= 0.4), -1, 0)
    rng = (h - l).rolling(9).sum()
    ov = (h.shift(1) - l).abs().rolling(9).sum()
    ratio = ov / rng.replace(0, np.nan)
    s_ov = np.where(ratio > 0.65, -1, np.where(ratio < 0.45, 1, 0))
    s = (pd.Series(s_ema, index=c.index).fillna(0)
         + pd.Series(s_ma, index=c.index).fillna(0)
         + pd.Series(s_hh, index=c.index).fillna(0)
         + pd.Series(s_bar, index=c.index).fillna(0)
         + pd.Series(s_ov, index=c.index).fillna(0))
    return s


def pa_position(df, thresh=3, fix=False):
    s = votes_series(df, fix_trend_bar=fix)
    return (s >= thresh).astype(int)


def main():
    sets = {}
    for sym in ('SPY', 'QQQ'):
        sets[sym] = load_etf(sym)
    sets['BTC'] = load_btc()
    costs = {'SPY': 0.0005, 'QQQ': 0.0005, 'BTC': 0.0025}

    print('=' * 78)
    print('他的六個回測策略（原樣照抄）vs 買進持有')
    print('=' * 78)
    for sym, df in sets.items():
        bh = buyhold(df)
        print(f"\n--- {sym}  {df.index[0].date()} ~ {df.index[-1].date()}  "
              f"{len(df)} 日 ---")
        print(f"  {'策略':26s} {'年化':>8s} {'回撤':>8s} {'夏普':>6s} "
              f"{'換手':>5s} {'曝險':>6s}   vs 買持")
        print(f"  {'買進持有':26s} {bh['cagr']*100:7.2f}% {bh['mdd']*100:7.1f}% "
              f"{bh['sharpe']:6.2f} {'-':>5s} {100:5.1f}%")
        for name, fn in STRATS.items():
            r = backtest(df, fn, costs[sym])
            print(f"  {name:26s} {r['cagr']*100:7.2f}% {r['mdd']*100:7.1f}% "
                  f"{r['sharpe']:6.2f} {r['trades']:5d} {r['expo']*100:5.1f}% "
                  f"  {r['cagr']-bh['cagr']:+.1%}")

    print('\n' + '=' * 78)
    print('他的五票制 PA 引擎（原樣，含 bug）vs 修正後')
    print('=' * 78)
    for sym, df in sets.items():
        bh = buyhold(df)
        print(f"\n--- {sym} ---")
        for lbl, fix in (('原樣（趨勢棒票永遠 -1）', False),
                         ('修正趨勢棒票後', True)):
            pos = pa_position(df, 3, fix)
            r = backtest(df, lambda d, p=pos: p, costs[sym])
            print(f"  {lbl:24s} 年化 {r['cagr']*100:6.2f}% 回撤 {r['mdd']*100:6.1f}% "
                  f"夏普 {r['sharpe']:5.2f} 曝險 {r['expo']*100:5.1f}% "
                  f"進場 {r['trades']:3d} 次   vs 買持 {r['cagr']-bh['cagr']:+.1%}")

    # 票數分布：看他的「看多」到底多罕見
    print('\n' + '=' * 78)
    print('投票總分 s 的分布（為什麼「看多」很難出現）')
    print('=' * 78)
    for sym, df in sets.items():
        for lbl, fix in (('原樣', False), ('修正', True)):
            s = votes_series(df, fix_trend_bar=fix)
            d = s.dropna()
            n = len(d)
            print(f"  {sym:4s} {lbl}  s>=3(看多) {(d>=3).mean()*100:5.1f}%  "
                  f"s<=-3(看空) {(d<=-3).mean()*100:5.1f}%  "
                  f"中性 {(d.abs()<3).mean()*100:5.1f}%   "
                  f"s 最大值 {d.max():.0f} 最小值 {d.min():.0f}")


if __name__ == '__main__':
    main()
