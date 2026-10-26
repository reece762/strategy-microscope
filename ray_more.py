#!/usr/bin/env python3
"""Rayner 其餘主張：動量（50 週新高）與布林通道回檔。

規則來源說明
------------
字幕與影片描述都被 YouTube 封鎖（Sign in to confirm you're not a bot），
所以這裡的規則全部取自他本人網站文章，逐條對應：

  動量／股票版順勢（/the-essential-guide-to-momentum-trading/）
    - 股票創 50 週新高 → 做多
    - 20% 移動停損
    - 候選太多時，取過去 50 週漲幅最大的前 20 支
    - 最多 20 支、每支不超過 5% 資金
    他宣稱：707 筆 / 勝率 48.66% / 年化 12.81% / 最大回撤 40.75%

  布林通道（/bollinger-bands-trading-strategy/）
    文章只給原則，沒有給帶數字的回測規則；影片標題宣稱
    「Bollinger Band Pullback System (+2834% Over The Last 25 Years)」。
    原則（文章原文）：上升趨勢中，價格回到下軌是買點；20 期均線是「價值區」；
    可用 20 均線或外軌移動停損。
    我把它操作化成最貼近原文的機械版本，並在報告中標明「這是我的操作化，
    不是他的原始程式」：
      - 趨勢過濾：收盤 > 200 日均線
      - 進場：收盤跌破布林下軌（20 期、2 標準差）
      - 出場：收盤回到 20 期中軌之上，或滿 10 個交易日
      - 隔天開盤進出
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from yahoo import fetch  # noqa: E402
from ray_rsi import run as rsi_run  # noqa: E402


def tstat(x):
    x = pd.Series(x).dropna()
    if len(x) < 3 or x.std() == 0:
        return float('nan')
    return float(x.mean() / (x.std() / np.sqrt(len(x))))


# ------------------------------------------------------------ 布林回檔
def bollinger_pullback(df, ma_n=200, bb_n=20, k=2.0, max_hold=10,
                       cost_bp=1.0):
    close, op = df['close'], df['open']
    ma = close.rolling(ma_n).mean()
    mid = close.rolling(bb_n).mean()
    sd = close.rolling(bb_n).std()
    lower = mid - k * sd

    trades = []
    i = ma_n
    n = len(df)
    while i < n - 1:
        if (pd.notna(ma.iloc[i]) and pd.notna(lower.iloc[i])
                and close.iloc[i] > ma.iloc[i] and close.iloc[i] < lower.iloc[i]):
            ei = i + 1
            ep = op.iloc[ei]
            xi = None
            for j in range(ei + 1, min(ei + max_hold + 1, n)):
                if pd.notna(mid.iloc[j]) and close.iloc[j] > mid.iloc[j]:
                    xi = j + 1 if j + 1 < n else n - 1
                    break
            if xi is None:
                xi = min(ei + max_hold, n - 1)
            xp = op.iloc[xi]
            trades.append(dict(entry=df.index[ei], exit=df.index[xi],
                               ret=xp / ep - 1 - cost_bp / 1e4 * 2,
                               days=(df.index[xi] - df.index[ei]).days))
            i = xi
        else:
            i += 1
    return pd.DataFrame(trades)


def summarize(t, df, label, claimed=''):
    if t.empty:
        print(f'{label}: 无交易')
        return None
    wins = t['ret'] > 0
    yrs = (df.index[-1] - df.index[0]).days / 365.25
    ev = float(t['ret'].mean())
    out = dict(n=len(t), win=float(wins.mean()), avg=ev,
               sum=float(t['ret'].sum()), years=yrs,
               t=tstat(t['ret']))
    print(f'\n{label}')
    print(f'  交易 {len(t)} 筆（每年 {len(t)/yrs:.1f}）  勝率 {wins.mean()*100:.2f}%  '
          f'平均 {ev*100:+.2f}%')
    print(f'  平均獲利 {t["ret"][wins].mean()*100:+.2f}%  '
          f'平均虧損 {t["ret"][~wins].mean()*100:+.2f}%  '
          f'賺賠比 {abs(t["ret"][wins].mean()/t["ret"][~wins].mean()):.2f}')
    print(f'  總和（未複利）{t["ret"].sum()*100:+.1f}%   t 值 {out["t"]:.2f}'
          f'   → {"显著" if abs(out["t"])>1.96 else "不显著"}')
    if claimed:
        print(f'  宣称：{claimed}')
    return out


# ------------------------------------------------------------ 動量：50 週新高
def momentum_portfolio(syms, lookback=250, trail=0.20, top_n=20,
                       weight=0.05, cost_bp=1.0):
    """多标的动量：创 lookback 日新高者入场，20% 移动停损，每支 5% 资金。"""
    data = {}
    for s in syms:
        try:
            df = fetch(s, '30y')
            if len(df) > lookback + 50:
                data[s] = df
        except Exception:
            pass
    if not data:
        return None

    idx = sorted(set().union(*[set(d.index) for d in data.values()]))
    idx = pd.DatetimeIndex(idx)
    close = pd.DataFrame({s: d['close'] for s, d in data.items()}).reindex(idx)
    high = pd.DataFrame({s: d['high'] for s, d in data.items()}).reindex(idx)

    rollmax = close.rolling(lookback).max().shift(1)
    rollret = close / close.shift(lookback) - 1

    n, m = close.shape
    cols = list(close.columns)
    pos = np.zeros((n, m))
    state = np.zeros(m)          # 目前持倉（1=持有）
    peak = np.full(m, np.nan)

    for i in range(lookback, n):
        for j in range(m):
            px = close.iloc[i, j]
            if not np.isfinite(px):
                continue
            if state[j] == 1:
                peak[j] = max(peak[j], px)
                if px < peak[j] * (1 - trail):
                    state[j] = 0.0
            else:
                rm = rollmax.iloc[i, j]
                if np.isfinite(rm) and px > rm:
                    state[j] = 1.0
                    peak[j] = px
        pos[i] = state

    # 名額限制：同一時點最多持有 top_n 支，按 50 週漲幅排序取前
    ranked = rollret.rank(axis=1, ascending=False)
    allowed = (ranked <= top_n).to_numpy()
    pos = pos * allowed

    w = pd.DataFrame(pos * weight, index=close.index, columns=cols)
    rets = close.pct_change().fillna(0.0).to_numpy()
    held = np.vstack([np.zeros((1, m)), w.to_numpy()[:-1]])
    turn = np.abs(np.diff(np.vstack([np.zeros((1, m)), held]), axis=0))
    pnl = (held * rets).sum(axis=1) - turn.sum(axis=1) * cost_bp / 1e4
    eq = pd.Series((1 + pnl).cumprod(), index=close.index)

    # 同曝險基準
    expo = np.abs(held).sum(axis=1)
    bh_ret = np.nanmean(rets, axis=1)
    em = pd.Series((1 + bh_ret * expo.mean()).cumprod(), index=close.index)
    bh = pd.Series((1 + bh_ret).cumprod(), index=close.index)

    # 交易筆數與勝率
    trades = []
    for j in range(m):
        held_j = w.iloc[:, j].to_numpy() > 0
        ent = None
        for i in range(n):
            if held_j[i] and ent is None:
                ent = i
            elif not held_j[i] and ent is not None:
                if i > ent and np.isfinite(close.iloc[ent, j]) and np.isfinite(close.iloc[i - 1, j]):
                    trades.append(close.iloc[i - 1, j] / close.iloc[ent, j] - 1)
                ent = None
    tr = pd.Series(trades)

    yrs = (close.index[-1] - close.index[0]).days / 365.25
    cagr = lambda e: float(e.iloc[-1] ** (1 / yrs) - 1)
    v = eq.to_numpy()
    return dict(eq=eq, em=em, bh=bh, pnl=pnl, expo=float(expo.mean()),
                cagr=cagr(eq), em_cagr=cagr(em), bh_cagr=cagr(bh),
                mdd=float((v / np.maximum.accumulate(v) - 1).min()),
                n_trades=len(tr), win=float((tr > 0).mean()) if len(tr) else float('nan'),
                t=tstat(pnl), years=yrs, n_assets=m)


if __name__ == '__main__':
    SYMS = ['SPY', 'QQQ', 'IWM', 'DIA', 'GLD', 'TLT', 'EEM',
            'XLE', 'XLB', 'XLF', 'XLK', 'XLV']

    print('=' * 78)
    print('布林通道回檔（SPY 25 年）')
    print('  影片標題宣稱：+2834% / 25 年')
    print('=' * 78)
    spy = fetch('SPY', '30y')
    t = bollinger_pullback(spy)
    summarize(t, spy, '布林回檔：收盤 < 下軌 且 > 200MA → 隔天開盤買，回中軌或 10 日後出場',
              '+2834% / 25 年（影片標題）')
    if not t.empty:
        t.to_csv(os.path.join(HERE, 'out', 'ray_bb_trades.csv'), index=False)

    print('\n' + '=' * 78)
    print('動量：50 週（250 日）新高 + 20% 移動停損，最多 20 支各 5%')
    print('  他宣稱（Russell 1000）：707 筆 / 勝率 48.66% / 年化 12.81% / 回撤 40.75%')
    print('=' * 78)
    r = momentum_portfolio(SYMS)
    if r:
        print(f'\n用 12 支 ETF 當股票池（{r["n_assets"]} 支，{r["years"]:.1f} 年）')
        print(f'  年化 {r["cagr"]*100:6.2f}%   回撤 {r["mdd"]*100:6.1f}%   '
              f'平均曝險 {r["expo"]*100:.1f}%')
        print(f'  同曝險基準 {r["em_cagr"]*100:6.2f}%   等權買進持有 {r["bh_cagr"]*100:6.2f}%')
        print(f'  交易 {r["n_trades"]} 筆  勝率 {r["win"]*100:.1f}%  '
              f't 值 {r["t"]:.2f}')
        print(f'  策略 vs 同曝險：{(r["cagr"]-r["em_cagr"])*100:+.2f} pp')
