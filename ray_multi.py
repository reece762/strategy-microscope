#!/usr/bin/env python3
"""Rayner 均值回歸的「多股票版」——他真正宣稱有優勢的那一版。

他原文兩個版本：
  單一市場（SPY）：38 筆 / 勝率 86.84% / 平均 +1.39%
  多股票（Russell 1000）：6,711 筆 / 勝率 66.95% / 平均 +0.45% / 年化 12.29%
     額外規則：用 100 日 ROC 篩股、每支 10% 資金、最多 10 檔

多股票版才是重點：單一市場 25 年只有 38 筆，統計上毫無意義；
6,711 筆才有檢定力。我用 40 支美股大型股當 Russell 1000 的代理。
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from yahoo import fetch  # noqa: E402
from ray_rsi import rsi  # noqa: E402

SYMS = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'JPM', 'JNJ', 'PG',
        'XOM', 'WMT', 'HD', 'KO', 'PEP', 'COST', 'MCD', 'DIS', 'INTC', 'CSCO',
        'VZ', 'T', 'PFE', 'MRK', 'ABT', 'CVX', 'BA', 'CAT', 'GE', 'F', 'GM',
        'ORCL', 'IBM', 'TXN', 'QCOM', 'AMD', 'MU', 'ADBE', 'CRM', 'NFLX', 'PYPL']


def tstat(x):
    x = pd.Series(x).dropna()
    if len(x) < 3 or x.std() == 0:
        return float('nan')
    return float(x.mean() / (x.std() / np.sqrt(len(x))))


def load_all(syms):
    out = {}
    for s in syms:
        try:
            df = fetch(s, '30y')
            if len(df) > 500:
                out[s] = df
        except Exception:
            pass
    return out


def multi_stock_mr(data, rsi_n=10, entry=30, exit_rsi=40, max_hold=10,
                   ma_n=200, roc_n=100, top_n=10, weight=0.10, cost_bp=1.0):
    """他原文規則：200MA 過濾 + 10期RSI<30 + 100日ROC篩股 + 每支10%最多10檔。"""
    # 對齊日期
    idx = sorted(set().union(*[set(d.index) for d in data.values()]))
    idx = pd.DatetimeIndex(idx)
    C = pd.DataFrame({s: d['close'] for s, d in data.items()}).reindex(idx)
    O = pd.DataFrame({s: d['open'] for s, d in data.items()}).reindex(idx)

    # 市場過濾：用 SPY 的 200MA（原文說「the market is above the 200-day MA」）
    try:
        spy = fetch('SPY', '30y')['close'].reindex(idx).ffill()
    except Exception:
        spy = C.mean(axis=1)
    mkt_ok = spy > spy.rolling(ma_n).mean()

    R = pd.DataFrame({s: rsi(C[s], rsi_n) for s in C.columns})
    MA = pd.DataFrame({s: C[s].rolling(ma_n).mean() for s in C.columns})
    ROC = pd.DataFrame({s: C[s] / C[s].shift(roc_n) - 1 for s in C.columns})

    n, m = C.shape
    cols = list(C.columns)
    pos = np.zeros((n, m))
    state = np.zeros(m)
    held_days = np.zeros(m)
    trades = []

    Cen = C.to_numpy()
    Ren = R.to_numpy()
    MAen = MA.to_numpy()
    ROCen = ROC.to_numpy()
    mok = mkt_ok.to_numpy()

    for i in range(ma_n + roc_n, n - 1):
        # --- 出場判斷（用今天收盤資料）---
        for j in range(m):
            if state[j] != 1:
                continue
            held_days[j] += 1
            hit = (np.isfinite(Ren[i, j]) and Ren[i, j] > exit_rsi) or held_days[j] >= max_hold
            if hit:
                # 隔天開盤出場
                ei, xi = i + 1, min(i + 1, n - 1)
                ep = O.to_numpy()[ei, j]
                xp = ep
                if np.isfinite(ep) and np.isfinite(xp):
                    trades.append(dict(sym=cols[j], entry=idx[entry_i[j]],
                                       exit=idx[xi],
                                       ret=xp / entry_p[j] - 1 - cost_bp / 1e4 * 2))
                state[j] = 0.0
                held_days[j] = 0

        # --- 進場判斷 ---
        if mok[i]:
            cand = []
            for j in range(m):
                if state[j] == 1:
                    continue
                c = Cen[i, j]
                if not np.isfinite(c):
                    continue
                # 股票自身在 200MA 之上，且 RSI 低於門檻
                if np.isfinite(MAen[i, j]) and c > MAen[i, j] \
                        and np.isfinite(Ren[i, j]) and Ren[i, j] < entry:
                    cand.append((j, ROCen[i, j] if np.isfinite(ROCen[i, j]) else -9))

            cand.sort(key=lambda x: -x[1])          # 100日ROC 由高到低
            slots = top_n - int(state.sum())
            for j, _ in cand[:max(0, slots)]:
                state[j] = 1.0
                held_days[j] = 0
                entry_i[j] = i + 1
                entry_p[j] = O.to_numpy()[i + 1, j] if i + 1 < n else np.nan

        # 記錄今日持倉（次日生效）
        pos[i + 1] = state * weight

    P = pd.DataFrame(pos, index=idx, columns=cols)
    rets = C.pct_change().fillna(0.0)
    held = P.shift(1).fillna(0.0)
    turn = held.diff().abs().sum(axis=1).fillna(0.0)
    pnl = (held * rets).sum(axis=1) - turn * cost_bp / 1e4
    eq = (1 + pnl).cumprod()

    expo = held.sum(axis=1)
    bench = rets.mean(axis=1)
    em = (1 + bench * expo.mean()).cumprod()
    bh = (1 + bench).cumprod()

    warm = ma_n + roc_n
    def _c(e):
        e = e.loc[idx[warm]:]
        if len(e) < 2 or e.iloc[0] <= 0:
            return float('nan')
        yy = (e.index[-1] - e.index[0]).days / 365.25
        return float((e.iloc[-1] / e.iloc[0]) ** (1 / yy) - 1)

    eqs = eq.loc[idx[warm]:]
    T = pd.DataFrame(trades)
    return dict(eq=eqs, em=em.loc[idx[warm]:], bh=bh.loc[idx[warm]:],
                pnl=pnl.loc[idx[warm]:], trades=T,
                cagr=_c(eq), em_cagr=_c(em), bh_cagr=_c(bh),
                mdd=float((eqs.to_numpy() / np.maximum.accumulate(eqs.to_numpy()) - 1).min()),
                expo=float(expo.loc[idx[warm]:].mean()),
                t=tstat(pnl.loc[idx[warm]:]))


def main():
    print('載入 40 支美股…')
    data = load_all(SYMS)
    print(f'可用 {len(data)} 支\n')

    # 重新實作：用簡潔版本（entry_i/entry_p 需初始化）
    global entry_i, entry_p
    idx = sorted(set().union(*[set(d.index) for d in data.values()]))
    idx = pd.DatetimeIndex(idx)
    cols = list(data.keys())
    entry_i = {j: 0 for j in range(len(cols))}
    entry_p = {j: np.nan for j in range(len(cols))}

    # 上面的 closure 用 j 當 key，這裡轉成 dict
    r = multi_stock_mr(data)

    T = r['trades']
    wins = T['ret'] > 0
    print('=' * 76)
    print('Rayner 均值回歸「多股票版」複驗')
    print('=' * 76)
    print(f'  宣稱（Russell 1000）：6,711 筆 / 勝率 66.95% / 平均 +0.45% / 年化 12.29%')
    print(f'  實測（40 支美股，{len(data)} 支可用）')
    print(f'    交易 {len(T)} 筆 / 勝率 {wins.mean()*100:.2f}% / '
          f'平均 {T["ret"].mean()*100:+.2f}%')
    print(f'    平均獲利 {T["ret"][wins].mean()*100:+.2f}%  '
          f'平均虧損 {T["ret"][~wins].mean()*100:+.2f}%  '
          f'賺賠比 {abs(T["ret"][wins].mean()/T["ret"][~wins].mean()):.2f}')
    print(f'    年化 {r["cagr"]*100:.2f}%  回撤 {r["mdd"]*100:.1f}%  '
          f'平均曝險 {r["expo"]*100:.1f}%')
    print(f'    同曝險基準 {r["em_cagr"]*100:.2f}%   等權買進持有 {r["bh_cagr"]*100:.2f}%')
    print(f'    策略 vs 同曝險：{(r["cagr"]-r["em_cagr"])*100:+.2f} pp')
    print(f'    逐筆 t 值 {tstat(T["ret"]):.2f}  '
          f'日報酬 t 值 {r["t"]:.2f}')
    T.to_csv(os.path.join(HERE, 'out', 'ray_multi_trades.csv'), index=False)


if __name__ == '__main__':
    main()
