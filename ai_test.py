#!/usr/bin/env python3
"""AI 寫交易系統能不能用？用「樣本內選最好、樣本外見真章」直接測。

為什麼這樣設計
--------------
Humbled Trader（92 萬觀看）與 SMB Capital 的影片都在示範
「叫 AI 產出策略 → 回測 → 看起來很賺」。這個流程最致命的問題不是 AI 會不會寫程式，
而是它預設「回測好看 ＝ 策略有效」。所以真正該測的不是模型能力，而是
**同一套流程產生的策略，在沒看過的資料上還剩多少**。

做法（完全機械、可重現）
  1. 用 SPY 日線，前 70% 當樣本內、後 30% 當樣本外。
  2. 建立 5 個策略家族，每家族掃一批參數（合計 240 組）——
     這模擬「請 AI 試幾種寫法」的真實行為。
  3. 每個家族挑「樣本內 Sharpe 最高」的那組（AI 與使用者都會這樣挑）。
  4. 把挑出來的拿到樣本外跑，看衰減多少。
  5. 順便統計：樣本內有多少組看起來「很賺」，藉此量化多重檢定的規模。

如果 AI 真的能寫出有用的系統，那麼「樣本內最佳」在樣本外也該站得住。
站不住，就證明那些影片展示的是選參數的運氣。
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from yahoo import fetch  # noqa: E402

COST_BP = 1.0
PPY = 252


def _sharp(pnl):
    pnl = pd.Series(pnl).dropna()
    sd = pnl.std()
    if len(pnl) < 20 or sd == 0 or not np.isfinite(sd):
        return float('nan')
    return float(pnl.mean() / sd * np.sqrt(PPY))


def _cagr(eq):
    """區間年化。必須先把切片 rebase 到 1.0 —— 直接拿累積淨值的切片算，
    起點不是 1，會把前面幾年的漲幅也算進這一區間，得出離譜的年化。"""
    if eq is None or len(eq) < 2:
        return float('nan')
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    if yrs <= 0 or eq.iloc[0] <= 0:
        return float('nan')
    v = float(eq.iloc[-1] / eq.iloc[0])
    if v <= 0:
        return float('nan')
    return float(v ** (1 / yrs) - 1)


def _mdd(eq):
    v = eq.to_numpy()
    return float((v / np.maximum.accumulate(v) - 1).min())


def _run(pos, ret, cost_bp=COST_BP):
    held = pos.shift(1).fillna(0.0)
    turn = held.diff().abs().fillna(held.abs())
    pnl = held * ret - turn * cost_bp / 1e4
    return (1 + pnl).cumprod(), pnl


# ---------------- 策略家族（每個都只有參數，沒有主觀判斷） ----------------
def f_ma_cross(df, fast=20, slow=100, **kw):
    c = df['close']
    f, s = c.rolling(fast).mean(), c.rolling(slow).mean()
    pos = pd.Series(np.where(f > s, 1.0, 0.0), index=c.index)
    pos[f.isna() | s.isna()] = 0.0
    return pos


def f_rsi_reversion(df, n=10, lo=30, hi=55, **kw):
    c = df['close']
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    r = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    entry = r < lo
    exit_ = r > hi
    pos = pd.Series(np.nan, index=c.index)
    pos[entry.fillna(False)] = 1.0
    pos[exit_.fillna(False)] = 0.0
    return pos.ffill().fillna(0.0)


def f_breakout(df, n=55, exit_n=20, **kw):
    c = df['close']
    hi = c.rolling(n).max().shift(1)
    lo = c.rolling(exit_n).min().shift(1)
    pos = pd.Series(np.nan, index=c.index)
    pos[c > hi] = 1.0
    pos[c < lo] = 0.0
    return pos.ffill().fillna(0.0)


def f_boll(df, n=20, k=2.0, exit_mid=True, **kw):
    c = df['close']
    mid = c.rolling(n).mean()
    sd = c.rolling(n).std()
    lo = mid - k * sd
    pos = pd.Series(np.nan, index=c.index)
    pos[c < lo] = 1.0
    pos[c > mid] = 0.0
    return pos.ffill().fillna(0.0)


def f_momentum(df, n=250, th=0.0, **kw):
    c = df['close']
    mom = c / c.shift(n) - 1
    pos = pd.Series(np.where(mom > th, 1.0, 0.0), index=c.index)
    pos[mom.isna()] = 0.0
    return pos


FAMILIES = {
    '均線交叉 MA cross': (
        f_ma_cross,
        [dict(fast=a, slow=b) for a in (5, 10, 20, 50)
         for b in (50, 100, 150, 200) if a < b]),
    'RSI 回歸 RSI reversion': (
        f_rsi_reversion,
        [dict(n=n, lo=lo, hi=hi) for n in (2, 5, 7, 10, 14)
         for lo in (20, 25, 30, 35) for hi in (45, 55, 65)]),
    '突破 Breakout': (
        f_breakout,
        [dict(n=n, exit_n=e) for n in (20, 40, 55, 100, 200)
         for e in (10, 20, 40)]),
    '布林 Bollinger': (
        f_boll,
        [dict(n=n, k=k) for n in (10, 20, 30, 50) for k in (1.5, 2.0, 2.5, 3.0)]),
    '動量 Momentum': (
        f_momentum,
        [dict(n=n, th=t) for n in (20, 60, 120, 200, 250)
         for t in (-0.05, 0.0, 0.05, 0.10)]),
}


def main():
    df = fetch('SPY', '30y')
    ret = df['close'].pct_change().fillna(0.0)
    n = len(df)
    split = int(n * 0.70)
    is_idx, oos_idx = df.index[:split], df.index[split:]

    print('=' * 80)
    print('AI 產出的策略，樣本外還剩多少？')
    print('=' * 80)
    print(f'資料 SPY 日線 {df.index[0].date()} ~ {df.index[-1].date()}（{n} 天）')
    print(f'樣本內 {is_idx[0].date()} ~ {is_idx[-1].date()}（{split} 天）')
    print(f'樣本外 {oos_idx[0].date()} ~ {oos_idx[-1].date()}（{n-split} 天）')
    print()

    all_rows = []
    for fam, (fn, grids) in FAMILIES.items():
        rows = []
        for g in grids:
            pos = fn(df, **g)
            eq, pnl = _run(pos, ret)
            sh_is = _sharp(pnl.loc[is_idx])
            sh_oos = _sharp(pnl.loc[oos_idx])
            rows.append(dict(fam=fam, params=g, is_sharpe=sh_is, oos_sharpe=sh_oos,
                             is_cagr=_cagr(eq.loc[is_idx]),
                             oos_cagr=_cagr(eq.loc[oos_idx]),
                             oos_mdd=_mdd(eq.loc[oos_idx]),
                             full_cagr=_cagr(eq), full_mdd=_mdd(eq)))
        t = pd.DataFrame(rows).dropna(subset=['is_sharpe'])
        all_rows.append(t)
        best = t.loc[t['is_sharpe'].idxmax()]
        # 隨機抽一個對照，避免「最佳」被誤讀為「家族水準」
        med = t['oos_sharpe'].median()
        print(f'【{fam}】{len(t)} 組參數')
        print(f'  樣本內最佳 {best["params"]}')
        print(f'    樣本內 Sharpe {best["is_sharpe"]:5.2f} 年化 {best["is_cagr"]*100:6.2f}%'
              f'  →  樣本外 Sharpe {best["oos_sharpe"]:5.2f} 年化 {best["oos_cagr"]*100:6.2f}%'
              f'  回撤 {best["oos_mdd"]*100:6.1f}%')
        print(f'    該家族樣本外 Sharpe 中位數 {med:5.2f}')
        dec = best['is_sharpe'] - best['oos_sharpe']
        print(f'    衰減 {dec:+.2f}（{"站不住" if dec > 0.4 else "尚可"}）')
        print()

    A = pd.concat(all_rows, ignore_index=True)
    A.to_csv(os.path.join(HERE, 'out', 'ai_test_all.csv'), index=False)

    print('=' * 80)
    print('整體統計')
    print('=' * 80)
    n_all = len(A)
    # 樣本內看起來「很棒」的有多少？
    good = A['is_sharpe'] > 1.0
    print(f'共測試 {n_all} 組策略參數')
    print(f'  樣本內 Sharpe > 1.0 的有 {int(good.sum())} 組（{good.mean()*100:.1f}%）'
          f' ← AI 會從中挑出「很棒的那個」')
    if good.any():
        g = A[good]
        print(f'  這 {int(good.sum())} 組在樣本外：Sharpe 中位 {g["oos_sharpe"].median():.2f}'
              f'  平均年化 {g["oos_cagr"].mean()*100:+.2f}%')
        print(f'  其中樣本外 Sharpe 仍 > 1.0 的只有 {int((g["oos_sharpe"]>1.0).sum())} 組'
              f'（{(g["oos_sharpe"]>1.0).mean()*100:.0f}%）')
        print(f'  樣本外年化為負的有 {int((g["oos_cagr"]<0).sum())} 組'
              f'（{(g["oos_cagr"]<0).mean()*100:.0f}%）')
    # 相關係數：樣本內表現能不能預測樣本外
    corr = A[['is_sharpe', 'oos_sharpe']].corr().iloc[0, 1]
    print(f'\n  樣本內 Sharpe 與樣本外 Sharpe 的相關係數：{corr:+.2f}'
          f'  （{"幾乎無預測力" if abs(corr) < 0.3 else "有些預測力"}）')

    # 買進持有對照
    bh, _ = _run(pd.Series(1.0, index=df.index), ret)
    print(f'\n  對照：買進持有（同期）年化 {_cagr(bh)*100:.2f}%  回撤 {_mdd(bh)*100:.1f}%')
    print(f'        樣本外年化 {_cagr(bh.loc[oos_idx])*100:.2f}%')


if __name__ == '__main__':
    main()
