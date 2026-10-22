#!/usr/bin/env python3
"""驗證「支撐／阻力真的守得住嗎」——這是 DENSITY-SR 的核心主張。

他的 README 自己寫：
  守住率 Brier 0.1836 vs 常數預測 0.1842 → 零區分度
  所以「不再輸出任何單個關鍵位的守住率概率」

我的測試比他的更直接：不看模型，只問一個問題——
  價格碰到事前定義好的支撐位之後，20 天內有沒有跌破？
  這個「守住率」跟「隨便一天往後看、會不會跌破同等距離」比，有沒有差別？

如果兩者一樣，那支撐位就沒有資訊，整個產品的地基是空的。
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DAILY = os.path.join(os.path.dirname(HERE), 'data_daily')


def load_etf(sym):
    df = pd.read_csv(os.path.join(DAILY, f'{sym}.csv'))
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
    for c in ['open', 'high', 'low', 'close', 'volume']:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df.set_index('date').sort_index().dropna(subset=['close'])


def atr_series(df, n=14):
    h, l, c = df['high'], df['low'], df['close']
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def zigzag_pivots(h, l, atr, k=2.0):
    """簡化 zigzag：反向走超過 k*ATR 才算轉折。

    回傳 [(pivot_idx, price, 'H'/'L', confirm_idx)]。

    confirm_idx 是關鍵：轉折點本身要等價格反向走夠 k*ATR 才「被確認」。
    在那之前，我們根本不知道它是轉折點——用它就是用了未來資訊。
    """
    piv = []
    n = len(h)
    if n < 3:
        return piv
    direction = 0
    ext_i, ext_p = 0, h[0]
    for i in range(1, n):
        a = atr[i] if not np.isnan(atr[i]) else atr[max(0, i - 1)]
        if np.isnan(a) or a <= 0:
            continue
        if direction >= 0:
            if h[i] > ext_p:
                ext_i, ext_p = i, h[i]
            elif ext_p - l[i] > k * a:
                piv.append((ext_i, ext_p, 'H', i))
                direction = -1
                ext_i, ext_p = i, l[i]
        if direction <= 0:
            if l[i] < ext_p:
                ext_i, ext_p = i, l[i]
            elif h[i] - ext_p > k * a:
                piv.append((ext_i, ext_p, 'L', i))
                direction = 1
                ext_i, ext_p = i, h[i]
    return piv


def evaluate(df, horizon=20, k_atr=2.0, touch_atr=0.5, width_atr=0.3):
    """因果版：轉折點必須先被確認，之後的觸及才算數。"""
    """回傳每筆「觸及事件」是否守住。"""
    h, l, c = df['high'].values, df['low'].values, df['close'].values
    atr = atr_series(df).values
    piv = zigzag_pivots(h, l, atr, k=k_atr)
    n = len(df)
    rows = []

    for pidx, pprice, kind, cidx in piv:
        a = atr[pidx]
        if np.isnan(a) or a <= 0:
            continue
        width = width_atr * a
        lo, hi = pprice - width, pprice + width
        # 只從「確認之後」開始找觸及：確認前不知道這是轉折點
        i = max(pidx + 1, cidx + 1)
        while i < n - horizon:
            ain = atr[i]
            if np.isnan(ain) or ain <= 0:
                break
            touched = (l[i] <= hi) and (h[i] >= lo)
            if touched:
                # 守住 = 之後 horizon 天內，收盤不曾有效跌破（超過 0.5 ATR 寬的緩衝）
                buf = 0.5 * ain
                window = c[i:i + horizon + 1]
                if kind == 'L':
                    broke = bool((window < (lo - buf)).any())
                    ref = lo - buf
                    base_broke = bool((c[i:i + horizon + 1] < (c[i] - 0.8 * ain)).any())
                else:
                    broke = bool((window > (hi + buf)).any())
                    ref = hi + buf
                    base_broke = bool((c[i:i + horizon + 1] > (c[i] + 0.8 * ain)).any())
                rows.append(dict(date=df.index[i], kind=kind, broke=broke,
                                 base_broke=base_broke, atr=ain,
                                 n_prior=sum(1 for p, _, kk, _c in piv
                                             if kk == kind and p < pidx)))
                break
            i += 1
    return pd.DataFrame(rows)


def base_rate(df, horizon=20, dist_atr=1.5):
    """無條件基準：隨便一天往後看 20 天，價格是否跌破（或漲破）1.5 ATR？"""
    c = df['close'].values
    atr = atr_series(df).values
    n = len(df)
    down = up = cnt = 0
    for i in range(n - horizon):
        a = atr[i]
        if np.isnan(a) or a <= 0:
            continue
        w = c[i:i + horizon + 1]
        down += (w < c[i] - dist_atr * a).any()
        up += (w > c[i] + dist_atr * a).any()
        cnt += 1
    return down / cnt, up / cnt


def main():
    print('=' * 76)
    print('支撐／阻力「守住率」檢驗（DENSITY-SR 的核心主張）')
    print('=' * 76)

    for sym in ('SPY', 'QQQ', 'IWM'):
        df = load_etf(sym)
        r = evaluate(df)
        if r.empty:
            print(f'\n--- {sym}: 無觸及事件 ---')
            continue
        # 距離必須一致：S/R 判定「破」＝離觸及價 0.3(區寬)+0.5(緩衝)=0.8 ATR，
        # 基準線就必須同樣用 0.8 ATR，否則基準太寬鬆、會高估 S/R 的表現。
        down_base, up_base = base_rate(df, dist_atr=0.8)
        sup = r[r['kind'] == 'L']
        res = r[r['kind'] == 'H']
        print(f"\n--- {sym}  {df.index[0].date()} ~ {df.index[-1].date()}  "
              f"{len(df)} 日 ---")
        print(f"  轉折點 {len(r)} 個觸及事件（支撐 {len(sup)}、壓力 {len(res)}）")
        if len(sup):
            hr = 1 - sup['broke'].mean()
            br = 1 - sup['base_broke'].mean()
            print(f"  支撐守住率      {hr*100:5.1f}%   （20 日內未有效跌破）")
            print(f"  無條件對照      {br*100:5.1f}%   （隨便一天、跌 0.8 ATR 的機率）")
            print(f"  差額            {(hr-br)*100:+5.1f} 個百分點")
        if len(res):
            hr = 1 - res['broke'].mean()
            br = 1 - res['base_broke'].mean()
            print(f"  壓力守住率      {hr*100:5.1f}%   （20 日內未有效漲破）")
            print(f"  無條件對照      {br*100:5.1f}%   （隨便一天、漲 0.8 ATR 的機率）")
            print(f"  差額            {(hr-br)*100:+5.1f} 個百分點")

        # 他的排序分只用「觸及次數」與「陳舊度」。檢驗觸及次數有沒有預測力。
        r2 = r.copy()
        r2['q'] = pd.qcut(r2['n_prior'], 3, labels=['少(1/3)', '中', '多(1/3)'],
                          duplicates='drop')
        print('  --- 依「歷史觸及次數」分組的守住率（他的排序主要因子）---')
        for q, g in r2.groupby('q', observed=True):
            print(f"    {str(q):8s} n={len(g):4d}  守住 {100*(1-g['broke'].mean()):5.1f}%")


if __name__ == '__main__':
    main()
