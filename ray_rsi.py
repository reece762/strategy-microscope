#!/usr/bin/env python3
"""Rayner Teo 均值回歸策略：複驗他宣稱的「86.84% 勝率、25 年」。

規則（完整取自他本人文章 /mean-reversion-trading-strategy/，非我推測）：
  1. 價格在 200 日均線之上
  2. 10 期 RSI 低於 30
  3. 隔天開盤買進
  4. 10 期 RSI 穿越 40 以上出場，或滿 10 個交易日強制出場

他宣稱（標普 500，25 年）：38 筆交易、勝率 86.84%、平均獲利 1.39%。

資料落差（必須寫進報告，不能藏）：
  - data_daily/SPY.csv 有完整 OHLC，但只有 2016-10 ~ 2025-12（約 10 年）
    → 這是「照規則原樣」的版本，用真開盤價。
  - xbx/data/SP500.csv 只有收盤價，也只有 10 年 → 無法做開盤進場。
  - xbx/data/NASDAQCOM.csv 有 55 年，但只有收盤價 → 用次日收盤近似開盤，
    這是近似，不是原規則。指數也非 ETF、無股息。
  三種來源都跑，讓「資料本身造成多少差異」自己講話。
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from tier1 import load_etf  # noqa: E402


def rsi(close, n=10):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def run(df, use_open=True, cost_bp=1.0, rsi_n=10, entry=30.0,
        exit_rsi=40.0, ma_n=200, max_hold=10):
    """逐一筆交易回傳（不是連續部位），因為原文是『一筆交易』計。"""
    close = df['close']
    ma = close.rolling(ma_n).mean()
    r = rsi(close, rsi_n)
    op = df['open'] if use_open else close

    trades = []
    i = ma_n
    n = len(df)
    while i < n - 1:
        # 訊號日 i：收盤後判斷
        if pd.notna(ma.iloc[i]) and pd.notna(r.iloc[i]) \
                and close.iloc[i] > ma.iloc[i] and r.iloc[i] < entry:
            # 隔天開盤進場
            ei = i + 1
            if ei >= n:
                break
            ep = op.iloc[ei]
            xi = None
            for j in range(ei + 1, min(ei + max_hold + 1, n)):
                if pd.notna(r.iloc[j]) and r.iloc[j] > exit_rsi:
                    xi = j + 1 if j + 1 < n else n - 1
                    break
            if xi is None:
                xi = min(ei + max_hold, n - 1)
            xp = op.iloc[xi] if use_open else close.iloc[xi]
            ret = xp / ep - 1 - cost_bp / 1e4 * 2
            trades.append(dict(entry=df.index[ei], exit=df.index[xi],
                               ret=ret, days=(df.index[xi] - df.index[ei]).days))
            i = xi
        else:
            i += 1

    t = pd.DataFrame(trades)
    if t.empty:
        return None
    wins = (t['ret'] > 0)
    # 以交易為單位、等權 10% 部位的粗略年化（僅供量級參考）
    years = (df.index[-1] - df.index[0]).days / 365.25
    return dict(
        n=len(t), win=float(wins.mean()),
        avg=float(t['ret'].mean()),
        med=float(t['ret'].median()),
        avg_win=float(t['ret'][wins].mean()) if wins.any() else float('nan'),
        avg_loss=float(t['ret'][~wins].mean()) if (~wins).any() else float('nan'),
        best=float(t['ret'].max()), worst=float(t['ret'].min()),
        sum_ret=float(t['ret'].sum()),
        years=years,
        per_year=len(t) / years,
        trades=t,
    )


def show(label, st, claimed=None):
    if st is None:
        print(f'{label}: 無交易')
        return
    print(f'\n=== {label} ===')
    print(f'  交易 {st["n"]} 筆（每年 {st["per_year"]:.1f} 筆，涵蓋 {st["years"]:.1f} 年）')
    print(f'  勝率 {st["win"]*100:.2f}%   平均 {st["avg"]*100:+.2f}%   中位 {st["med"]*100:+.2f}%')
    print(f'  平均獲利 {st["avg_win"]*100:+.2f}%   平均虧損 {st["avg_loss"]*100:+.2f}%'
          f'   賺賠比 {abs(st["avg_win"]/st["avg_loss"]):.2f}')
    print(f'  最佳 {st["best"]*100:+.2f}%   最差 {st["worst"]*100:+.2f}%'
          f'   總和（未複利）{st["sum_ret"]*100:+.1f}%')
    # 期望值
    ev = st['win'] * st['avg_win'] + (1 - st['win']) * st['avg_loss']
    print(f'  每筆期望值 {ev*100:+.3f}%')
    if claimed:
        print(f'  宣稱：{claimed}')


def main():
    print('=' * 78)
    print('Rayner 均值回歸：複驗「86.84% 勝率、25 年、38 筆」')
    print('=' * 78)

    # 1) SPY 真開盤價（10 年）
    spy = load_etf('SPY')
    st = run(spy, use_open=True)
    show('SPY（真開盤價進出，2016-10~2025-12）', st,
         '38 筆 / 勝率 86.84% / 平均 +1.39%（標普 500，25 年）')
    if st is not None:
        st['trades'].to_csv(os.path.join(HERE, 'out', 'rsi2_spy_trades.csv'),
                            index=False)

    # 2) 納斯達克 55 年（收盤近似開盤）
    nq = pd.read_csv(os.path.join(HERE, 'data', 'NASDAQCOM.csv'))
    nq.columns = ['date', 'close']
    nq['date'] = pd.to_datetime(nq['date'], errors='coerce')
    nq['close'] = pd.to_numeric(nq['close'], errors='coerce')
    nq = nq.dropna().set_index('date').sort_index()
    nq['open'] = nq['close']
    st2 = run(nq, use_open=False)
    show('納斯達克綜合指數（55 年，收盤近似開盤）', st2)

    # 3) 參數敏感度：換 RSI 期數與門檻，看結論穩不穩
    print('\n' + '=' * 78)
    print('參數敏感度（SPY，真開盤價）')
    print('=' * 78)
    print(f'{"RSI":>4}{"進場":>6}{"出場":>6}{"筆數":>6}{"勝率":>8}{"平均":>8}{"期望值":>9}')
    for rn in (2, 5, 10, 14):
        for en, ex in ((30, 40), (30, 50), (25, 40), (20, 40)):
            s = run(spy, use_open=True, rsi_n=rn, entry=en, exit_rsi=ex)
            if s is None:
                continue
            ev = s['win'] * s['avg_win'] + (1 - s['win']) * s['avg_loss']
            print(f'{rn:>4}{en:>6}{ex:>6}{s["n"]:>6}{s["win"]*100:>7.1f}%'
                  f'{s["avg"]*100:>+7.2f}%{ev*100:>+8.3f}%')


if __name__ == '__main__':
    main()
