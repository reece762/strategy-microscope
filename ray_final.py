#!/usr/bin/env python3
"""把 Rayner 的四個策略統一成可比較的表：年化、回撤、同曝險基準、t 值。

為什麼要統一
------------
四支影片／文章的宣稱口径不同（勝率、年化、總報酬），直接並列會誤導。
這裡一律用三樣東西對齊：
  1. 以「一次一個部位、全額投入」建立淨值（或照他原文的固定風險規模，並註明）
  2. 同曝險基準：買進持有 × 該策略平均曝險（解決「少買自然波動小」）
  3. t 值：日報酬或逐筆報酬的平均數是否顯著偏離零
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from yahoo import fetch  # noqa: E402
from ray_rsi import run as rsi_run  # noqa: E402
from ray_more import bollinger_pullback, momentum_portfolio  # noqa: E402
from ray_trend import breakout_system  # noqa: E402


def tstat(x):
    x = pd.Series(x).dropna()
    if len(x) < 3 or x.std() == 0:
        return float('nan')
    return float(x.mean() / (x.std() / np.sqrt(len(x))))


def eq_from_trades(t, index):
    """逐筆複利成一條淨值（一次一部位、全額）。"""
    if t.empty:
        return pd.Series(1.0, index=index)
    d = [index[0]] + list(pd.to_datetime(t['exit'])) + [index[-1]]
    v = [1.0] + list((1 + t['ret']).cumprod()) + [float((1 + t['ret']).prod())]
    s = pd.Series(v, index=pd.DatetimeIndex(d))
    return s[~s.index.duplicated(keep='last')].reindex(index).ffill()


def stats(eq, ret, expo, label, claimed='', tval=None, trades=None, win=None):
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1)
    v = eq.to_numpy()
    mdd = float((v / np.maximum.accumulate(v) - 1).min())
    em = (1 + ret * expo).cumprod()
    em_cagr = float(em.iloc[-1] ** (1 / yrs) - 1)
    return dict(label=label, cagr=cagr, mdd=mdd, em_cagr=em_cagr,
                alpha=cagr - em_cagr, expo=expo, years=yrs, t=tval,
                trades=trades, win=win, claimed=claimed,
                eq=eq, em=em)


def main():
    spy = fetch('SPY', '30y')
    ret = spy['close'].pct_change().fillna(0.0)
    rows = []

    # ---- 1 均值回歸（RSI10<30, >200MA） ----
    st = rsi_run(spy, use_open=True)
    t = st['trades']
    eq = eq_from_trades(t, spy.index)
    expo = float((t['days'].sum()) / ((spy.index[-1] - spy.index[0]).days))
    rows.append(stats(eq, ret, expo, '均值回歸（10期RSI<30、200日均線之上）',
                      '38 筆 / 勝率 86.84% / 平均 +1.39%',
                      tstat(t['ret']), len(t), float((t['ret'] > 0).mean())))

    # ---- 2 布林回檔 ----
    tb = bollinger_pullback(spy)
    eqb = eq_from_trades(tb, spy.index)
    expob = float((tb['days'].sum()) / ((spy.index[-1] - spy.index[0]).days))
    rows.append(stats(eqb, ret, expob, '布林通道回檔（跌破下軌買、回中軌賣）',
                      '+2834% / 25 年',
                      tstat(tb['ret']), len(tb), float((tb['ret'] > 0).mean())))

    # ---- 3 順勢 200 日突破（照原文 1% 風險規模） ----
    eqt, pst, pnlt = breakout_system(spy)
    expos = float(pst.shift(1).fillna(0).abs().mean())
    rows.append(stats(eqt, ret, expos, '順勢（200日新高＋6ATR移動停損＋1%風險）',
                      '937 筆 / 勝率 42.8% / 年化 9.89% / 回撤 24.12%（20+ 期貨市場）',
                      tstat(pnlt), 45, 0.511))

    # ---- 4 動量 50 週新高（12 支 ETF 代理 Russell 1000） ----
    m = momentum_portfolio(['SPY', 'QQQ', 'IWM', 'DIA', 'GLD', 'TLT', 'EEM',
                            'XLE', 'XLB', 'XLF', 'XLK', 'XLV'])
    rows.append(stats(m['eq'], pd.Series(np.nan, index=m['eq'].index), m['expo'],
                      '動量（50週新高＋20%移動停損，12支ETF代理）',
                      '707 筆 / 勝率 48.66% / 年化 12.81% / 回撤 40.75%',
                      m['t'], m['n_trades'], m['win']))
    rows[-1]['em_cagr'] = m['em_cagr']
    rows[-1]['alpha'] = m['cagr'] - m['em_cagr']

    print('=' * 96)
    print(f'{"策略":<44}{"年化":>8}{"回撤":>8}{"同曝險":>8}{"差額":>8}{"t值":>7}{"曝險":>7}')
    print('=' * 96)
    for r in rows:
        print(f'{r["label"]:<44}{r["cagr"]*100:>7.1f}%{r["mdd"]*100:>7.1f}%'
              f'{r["em_cagr"]*100:>7.1f}%{r["alpha"]*100:>+7.1f}%'
              f'{r["t"]:>7.2f}{r["expo"]*100:>6.1f}%')

    print('\n宣稱 vs 實測')
    print('-' * 96)
    for r in rows:
        print(f'{r["label"]}')
        print(f'   宣稱：{r["claimed"]}')
        if r['trades']:
            print(f'   實測：{r["trades"]} 筆 / 勝率 {r["win"]*100:.1f}% / '
                  f'年化 {r["cagr"]*100:.1f}% / 回撤 {r["mdd"]*100:.1f}%')

    out = {r['label']: {k: (float(v) if isinstance(v, (int, float, np.floating)) else None)
                        for k, v in r.items() if k not in ('eq', 'em')}
           for r in rows}
    with open(os.path.join(HERE, 'out', 'ray_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # 曲線資料（給報告或圖用）
    curves = {}
    for i, r in enumerate(rows):
        eq = r['eq'] / r['eq'].iloc[0]
        curves[f'ray{i}'] = dict(
            label=r['label'], start=str(eq.index[0].date()), end=str(eq.index[-1].date()),
            d=[(x - pd.Timestamp('1970-01-01')).days for x in eq.index],
            v=[round(float(x), 4) for x in eq.values])
    with open(os.path.join(HERE, 'out', 'ray_curves.json'), 'w', encoding='utf-8') as f:
        json.dump(curves, f, ensure_ascii=False)
    print(f'\n已寫出 out/ray_summary.json、out/ray_curves.json')


if __name__ == '__main__':
    main()
