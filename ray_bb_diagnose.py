#!/usr/bin/env python3
"""追查「平均曝險只有 9%」——這是不是結論的關鍵？

為什麼要做這個
--------------
照影片原話跑，288 支股票、18.2 年只成交 443 筆，平均曝險 9%。
也就是說，九成時間資金是閒置的。三年化 3.75% 看起來很差，
但那不是「策略沒效」，而是「部位填不滿」——這兩件事必須分開。

他自稱在 Russell 10000（數千支）裡篩選。若訊號數量隨股票數等比放大，
5 個部位自然會被填滿。所以要回答的是：**給它足夠多的訊號，它撐得住嗎？**

作法：放寬持股上限（同時縮小每檔權重，維持不融資），
模擬「股票池變大」的效果。若放大後年化顯著上升且不崩壞，
代表限制是「池子太小」而非「策略無效」。
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, '.')

from ray_bb_opend import load, portfolio


def summarize(r, label):
    T = r['trades']
    if T.empty:
        print(f'  {label:<28} 無交易')
        return None
    w = T['ret'] > 0
    print(f'  {label:<28} 年化 {r["cagr"]*100:6.2f}%  總 {r["eq"].iloc[-1]*100-100:>7.0f}%  '
          f'曝險 {r["expo"]*100:5.1f}%  回撤 {r["mdd"]*100:6.1f}%  '
          f'{len(T):>4} 筆  勝率 {w.mean()*100:5.1f}%')
    return dict(cagr=r['cagr'], expo=r['expo'], mdd=r['mdd'],
                n=len(T), win=w.mean(), total=r['eq'].iloc[-1] - 1)


def main():
    data = load()
    print(f'載入 {len(data)} 支\n')
    print('=' * 96)
    print('A. 放寬持股上限（每檔權重同步縮小，維持不融資）')
    print('=' * 96)
    rows = {}
    for npos in (5, 10, 20, 40, 60):
        w = 1.0 / npos
        r = portfolio(data, max_pos=npos, weight=w)
        rows[npos] = summarize(r, f'{npos} 檔 × {w*100:.0f}%')

    print('\n' + '=' * 96)
    print('B. 逐年報酬（原話規則，5 檔）— 看是不是靠少數年份')
    print('=' * 96)
    base = portfolio(data)
    yr = (1 + base['pnl']).groupby(base['pnl'].index.year).prod() - 1
    pos_years = (yr > 0).sum()
    print(f'  有交易的年份 {len(yr)}，為正 {pos_years}/{len(yr)} '
          f'({pos_years/len(yr)*100:.0f}%)')
    for y, v in yr.items():
        bar = '█' * max(0, int(abs(v) * 120))
        sign = '+' if v >= 0 else '-'
        print(f'    {y}  {v*100:+7.2f}%  {sign}{bar}')

    print('\n' + '=' * 96)
    print('C. 集中度檢驗：剔除最大幾筆報酬後是否還成立')
    print('=' * 96)
    T = base['trades'].sort_values('ret', ascending=False)
    for k in (0, 1, 5, 10, 22):
        sub = T.iloc[k:]
        if len(sub) < 10:
            continue
        eq = float((1 + sub['ret']).prod())
        yrs = (sub['exit'].max() - sub['entry'].min()).days / 365.25
        cg = eq ** (1 / yrs) - 1 if yrs > 0 else float('nan')
        t = sub['ret'].mean() / (sub['ret'].std() / np.sqrt(len(sub)))
        print(f'  剔除前 {k:>2} 筆：剩 {len(sub):>4} 筆，'
              f'平均 {sub["ret"].mean()*100:+.2f}%，t={t:5.2f}，'
              f'累積 {eq*100-100:+7.0f}%')

    print('\n' + '=' * 96)
    print('D. 對照基準（同期間）')
    print('=' * 96)
    print(f'  同曝險基準（照策略實際曝險投資大盤）  {base["em_cagr"]*100:6.2f}%')
    print(f'  等權買進持有 288 支               {base["bh_cagr"]*100:6.2f}%')
    print(f'  策略本身                       {base["cagr"]*100:6.2f}%')

    print('\n' + '=' * 96)
    print('E. 要把 3.75% 拉到「買進持有 13.6%」需要多少槓桿')
    print('=' * 96)
    need = 0.1360 / base['cagr']
    print(f'  需要約 {need:.1f} 倍槓桿')
    print(f'  回撤會由 {base["mdd"]*100:.1f}% 放大到約 {base["mdd"]*100*need:.0f}%')


if __name__ == '__main__':
    main()
