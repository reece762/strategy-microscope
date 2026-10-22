#!/usr/bin/env python3
"""排除「波動率聚集」這個替代解釋。

問題：支撐位被觸及時，往往正好是低波動期。低波動期未來 20 天本來就不容易
再走 0.8 ATR。所以「守住率較高」可能只是波動聚集，與支撐位本身無關。

作法：對每一筆觸及事件，找出全樣本中「20 日滾動波動率落在同一十分位」的日子，
用這些日子當對照組。若控制了波動水準後差距消失，那支撐位就沒有額外資訊。
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from probe_alpha_sr import load_etf, atr_series, zigzag_pivots  # noqa


def build(df, horizon=20, width_atr=0.3, buf_atr=0.5):
    h, l, c = df['high'].values, df['low'].values, df['close'].values
    atr = atr_series(df).values
    n = len(df)
    ret = np.diff(np.log(np.maximum(c, 1e-12)), prepend=np.log(c[0]))
    vol20 = pd.Series(ret).rolling(20).std().values
    volpct = pd.Series(vol20).rank(pct=True).values      # 0..1 波動百分位

    piv = zigzag_pivots(h, l, atr, k=2.0)
    ev = []
    for pidx, pprice, kind, cidx in piv:
        a = atr[pidx]
        if np.isnan(a) or a <= 0:
            continue
        w = width_atr * a
        lo, hi = pprice - w, pprice + w
        i = max(pidx + 1, cidx + 1)
        while i < n - horizon:
            ain = atr[i]
            if np.isnan(ain) or ain <= 0:
                break
            if (l[i] <= hi) and (h[i] >= lo):
                b = buf_atr * ain
                win = c[i:i + horizon + 1]
                if kind == 'L':
                    broke = bool((win < (lo - b)).any())
                    tgt = -0.8 * ain
                else:
                    broke = bool((win > (hi + b)).any())
                    tgt = +0.8 * ain
                ev.append(dict(i=i, kind=kind, broke=broke, atr=ain,
                               volpct=volpct[i] if not np.isnan(volpct[i]) else 0.5))
                break
            i += 1

    # 全樣本基準：每一天往後 20 天，是否走出 ±0.8 ATR
    base = []
    for j in range(n - horizon):
        a = atr[j]
        if np.isnan(a) or a <= 0:
            continue
        win = c[j:j + horizon + 1]
        dn = bool((win < c[j] - 0.8 * a).any())
        up = bool((win > c[j] + 0.8 * a).any())
        base.append(dict(j=j, dn=dn, up=up,
                         volpct=volpct[j] if not np.isnan(volpct[j]) else 0.5))
    return pd.DataFrame(ev), pd.DataFrame(base)


def main():
    print('=' * 78)
    print('控制波動率後，支撐／阻力還有優勢嗎？')
    print('=' * 78)
    print('作法是：每筆觸及事件，只跟「同一波動率十分位」的日子比。\n')

    for sym in ('SPY', 'QQQ', 'IWM'):
        df = load_etf(sym)
        ev, base = build(df)
        if ev.empty:
            print(f'{sym}: 無事件'); continue

        # 依波動百分位分十檔，逐檔配對
        print(f'--- {sym}（{len(ev)} 筆觸及）---')
        print(f"  {'波動分位':>10s} {'事件數':>6s} {'支撐守住':>8s} "
              f"{'壓力守住':>8s} {'對照(跌)':>9s} {'對照(漲)':>9s} {'支撐差':>7s} {'壓力差':>7s}")

        tot_ev = tot_base_dn = tot_base_up = 0
        tot_sup_hold = tot_res_hold = 0
        n_sup = n_res = 0
        for q in range(10):
            lo_q, hi_q = q / 10, (q + 1) / 10
            be = base[(base['volpct'] >= lo_q) & (base['volpct'] < hi_q)]
            ee = ev[(ev['volpct'] >= lo_q) & (ev['volpct'] < hi_q)]
            if ee.empty or be.empty:
                continue
            sup = ee[ee['kind'] == 'L']
            res = ee[ee['kind'] == 'H']
            brk_dn = be['dn'].mean()
            brk_up = be['up'].mean()
            sup_hold = 1 - sup['broke'].mean() if len(sup) else np.nan
            res_hold = 1 - res['broke'].mean() if len(res) else np.nan
            d_sup = (sup_hold - (1 - brk_dn)) if len(sup) else np.nan
            d_res = (res_hold - (1 - brk_up)) if len(res) else np.nan
            print(f"  {lo_q*100:3.0f}-{hi_q*100:3.0f}%  {len(ee):6d} "
                  f"{sup_hold*100 if len(sup) else float('nan'):7.1f}% "
                  f"{res_hold*100 if len(res) else float('nan'):7.1f}% "
                  f"{(1-brk_dn)*100:8.1f}% {(1-brk_up)*100:8.1f}% "
                  f"{d_sup*100 if len(sup) else float('nan'):+6.1f} "
                  f"{d_res*100 if len(res) else float('nan'):+6.1f}")
            if len(sup):
                tot_sup_hold += sup['broke'].eq(False).sum(); n_sup += len(sup)
            if len(res):
                tot_res_hold += res['broke'].eq(False).sum(); n_res += len(res)
            tot_base_dn += (1 - brk_dn) * len(sup) if len(sup) else 0
            tot_base_up += (1 - brk_up) * len(res) if len(res) else 0

        if n_sup:
            h_sup = tot_sup_hold / n_sup
            b_dn = tot_base_dn / n_sup
            print(f"  {'加權合計':>10s} {n_sup+n_res:6d} "
                  f"{h_sup*100:7.1f}% {'':>7s} {b_dn*100:8.1f}% {'':>8s} "
                  f"{(h_sup-b_dn)*100:+6.1f}")
        if n_res:
            h_res = tot_res_hold / n_res
            b_up = tot_base_up / n_res
            print(f"  {'壓力加權':>10s} {n_res:6d} {'':>7s} {h_res*100:6.1f}% "
                  f"{'':>8s} {b_up*100:7.1f}% {'':>6s} {(h_res-b_up)*100:+6.1f}")
        print()


if __name__ == '__main__':
    main()
