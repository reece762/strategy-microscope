#!/usr/bin/env python3
"""AI 產出策略的樣本外檢驗（滾動多窗口版），以及相對強度突破的概念測試。

為什麼要重做第一版
------------------
第一版只用單一 70/30 切分，樣本外剛好是 2019-2026 的大多頭，
所有策略在樣本外年化都很好看（14~33%），但那是市場给的，不是策略赚的。
正确做法是滚動多窗口、每個窗口都和「同期買進持有」比。

第一部分：AI 選參數到底有没有用
  每一折：用前一段資料挑 Sharpe 最高的參數，拿到下一段跑。
  關鍵指標不是報酬，而是「挑出來的那組，在樣本外贏過買進持有的比例」。

第二部分：相對強度突破（SMB Capital 的概念，非他本人的原始規則）
  我沒有他的影片字幕與說明欄（YouTube 全封），所以這裡測的是
  「相對強度突破」這個公開概念，用產業 ETF 當標的，
  明確標示為我的操作化，不冒充他的原始規則。
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from yahoo import fetch  # noqa: E402
from ai_test import FAMILIES, _run, _sharp, _cagr, _mdd  # noqa: E402

PPY = 252


# ============================================================ 第一部分
def walk_forward(syms, n_folds=4, train_frac=0.70):
    """滾動窗口：訓練段挑最佳參數，測試段驗證。"""
    rows = []
    for sym in syms:
        try:
            df = fetch(sym, '30y')
        except Exception:
            continue
        if len(df) < 1500:
            continue
        ret = df['close'].pct_change().fillna(0.0)
        n = len(df)

        for fam, (fn, grids) in FAMILIES.items():
            # 先算好所有參數的完整 pnl（避免每折重算）
            pnls = {}
            for g in grids:
                pos = fn(df, **g)
                _, pnl = _run(pos, ret)
                pnls[str(g)] = pnl

            for k in range(n_folds):
                # 滾動窗口：每折往前推
                span = n // (n_folds + 1)
                start = k * span // 2
                end = min(start + int(n * train_frac), n)
                if end >= n - 60:
                    break
                test_end = min(end + span, n)
                is_idx = df.index[start:end]
                oos_idx = df.index[end:test_end]
                if len(is_idx) < 250 or len(oos_idx) < 60:
                    continue

                # 訓練段挑 Sharpe 最高
                best_key, best_sh = None, -9e9
                for key, pnl in pnls.items():
                    sh = _sharp(pnl.loc[is_idx])
                    if np.isfinite(sh) and sh > best_sh:
                        best_sh, best_key = sh, key
                if best_key is None:
                    continue

                pnl = pnls[best_key]
                eq, _ = _run(fn(df, **eval(best_key)), ret)
                oos_cagr = _cagr(eq.loc[oos_idx])
                oos_sh = _sharp(pnl.loc[oos_idx])
                bh = (1 + ret.loc[oos_idx]).cumprod()
                bh_cagr = _cagr(bh)
                rows.append(dict(sym=sym, fam=fam, fold=k, params=best_key,
                                 is_sharpe=best_sh, oos_sharpe=oos_sh,
                                 oos_cagr=oos_cagr, bh_cagr=bh_cagr,
                                 beat=oos_cagr > bh_cagr,
                                 oos_mdd=_mdd(eq.loc[oos_idx])))
    return pd.DataFrame(rows)


# ============================================================ 第二部分
def rel_strength(syms, top_n=3, lookback=126, ma_n=200, cost_bp=2.0,
                 rebal=21):
    """相對強度突破：按 6 個月相對強度排序，買最強且站上 200MA 的前 N 支。"""
    data = {}
    for s in syms:
        try:
            df = fetch(s, '30y')
            if len(df) > ma_n + lookback + 100:
                data[s] = df['close']
        except Exception:
            pass
    if len(data) < 3:
        return None

    C = pd.DataFrame(data).dropna(how='all')
    ma = C.rolling(ma_n).mean()
    rs = C / C.shift(lookback) - 1          # 絕對動能（相對強度的常用代理）
    # 相對強度：扣掉同期全體平均，才是「相對」而非「絕對」
    rs_rel = rs.sub(rs.mean(axis=1), axis=0)

    dates = C.index
    pos = pd.DataFrame(0.0, index=dates, columns=C.columns)
    cur = pd.Series(0.0, index=C.columns)
    last = -10 ** 9

    for i in range(ma_n + lookback, len(dates)):
        if i - last >= rebal:
            ok = (C.iloc[i] > ma.iloc[i]) & rs_rel.iloc[i].notna()
            cand = rs_rel.iloc[i][ok].sort_values(ascending=False)
            cur = pd.Series(0.0, index=C.columns)
            if len(cand):
                for s in cand.index[:top_n]:
                    cur[s] = 1.0 / top_n
            last = i
        pos.iloc[i] = cur.values

    held = pos.shift(1).fillna(0.0)
    rets = C.pct_change().fillna(0.0)
    turn = held.diff().abs().sum(axis=1).fillna(0.0)
    pnl = (held * rets).sum(axis=1) - turn * cost_bp / 1e4
    eq = (1 + pnl).cumprod()

    expo = held.sum(axis=1)
    bench = rets.mean(axis=1)
    em = (1 + bench * expo.mean()).cumprod()
    bh = (1 + bench).cumprod()

    yrs = (dates[-1] - dates[0]).days / 365.25
    warm = ma_n + lookback

    def _c(e):
        """区间年化：先 rebase 到 1.0，且用切片自己的年限。"""
        if e is None or len(e) < 2 or e.iloc[0] <= 0:
            return float('nan')
        yy = (e.index[-1] - e.index[0]).days / 365.25
        if yy <= 0:
            return float('nan')
        return float((e.iloc[-1] / e.iloc[0]) ** (1 / yy) - 1)

    pnl_clean = pnl.loc[dates[warm]:]
    # 基準也必须在同一段重新起算，否则预热期的报酬会提前累积进去
    expo_s = expo.loc[dates[warm]:]
    bench_s = bench.loc[dates[warm]:]
    eq_s = eq.loc[dates[warm]:]
    em_s = (1 + bench_s * expo_s.mean()).cumprod()
    bh_s = (1 + bench_s).cumprod()
    return dict(eq=eq_s, em=em_s, bh=bh_s, pnl=pnl_clean,
                cagr=_c(eq_s), em_cagr=_c(em_s), bh_cagr=_c(bh_s),
                mdd=_mdd(eq_s), expo=float(expo_s.mean()),
                years=(eq_s.index[-1] - eq_s.index[0]).days / 365.25,
                n_assets=len(data), assets=list(data))


def main():
    print('=' * 82)
    print('第一部分：AI 選參數，樣本外還站得住嗎？（滾動 4 折）')
    print('=' * 82)
    syms = ['SPY', 'QQQ', 'IWM', 'GLD', 'TLT']
    W = walk_forward(syms, n_folds=4)
    if not W.empty:
        print(f'共 {len(W)} 個「資產×策略家族×折」組合')
        print(f'  樣本外贏過同期買進持有的比例：{W["beat"].mean()*100:.1f}%'
              f'（{int(W["beat"].sum())}/{len(W)}）')
        print(f'  樣本內 Sharpe 中位 {W["is_sharpe"].median():.2f}'
              f'  →  樣本外 Sharpe 中位 {W["oos_sharpe"].median():.2f}')
        print(f'  樣本內→外 Sharpe 相關係數 {W[["is_sharpe","oos_sharpe"]].corr().iloc[0,1]:+.2f}')
        print(f'  樣本外年化中位 {W["oos_cagr"].median()*100:+.2f}%'
              f'   同期買進持有中位 {W["bh_cagr"].median()*100:+.2f}%')
        print()
        by = W.groupby('fam').agg(n=('beat', 'size'), win_rate=('beat', 'mean'),
                                  oos=('oos_cagr', 'median'), bh=('bh_cagr', 'median'))
        print(f'{"策略家族":<26}{"組合數":>6}{"贏過基準":>9}{"樣本外年化":>12}{"買進持有":>10}')
        for fam, r in by.iterrows():
            print(f'{fam:<26}{int(r["n"]):>6}{r["win_rate"]*100:>8.0f}%'
                  f'{r["oos"]*100:>11.1f}%{r["bh"]*100:>9.1f}%')
        W.to_csv(os.path.join(HERE, 'out', 'ai_walkforward.csv'), index=False)

    print('\n' + '=' * 82)
    print('第二部分：相對強度突破（概念測試，非 SMB 原始規則）')
    print('=' * 82)
    syms2 = ['SPY', 'QQQ', 'IWM', 'DIA', 'GLD', 'TLT', 'EEM',
             'XLE', 'XLB', 'XLF', 'XLK', 'XLV']
    r = rel_strength(syms2)
    if r:
        print(f'{r["n_assets"]} 支 ETF，{r["years"]:.1f} 年')
        print(f'  年化 {r["cagr"]*100:6.2f}%   回撤 {r["mdd"]*100:6.1f}%'
              f'   平均曝險 {r["expo"]*100:.1f}%')
        print(f'  同曝險基準 {r["em_cagr"]*100:6.2f}%   等權買進持有 {r["bh_cagr"]*100:6.2f}%')
        print(f'  策略 vs 同曝險：{(r["cagr"]-r["em_cagr"])*100:+.2f} pp')
        print(f'  t 值 {_sharp(r["pnl"]):.2f}（Sharpe）')


if __name__ == '__main__':
    main()
