#!/usr/bin/env python3
"""Rayner Teo 顺势／动量系统：複验他文章裡给的可执行规则。

规则来源（他本人文章，非我推测）：
  A) 系统化顺势（多市场期货版）
     - 收盘价创 200 日新高 → 做多
     - 收盘价创 200 日新低 → 做空
     - 6 ATR 移动停损
     - 每笔风险 1% 资金
     他宣称：937 笔 / 勝率 42.8% / 年化 9.89% / 最大回撤 24.12%
     市场：黄金、铜、白银、钯、铂、标普 500、EUR/JPY、EUR/USD、披索、英镑、
           美债、BOBL、BUXL、BTP、加拿大十年债、取暖油、小麦、玉米、木材、糖

  B) 股票版顺势
     - 股票创 50 週新高 → 做多
     - 20% 移动停损
     - 候选太多时，取过去 50 週涨幅最大的前 20 支
     - 最多 20 支、每支不超过 5% 资金
     他宣称：707 笔 / 勝率 48.66% / 年化 12.81% / 最大回撤 40.75%

重要限制：我手上没有期货资料，也没有 Russell 1000 成分股。
所以 A 只能测「同样规则套在能拿到的资产上」，B 只能测「用 ETF 当股票池」。
这会在报告里讲清楚，不会假装测了他的原始市场。
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from yahoo import fetch  # noqa: E402


def atr(df, n=20):
    h, l, c = df['high'], df['low'], df['close']
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def breakout_system(df, lookback=200, atr_n=20, atr_mult=6.0,
                    risk=0.01, allow_short=True, cost_bp=2.0):
    """200 日突破 + N×ATR 移动停损 + 每笔风险固定比例。

    部位规模 = risk / (atr_mult * ATR / price)
    这样每笔最大亏损（价格触及停损时）恰好是 risk × 资金，
    与他「Risk 1% of your capital on each trade」一致。
    """
    c = df['close'].to_numpy()
    a = atr(df, atr_n).to_numpy()
    n = len(c)

    hi = df['close'].rolling(lookback).max().shift(1).to_numpy()
    lo = df['close'].rolling(lookback).min().shift(1).to_numpy()

    pos = np.zeros(n)          # 目标权重（带符号）
    cur = 0.0                  # 当前方向 +1/-1/0
    stop = np.nan              # 多头停损价
    peak = np.nan

    for i in range(1, n):
        if not np.isfinite(a[i]) or not np.isfinite(hi[i]):
            pos[i] = 0.0
            continue

        px = c[i]
        if cur == 0.0:
            if px > hi[i]:
                cur = 1.0
                peak = px
                stop = px - atr_mult * a[i]
            elif allow_short and px < lo[i]:
                cur = -1.0
                peak = px
                stop = px + atr_mult * a[i]
        else:
            if cur > 0:
                peak = max(peak, px)
                stop = max(stop, peak - atr_mult * a[i])
                if px < stop:
                    cur = 0.0
            elif cur < 0:
                peak = min(peak, px)
                stop = min(stop, peak + atr_mult * a[i])
                if px > stop:
                    cur = 0.0

        riskdist = atr_mult * a[i] / px
        w = min(1.0, risk / riskdist) if riskdist > 0 else 0.0
        pos[i] = cur * w

    ps = pd.Series(pos, index=df.index)
    held = ps.shift(1).fillna(0.0)
    ret = df['close'].pct_change().fillna(0.0)
    turn = held.diff().abs().fillna(held.abs())
    pnl = held * ret - turn * cost_bp / 1e4
    eq = (1 + pnl).cumprod()
    return eq, ps, pnl


def perfs(eq, pnl, ppy=252):
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1)
    v = eq.to_numpy()
    mdd = float((v / np.maximum.accumulate(v) - 1).min())
    sd = pnl.std()
    shp = float(pnl.mean() / sd * np.sqrt(ppy)) if sd > 0 else float('nan')
    return dict(cagr=cagr, mdd=mdd, sharpe=shp, multiple=float(eq.iloc[-1]),
                years=yrs)


def tstat(x):
    x = pd.Series(x).dropna()
    if len(x) < 3 or x.std() == 0:
        return float('nan')
    return float(x.mean() / (x.std() / np.sqrt(len(x))))


def exposure_match(ret, pos):
    """同曝險基準：買進持有 × 平均曝險（不含方向，只看投入多少）。"""
    expo = pos.shift(1).fillna(0).abs().mean()
    return (1 + ret * expo).cumprod(), expo


def run_single(sym, rng='30y', **kw):
    df = fetch(sym, rng)
    eq, ps, pnl = breakout_system(df, **kw)
    p = perfs(eq, pnl)
    ret = df['close'].pct_change().fillna(0.0)
    em, expo = exposure_match(ret.dropna(), ps)
    bh = (1 + ret).cumprod()

    # 勝率：以「進場到出場」為一筆交易
    d = ps.diff().fillna(ps)
    trades = []
    entry = None
    for i in range(len(ps)):
        if entry is None and ps.iloc[i] != 0:
            entry = i
        elif entry is not None and ps.iloc[i] == 0:
            seg = df['close'].iloc[entry:i]
            if len(seg) > 1:
                trades.append(float(seg.iloc[-1] / seg.iloc[0] - 1) *
                              (1 if ps.iloc[entry] > 0 else -1))
            entry = None
    tr = pd.Series(trades)
    win = float((tr > 0).mean()) if len(tr) else float('nan')

    return dict(sym=sym, perf=p, expo=expo,
                em_cagr=float(em.iloc[-1] ** (1 / ((em.index[-1] - em.index[0]).days / 365.25)) - 1),
                bh_cagr=float(bh.iloc[-1] ** (1 / ((bh.index[-1] - bh.index[0]).days / 365.25)) - 1),
                n_trades=len(tr), win=win,
                t=tstat(pnl), pnl=pnl, eq=eq, em=em, bh=bh, df=df)


def multi_asset_pf(syms, rng='30y', **kw):
    """多资产组合：每笔风险 1%，各资产等权风险贡献。

    这是最接近他原始设计的版本（他交易 20 多个市场）。
    """
    pnls = {}
    for s in syms:
        try:
            df = fetch(s, rng)
        except Exception as e:
            print(f'    {s} 资料取得失败：{type(e).__name__}')
            continue
        if len(df) < 400:
            continue
        _, _, pnl = breakout_system(df, **kw)
        pnls[s] = pnl
    if not pnls:
        return None
    P = pd.DataFrame(pnls).fillna(0.0)
    # 组合：各资产等权（每笔风险 1% 已内建在部位里）
    port = P.mean(axis=1)
    eq = (1 + port).cumprod()
    p = perfs(eq, port)
    p['n_assets'] = len(pnls)
    p['t'] = tstat(port)
    p['assets'] = list(pnls)
    return dict(perf=p, port=port, eq=eq, pnl_map=P)


if __name__ == '__main__':
    print('=' * 78)
    print('A) 200 日突破 + 6ATR 移动停损 + 每笔风险 1%')
    print('   宣称（20+ 个期货市场）：937 笔 / 勝率 42.8% / 年化 9.89% / 回撤 24.12%')
    print('=' * 78)
    for sym in ('SPY', 'QQQ'):
        try:
            r = run_single(sym)
        except Exception as e:
            print(f'{sym}: {type(e).__name__} {str(e)[:60]}')
            continue
        p, w = r['perf'], r['win']
        print(f'\n{sym}  年化 {p["cagr"]*100:6.2f}%  回撤 {p["mdd"]*100:6.1f}%  '
              f'Sharpe {p["sharpe"]:.2f}  倍数 {p["multiple"]:.2f}x')
        print(f'      交易 {r["n_trades"]} 笔  勝率 {w*100:.1f}%  '
              f'平均曝險 {r["expo"]*100:.1f}%  t值 {r["t"]:.2f}')
        print(f'      同曝險基準年化 {r["em_cagr"]*100:6.2f}%   '
              f'買進持有 {r["bh_cagr"]*100:6.2f}%')

    print('\n' + '=' * 78)
    print('B) 多资产组合（12 支 ETF 当「多市场」代理）')
    print('=' * 78)
    syms = ['SPY', 'QQQ', 'IWM', 'DIA', 'GLD', 'TLT', 'EEM',
            'XLE', 'XLB', 'XLF', 'XLK', 'XLV']
    m = multi_asset_pf(syms)
    if m:
        p = m['perf']
        print(f'\n组合（{p["n_assets"]} 支）')
        print(f'  年化 {p["cagr"]*100:6.2f}%  回撤 {p["mdd"]*100:6.1f}%  '
              f'Sharpe {p["sharpe"]:.2f}  倍数 {p["multiple"]:.2f}x  t值 {p["t"]:.2f}')
        print(f'  涵盖 {p["years"]:.1f} 年')
        print(f'  资产：{", ".join(p["assets"])}')
