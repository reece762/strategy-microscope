#!/usr/bin/env python3
"""替「有機械策略回測、卻沒有畫曲線」的卡片補上真實淨值曲線。

背景：MACD 背離、KDJ、量比這三張卡在正文裡都寫了年化數字
（例如 MACD 納斯達克 2.98% vs 買進持有 10.55%），但詳情頁卻顯示
「無法誠實回測」。原因是這些卡片沒有被放進 series.json。

這支腳本用與 tier1.py 完全相同的規則重算淨值，
並且把算出來的最終年化與正文數字比對——對不上就報錯，不默默放行。

輸出 out/tier1_curves.json（格式同 series.json 的 card）
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, 'out')
EPOCH = pd.Timestamp('1970-01-01')

import tier1 as T

RED = '#f85149'
BLUE = '#58a6ff'
GREEN = '#3fb950'
GOLD = '#f0b429'
GRAY = '#8b949e'


def _np(eq, ret, ppy=252):
    """回傳 (淨值序列, 年化, 回撤)。"""
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1)
    mdd = float((eq.values / np.maximum.accumulate(eq.values) - 1).min())
    return eq, cagr, mdd


def _card(cid, label, eq, bh, ppy=252, strat_name='策略', bh_name='買進持有'):
    # 兩條線從同一天起算、都 rebase 到 1.0
    df = pd.DataFrame({'s': eq, 'b': bh}).dropna()
    df = df / df.iloc[0]
    idx = list(df.index)
    return dict(
        id=cid, label=label, ppy=ppy,
        d=[(pd.Timestamp(x).normalize() - EPOCH).days for x in idx],
        series=[
            dict(id='strat', label=strat_name, color=RED, ppy=ppy,
                 v=[round(float(v), 4) for v in df['s']]),
            dict(id='bh', label=bh_name, color=BLUE, ppy=ppy,
                 v=[round(float(v), 4) for v in df['b']]),
        ],
        start=str(idx[0].date()), end=str(idx[-1].date()),
    )


def macd_curves():
    """完全照 tier1.run_macd_divergence 的機械策略：底背離買、頂背離平倉。"""
    out = {}
    for name, df in (('納斯達克綜合指數', T.load_fred('NASDAQCOM')),
                     ('SPY', T.load_etf('SPY')),
                     ('QQQ', T.load_etf('QQQ')),
                     ('BTC', T.load_binance('BTCUSDT'))):
        close = df['close']
        b_idx, _, _ = T.divergence(close, top=False)
        t_idx, _, _ = T.divergence(close, top=True)
        bs, ts = set(b_idx), set(t_idx)
        pos = pd.Series(0.0, index=close.index)
        cur = 0.0
        for i in range(len(close)):
            if i in bs:
                cur = 1.0
            elif i in ts:
                cur = 0.0
            pos.iloc[i] = cur
        held = pos.shift(1).fillna(0.0)
        ret = close.pct_change().fillna(0.0)
        cost = held.diff().abs().fillna(0.0) * 0.0001
        strat = (1 + held * ret - cost).cumprod()
        bh = (1 + ret).cumprod()
        ppy = 365 if name == 'BTC' else 252
        _, cagr, mdd = _np(strat, ret, ppy)
        out[name] = dict(eq=strat, bh=bh, ppy=ppy, cagr=cagr, mdd=mdd)
    return out


def kdj_curves():
    """照 tier1.run_kdj：金叉買、死叉賣，隔日生效。"""
    out = {}
    for name, df in (('SPY', T.load_etf('SPY')),
                     ('QQQ', T.load_etf('QQQ')),
                     ('BTC', T.load_binance('BTCUSDT'))):
        k, d, _ = T.kdj(df)
        close = df['close']
        cu = (k > d) & (k.shift(1) <= d.shift(1))
        cd = (k < d) & (k.shift(1) >= d.shift(1))
        pos = pd.Series(np.nan, index=close.index)
        pos[cu.fillna(False)] = 1.0
        pos[cd.fillna(False)] = 0.0
        pos = pos.ffill().fillna(0.0)
        held = pos.shift(1).fillna(0.0)
        ret = close.pct_change().fillna(0.0)
        cost = held.diff().abs().fillna(0.0) * 0.0001
        strat = (1 + held * ret - cost).cumprod()
        bh = (1 + ret).cumprod()
        ppy = 365 if name == 'BTC' else 252
        _, cagr, mdd = _np(strat, ret, ppy)
        out[name] = dict(eq=strat, bh=bh, ppy=ppy, cagr=cagr, mdd=mdd)
    return out


def vr_curves():
    """照 tier1.run_volume_ratio：量比 2.5–5 買、持有 2 日。"""
    out = {}
    for name in ('SPY', 'QQQ'):
        df = T.load_etf(name)
        vr = T.volume_ratio(df)
        close = df['close']
        buy = (vr > 2.5) & (vr < 5)
        sig = buy.shift(1).fillna(False)
        hold = pd.Series(0.0, index=close.index)
        hold[sig] = 1.0
        hold = hold.rolling(2, min_periods=1).max()
        ret = close.pct_change().fillna(0.0)
        cost = hold.diff().abs().fillna(0.0) * 0.0001
        strat = (1 + hold * ret - cost).cumprod()
        bh = (1 + ret).cumprod()
        _, cagr, mdd = _np(strat, ret, 252)
        out[name] = dict(eq=strat, bh=bh, ppy=252, cagr=cagr, mdd=mdd)
    return out


def main():
    t1 = json.load(open(os.path.join(OUT, 'tier1.json')))
    cards = {}
    report = []

    # ---- MACD：用納斯達克當代表（正文引用的就是這一組）----
    m = macd_curves()
    ref = t1['macd']['納斯達克綜合指數']['strategy']['cagr']
    got = m['納斯達克綜合指數']['cagr']
    assert abs(got - ref) < 1e-6, f'MACD 年化對不上：重算 {got:.6f} vs tier1 {ref:.6f}'
    cards['t1-macd'] = _card('t1-macd', 'MACD 背離（納斯達克）',
                             m['納斯達克綜合指數']['eq'], m['納斯達克綜合指數']['bh'],
                             252, '底背離買進／頂背離平倉', '買進持有納斯達克')
    for k, v in m.items():
        report.append(f"  MACD {k:16s} 年化 {v['cagr']*100:6.2f}%  回撤 {v['mdd']*100:7.2f}%")

    # ---- KDJ：用 SPY 當代表 ----
    kd = kdj_curves()
    ref = t1['kdj']['SPY']['strategy']['cagr']
    got = kd['SPY']['cagr']
    assert abs(got - ref) < 1e-6, f'KDJ 年化對不上：重算 {got:.6f} vs tier1 {ref:.6f}'
    cards['t1-kdj'] = _card('t1-kdj', 'KDJ 金叉死叉（SPY）',
                            kd['SPY']['eq'], kd['SPY']['bh'], 252,
                            '金叉買進／死叉賣出', '買進持有 SPY')
    for k, v in kd.items():
        report.append(f"  KDJ  {k:16s} 年化 {v['cagr']*100:6.2f}%  回撤 {v['mdd']*100:7.2f}%")

    # ---- 量比：SPY ----
    v = vr_curves()
    ref = t1['vr']['SPY']['strategy']['cagr']
    got = v['SPY']['cagr']
    assert abs(got - ref) < 1e-6, f'量比年化對不上：重算 {got:.6f} vs tier1 {ref:.6f}'
    cards['t1-vr'] = _card('t1-vr', '量比 2.5–5（SPY）',
                           v['SPY']['eq'], v['SPY']['bh'], 252,
                           '量比 2.5–5 買進／持有 2 日', '買進持有 SPY')
    for k, x in v.items():
        report.append(f"  量比 {k:16s} 年化 {x['cagr']*100:6.2f}%  回撤 {x['mdd']*100:7.2f}%")

    json.dump(cards, open(os.path.join(OUT, 'tier1_curves.json'), 'w'))
    print('=== 重算結果（已與 tier1.json 對帳，全部相符）===')
    print('\n'.join(report))
    print(f'\nwritten out/tier1_curves.json  卡片 {list(cards)}')


if __name__ == '__main__':
    main()
