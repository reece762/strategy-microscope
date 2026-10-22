#!/usr/bin/env python3
"""替 tier2 三張「有真實回測、卻沒有曲線」的卡片補上淨值。

為什麼要補：馬克維茨月配置、比特幣均線定投、BTC/ETH 輪動，
正文都給了年化與淨值倍數，屬於真的跑過的回測；
若在詳情頁寫「統計檢定」或「無法回測」都是錯的。

作法：用 tier2.py 完全相同的規則重算，並斷言年化／倍數與 tier2.json 相符。
- 馬克維茨：月度再平衡，取等權版（正文引用的對照是 SPY 買進持有）
- BTC/ETH 輪動：N=20 純多版，對照 BTC 單獨持有
- 定投：現金流型，沒有單一淨值曲線（投入與市值是兩條現金流），
        因此不畫圖，改用卡片上的 chart_note 說明。
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

import tier2 as T2

RED = '#f85149'
BLUE = '#58a6ff'


def _card(cid, label, pairs, ppy=252):
    """pairs = [(label, series, color)]"""
    df = pd.DataFrame({lb: s for lb, s, _ in pairs}).dropna()
    df = df / df.iloc[0]
    idx = list(df.index)
    return dict(
        id=cid, label=label, ppy=ppy,
        d=[(pd.Timestamp(x).normalize() - EPOCH).days for x in idx],
        series=[dict(id=lb, label=lb, color=co, ppy=ppy,
                     v=[round(float(v), 4) for v in df[lb]])
                for lb, _, co in pairs],
        start=str(idx[0].date()), end=str(idx[-1].date()),
    )


def markowitz_curve():
    syms = ('SPY', 'EEM', 'TLT', 'GLD', 'DBC', 'IWM')
    px = pd.DataFrame({s: T2.load_etf(s)['close'] for s in syms}).dropna()
    rets = px.pct_change().dropna()
    month_end = rets.index.to_series().groupby(
        [rets.index.year, rets.index.month]).max()

    eq = [1.0]
    dates = [rets.index[0]]
    prev_w = np.repeat(1.0 / len(syms), len(syms))
    for idx, rday in enumerate(month_end.values):
        if idx == 0:
            continue
        hist = rets.loc[:rday].tail(6 * 21)
        if len(hist) < 6 * 15:
            continue
        w = np.repeat(1.0 / len(syms), len(syms))     # 等權版
        nxt = month_end.values[idx + 1] if idx + 1 < len(month_end) else rets.index[-1]
        seg = rets.loc[pd.Timestamp(rday):pd.Timestamp(nxt)].iloc[1:]
        for _, row in seg.iterrows():
            r = float(np.dot(w, row.values))
            turnover = float(np.abs(w - prev_w).sum())
            r -= turnover * 0.0001
            eq.append(eq[-1] * (1 + r))
            dates.append(row.name)
        prev_w = w
    e = pd.Series(eq, index=pd.DatetimeIndex(dates)).groupby(level=0).last()
    spy = px['SPY'].reindex(e.index).ffill()
    return e, spy / spy.iloc[0]


def rotation_curve():
    def load(sym):
        import glob, zipfile, io
        frames = []
        for f in sorted(glob.glob(os.path.join(T2.DATA, f'{sym}-1d-*.zip'))):
            with zipfile.ZipFile(f) as z:
                raw = z.read(z.namelist()[0]).decode()
            d = pd.read_csv(io.StringIO(raw), header=None).iloc[:, :6]
            d.columns = ['ts', 'open', 'high', 'low', 'close', 'volume']
            frames.append(d)
        d = pd.concat(frames, ignore_index=True)
        ts = pd.to_numeric(d['ts'], errors='coerce')
        ts = ts.where(ts < 1e14, ts / 1000.0)
        d['date'] = pd.to_datetime(ts, unit='ms').dt.normalize()
        d['close'] = pd.to_numeric(d['close'], errors='coerce')
        return d.dropna(subset=['close']).drop_duplicates('date') \
                .set_index('date').sort_index()['close']

    px = pd.DataFrame({'BTC': load('BTCUSDT'), 'ETH': load('ETHUSDT')}).dropna()
    rets = px.pct_change().fillna(0.0)
    N = 20
    mb = px['BTC'] / px['BTC'].shift(N) - 1
    ms = px['ETH'] / px['ETH'].shift(N) - 1
    pick_btc = mb >= ms
    wb = pick_btc.astype(float).shift(1).fillna(0.0)
    we = (~pick_btc).astype(float).shift(1).fillna(0.0)
    turn = wb.diff().abs().fillna(0.0) + we.diff().abs().fillna(0.0)
    r = wb * rets['BTC'] + we * rets['ETH'] - turn * 0.0025
    eq = (1 + r).cumprod()
    btc_bh = px['BTC'] / px['BTC'].iloc[0]
    return eq, btc_bh


def main():
    t2 = json.load(open(os.path.join(OUT, 'tier2.json')))
    cards = {}

    e, spy = markowitz_curve()
    yrs = (e.index[-1] - e.index[0]).days / 365.25
    cagr = float(e.iloc[-1] ** (1 / yrs) - 1)
    ref = t2['markowitz']['equal']['cagr']
    assert abs(cagr - ref) < 1e-6, f'馬克維茨年化對不上：{cagr:.6f} vs {ref:.6f}'
    cards['t2-markowitz'] = _card(
        't2-markowitz', '馬克維茨（等權）',
        [('馬克維茨等權', e, RED), ('單獨持有 SPY', spy, BLUE)])
    print(f"  馬克維茨 年化 {cagr*100:6.2f}%  ✓ 與 tier2.json 相符")

    e2, btc = rotation_curve()
    yrs2 = (e2.index[-1] - e2.index[0]).days / 365.25
    cagr2 = float(e2.iloc[-1] ** (1 / yrs2) - 1)
    ref2 = t2['rotation']['N20_plain']['cagr']
    assert abs(cagr2 - ref2) < 1e-6, f'輪動年化對不上：{cagr2:.6f} vs {ref2:.6f}'
    cards['t2-btceth'] = _card(
        't2-btceth', 'BTC/ETH 輪動（N=20）',
        [('20 日動能輪動', e2, RED), ('單獨持有 BTC', btc, BLUE)], ppy=365)
    print(f"  BTC/ETH 輪動 年化 {cagr2*100:6.2f}%  ✓ 與 tier2.json 相符")

    json.dump(cards, open(os.path.join(OUT, 'tier2_curves.json'), 'w'))
    print(f'written out/tier2_curves.json  {list(cards)}')


if __name__ == '__main__':
    main()
