#!/usr/bin/env python3
"""把槓鈴與永久投資組合的逐日淨值，接成詳情頁可畫的曲線格式。

為什麼要獨立一支：series.json 由 prep_series.py 產生，重跑會被覆蓋。
把這裡的結果在 build_page 載入後 merge 進去，才不會下次重建就消失。

格式（沿用 series.json 既有結構）：
    id / label / ppy / d（1970-01-01 起算天數）/ series[{id,label,color,ppy,v}] / start / end
"""
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
DATA = os.path.join(os.path.dirname(HERE), 'data_daily')
EPOCH = pd.Timestamp('1970-01-01')

RED = '#f85149'
BLUE = '#58a6ff'
GOLD = '#f0b429'
GREEN = '#3fb950'
GRAY = '#8b949e'
PURPLE = '#bc8cff'


def _days(idx):
    return [(pd.Timestamp(x).normalize() - EPOCH).days for x in idx]


def _series(label, vals, color='#f85149', ppy=252):
    return dict(id=label, label=label, color=color, ppy=ppy,
                v=[round(float(x), 4) for x in vals])


def _card(cid, label, idx, rows, ppy=252):
    """rows = [(label, values, color)]，第一條視為策略本身。"""
    return dict(id=cid, label=label, ppy=ppy, d=_days(idx),
                series=[_series(lb, v, co, ppy) for lb, v, co in rows],
                start=str(pd.Timestamp(idx[0]).date()),
                end=str(pd.Timestamp(idx[-1]).date()))


def _bh_curve(sym, idx):
    """把單一 ETF 在其自身日曆上 rebase 到 1.0，再對齊到 idx。"""
    d = pd.read_csv(os.path.join(DATA, f'{sym}.csv'))
    d.columns = [c.strip().lower() for c in d.columns]
    d['date'] = pd.to_datetime(d['date'], format='%m/%d/%Y', errors='coerce')
    d = d.dropna(subset=['date']).set_index('date').sort_index()
    s = pd.to_numeric(d['close'], errors='coerce').dropna()
    s = s.reindex(pd.DatetimeIndex(idx)).ffill().bfill()
    return (s / s.iloc[0]).values


def build():
    cards = {}

    p = os.path.join(OUT, 'barbell2.json')
    if os.path.exists(p):
        d = json.load(open(p))
        a = d.get('cash_10')
        b = d.get('tlt_10')
        if a and b:
            idx = a['idx']
            rows = [
                ('90% 短債 + 10% 小型股', a['series'], RED),
                ('90% 長債 TLT + 10% 小型股', b['series'], BLUE),
                ('單獨持有 小型股 IWM', _bh_curve('IWM', idx), GREEN),
            ]
            cards['n4-barbell'] = _card('n4-barbell', '槓鈴 90/10', idx, rows)

    p = os.path.join(OUT, 'perm.json')
    if os.path.exists(p):
        d = json.load(open(p))
        sp = d.get('variants', {}).get('spy')
        if sp:
            idx = sp['idx']
            rows = [
                ('永久投資組合（SPY 版）', sp['series'], RED),
                ('單獨持有 SPY', _bh_curve('SPY', idx), BLUE),
            ]
            qq = d.get('variants', {}).get('qqq')
            cards['n4-perm'] = _card('n4-perm', '永久投資組合', idx, rows)

    return cards


if __name__ == '__main__':
    cs = build()
    json.dump(cs, open(os.path.join(OUT, 'new4_curves.json'), 'w'))
    for k, v in cs.items():
        print(f"{k:12s} {v['start']}~{v['end']}  {len(v['d'])} 點  "
              f"{[s['label'] for s in v['series']]}")
    print(f'written out/new4_curves.json  {list(cs)}')
