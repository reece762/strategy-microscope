#!/usr/bin/env python3
"""為「六策略對比」那張卡片產生淨值曲線。

用唐奇安突破（六個策略裡唯一顯著偏離同曝險基準的一個）畫三條線：
策略本身、同曝險不擇時、買進持有。三條都從同一天、同樣 rebase 到 1.0。
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

from alpha_data import load_etf, run           # noqa: E402
from probe_alpha_pa import s_breakout          # noqa: E402

RED = '#f85149'
BLUE = '#58a6ff'
GREEN = '#3fb950'


def main():
    df = load_etf('SPY')
    pos = s_breakout(df).values
    r = run(df, pos, 5.0, 252)

    s = pd.Series(r['equity_strat'], index=df.index)
    e = pd.Series(r['equity_em'], index=df.index)
    b = pd.Series(r['equity_bh'], index=df.index)

    d = pd.DataFrame({'s': s, 'e': e, 'b': b}).dropna()
    d = d / d.iloc[0]
    idx = list(d.index)

    card = dict(
        id='al-six', label='唐奇安突破（20 日）vs 同曝險 vs 買進持有',
        ppy=252,
        d=[int((pd.Timestamp(x).normalize() - EPOCH).days) for x in idx],
        series=[
            dict(id='strat', label='唐奇安突破（做多訊號）', color=RED, ppy=252,
                 v=[round(float(v), 4) for v in d['s']]),
            dict(id='em', label='同曝險、不擇時', color=BLUE, ppy=252,
                 v=[round(float(v), 4) for v in d['e']]),
            dict(id='bh', label='買進持有 SPY', color=GREEN, ppy=252,
                 v=[round(float(v), 4) for v in d['b']]),
        ],
        start=str(idx[0].date()), end=str(idx[-1].date()),
    )

    # 斷言：曲線終點必須與 alpha.json 的數字一致
    a = json.load(open(os.path.join(OUT, 'alpha.json')))
    ref = a['six']['SPY']['唐奇安突破(20)']
    got_s = float(d['s'].iloc[-1])
    yrs = (idx[-1] - idx[0]).days / 365.25
    got_cagr = got_s ** (1 / yrs) - 1
    assert abs(got_cagr - ref['strat_cagr']) < 2e-3, \
        f'唐奇安年化對不上：曲線 {got_cagr:.4f} vs alpha.json {ref["strat_cagr"]:.4f}'

    p = os.path.join(OUT, 'alpha_curves.json')
    json.dump({'al-six': card}, open(p, 'w'), ensure_ascii=False)
    print(f'wrote {p}: strat {got_cagr*100:.2f}%  '
          f'em {(d["e"].iloc[-1]**(1/yrs)-1)*100:.2f}%  '
          f'bh {(d["b"].iloc[-1]**(1/yrs)-1)*100:.2f}%')


if __name__ == '__main__':
    main()
