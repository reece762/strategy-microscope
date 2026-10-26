#!/usr/bin/env python3
"""對照組：把 VCP 的「型態條件」拿掉，只留趨勢過濾與突破。

這是要回答一個問題：**VCP 這個型態本身有沒有貢獻？**

VCP 策略的規則可以拆成三塊：
  (A) 趨勢過濾：股價 > MA50 > MA200，兩條均線都向上
  (B) 突破：收盤價創近期新高，且量能放大 1.2 倍
  (C) VCP 型態：第一段收縮 20-60%、最後一段 ≤10%、逐段遞減

如果 (A)+(B) 自己就能賺到差不多的錢，那 (C) 就是裝飾品，
「學看 VCP 型態」這件事在資料上沒有價值。
"""
import glob
import json
import os

import numpy as np
import pandas as pd

STOCK = '/root/.openclaw/stock_data'
OUT = '/root/.openclaw/workspace/reports/_vcp'
os.makedirs(OUT, exist_ok=True)

STOP_PCT = 0.07
MAX_HOLD = 120
VOL_MULT = 1.2
BREAK_LOOKBACK = 60      # 「近期新高」＝ 60 日內最高


def load():
    files = sorted(glob.glob(f'{STOCK}/live/*.csv') + glob.glob(f'{STOCK}/extra/*.csv'))
    out = {}
    for f in files:
        s = os.path.basename(f)[:-4].upper()
        d = pd.read_csv(f, usecols=['date', 'open', 'high', 'low', 'close', 'volume'])
        d['date'] = pd.to_datetime(d['date'], errors='coerce')
        d = d.dropna(subset=['date', 'close']).groupby('date').last().sort_index()
        if len(d) < 400:
            continue
        out[s] = d
    return out


def signals(df):
    """趨勢過濾 + 突破（完全不看 VCP 型態）。"""
    close = df['close'].to_numpy(float)
    high = df['high'].to_numpy(float)
    vol = df['volume'].to_numpy(float)
    n = len(close)
    ma50 = pd.Series(close).rolling(50).mean().to_numpy()
    ma200 = pd.Series(close).rolling(200).mean().to_numpy()
    vol20 = pd.Series(vol).rolling(20).mean().to_numpy()
    hh = pd.Series(high).rolling(BREAK_LOOKBACK).max().shift(1).to_numpy()

    out = []
    last = -999
    for j in range(210, n - 1):
        if j - last < 20:            # 同一檔不連續重複進場，間隔至少 20 日
            continue
        c = close[j]
        if not (c > ma50[j] > ma200[j]):
            continue
        if not (ma50[j] > ma50[j - 20] and ma200[j] > ma200[j - 20]):
            continue
        if not (hh[j] == hh[j] and c > hh[j]):   # 創 60 日新高
            continue
        if not (vol20[j] > 0 and vol[j] >= VOL_MULT * vol20[j]):
            continue
        out.append(j)
        last = j
    return out


def run(stocks):
    tr = []
    for s, df in stocks.items():
        close = df['close'].to_numpy(float)
        low = df['low'].to_numpy(float)
        n = len(close)
        for j in signals(df):
            stop = close[j] * (1 - STOP_PCT)
            ej, reason, px = None, None, None
            for k in range(j + 1, min(j + MAX_HOLD + 1, n)):
                if low[k] <= stop:
                    ej, reason, px = k, 'stop', stop
                    break
            if ej is None:
                ej = min(j + MAX_HOLD, n - 1)
                reason, px = 'time', close[ej]
            tr.append(dict(symbol=s, entry_date=str(df.index[j].date()),
                           exit_date=str(df.index[ej].date()),
                           entry=float(close[j]), exit=float(px),
                           ret=float(px / close[j] - 1), reason=reason,
                           hold=int(ej - j)))
    return pd.DataFrame(tr)


def stats(t):
    if t.empty:
        return {}
    r = t['ret']
    w = (r > 0).mean()
    return dict(n=int(len(t)), win=float(w), avg=float(r.mean()),
                med=float(r.median()),
                avg_win=float(r[r > 0].mean()) if (r > 0).any() else 0.0,
                avg_loss=float(r[r <= 0].mean()) if (r <= 0).any() else 0.0,
                stop_rate=float((t['reason'] == 'stop').mean()),
                avg_hold=float(t['hold'].mean()),
                big_win_share=float((r > 0.20).mean()),
                trimmed=float(t[r < r.quantile(0.95)]['ret'].mean()))


def main():
    print('載入…', flush=True)
    S = load()
    print(f'  {len(S)} 檔', flush=True)
    t = run(S)
    print(f'  交易 {len(t)} 筆', flush=True)
    t.to_csv(f'{OUT}/ctrl_trades.csv', index=False)
    st = stats(t)

    # 逐年
    t['year'] = pd.to_datetime(t['entry_date']).dt.year
    yr = t.groupby('year')['ret'].agg(['count', 'mean']).to_dict('index')

    json.dump(dict(stats=st, by_year={str(k): v for k, v in yr.items()}),
              open(f'{OUT}/ctrl_result.json', 'w'), ensure_ascii=False, indent=1)
    print('\n=== 對照組（純趨勢+突破，無 VCP）===')
    for k, v in st.items():
        print(f'  {k:<12} {v:.4f}' if isinstance(v, float) else f'  {k:<12} {v}')

    # 與 VCP 對比
    vp = json.load(open(f'{OUT}/vcp_result.json'))
    v = vp['stats']
    print('\n=== VCP vs 對照組 ===')
    rows = [('交易數', v['n'], st['n']),
            ('勝率', v['win'], st['win']),
            ('平均報酬', v['avg'], st['avg']),
            ('中位報酬', v['med'], st['med']),
            ('平均獲利', v['avg_win'], st['avg_win']),
            ('停損比例', v['stop_rate'], st['stop_rate']),
            ('大賺(>20%)比例', v['big_win_share'] if 'big_win_share' in v else
             float((pd.read_csv(f'{OUT}/vcp_trades.csv')['ret'] > 0.20).mean()),
             st['big_win_share']),
            ('剔除最好5%後', vp['trimmed_mean'], st['trimmed'])]
    print(f"  {'項目':<16}{'VCP':>12}{'對照組':>12}{'差':>12}")
    for name, a, b in rows:
        if name in ('交易數',):
            print(f'  {name:<16}{a:>12,}{b:>12,}{b-a:>12,}')
        else:
            print(f'  {name:<16}{a*100:>11.2f}%{b*100:>11.2f}%{(a-b)*100:>11.2f}pp')


if __name__ == '__main__':
    main()
