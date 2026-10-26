#!/usr/bin/env python3
"""Yahoo Finance 日線 OHLC 抓取與快取。

為什麼要這支
------------
專案原有的美股資料有兩個硬限制：
  - data_daily/*.csv 只有 2016-10 起（約 10 年），且只有 12 支 ETF
  - xbx/data/*.csv 的指數雖然有 55 年，但只有收盤價，沒有開盤
Rayner Teo 的宣稱動輒「25 年」，且他的均值回歸規則明確要求「隔天開盤買進」，
沒有開盤價就無法照規則跑。Yahoo 的 chart API 免金鑰、回傳完整 OHLC，
剛好補上這兩點。

注意：close 是原始價（未調整），open/high/low 也是原始價。
      四者同源，做「單筆交易報酬」是對的；不要拿 close 去算多年複利
      （那會漏掉股息，但對「策略 vs 基準」的相對比較影響一致）。
"""
import json
import os
import time
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, 'data_yahoo')
os.makedirs(CACHE, exist_ok=True)

UA = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
      '(KHTML, like Gecko) Chrome/126.0 Safari/537.36')


def fetch(sym, rng='30y', interval='1d', force=False):
    """回傳 DataFrame(index=date, open/high/low/close/volume)。"""
    path = os.path.join(CACHE, f'{sym}_{rng}.csv')
    if os.path.exists(path) and not force:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df

    url = (f'https://query1.finance.yahoo.com/v8/finance/chart/{sym}'
           f'?range={rng}&interval={interval}')
    req = urllib.request.Request(url, headers={'User-Agent': UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.load(r)

    res = d['chart']['result'][0]
    ts = res['timestamp']
    q = res['indicators']['quote'][0]
    df = pd.DataFrame({
        'open': q['open'], 'high': q['high'], 'low': q['low'],
        'close': q['close'], 'volume': q.get('volume'),
    }, index=pd.to_datetime(ts, unit='s', utc=True).tz_convert('America/New_York')
       .normalize().tz_localize(None))
    df = df.dropna(subset=['close']).sort_index()
    df = df[~df.index.duplicated(keep='last')]
    df.to_csv(path)
    time.sleep(1.2)          # 對 Yahoo 客氣一點
    return df


def multi(syms, rng='30y'):
    out = {}
    for s in syms:
        try:
            out[s] = fetch(s, rng)
            print(f'  {s:<6} {len(out[s]):>6} 列  '
                  f'{out[s].index[0].date()} ~ {out[s].index[-1].date()}', flush=True)
        except Exception as e:
            print(f'  {s:<6} 失敗 {type(e).__name__}: {str(e)[:80]}', flush=True)
    return out


if __name__ == '__main__':
    print('抓取長歷史日線…')
    multi(['SPY', 'QQQ', 'IWM', 'DIA', 'GLD', 'TLT', 'EEM', 'XLE',
           'XLB', 'XLF', 'XLK', 'XLV'], '30y')
