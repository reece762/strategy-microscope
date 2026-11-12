#!/usr/bin/env python3
"""Fetch ~24 months of 1-minute US bars via Futu OpenD for the ORB test.

Output: out/orb1m/<SYM>.csv (gitignored). Resumable: skips files already written.
Futu's intraday depth is capped at 200 pages / 200,000 bars per request chain,
so 24 months (~200k bars) is the practical maximum.
"""
import os
import sys
import time

import pandas as pd
from futu import OpenQuoteContext, RET_OK

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out', 'orb1m')
os.makedirs(OUT, exist_ok=True)

SYMS = sys.argv[1:] or ['QQQ', 'SPY', 'IWM', 'NVDA', 'AAPL']
START = '2024-11-01'
END = '2026-11-12'
COLS = ['code', 'name', 'time_key', 'open', 'close', 'high', 'low',
        'pe_ratio', 'turnover_rate', 'volume', 'turnover', 'change_rate', 'last_close']


def fetch(ctx, sym):
    code = f'US.{sym}'
    path = os.path.join(OUT, f'{sym}.csv')
    if os.path.exists(path) and os.path.getsize(path) > 100_000:
        print(f'{sym}: cached {os.path.getsize(path)/1e6:.1f} MB', flush=True)
        return
    rows, key, pages = [], None, 0
    while True:
        ret, data, key = ctx.request_history_kline(
            code, start=START, end=END, ktype='K_1M', autype='qfq',
            max_count=1000, page_req_key=key)
        if ret != RET_OK:
            print(f'{sym}: ERROR page {pages}: {data}', flush=True)
            break
        rows.append(data)
        pages += 1
        if pages % 20 == 0:
            print(f'{sym}: {pages} pages / {sum(len(x) for x in rows)} bars', flush=True)
        if key is None:
            break
        time.sleep(0.55)  # stay under ~60 requests / 30 s
    if not rows:
        print(f'{sym}: nothing fetched', flush=True)
        return
    df = pd.concat(rows, ignore_index=True)
    df = df[COLS].drop_duplicates('time_key').sort_values('time_key')
    df.to_csv(path, index=False)
    print(f'{sym}: {len(df)} bars {df.time_key.min()} -> {df.time_key.max()}', flush=True)


def main():
    ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
    try:
        for s in SYMS:
            t0 = time.time()
            fetch(ctx, s)
            print(f'{s}: {time.time()-t0:.0f}s', flush=True)
    finally:
        ctx.close()


if __name__ == '__main__':
    main()
