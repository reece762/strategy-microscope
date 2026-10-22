#!/usr/bin/env python3
"""Fetch the monthly ETHUSDT daily klines that match the BTC series we already have.

Binance publishes one zip per month at data.binance.vision. We mirror exactly the
span used for BTC (2017-08 .. 2026-09) so both benchmarks start on the same day.
Existing files are skipped, so re-running is cheap.
"""
import os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
BASE = 'https://data.binance.vision/data/spot/monthly/klines/ETHUSDT/1d'


def months():
    out = []
    y, m = 2017, 8
    while (y, m) <= (2026, 9):
        out.append(f'{y}-{m:02d}')
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def fetch(ym):
    name = f'ETHUSDT-1d-{ym}.zip'
    path = os.path.join(DATA, name)
    if os.path.exists(path) and os.path.getsize(path) > 500:
        return name, 'skip'
    r = subprocess.run(['curl', '-sS', '-m', '60', '-o', path, '-w', '%{http_code}',
                        f'{BASE}/{name}'], capture_output=True, text=True)
    code = (r.stdout or '').strip()
    if code != '200' or os.path.getsize(path) < 500:
        if os.path.exists(path):
            os.remove(path)
        return name, f'FAIL {code}'
    return name, 'ok'


if __name__ == '__main__':
    ms = months()
    print(f'downloading {len(ms)} monthly files ...')
    ok = skip = fail = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for name, status in ex.map(fetch, ms):
            if status == 'ok':
                ok += 1
            elif status == 'skip':
                skip += 1
            else:
                fail += 1
                print(' ', name, status)
    print(f'ok={ok} skip={skip} fail={fail}')
    have = [f for f in os.listdir(DATA) if f.startswith('ETHUSDT-1d-')]
    print(f'ETH files on disk: {len(have)}')
    sys.exit(1 if fail else 0)
