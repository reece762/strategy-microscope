#!/usr/bin/env python3
"""用富途 OpenD 抓美股股票池的日 K（單一來源、不與 Yahoo 混接）。

為什麼整批重抓而不沿用 Yahoo 的 65 支
------------------------------------
OpenD 的 `autype='qfq'` 是「拆股＋股利」都調整過的序列；Yahoo 的 close 一般只調拆股。
把兩種基準串在一起會在接縫處製造假跳空（既有筆記量到中位 +18%）。
而且日期連續看不出來，只有把兩個來源在重疊日相除才會現形。
所以整個股票池一律從 OpenD 重抓，不拼接。

額度
----
`get_history_kl_quota` 回 (已用, 剩餘)。額度按「7 天滾動窗內的不同代碼」計，
同一支重抓不另計。抓之前先讀，不夠就中止而不是抓到一半失敗。

進度可續跑
----------
每支成功就落地一個 CSV；重跑時跳過已完成的。這樣中途停掉不會重做。
"""
import argparse
import os
import sys
import time

import pandas as pd
from futu import RET_OK, OpenQuoteContext

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'data_opend')
os.makedirs(OUT, exist_ok=True)

HOST, PORT = '127.0.0.1', 11111
START, END = '2006-01-01', '2025-10-25'
MAX_COUNT = 1000
SLEEP = 0.55          # 速率上限約 60 次/30 秒 ⇒ 每次至少 0.5 秒

BENCHMARKS = ['SPY', 'QQQ', 'IWM', 'DIA']


def tickers():
    """從 fetch_universe.py 取清單，去掉重複標記與非字母項，再加上基準 ETF。"""
    sys.path.insert(0, HERE)
    from fetch_universe import TICKERS as RAW
    seen, out = set(), []
    for t in RAW:
        t = t.upper()
        if '(dup)' in t.lower() or not t.isalpha() or t in seen:
            continue
        seen.add(t)
        out.append(t)
    for b in BENCHMARKS:
        if b not in seen:
            seen.add(b)
            out.append(b)
    return out


def log(msg):
    print(f'{time.strftime("%H:%M:%S")} {msg}', flush=True)


def pull(ctx, sym):
    code = 'US.' + sym.replace('-', '.')
    frames, page, calls = [], None, 0
    while True:
        r, d, page = ctx.request_history_kline(
            code, start=START, end=END, autype='qfq',
            max_count=MAX_COUNT, page_req_key=page)
        calls += 1
        if r != RET_OK:
            return None, str(d)[:120], calls
        if len(d):
            frames.append(d)
        if page is None:
            break
        time.sleep(SLEEP)
    if not frames:
        return None, 'empty', calls
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=['time_key']).sort_values('time_key')
    return df, None, calls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-minutes', type=float, default=25)
    args = ap.parse_args()
    t0 = time.time()

    syms = tickers()
    log(f'目標 {len(syms)} 支（含基準 {BENCHMARKS}）')

    ctx = OpenQuoteContext(host=HOST, port=PORT)
    try:
        r, st = ctx.get_global_state()
        if r != RET_OK or not st.get('qot_logined'):
            log(f'✗ 未登入，中止。state={st}')
            return 1

        r, q = ctx.get_history_kl_quota(get_detail=False)
        used, remain = (q[0], q[1]) if r == RET_OK else (None, None)
        todo = [s for s in syms
                if not os.path.exists(os.path.join(OUT, f'{s}.csv'))]
        log(f'額度 已用 {used} / 剩 {remain}；待抓 {len(todo)} 支')

        if remain is not None and len(todo) > remain:
            log(f'⚠ 待抓超過剩餘額度 {remain}，只抓前 {remain} 支')

        ok = skip = fail = 0
        for i, s in enumerate(todo, 1):
            if time.time() - t0 > args.max_minutes * 60:
                log(f'⏱ 達時間上限，停在 {i}/{len(todo)}（可續跑）')
                break
            if remain is not None and ok >= remain:
                log('額度用完，停止')
                break

            path = os.path.join(OUT, f'{s}.csv')
            if os.path.exists(path):
                skip += 1
                continue

            df, err, calls = pull(ctx, s)
            if err:
                fail += 1
                log(f'  [{i}/{len(todo)}] ✗ {s} {err}')
                time.sleep(SLEEP)
                continue

            df.to_csv(path, index=False)
            ok += 1
            span = f'{df["time_key"].iloc[0][:10]}~{df["time_key"].iloc[-1][:10]}'
            log(f'  [{i}/{len(todo)}] ✓ {s} {len(df):>5} 筆 {span} ({calls} 呼叫)')
            time.sleep(SLEEP)

        r, q2 = ctx.get_history_kl_quota(get_detail=False)
        log(f'完成：成功 {ok}、跳過 {skip}、失敗 {fail}')
        if r == RET_OK:
            log(f'額度 已用 {q2[0]} / 剩 {q2[1]}')
    finally:
        ctx.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
