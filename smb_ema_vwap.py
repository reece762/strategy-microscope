#!/usr/bin/env python3
"""SMB「Fashionably Late」9 EMA / VWAP 盤中剝頭皮策略回測。

規則（全部出自 Justin Spiro 在 SMB 影片的原話）
-----------------------------------------------
1. 個股從當日低點翻上去（先下殺、做出低點、不再破底）
2. 動能往 VWAP 回頭（VWAP＝當日成交量加權均價，每天重算）
3. 觸發：**9 EMA 向上穿越 VWAP**（就是進場那一刻，不等 K 棒收盤）
4. 進場在交叉價位
5. 停損與目標用「等幅測量」：
   低點 → 交叉價 的距離 = 1 個單位
   目標 = 交叉價 + 1 單位
   停損 = 交叉價 − 1/3 單位  ⇒ 約 3:1 的賺賠比
6. 時間窗：10:00–10:45 與 10:46–13:30（美東時間），其餘時段不交易
7. 過濾：9 EMA 在進場前若是「完全走平」則不做；從低點回來的推升要「穩」不要「暴衝」

他自稱：勝率約 60%，賺賠比約 3:1。

資料
----
OpenD 1 分鐘 K（autype='qfq'）。time_key 已驗證為美東時間
（09:31 起、16:00 收，符合美股盤中）。
"""
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
D1 = os.path.join(HERE, 'data_1m')
OUT = os.path.join(HERE, 'out')
os.makedirs(D1, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

START, END = '2026-04-01', '2026-10-24'
SLEEP = 0.55
COST = 1e-4          # 單邊 1bp；來回 2bp

# 近一年成交金額最大的幾支（流動性足夠做盤中）
SYMS = ['SPY', 'QQQ', 'NVDA', 'AAPL', 'MSFT', 'IWM', 'UNH', 'LLY',
        'V', 'MA', 'WFC', 'C', 'TMO', 'ACN', 'NKE']

WIN_START = (10, 0)
WIN_END = (13, 30)


def log(m):
    print(f'{time.strftime("%H:%M:%S")} {m}', flush=True)


# --------------------------------------------------------------- 抓資料
def fetch(force=False):
    from futu import RET_OK, OpenQuoteContext
    ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
    try:
        r, st = ctx.get_global_state()
        if r != RET_OK or not st.get('qot_logined'):
            log(f'✗ 未登入 {st}')
            return False
        for k, s in enumerate(SYMS, 1):
            path = os.path.join(D1, f'{s}.csv')
            if os.path.exists(path) and not force:
                log(f'  [{k}/{len(SYMS)}] {s} 已有，跳過')
                continue
            frames, page, pages = [], None, 0
            t0 = time.time()
            while True:
                r, d, page = ctx.request_history_kline(
                    'US.' + s, start=START, end=END,
                    ktype='K_1M', autype='qfq',
                    max_count=1000, page_req_key=page)
                if r != RET_OK:
                    log(f'  [{k}/{len(SYMS)}] ✗ {s} {str(d)[:70]}')
                    break
                if len(d):
                    frames.append(d)
                    pages += 1
                if page is None:
                    break
                time.sleep(SLEEP)
            if frames:
                df = pd.concat(frames, ignore_index=True)
                df = df.drop_duplicates('time_key').sort_values('time_key')
                df.to_csv(path, index=False)
                log(f'  [{k}/{len(SYMS)}] ✓ {s} {len(df):>6} 筆 '
                    f'{pages} 頁 {time.time()-t0:.0f}s')
            time.sleep(SLEEP)
        r, q = ctx.get_history_kl_quota(get_detail=False)
        if r == RET_OK:
            log(f'額度 已用 {q[0]} / 剩 {q[1]}')
        return True
    finally:
        ctx.close()


# --------------------------------------------------------------- 回測
def one_day(g, min_d_pct=0.0015, flat_tol=0.02):
    """單日回測。g 為當日 1 分鐘 K（含 open/high/low/close/volume）。"""
    if len(g) < 30:
        return None
    tp = (g['high'] + g['low'] + g['close']) / 3.0
    v = g['volume'].astype(float)
    cum_v = v.cumsum()
    cum_pv = (tp * v).cumsum()
    vwap = (cum_pv / cum_v.replace(0, np.nan)).ffill()
    ema9 = g['close'].ewm(span=9, adjust=False).mean()
    low_run = g['low'].cummin()

    ts = pd.to_datetime(g['time_key'])
    minute = ts.dt.hour * 60 + ts.dt.minute
    lo_w = WIN_START[0] * 60 + WIN_START[1]
    hi_w = WIN_END[0] * 60 + WIN_END[1]

    C = g['close'].to_numpy()
    H = g['high'].to_numpy()
    L = g['low'].to_numpy()
    V = vwap.to_numpy()
    E = ema9.to_numpy()
    LO = low_run.to_numpy()
    M = minute.to_numpy()
    n = len(g)

    for i in range(10, n - 2):
        if not (lo_w <= M[i] <= hi_w):
            continue
        if not (np.isfinite(V[i]) and np.isfinite(E[i]) and np.isfinite(V[i - 1])):
            continue
        # 9 EMA 上穿 VWAP
        if not (E[i - 1] <= V[i - 1] and E[i] > V[i]):
            continue
        # 交叉前必須在 VWAP 下方（就是「從低點回來」）
        if C[i - 1] > V[i - 1]:
            continue
        entry = V[i]                       # 進場在交叉處（≈VWAP）
        low = LO[i]
        d = entry - low
        if d <= 0 or d / entry < min_d_pct:
            continue
        # 過濾：進場前 9 EMA 若完全走平就不做
        seg = E[max(0, i - 3):i + 1]
        if len(seg) >= 3:
            rng = (seg.max() - seg.min()) / entry
            if rng < flat_tol * (d / entry):
                continue
        target = entry + d
        stop = entry - d / 3.0
        # 往前找：先觸停損還是先達標（同一根 K 內保守假設先停損）
        out_i, out_px, why = None, None, None
        for j in range(i + 1, n):
            if L[j] <= stop:
                out_i, out_px, why = j, stop, 'stop'
                break
            if H[j] >= target:
                out_i, out_px, why = j, target, 'target'
                break
        if out_i is None:
            out_i, out_px, why = n - 1, C[-1], 'eod'
        return dict(entry_time=ts.iloc[i], entry=entry,
                    ret=out_px / entry - 1 - COST * 2,
                    why=why, bars=out_i - i,
                    d_pct=d / entry, target=target, stop=stop)
    return None


def run(syms):
    trades = []
    for s in syms:
        f = os.path.join(D1, f'{s}.csv')
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f)
        df['ts'] = pd.to_datetime(df['time_key'])
        df['day'] = df['ts'].dt.normalize()
        for day, g in df.groupby('day'):
            if len(g) < 100:
                continue
            t = one_day(g.reset_index(drop=True))
            if t:
                t['sym'] = s
                t['day'] = day
                trades.append(t)
    return pd.DataFrame(trades)


def report(T, label):
    if T.empty:
        print(f'\n{label}: 無交易')
        return
    w = T['ret'] > 0
    aw = T['ret'][w].mean() if w.any() else np.nan
    al = T['ret'][~w].mean() if (~w).any() else np.nan
    ts = T['ret'].mean() / (T['ret'].std() / np.sqrt(len(T))) if len(T) > 2 else np.nan
    print(f'\n{label}')
    print(f'  交易 {len(T):>4} 筆   勝率 {w.mean()*100:5.2f}%   '
          f'每筆平均 {T["ret"].mean()*100:+.3f}%')
    print(f'  平均獲利 {aw*100:+.3f}%   平均虧損 {al*100:+.3f}%   '
          f'賺賠比 {abs(aw/al):.2f}')
    print(f'  累積 {((1+T["ret"]).prod()-1)*100:+.1f}%   '
          f'逐筆 t 值 {ts:.2f}   平均持倉 {T["bars"].mean():.0f} 分鐘')
    print(f'  出場原因: ' + ', '.join(
        f'{k} {v}' for k, v in T['why'].value_counts().items()))


def main():
    if '--fetch' in sys.argv:
        log('開始抓 1 分鐘 K')
        if not fetch(force='--force' in sys.argv):
            return 1
        log('抓取完成')

    T = run(SYMS)
    print('=' * 78)
    print('SMB「Fashionably Late」9 EMA / VWAP 盤中剝頭皮（照原話規則）')
    print('=' * 78)
    print(f'  標的 {len(SYMS)} 支   期間 {START} ~ {END}')
    print(f'  他自稱：勝率約 60%、賺賠比約 3:1')
    if T.empty:
        print('\n  沒有任何符合條件的交易。')
        return 0

    report(T, '★ 原話規則（10:00–13:30 窗口、-1/3 單位停損、1 單位目標）')

    print('\n' + '=' * 78)
    print('敏感度：每次只改一個細節')
    print('=' * 78)
    # 全域 vs 只有窗口
    for lab, lo, hi in [('全天都可做（不限時間窗）', 0, 24 * 60),
                        ('只有早盤 10:00–10:45', 600, 645),
                        ('只有午盤 10:46–13:30', 646, 810)]:
        global WIN_START, WIN_END
        WIN_START = (lo // 60, lo % 60)
        WIN_END = (hi // 60, hi % 60)
        r = run(SYMS)
        if not r.empty:
            w = r['ret'] > 0
            aw = r['ret'][w].mean() if w.any() else np.nan
            al = r['ret'][~w].mean() if (~w).any() else np.nan
            print(f'  {lab:<26} {len(r):>4} 筆  勝率 {w.mean()*100:5.1f}%  '
                  f'平均 {r["ret"].mean()*100:+.3f}%  '
                  f'累積 {((1+r["ret"]).prod()-1)*100:+.1f}%')
        else:
            print(f'  {lab:<26} 無交易')
    WIN_START, WIN_END = (10, 0), (13, 30)

    # 停損放寬到 1 單位（1:1）
    global one_day
    orig = one_day

    def one_day_1r(g, **kw):
        r = orig(g, **kw)
        return r
    # 用重新計算的方式測 1:1
    print()
    for ratio_lab, stop_div in [('停損改 1 單位（1:1）', 1.0),
                                ('停損改 1/2 單位', 2.0)]:
        # 臨時改寫：借用內部邏輯
        trades = []
        for s in SYMS:
            f = os.path.join(D1, f'{s}.csv')
            if not os.path.exists(f):
                continue
            df = pd.read_csv(f)
            df['ts'] = pd.to_datetime(df['time_key'])
            df['day'] = df['ts'].dt.normalize()
            for day, g in df.groupby('day'):
                if len(g) < 100:
                    continue
                gg = g.reset_index(drop=True)
                tp = (gg['high'] + gg['low'] + gg['close']) / 3.0
                v = gg['volume'].astype(float)
                vwap = ((tp * v).cumsum() / v.cumsum().replace(0, np.nan)).ffill()
                ema9 = gg['close'].ewm(span=9, adjust=False).mean()
                low_run = gg['low'].cummin()
                ts = pd.to_datetime(gg['time_key'])
                minute = ts.dt.hour * 60 + ts.dt.minute
                C = gg['close'].to_numpy(); H = gg['high'].to_numpy()
                L = gg['low'].to_numpy(); V = vwap.to_numpy()
                E = ema9.to_numpy(); LO = low_run.to_numpy(); M = minute.to_numpy()
                n = len(gg)
                for i in range(10, n - 2):
                    if not (600 <= M[i] <= 810):
                        continue
                    if not (np.isfinite(V[i]) and np.isfinite(E[i])
                            and np.isfinite(V[i - 1])):
                        continue
                    if not (E[i - 1] <= V[i - 1] and E[i] > V[i]):
                        continue
                    if C[i - 1] > V[i - 1]:
                        continue
                    entry = V[i]; d = entry - LO[i]
                    if d <= 0 or d / entry < 0.0015:
                        continue
                    tgt = entry + d; stp = entry - d / stop_div
                    res = None
                    for j in range(i + 1, n):
                        if L[j] <= stp:
                            res = stp; break
                        if H[j] >= tgt:
                            res = tgt; break
                    if res is None:
                        res = C[-1]
                    trades.append(res / entry - 1 - COST * 2)
        if trades:
            t = pd.Series(trades)
            w = t > 0
            print(f'  {ratio_lab:<26} {len(t):>4} 筆  勝率 {w.mean()*100:5.1f}%  '
                  f'平均 {t.mean()*100:+.3f}%  累積 {((1+t).prod()-1)*100:+.1f}%')

    T.to_csv(os.path.join(OUT, 'smb_ema_vwap_trades.csv'), index=False)
    print(f'\n逐筆交易 → out/smb_ema_vwap_trades.csv')
    return 0


if __name__ == '__main__':
    sys.exit(main())
