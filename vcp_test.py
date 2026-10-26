#!/usr/bin/env python3
"""J Law「VCP 三招」選股規則的形式化回測。

來源：YouTube qBAcdEshRX8《買中超強勢股票的3個交易技巧》逐字稿（5,383 字）。
他給的規則（有明確數值，因此可寫成程式）：

  第一招 趨勢過濾：股價 > 50 日均線、50 日均線 > 200 日均線、兩條均線都向上
  第二招 第一個收縮：幅度要在 50-60% 以內，理想 20-35%
  第三招 最後一個收縮：幅度在 10% 以內，且收縮期間下跌日成交量少
       （他稱之為 Tight Area）；買點 = 突破 Tight Area 上緣

他的自述績效：勝率 45-50%、平均虧損 4-5%、平均獲利 12-14%。
（換算損益兩平勝率僅 25.7%，所以如果數字為真，數學上很寬鬆。
  但這些是他自述，沒有對帳單，本檔只驗證「規則本身」有沒有優勢。）

── 形式化時我必須做選擇，以下是選擇與理由 ──
1. 什麼是「一個收縮」？
   用 5 根 K 線的分形（fractal）認出轉折點：某根高點高於左右各 5 根，
   就是一個擺動高點（swing high）；低點同理。兩個相鄰擺動高點之間的最低點
   就是該次收縮的底。收縮幅度 =（擺動高點 − 該段最低點）／擺動高點。
   這是他看圖時實際在量的東西，我只是把它寫成可以重複的定義。

2. 什麼時候算「VCP 成立」？
   從最近一個擺動高點往回，取連續的收縮段，要求：
     - 至少 2 段（他說「最少有兩個收縮已經夠了，一般 3~4 個」）
     - 第一段在 20%~60%（他給的區間放寬到上限 60%）
     - 最後一段 ≤ 10%（第三招）
     - 收縮幅度大致遞減（後一段 ≤ 前一段 × 1.15，容許一點誤差）
   
3. 買點：收盤價突破「最後一段的擺動高點」，且成交量 > 20 日均量的 1.2 倍
   （他說「有成交量配合」，但沒給倍數，1.2 是我選的保守值）

4. 出場：跌破停損價（進場價 −7%，他自述虧損控制在 4-5%、停損設 6-8%），
   或持有超過 120 個交易日。**沒有設獲利目標**——因為他的獲利 12-14%
   是「平均」，不是每筆的目標價，硬設一個會扭曲結果。

5. 前視偏差：所有判斷只用當日與之前的資料；進場在「突破當天的收盤」，
   不是當天的最低價。
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

STOCK = '/root/.openclaw/stock_data'
OUT = '/root/.openclaw/workspace/reports/_vcp'
os.makedirs(OUT, exist_ok=True)

PIVOT_W = 5          # 分形窗口：左右各 5 根
LOOKBACK = 180       # 最多往回找幾個交易日建立底部
MIN_SEGS = 2
FIRST_MAX = 0.60     # 第二招上限
FIRST_MIN = 0.20     # 第二招理想下限
FIRST_HARD_MAX = 0.60
LAST_MAX = 0.10      # 第三招
VOL_MULT = 1.2       # 突破量能倍數
STOP_PCT = 0.07      # 停損
MAX_HOLD = 120       # 最長持有


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


def pivots(high, low, w=PIVOT_W):
    """回傳 [(idx, 'H'|'L', price)]，按時間排序。向量化找分形。"""
    n = len(high)
    if n < 2 * w + 1:
        return []
    res = []
    for i in range(w, n - w):
        hseg = high[i - w:i + w + 1]
        lseg = low[i - w:i + w + 1]
        if high[i] == hseg.max() and (hseg == high[i]).sum() == 1:
            res.append((i, 'H', float(high[i])))
        if low[i] == lseg.min() and (lseg == low[i]).sum() == 1:
            res.append((i, 'L', float(low[i])))
    return res


def find_patterns(df):
    """回傳訊號列表 [(entry_idx, stop, tight_high, info)]"""
    high = df['high'].to_numpy(float)
    low = df['low'].to_numpy(float)
    close = df['close'].to_numpy(float)
    vol = df['volume'].to_numpy(float)
    n = len(close)

    ma50 = pd.Series(close).rolling(50).mean().to_numpy()
    ma200 = pd.Series(close).rolling(200).mean().to_numpy()
    vol20 = pd.Series(vol).rolling(20).mean().to_numpy()

    pv = pivots(high, low)
    if len(pv) < 6:
        return []

    # 依時間走訪每個擺動高點，試著把它當成「底部起點」
    sig = []
    hidx = [i for i, (j, k, p) in enumerate(pv) if k == 'H']
    for hi_pos in hidx:
        j, _, H = pv[hi_pos]
        # 收集之後的擺動高點序列，每段：H -> 最低 -> 下一個 H
        segs = []
        cur_h = H
        cur_j = j
        k = hi_pos + 1
        while k < len(pv) and len(segs) < 5:
            # 找下一個低點、再找下一個高點
            if pv[k][1] != 'L':
                k += 1
                continue
            lj, _, L = pv[k]
            k2 = k + 1
            while k2 < len(pv) and pv[k2][1] != 'H':
                k2 += 1
            if k2 >= len(pv):
                break
            hj, _, H2 = pv[k2]
            if hj - cur_j > LOOKBACK:
                break
            depth = (cur_h - L) / cur_h if cur_h > 0 else 1
            if depth <= 0 or depth > 0.95:
                break
            segs.append(dict(depth=depth, low=L, low_j=lj,
                             high=H2, high_j=hj, prev_high=cur_h))
            cur_h, cur_j = H2, hj
            k = k2
            if len(segs) >= MIN_SEGS:
                # ---- 用目前這組 segs 檢查是否符合 VCP ----
                d = [s['depth'] for s in segs]
                first_ok = FIRST_MIN <= d[0] <= FIRST_HARD_MAX
                last_ok = d[-1] <= LAST_MAX
                dec_ok = all(d[i + 1] <= d[i] * 1.15 for i in range(len(d) - 1))
                if first_ok and last_ok and dec_ok:
                    tight_high = segs[-1]['high']
                    # 擺動高點要再過 PIVOT_W 根才被確認，所以最早只能在
                    # highlight_j + PIVOT_W 之後才「看得到」這個型態。
                    # 之前的版本直接拿 high_j 當進場日，那是前視偏差——修正。
                    confirm_j = segs[-1]['high_j'] + PIVOT_W
                    entry_j = None
                    for jj in range(confirm_j, min(confirm_j + 60, n - 1)):
                        if jj < 210:
                            continue
                        c = close[jj]
                        if c <= tight_high:
                            continue
                        if not (c > ma50[jj] > ma200[jj]):
                            continue
                        if not (ma50[jj] > ma50[jj - 20]
                                and ma200[jj] > ma200[jj - 20]):
                            continue
                        if not (vol20[jj] > 0 and vol[jj] >= VOL_MULT * vol20[jj]):
                            continue
                        entry_j = jj
                        break
                    if entry_j is None:
                        continue
                    # 最後收縮期間下跌日量能（第三招另一半）
                    seg = segs[-1]
                    win = slice(max(0, seg['low_j']), seg['high_j'])
                    dn = close[win][1:] < close[win][:-1]
                    v = vol[win][1:]
                    quiet = True
                    if dn.sum() >= 3:
                        quiet = v[dn].mean() <= v[~dn].mean() if (~dn).sum() else True
                    if not quiet:
                        continue
                    c = close[entry_j]
                    sig.append(dict(
                        entry_j=int(entry_j),
                        stop=float(c * (1 - STOP_PCT)),
                        entry=float(c),
                        tight_high=float(tight_high),
                        n_segs=len(segs),
                        d0=float(d[0]), dlast=float(d[-1]),
                    ))
                    break   # 這個底部起點只取第一個合格訊號
    return sig


def backtest(stocks):
    trades = []
    for s, df in stocks.items():
        close = df['close'].to_numpy(float)
        low = df['low'].to_numpy(float)
        n = len(close)
        # 同一個底部會從不同「起點擺動高點」被重複認出，導致同一天同一檔
        # 出現 2~3 筆一模一樣的交易（實測 LITE ×2、ARWR ×3）。
        # 這會讓平均值被同一個暴漲案例灌高，必須去重。
        seen_entry = set()
        for g in find_patterns(df):
            if g['entry_j'] in seen_entry:
                continue
            seen_entry.add(g['entry_j'])
            j = g['entry_j']
            exit_j, reason, px = None, None, None
            for k in range(j + 1, min(j + MAX_HOLD + 1, n)):
                if low[k] <= g['stop']:
                    exit_j, reason, px = k, 'stop', g['stop']
                    break
            if exit_j is None:
                exit_j = min(j + MAX_HOLD, n - 1)
                reason = 'time'
                px = close[exit_j]
            trades.append(dict(
                symbol=s, entry_date=str(df.index[j].date()),
                exit_date=str(df.index[exit_j].date()),
                entry=g['entry'], exit=float(px),
                ret=float(px / g['entry'] - 1), reason=reason,
                hold=int(exit_j - j), n_segs=g['n_segs'],
                d0=g['d0'], dlast=g['dlast'],
                entry_j=int(j), exit_j=int(exit_j),
            ))
    return pd.DataFrame(trades)


def benchmark(stocks, trades):
    """同曝險基準：把每筆交易的持有期間串起來，算那段時間大盤（全體等權）的報酬。

    這是本站一貫的檢驗方式：策略賺的錢，是不是只是「有持倉」造成的？
    """
    if trades.empty:
        return {}
    # 建立全體等權日報酬
    px = pd.DataFrame({s: d['close'] for s, d in stocks.items()})
    ret = px.pct_change()
    mkt = ret.mean(axis=1)

    # 策略：每筆交易獨立的報酬序列（等權、資金分成 N 份同時持有）
    tr = trades.copy()
    tr['entry_date'] = pd.to_datetime(tr['entry_date'])
    tr['exit_date'] = pd.to_datetime(tr['exit_date'])
    # 逐日計算「當天有幾筆部位、平均報酬」
    daily = {}
    for _, r in tr.iterrows():
        seg = ret.loc[(ret.index >= r['entry_date']) & (ret.index <= r['exit_date']), r['symbol']]
        for dt, v in seg.items():
            daily.setdefault(dt, []).append(v)
    ser = pd.Series({k: np.mean(v) for k, v in daily.items()}).sort_index()
    return dict(strategy=ser, market=mkt.reindex(ser.index))


def stats(tr):
    if tr.empty:
        return {}
    r = tr['ret']
    win = (r > 0).mean()
    return dict(
        n=int(len(tr)), win=float(win),
        avg=float(r.mean()), med=float(r.median()),
        avg_win=float(r[r > 0].mean()) if (r > 0).any() else 0.0,
        avg_loss=float(r[r <= 0].mean()) if (r <= 0).any() else 0.0,
        best=float(r.max()), worst=float(r.min()),
        avg_hold=float(tr['hold'].mean()),
        stop_rate=float((tr['reason'] == 'stop').mean()),
        expectancy=float(win * (r[r > 0].mean() if (r > 0).any() else 0)
                         + (1 - win) * (r[r <= 0].mean() if (r <= 0).any() else 0)),
    )


def main():
    print('載入…', flush=True)
    stocks = load()
    print(f'  {len(stocks)} 檔', flush=True)

    print('找訊號…', flush=True)
    tr = backtest(stocks)
    print(f'  訊號/交易 {len(tr)} 筆', flush=True)
    if tr.empty:
        json.dump({'error': 'no signals'}, open(f'{OUT}/vcp_result.json', 'w'))
        return

    tr.to_csv(f'{OUT}/vcp_trades.csv', index=False)
    st = stats(tr)

    print('基準…', flush=True)
    bm = benchmark(stocks, tr)
    bstat = {}
    if bm:
        ser, mkt = bm['strategy'], bm['market']
        bstat = dict(
            days=int(len(ser)),
            strat_cum=float((1 + ser).prod() - 1),
            mkt_cum=float((1 + mkt.fillna(0)).prod() - 1),
            strat_ann=float((1 + ser).prod() ** (252 / len(ser)) - 1),
            mkt_ann=float((1 + mkt.fillna(0)).prod() ** (252 / len(ser)) - 1),
        )

    # 逐年
    tr['year'] = pd.to_datetime(tr['entry_date']).dt.year
    yr = tr.groupby('year')['ret'].agg(['count', 'mean', 'sum'])
    yr.to_csv(f'{OUT}/vcp_by_year.csv')

    # 剔除最好 5% 的交易
    cut = tr['ret'].quantile(0.95)
    trimmed = tr[tr['ret'] < cut]['ret']
    trim_mean = float(trimmed.mean())

    res = dict(stats=st, bench=bstat,
               trimmed_mean=trim_mean, n_trimmed=int(len(trimmed)),
               cfg=dict(PIVOT_W=PIVOT_W, LOOKBACK=LOOKBACK, MIN_SEGS=MIN_SEGS,
                        FIRST_MIN=FIRST_MIN, FIRST_MAX=FIRST_MAX,
                        LAST_MAX=LAST_MAX, VOL_MULT=VOL_MULT,
                        STOP_PCT=STOP_PCT, MAX_HOLD=MAX_HOLD))
    json.dump(res, open(f'{OUT}/vcp_result.json', 'w'), ensure_ascii=False, indent=1)

    print('\n=== 結果 ===')
    for k, v in st.items():
        print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')
    print('\n=== 同曝險基準 ===')
    for k, v in bstat.items():
        print(f'  {k}: {v:.4f}' if isinstance(v, float) else f'  {k}: {v}')
    print(f'\n  剔除最好 5% 後平均：{trim_mean*100:+.2f}%（原 {st["avg"]*100:+.2f}%）')
    print(f'\n逐年：')
    print(yr.to_string())


if __name__ == '__main__':
    main()
