#!/usr/bin/env python3
"""趨勢線突破（春哥 XClOoOIiwpE + J Law zbb09zSKYts）的形式化回測。

兩人教同一件事，我把規則合併成一套可執行的定義。

## 逐字稿抄出來的規則

春哥的四個「可靠趨勢線」準則（原文）：
  1.「有效的趨勢線至少能連起三個點」
  2.「一條趨勢線能連起的點越多，這條趨勢線就越可靠」
  3.「趨勢線至少要涵蓋三個禮拜的K線，最好是涵蓋七個禮拜以上」
  4.「趨勢線的斜率要相對平坦，避開太過斜的趨勢線」
     理由：「太斜代表賣壓可能還是很大，即使突破了，往往也只是進入橫行整理」

買賣（他的 2021 蘋果短線實例）：
  - 進場：股價以「大成交量」突破下降趨勢線
  - 停損：「設在前低支撐的下面，停損幅度大約是 3%」
  - 停利：「只要收盤價保持在二十日均線之上就一直持有，直到跌破二十日均線賣出」
  - 「跌破二十日均線和上升趨勢線」是他判斷短期趨勢結束的依據

J Law 的補充：
  - 支撐壓力是「一個區間一個範圍」，不是一條死線（要有容差）
  - 「越傾斜的趨勢線，它的支持或阻力力量是越低的」
  - 跌破後會「向下一條沒那麼斜的趨勢線進發」

## 我如何把它寫成程式

- 擺動點：左右各 w 根的分形（w=5）
- 下降趨勢線：連接兩個擺動高點。要求斜率為負（遞降）。
- 觸及：K 線高點與線值差距在容差內（tol=1.5% 的價格）
- 可靠性門檻：連起 ≥3 點、涵蓋 ≥15 個交易日
- 斜率平坦度：每日跌幅 / 價格 ≤ 0.4%（避開「太斜」）
  這一條是春哥準則 4 的量化，也是唯一需要我選數字的地方。
- 進場：收盤 > 線值（突破），且成交量 ≥ 20 日均量 1.2 倍
- 停損：進場價 × (1 − 3%)，照他蘋果例子的「大約 3%」
- 停利：收盤 < 20 日均線（他的短線規則）；另設最長持有 60 日

## 防前視偏差

擺動高點要再過 w 根才被確認，所以候選線只能用「已確認」的點。
突破當天的收盤價才進場，不用當天最低價。
"""
import glob
import json
import os

import numpy as np
import pandas as pd

STOCK = '/root/.openclaw/stock_data'
OUT = '/root/.openclaw/workspace/reports/_agent_lines'
os.makedirs(OUT, exist_ok=True)

W = 5                 # 分形窗口
TOL = 0.015           # 觸及容差（價格百分比）
MIN_TOUCH = 3         # 春哥準則 1
MIN_SPAN = 15         # 春哥準則 3：至少 3 週（15 交易日）
MAX_SLOPE = 0.004     # 春哥準則 4：每日跌幅上限 0.4%
VOL_MULT = 1.2
STOP_PCT = 0.03       # 他的蘋果例子「大約 3%」
MAX_HOLD = 60


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


def swing_highs(high, low, w=W):
    """回傳已確認的擺動高點索引與價格（左右各 w 根）。"""
    n = len(high)
    hs = []
    for i in range(w, n - w):
        seg = high[i - w:i + w + 1]
        if high[i] == seg.max() and (seg == high[i]).sum() == 1:
            hs.append(i)
    return hs


def find_signals(df):
    """回傳 [(entry_idx, stop, line_slope, touches, span)]"""
    high = df['high'].to_numpy(float)
    low = df['low'].to_numpy(float)
    close = df['close'].to_numpy(float)
    vol = df['volume'].to_numpy(float)
    n = len(close)

    ma20 = pd.Series(close).rolling(20).mean().to_numpy()
    vol20 = pd.Series(vol).rolling(20).mean().to_numpy()

    hs = swing_highs(high, low)
    if len(hs) < 3:
        return []

    sig = []
    used = set()
    for a in range(len(hs) - 1):
        i = hs[a]
        for b in range(a + 1, min(a + 6, len(hs))):
            j = hs[b]
            if j - i < MIN_SPAN:
                continue
            # 線：經過 (i, H_i) 與 (j, H_j)，要求下降
            hi_i, hi_j = high[i], high[j]
            if hi_j >= hi_i:
                continue
            slope = (hi_j - hi_i) / (j - i)
            if slope >= 0:
                continue
            # 春哥準則 4：斜率要相對平坦
            if abs(slope) / hi_i > MAX_SLOPE:
                continue
            # 用這條線檢查從 i 到 j 的觸及次數
            # 注意：不可以把「連續幾根 K 線都在容差內」算成好幾次觸及——
            # 那只是同一段盤整被切碎，不是市場反覆測試這條線。
            # 正確做法是算「觸及叢集」：新叢集 = 前一根不在容差內。
            touches = 0
            prev_in = False
            for k in range(i, j + 1):
                linev = hi_i + slope * (k - i)
                if linev <= 0:
                    break
                in_band = abs(high[k] - linev) / linev <= TOL
                if in_band and not prev_in:
                    touches += 1
                prev_in = in_band
            # 連起 3 點：a、b 本身 + 至少 1 個中間觸及
            if touches < MIN_TOUCH:
                continue
            span = j - i
            if span < MIN_SPAN:
                continue

            # 從 j 之後找突破（j 之後 w 根才確認線，但線的端點 j 已確認）
            start = j + 1
            for k in range(start, min(start + 120, n - 1)):
                linev = hi_i + slope * (k - i)
                if linev <= 0 or k < 60:
                    continue
                if close[k] <= linev:
                    continue
                if not (vol20[k] > 0 and vol[k] >= VOL_MULT * vol20[k]):
                    continue
                if k in used:
                    break
                entry = close[k]
                # 停損：前低之下，幅度不超過 3%（取兩者較近者）
                prior_low = low[max(0, k - 20):k].min()
                stop = max(entry * (1 - STOP_PCT), prior_low * 0.995)
                if stop >= entry:
                    stop = entry * (1 - STOP_PCT)
                sig.append(dict(entry_j=k, entry=float(entry), stop=float(stop),
                                slope=float(slope), touches=int(touches),
                                span=int(span)))
                used.add(k)
                break
    return sig


def backtest(stocks):
    trades = []
    for s, df in stocks.items():
        close = df['close'].to_numpy(float)
        low = df['low'].to_numpy(float)
        n = len(close)
        ma20 = pd.Series(close).rolling(20).mean().to_numpy()
        seen = set()
        for g in find_signals(df):
            j = g['entry_j']
            if j in seen:
                continue
            seen.add(j)
            exit_j, reason, px = None, None, None
            for k in range(j + 1, min(j + MAX_HOLD + 1, n)):
                if low[k] <= g['stop']:
                    exit_j, reason, px = k, 'stop', g['stop']
                    break
                # 春哥短線停利：收盤跌破 20 日均線
                if close[k] < ma20[k]:
                    exit_j, reason, px = k, 'ma20', close[k]
                    break
            if exit_j is None:
                exit_j = min(j + MAX_HOLD, n - 1)
                reason, px = 'time', close[exit_j]
            trades.append(dict(
                symbol=s, entry_date=str(df.index[j].date()),
                exit_date=str(df.index[exit_j].date()),
                entry=g['entry'], exit=float(px),
                ret=float(px / g['entry'] - 1), reason=reason,
                hold=int(exit_j - j), touches=g['touches'], span=g['span'],
            ))
    return pd.DataFrame(trades)


def market_cum(stocks):
    px = pd.DataFrame({s: d['close'] for s, d in stocks.items()}).sort_index()
    # 先 mean 再 fillna：反過來會把早期 NaN 當 0，稀釋市場報酬
    mkt = px.pct_change().mean(axis=1).fillna(0)
    return (1 + mkt).cumprod()


def excess(df, cum):
    a = pd.to_datetime(df['entry_date'])
    b = pd.to_datetime(df['exit_date'])
    s = cum.reindex(a, method='ffill').to_numpy()
    e = cum.reindex(b, method='ffill').to_numpy()
    return df['ret'].to_numpy() - (e / s - 1)


def nwt(x, lag=5):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return float('nan')
    mu = x.mean()
    ee = x - mu
    s = ee @ ee / n
    for L in range(1, lag + 1):
        s += 2 * (1 - L / (lag + 1)) * (ee[L:] @ ee[:-L]) / n
    return float(mu / np.sqrt(s / n))


def stats(t, cum):
    r = t['ret']
    w = (r > 0).mean()
    ex = excess(t, cum)
    return dict(
        n=int(len(t)), win=float(w), avg=float(r.mean()), med=float(r.median()),
        avg_win=float(r[r > 0].mean()) if (r > 0).any() else 0.0,
        avg_loss=float(r[r <= 0].mean()) if (r <= 0).any() else 0.0,
        stop_rate=float((t['reason'] == 'stop').mean()),
        hold=float(t['hold'].mean()),
        excess=float(ex.mean()), excess_t=nwt(ex),
        trimmed=float(t[r < r.quantile(0.95)]['ret'].mean()),
    )


def control(stocks, cum):
    """對照組：同樣的停損/停利，但訊號改成「突破 60 日新高」，不看趨勢線。"""
    trades = []
    for s, df in stocks.items():
        close = df['close'].to_numpy(float)
        high = df['high'].to_numpy(float)
        low = df['low'].to_numpy(float)
        vol = df['volume'].to_numpy(float)
        n = len(close)
        ma20 = pd.Series(close).rolling(20).mean().to_numpy()
        vol20 = pd.Series(vol).rolling(20).mean().to_numpy()
        hh = pd.Series(high).rolling(60).max().shift(1).to_numpy()
        last = -999
        for k in range(60, n - 1):
            if k - last < 10:
                continue
            if not (hh[k] == hh[k] and close[k] > hh[k]):
                continue
            if not (vol20[k] > 0 and vol[k] >= VOL_MULT * vol20[k]):
                continue
            entry = close[k]
            stop = entry * (1 - STOP_PCT)
            ej, reason, px = None, None, None
            for q in range(k + 1, min(k + MAX_HOLD + 1, n)):
                if low[q] <= stop:
                    ej, reason, px = q, 'stop', stop
                    break
                if close[q] < ma20[q]:
                    ej, reason, px = q, 'ma20', close[q]
                    break
            if ej is None:
                ej = min(k + MAX_HOLD, n - 1)
                reason, px = 'time', close[ej]
            trades.append(dict(symbol=s, entry_date=str(df.index[k].date()),
                               exit_date=str(df.index[ej].date()),
                               entry=float(entry), exit=float(px),
                               ret=float(px / entry - 1), reason=reason,
                               hold=int(ej - k)))
            last = k
    return pd.DataFrame(trades)


def main():
    print('載入…', flush=True)
    S = load()
    print(f'  {len(S)} 檔', flush=True)
    cum = market_cum(S)

    print('趨勢線訊號…', flush=True)
    t = backtest(S)
    print(f'  交易 {len(t)} 筆', flush=True)
    if t.empty:
        json.dump({'error': 'no signals'}, open(f'{OUT}/trendline_result.json', 'w'))
        return

    t.to_csv(f'{OUT}/trendline_trades.csv', index=False)
    st = stats(t, cum)
    print(f"  勝率 {st['win']*100:.1f}% 原始 {st['avg']*100:+.2f}% "
          f"超額 {st['excess']*100:+.2f}% t={st['excess_t']:+.2f}", flush=True)

    print('對照組…', flush=True)
    ct = control(S, cum)
    ct.to_csv(f'{OUT}/trendline_control.csv', index=False)
    cst = stats(ct, cum)
    print(f"  對照 勝率 {cst['win']*100:.1f}% 原始 {cst['avg']*100:+.2f}% "
          f"超額 {cst['excess']*100:+.2f}% t={cst['excess_t']:+.2f}", flush=True)

    # 逐年
    t['year'] = pd.to_datetime(t['entry_date']).dt.year
    ex = excess(t, cum)
    t['ex'] = ex
    yr = t.groupby('year').agg(n=('ret', 'size'), mean=('ret', 'mean'),
                               ex=('ex', 'mean')).reset_index()
    yr.to_csv(f'{OUT}/trendline_by_year.csv', index=False)

    res = dict(
        stats=st, control=cst,
        years_pos=int((yr['ex'] > 0).sum()), years_n=int(len(yr)),
        cfg=dict(W=W, TOL=TOL, MIN_TOUCH=MIN_TOUCH, MIN_SPAN=MIN_SPAN,
                 MAX_SLOPE=MAX_SLOPE, VOL_MULT=VOL_MULT,
                 STOP_PCT=STOP_PCT, MAX_HOLD=MAX_HOLD),
        by_year=yr.to_dict('records'),
    )
    json.dump(res, open(f'{OUT}/trendline_result.json', 'w'),
              ensure_ascii=False, indent=1)
    print('\n逐年超額為正：', res['years_pos'], '/', res['years_n'])
    print(yr.to_string(index=False))


if __name__ == '__main__':
    main()
