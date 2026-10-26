#!/usr/bin/env python3
"""平行通道（J Law BM3jm780w4Q）與恐慌進場（J Law xVXB5AIg9dk）回測。

## 平行通道

逐字稿抄出來的用法：
  - 價格常在平行的通道裡運行，通道由上軌（阻力）與下軌（支撐）構成
  - 買點：「低量拉回」到通道下緣附近，且最好有 10/20/50 日均線同時支持
  - 停損：放在該均線下方
  - 賣點：到達通道上緣阻力，若出現 overshoot（輕微穿越後反轉）就獲利

我的實作：
  - 用兩個擺動低點連成下軌（支撐線），再用「通道內最高點到下軌的距離」
    平移出上軌（真正的平行通道，不是各自獨立擬合）
  - 買：收盤接近下軌（容差 2%）且收盤 > MA20，量能萎縮（低於 20 日均量）
  - 停損：下軌下方 3%
  - 停利：觸及上軌，或跌破 MA20，或 60 日

## 恐慌進場

**這支影片不是教學，是一支訪談（feat. 真倉大賽雙冠軍 J Law x Martin Luk）。**
逐字稿裡他講的是自己的實戰經驗，沒有可照做的規則：
  - 2024-08-05 極度恐慌那天，「我取消了所有的 stop」
  - 他說自己勝率 22–23%，停損通常 3% 以下
  - 「你不需要任何 indicator，只要成交量和陰陽燭」
這幾句都不是可執行的進場條件，而「取消停損」更是反規則的操作。

**所以我測的是他自己舉的那個具體情境的形式化版本：**
  極度恐慌 = 市場單日大幅下跌，且跌到長期均線附近
  然後看「這種日子買進，之後 N 天的報酬如何」。

這是把「他描述的那一天」變成可重複的條件，不是他那套判斷力。
我會在報告裡明確標示這個落差。
"""
import glob
import json
import os

import numpy as np
import pandas as pd

STOCK = '/root/.openclaw/stock_data'
OUT = '/root/.openclaw/workspace/reports/_agent_pattern'
os.makedirs(OUT, exist_ok=True)

BG = None

# ---- 通道參數
CH_W = 5
CH_TOL = 0.02
CH_MIN_SPAN = 30
CH_STOP = 0.03
CH_MAX_HOLD = 60

# ---- 恐慌參數
PANIC_DROP = -0.03        # 市場單日跌幅門檻
PANIC_NEAR_MA = 0.06      # 距 200 日均線的距離容差
PANIC_HOLD = [5, 10, 20, 60]


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


def market_cum(stocks):
    px = pd.DataFrame({s: d['close'] for s, d in stocks.items()}).sort_index()
    mkt = px.pct_change().mean(axis=1).fillna(0)
    return (1 + mkt).cumprod(), mkt


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


def summ(t, cum, name):
    if t.empty:
        return dict(name=name, n=0)
    r = t['ret']
    ex = excess(t, cum)
    w = (r > 0).mean()
    return dict(
        name=name, n=int(len(t)), win=float(w),
        avg=float(r.mean()), med=float(r.median()),
        avg_win=float(r[r > 0].mean()) if (r > 0).any() else 0.0,
        avg_loss=float(r[r <= 0].mean()) if (r <= 0).any() else 0.0,
        stop_rate=float((t['reason'] == 'stop').mean()) if 'reason' in t else float('nan'),
        hold=float(t['hold'].mean()) if 'hold' in t else float('nan'),
        excess=float(ex.mean()), excess_t=nwt(ex),
        trimmed=float(t[r < r.quantile(0.95)]['ret'].mean()),
    )


# ============================================================ 平行通道
def swings(high, low, w=CH_W):
    n = len(high)
    lo, hi = [], []
    for i in range(w, n - w):
        sl = low[i - w:i + w + 1]
        sh = high[i - w:i + w + 1]
        if low[i] == sl.min() and (sl == low[i]).sum() == 1:
            lo.append(i)
        if high[i] == sh.max() and (sh == high[i]).sum() == 1:
            hi.append(i)
    return lo, hi


def channel_trades(stocks):
    trades = []
    for s, df in stocks.items():
        high = df['high'].to_numpy(float)
        low = df['low'].to_numpy(float)
        close = df['close'].to_numpy(float)
        vol = df['volume'].to_numpy(float)
        n = len(close)
        ma20 = pd.Series(close).rolling(20).mean().to_numpy()
        vol20 = pd.Series(vol).rolling(20).mean().to_numpy()

        lo_idx, _ = swings(high, low)
        if len(lo_idx) < 2:
            continue
        seen = set()
        for a in range(len(lo_idx) - 1):
            i = lo_idx[a]
            for b in range(a + 1, min(a + 5, len(lo_idx))):
                j = lo_idx[b]
                if j - i < CH_MIN_SPAN:
                    continue
                lo_i, lo_j = low[i], low[j]
                # 下軌：兩低點連線，要求上升或至少平（支撐）
                slope = (lo_j - lo_i) / (j - i)
                if slope < 0:
                    continue
                # 通道寬度：期間最高點到下軌的距離
                seg_hi = high[i:j + 1].max()
                width = seg_hi - (lo_i + slope * (0))
                if width <= 0:
                    continue
                # 找下軌觸及後反彈進場
                start = j + 1
                for k in range(start, min(start + 150, n - 1)):
                    if k < 60 or k in seen:
                        continue
                    lowv = lo_i + slope * (k - i)
                    upv = lowv + width
                    if lowv <= 0:
                        break
                    # 低量拉回接近下軌
                    if not (low[k] <= lowv * (1 + CH_TOL)):
                        continue
                    if not (close[k] > ma20[k]):
                        continue
                    if not (vol20[k] > 0 and vol[k] < vol20[k]):   # 低量
                        continue
                    entry = close[k]
                    stop = entry * (1 - CH_STOP)
                    ej, reason, px = None, None, None
                    for q in range(k + 1, min(k + CH_MAX_HOLD + 1, n)):
                        if low[q] <= stop:
                            ej, reason, px = q, 'stop', stop
                            break
                        if close[q] >= upv:                 # 到上軌
                            ej, reason, px = q, 'target', close[q]
                            break
                        if close[q] < ma20[q]:
                            ej, reason, px = q, 'ma20', close[q]
                            break
                    if ej is None:
                        ej = min(k + CH_MAX_HOLD, n - 1)
                        reason, px = 'time', close[ej]
                    trades.append(dict(symbol=s,
                                       entry_date=str(df.index[k].date()),
                                       exit_date=str(df.index[ej].date()),
                                       entry=float(entry), exit=float(px),
                                       ret=float(px / entry - 1), reason=reason,
                                       hold=int(ej - k)))
                    seen.add(k)
                    break
    return pd.DataFrame(trades)


def channel_control(stocks):
    """對照：同樣進出場，但訊號改成「跌破 20 日均線後回到均線上方」，不看通道。"""
    trades = []
    for s, df in stocks.items():
        close = df['close'].to_numpy(float)
        low = df['low'].to_numpy(float)
        vol = df['volume'].to_numpy(float)
        n = len(close)
        ma20 = pd.Series(close).rolling(20).mean().to_numpy()
        vol20 = pd.Series(vol).rolling(20).mean().to_numpy()
        last = -999
        for k in range(60, n - 1):
            if k - last < 10:
                continue
            if not (close[k] > ma20[k] and close[k - 1] < ma20[k - 1]):
                continue
            if not (vol20[k] > 0 and vol[k] < vol20[k]):
                continue
            entry = close[k]
            stop = entry * (1 - CH_STOP)
            ej, reason, px = None, None, None
            for q in range(k + 1, min(k + CH_MAX_HOLD + 1, n)):
                if low[q] <= stop:
                    ej, reason, px = q, 'stop', stop
                    break
                if close[q] < ma20[q]:
                    ej, reason, px = q, 'ma20', close[q]
                    break
            if ej is None:
                ej = min(k + CH_MAX_HOLD, n - 1)
                reason, px = 'time', close[ej]
            trades.append(dict(symbol=s, entry_date=str(df.index[k].date()),
                               exit_date=str(df.index[ej].date()),
                               entry=float(entry), exit=float(px),
                               ret=float(px / entry - 1), reason=reason,
                               hold=int(ej - k)))
            last = k
    return pd.DataFrame(trades)


# ============================================================ 恐慌進場
def panic_events(stocks, cum, mkt):
    """找出「市場極度恐慌」的日子：單日大跌且跌到長期均線附近。"""
    px = pd.DataFrame({s: d['close'] for s, d in stocks.items()}).sort_index()
    mktpx = (1 + mkt).cumprod()
    ma200 = mktpx.rolling(200).mean()
    dist = mktpx / ma200 - 1

    events = []
    for dt in mkt.index:
        r = mkt.get(dt)
        if r is None or not np.isfinite(r):
            continue
        if r > PANIC_DROP:
            continue
        d = dist.get(dt)
        if d is None or not np.isfinite(d):
            continue
        if abs(d) > PANIC_NEAR_MA:
            continue
        events.append(dt)
    return events


def panic_trades(stocks, cum, mkt):
    """恐慌日買進，持有 N 天（市場等權），看超額。"""
    events = panic_events(stocks, cum, mkt)
    if not events:
        return pd.DataFrame(), {}
    px = pd.DataFrame({s: d['close'] for s, d in stocks.items()}).sort_index()
    mktpx = (1 + mkt).cumprod()
    idx = list(mktpx.index)
    pos = {d: i for i, d in enumerate(idx)}

    rows = []
    out = {}
    for H in PANIC_HOLD:
        rets, exs = [], []
        for dt in events:
            i = pos.get(dt)
            if i is None or i + H >= len(idx):
                continue
            a, b = idx[i + 1], idx[i + 1 + H]       # 隔天進場，避免當日
            if b not in pos or a not in pos:
                continue
            r = mktpx.loc[b] / mktpx.loc[a] - 1
            rets.append(r)
            # 「超額」對市場訊號沒有意義（它就是市場），改用勝率與無條件基準比較
            exs.append(r)
        if rets:
            out[H] = dict(n=len(rets), mean=float(np.mean(rets)),
                          win=float(np.mean(np.array(rets) > 0)),
                          t=nwt(rets))
    # 無條件基準：任一天往後 H 天的市場報酬
    base = {}
    for H in PANIC_HOLD:
        base[H] = float(mktpx.pct_change(H).dropna().mean())
    return pd.DataFrame(rows), dict(events=len(events), hold=out, base=base)


def main():
    print('載入…', flush=True)
    S = load()
    print(f'  {len(S)} 檔', flush=True)
    cum, mkt = market_cum(S)

    print('\n[1/2] 平行通道…', flush=True)
    ch = channel_trades(S)
    ch.to_csv(f'{OUT}/channel_trades.csv', index=False)
    chs = summ(ch, cum, '平行通道')
    print(f"  n={chs.get('n')} 勝率 {chs.get('win',0)*100:.1f}% "
          f"原始 {chs.get('avg',0)*100:+.2f}% 超額 {chs.get('excess',0)*100:+.2f}% "
          f"t={chs.get('excess_t',0):+.2f}", flush=True)

    print('  對照組…', flush=True)
    chc = channel_control(S)
    chc.to_csv(f'{OUT}/channel_control.csv', index=False)
    chcs = summ(chc, cum, '對照(均線回檔)')
    print(f"  n={chcs.get('n')} 原始 {chcs.get('avg',0)*100:+.2f}% "
          f"超額 {chcs.get('excess',0)*100:+.2f}% t={chcs.get('excess_t',0):+.2f}", flush=True)

    print('\n[2/2] 恐慌進場…', flush=True)
    _, pd_stats = panic_trades(S, cum, mkt)
    print(f"  恐慌日 {pd_stats.get('events')} 天", flush=True)
    for H, v in sorted(pd_stats.get('hold', {}).items()):
        b = pd_stats['base'].get(H, float('nan'))
        print(f"    持有 {H:>2} 日：n={v['n']:>3} 平均 {v['mean']*100:+.2f}% "
              f"勝率 {v['win']*100:.1f}% t={v['t']:+.2f} ｜ 無條件基準 {b*100:+.2f}%",
              flush=True)

    res = dict(channel=chs, channel_control=chcs, panic=pd_stats,
               cfg=dict(CH_TOL=CH_TOL, CH_MIN_SPAN=CH_MIN_SPAN, CH_STOP=CH_STOP,
                        CH_MAX_HOLD=CH_MAX_HOLD, PANIC_DROP=PANIC_DROP,
                        PANIC_NEAR_MA=PANIC_NEAR_MA))
    json.dump(res, open(f'{OUT}/channel_panic_result.json', 'w'),
              ensure_ascii=False, indent=1)
    print(f'\n已寫 {OUT}/channel_panic_result.json')


if __name__ == '__main__':
    main()
