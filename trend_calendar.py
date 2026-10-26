#!/usr/bin/env python3
"""驗證兩件事：日曆時間組合的年化報酬，以及「固定持有 20 日」下的觸及數效應。

## 為什麼要另外驗證

趨勢線子代理報了兩個很有說服力的說法：
  1. 月曆時間組合年化 12.29%，同期市場 17.37%
  2. 依觸及點數分組呈單調遞減（3 點 +0.974% → 6 點以上 −0.275%）

但它用的是「固定持有 20 日」，我用的是「停損 3% + 跌破 20 日均線」。
兩套出場規則下的結論可能不同，而且我先前用自己那套跑出來的
觸及數效應並不是單調的（3 點 +0.546%、6 點以上 +0.578%）。
所以這裡把兩件事都用我的資料重算一遍，確認差異是出在出場規則，不是誰算錯。

## 為什麼年化報酬值得算

「每筆超額 −0.02%」對讀者很抽象。但換成「20 年下來年化 12.3%，
而同期市場 17.4%」就很有感——同樣的錢放著不動買大盤反而賺更多。
這是同一個事實的另一種講法，而且更難反駁。
"""
import glob
import json
import os

import numpy as np
import pandas as pd

STOCK = '/root/.openclaw/stock_data'
LINES = '/root/.openclaw/workspace/reports/_agent_lines'
OUT = '/root/.openclaw/workspace/backtest_videos/xbx/out'


def load_prices():
    px = {}
    for f in sorted(glob.glob(f'{STOCK}/live/*.csv') + glob.glob(f'{STOCK}/extra/*.csv')):
        try:
            d = pd.read_csv(f, usecols=lambda c: c.lower() in ('date', 'close'))
        except Exception:
            continue
        d.columns = [c.lower() for c in d.columns]
        if 'date' not in d.columns or 'close' not in d.columns or len(d) < 300:
            continue
        d['date'] = pd.to_datetime(d['date'])
        px[os.path.basename(f).replace('.csv', '')] = (
            d.dropna(subset=['close']).set_index('date')['close'].sort_index())
    return px


def dedup(df):
    keep = []
    for _, g in df.sort_values('entry_date').groupby('symbol'):
        last = None
        for i, r in g.iterrows():
            if last is None or r.entry_date >= last:
                keep.append(i)
                last = r.exit_date
            elif r.exit_date > last:
                last = r.exit_date
    return df.loc[keep]


def calendar_time(trades, R, label):
    """把每筆交易鋪成每日持倉，等權相加，得到一條組合淨值曲線。

    進場日當天不計報酬（當天收盤才買進），從次一交易日起算。
    """
    sums = pd.Series(0.0, index=R.index)
    cnts = pd.Series(0.0, index=R.index)
    for _, r in trades.iterrows():
        s = r['symbol']
        if s not in R.columns:
            continue
        seg = R[s].loc[r['entry_date']:r['exit_date']].iloc[1:]
        ok = seg.notna()
        if not ok.any():
            continue
        sums.loc[seg.index[ok]] += seg[ok].to_numpy()
        cnts.loc[seg.index[ok]] += 1

    port = (sums / cnts.replace(0, np.nan)).dropna()
    if len(port) < 100:
        return None
    nav = (1 + port).cumprod()
    years = len(port) / 252.0
    ann = nav.iloc[-1] ** (1 / years) - 1
    sharpe = port.mean() / port.std(ddof=1) * np.sqrt(252) if port.std(ddof=1) > 0 else np.nan
    mdd = (nav / nav.cummax() - 1).min()
    # 存月線淨值供畫圖（原始日資料太大）
    m = nav.resample('ME').last().dropna()
    curve = {'dates': [d.strftime('%Y-%m') for d in m.index],
             'nav': [round(float(v), 4) for v in m.values]}
    return dict(label=label, days=int(len(port)), ann=float(ann),
                sharpe=float(sharpe), mdd=float(mdd),
                years=float(years), curve=curve)


def main():
    px = load_prices()
    P = pd.concat(px, axis=1, sort=True).sort_index()
    R = P.pct_change()
    mkt = R.mean(axis=1).fillna(0)

    nav = (1 + mkt).cumprod()
    years = len(mkt) / 252.0
    mkt_ann = nav.iloc[-1] ** (1 / years) - 1
    mkt_sharpe = mkt.mean() / mkt.std(ddof=1) * np.sqrt(252)
    mdd = (nav / nav.cummax() - 1).min()

    t = pd.read_csv(f'{LINES}/trendline_trades.csv')
    t['entry_date'] = pd.to_datetime(t.entry_date)
    t['exit_date'] = pd.to_datetime(t.exit_date)
    t = dedup(t)
    print(f'趨勢線去重後 {len(t):,} 筆', flush=True)

    res = {}
    res['trend_ma20'] = calendar_time(t, R, '趨勢線（3% 停損＋跌破 MA20 出場）')
    _mnav = (1 + mkt).cumprod()
    _mm = _mnav.resample('ME').last().dropna()
    res['market'] = dict(label='等權市場（買進持有）', days=int(len(mkt)),
                         ann=float(mkt_ann), sharpe=float(mkt_sharpe),
                         mdd=float(mdd), years=float(years),
                         curve={'dates': [d.strftime('%Y-%m') for d in _mm.index],
                                'nav': [round(float(v), 4) for v in _mm.values]})

    # ---- 固定持有 20 日：驗證子代理的觸及數單調遞減
    rows = []
    for _, r in t.iterrows():
        s = r['symbol']
        if s not in px:
            continue
        ser = px[s]
        i = ser.index.searchsorted(r['entry_date'])
        if i >= len(ser) or i + 20 >= len(ser):
            continue
        r20 = ser.iloc[i + 20] / ser.iloc[i] - 1
        d0, d1 = ser.index[i], ser.index[i + 20]
        try:
            mret = nav.loc[d1] / nav.loc[d0] - 1
        except KeyError:
            continue
        rows.append(dict(symbol=s, entry_date=r['entry_date'], touches=int(r['touches']),
                         ret20=float(r20), excess20=float(r20 - mret)))
    f20 = pd.DataFrame(rows)
    f20['grp'] = np.where(f20.touches >= 6, '6+', f20.touches.astype(str))

    groups = {}
    for k in ('3', '4', '5', '6+'):
        s = f20[f20.grp == k]
        if len(s) < 50:
            continue
        groups[k] = dict(n=int(len(s)), ret=float(s.ret20.mean()),
                         excess=float(s.excess20.mean()),
                         win=float((s.ret20 > 0).mean()))
    res['touch_fixed20'] = groups

    # 同一個觸及數分組，改用我的出場規則（已在卡片主結果中）
    groups_ma20 = {}
    m0 = nav.reindex(t.entry_date).ffill().values
    m1 = nav.reindex(t.exit_date).ffill().values
    t2 = t.copy()
    t2['excess'] = t2.ret.values - (m1 / m0 - 1)
    t2['grp'] = np.where(t2.touches >= 6, '6+', t2.touches.astype(str))
    for k in ('3', '4', '5', '6+'):
        s = t2[t2.grp == k]
        if len(s) < 50:
            continue
        groups_ma20[k] = dict(n=int(len(s)), ret=float(s.ret.mean()),
                              excess=float(s.excess.mean()),
                              win=float(s.ret.gt(0).mean()))
    res['touch_ma20'] = groups_ma20

    json.dump(res, open(f'{OUT}/trend_calendar.json', 'w'), ensure_ascii=False, indent=1)

    print('\n=== 日曆時間組合（年化）===')
    for k in ('trend_ma20', 'market'):
        v = res[k]
        if v:
            print(f"  {v['label']:28s} 年化 {v['ann']*100:6.2f}%  夏普 {v['sharpe']:.3f}  "
                  f"最大回撤 {v['mdd']*100:6.2f}%  {v['years']:.1f} 年")

    print('\n=== 觸及數 vs 報酬：兩種出場規則 ===')
    print(f"  {'分組':>4} {'固定20日 n':>10} {'固定20日原始':>12} {'MA20出場 n':>10} {'MA20出場原始':>12}")
    for k in ('3', '4', '5', '6+'):
        a = res['touch_fixed20'].get(k)
        b = res['touch_ma20'].get(k)
        if a and b:
            print(f"  {k:>4} {a['n']:>10,} {a['ret']*100:>11.3f}% {b['n']:>10,} {b['ret']*100:>11.3f}%")
    print('已寫 trend_calendar.json')


if __name__ == '__main__':
    main()
