#!/usr/bin/env python3
"""同一個策略，兩種基準，看結論會不會翻。

## 為什麼要做這個

我先前用「等權市場指數的複利報酬」當基準，得到趨勢線超額 −0.02%。
改成「同期的橫斷面平均個股報酬」當基準，同一批交易變成 +0.03%。
**同一份交易、同一個期間，只因基準定義不同就換了正負號。**

差別來自**波動拖累（volatility drag）**：
單一檔股票的複利報酬被自己的高波動拖下來，指數（616 檔平均）拖得少。
所以指數複利會系統性地高於「隨機挑一檔股票的期望報酬」。
實測 20 日差距約 0.19 個百分點——而這剛好和我量到的效應同一個數量級。

## 兩個基準各自回答什麼

基準 A（指數複利）：**「我如果去買大盤，會不會更好？」**
  這是投資人真正在意的問題，也是經濟上正確的口徑。
  但它會懲罰高波動的個股——那是分散化的真實價值，不算偏差。

基準 B（橫斷面平均個股）：**「我選的股票有沒有高於同期平均？」**
  這是「選股能力」的口徑。隨機挑一檔的期望超額恰好為 0。

## 結論怎麼寫

如果兩個基準給出的答案方向不一致，而且差距（0.19 個百分點）
大於效應本身（0.02~0.06 個百分點），
**那正確的結論不是「有效」或「無效」，而是「這個效應小於基準選擇的不確定性，量不出來」。**
這比任選一個基準宣稱結論誠實得多。
"""
import glob
import json
import os

import numpy as np
import pandas as pd

STOCK = '/root/.openclaw/stock_data'
OUT = '/root/.openclaw/workspace/backtest_videos/xbx/out'
LINES = '/root/.openclaw/workspace/reports/_agent_lines'
PATT = '/root/.openclaw/workspace/reports/_agent_pattern'


def load_prices():
    cols = {}
    for f in sorted(glob.glob(f'{STOCK}/live/*.csv') + glob.glob(f'{STOCK}/extra/*.csv')):
        try:
            d = pd.read_csv(f, usecols=lambda c: c.lower() in ('date', 'close'))
        except Exception:
            continue
        d.columns = [c.lower() for c in d.columns]
        if 'date' not in d.columns or 'close' not in d.columns or len(d) < 300:
            continue
        d['date'] = pd.to_datetime(d['date'])
        cols[os.path.basename(f).replace('.csv', '')] = (
            d.dropna(subset=['close']).set_index('date')['close'])
    P = pd.concat(cols, axis=1, sort=True).sort_index()
    return P.where(P > 0)


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


def nwt(ex, lag=10):
    n = len(ex)
    mu = ex.mean()
    e = ex - mu
    s = (e ** 2).mean()
    for l in range(1, lag + 1):
        s += 2 * (1 - l / (lag + 1)) * (e[l:] * e[:-l]).mean()
    return mu / np.sqrt(max(s, 1e-18) / n)


def main():
    P = load_prices()
    idx = P.index
    A = P.to_numpy(float)

    # 等權市場指數淨值（複利）
    daily = np.nanmean(A[1:] / A[:-1] - 1.0, axis=1)
    nav = np.concatenate([[1.0], np.cumprod(1 + np.nan_to_num(daily))])
    navs = pd.Series(nav, index=idx)

    # 橫斷面平均 h 日報酬
    benchX = {}
    for h in range(1, 62):
        r = A[h:] / A[:-h] - 1.0
        with np.errstate(invalid='ignore'):
            cnt = np.sum(~np.isnan(r), axis=1)
            ssum = np.nansum(r, axis=1)
        b = np.full(len(idx), np.nan)
        g = cnt >= 30
        b[:len(idx) - h][g] = ssum[g] / cnt[g]
        benchX[h] = b

    def run(path, label):
        df = pd.read_csv(path)
        df['entry_date'] = pd.to_datetime(df.entry_date)
        df['exit_date'] = pd.to_datetime(df.exit_date)
        df = dedup(df)
        h = df['hold'].astype(int).clip(1, 61).to_numpy()
        m0 = navs.reindex(df.entry_date).ffill().to_numpy()
        m1 = navs.reindex(df.exit_date).ffill().to_numpy()
        exA = df.ret.to_numpy() - (m1 / m0 - 1)

        pi = pd.Series(np.arange(len(idx)), index=idx).reindex(
            pd.DatetimeIndex(df.entry_date)).to_numpy()
        bm = np.array([benchX[int(hh)][int(p)] if not np.isnan(p) else np.nan
                       for hh, p in zip(h, pi)], dtype=float)
        exB = df.ret.to_numpy() - bm
        ok = ~np.isnan(exB)
        exA, exB, ret = exA[ok], exB[ok], df.ret.to_numpy()[ok]

        def stats(ex):
            q = np.quantile(ex, .95)
            return dict(excess=float(ex.mean()), t=float(nwt(ex)),
                        trimmed=float(ex[ex <= q].mean()))

        return dict(label=label, n=int(ok.sum()),
                    avg=float(ret.mean()), win=float((ret > 0).mean()),
                    A=stats(exA), B=stats(exB), hold=float(df['hold'].mean()))

    jobs = [
        (f'{LINES}/trendline_trades.csv', '趨勢線突破'),
        (f'{LINES}/trendline_control.csv', '突破 20 日新高'),
        (f'{PATT}/channel_trades.csv', '平行通道'),
        (f'{PATT}/channel_control.csv', '通道對照（跌破MA20後站回）'),
    ]
    res = [run(p, l) for p, l in jobs]

    print(f"{'策略':<24}{'n':>8}{'原始':>9}{'基準A':>9}{'t':>7}"
          f"{'基準B':>9}{'t':>7}{'剔除5%(A)':>11}")
    for r in res:
        print(f"{r['label']:<24}{r['n']:>8,}{r['avg']*100:>8.3f}%"
              f"{r['A']['excess']*100:>8.3f}%{r['A']['t']:>+7.2f}"
              f"{r['B']['excess']*100:>8.3f}%{r['B']['t']:>+7.2f}"
              f"{r['A']['trimmed']*100:>10.3f}%")

    print()
    print('基準A＝等權指數複利（回答「不如買大盤？」）')
    print('基準B＝同期橫斷面平均個股（回答「選股有沒有高於平均？」）')
    print()
    print('--- 基準敏感度 ---')
    for r in res:
        gap = r['A']['excess'] - r['B']['excess']
        eff = max(abs(r['A']['excess']), abs(r['B']['excess']))
        print(f"  {r['label']:<24} 兩基準差 {gap*100:+.3f}pp  "
              f"效應最大 {eff*100:.3f}pp  "
              f"{'效應 < 基準差（量不出來）' if eff < abs(gap) else '效應 > 基準差'}")

    json.dump(res, open(f'{OUT}/bench_dual.json', 'w'), ensure_ascii=False, indent=1)
    print('\n已寫 bench_dual.json')


if __name__ == '__main__':
    main()
