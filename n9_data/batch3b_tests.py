#!/usr/bin/env python3
"""第三批修正版：修掉兩個實質錯誤。

## 錯誤一：EP1 樣本為 0

`pd.Series(x).rolling(10).max()` 的 `min_periods` 預設等於視窗長度，
也就是「視窗內必須有 10 個非 NaN 值」才會有結果。
而帶量大陽燭很稀有（260 萬股票日裡只有 1,399 次），
所以每一格幾乎都是 NaN → 突破條件恆為 False → 樣本 0。
修正：`rolling(10, min_periods=1)`。

## 錯誤二：市場寬度的指數被污染

第一版用「611 檔每日報酬的等權平均」當市場指數，
但母體裡有 +2396%／−2714% 這種單日極端值（低價股、分割未調整），
複利下去會長出一個不存在的指數——所以才會算出「背馳之後 60 日 +38%」
這種在真實指數上不可能出現的數字。
修正：**指數一律改用真實 SPY / QQQ**，寬度（站上 50MA / 200MA 的比例）
仍然由 611 檔計算——那部分只用到收盤價與均線，不受報酬極端值影響。

## 錯誤三（方法論）：唐奇安突破用「20 日固定前瞻報酬」測不對

海龜法的優勢不在「突破後 20 日漲多少」，而在「用 10 日低點出場」的
不對稱停利——讓趨勢跑、把虧損砍短。用固定 20 日持有期去測，
等於把它的核心機制拆掉。
修正：改用**真實的進出場模擬**（突破進、收盤跌破 10 日低點出），
統計每筆交易的報酬、勝率、賺賠比、期望值。
"""
import glob
import json
import os

import numpy as np
import pandas as pd

STOCK = '/root/.openclaw/stock_data'
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')


def nwt(e, lag):
    e = np.asarray(e, float)
    e = e[~np.isnan(e)]
    n = len(e)
    if n < 40:
        return np.nan
    mu = e.mean()
    x = e - mu
    s = (x ** 2).mean()
    for l in range(1, min(lag, n - 1) + 1):
        s += 2 * (1 - l / (lag + 1)) * (x[l:] * x[:-l]).mean()
    return mu / np.sqrt(max(s, 1e-18) / n)


def trimmed(x, p=0.01):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 50:
        return float(np.median(x)) if len(x) else np.nan
    lo, hi = np.quantile(x, [p, 1 - p])
    return float(x[(x >= lo) & (x <= hi)].mean())


def load_stocks():
    out = {}
    for f in sorted(glob.glob(f'{STOCK}/live/*.csv') + glob.glob(f'{STOCK}/extra/*.csv')):
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        d.columns = [c.lower() for c in d.columns]
        if not {'date', 'open', 'high', 'low', 'close', 'volume'}.issubset(d.columns):
            continue
        if len(d) < 400:
            continue
        d['date'] = pd.to_datetime(d['date'])
        d = d.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
        out[os.path.basename(f).replace('.csv', '')] = d
    return out


def load_index(path):
    d = pd.read_csv(path)
    d.columns = [c.strip().lower() for c in d.columns]
    d = d.rename(columns={d.columns[0]: 'date'})
    d['date'] = pd.to_datetime(d['date'])
    d = d.dropna(subset=['close']).sort_values('date').reset_index(drop=True)
    return d[['date', 'close']]


def turtle_sim(stocks, n_enter, n_exit):
    """真實進出場：突破 n_enter 日高點買進，收盤跌破 n_exit 日低點賣出。"""
    trades = []
    for sym, d in stocks.items():
        close = d['close'].to_numpy(float)
        high = d['high'].to_numpy(float)
        low = d['low'].to_numpy(float)
        n = len(close)
        if n < 300:
            continue
        hi_e = pd.Series(high).rolling(n_enter).max().shift(1).to_numpy()
        lo_x = pd.Series(low).rolling(n_exit).min().shift(1).to_numpy()

        pos = 0.0
        entry_i = -1
        for i in range(max(n_enter, n_exit) + 1, n):
            if pos == 0:
                if not np.isnan(hi_e[i]) and close[i] > hi_e[i]:
                    pos = close[i]
                    entry_i = i
            else:
                if not np.isnan(lo_x[i]) and close[i] < lo_x[i]:
                    trades.append(dict(symbol=sym, entry_date=d['date'].iloc[entry_i],
                                       exit_date=d['date'].iloc[i],
                                       entry=pos, exit=float(close[i]),
                                       ret=float(close[i]) / pos - 1,
                                       bars=int(i - entry_i)))
                    pos = 0.0
        # 未平倉：以最後收盤結算（標記為未平倉）
        if pos > 0:
            trades.append(dict(symbol=sym, entry_date=d['date'].iloc[entry_i],
                               exit_date=d['date'].iloc[n - 1],
                               entry=pos, exit=float(close[-1]),
                               ret=float(close[-1]) / pos - 1,
                               bars=int(n - 1 - entry_i)))
    return pd.DataFrame(trades)


def trade_stats(t, label, hold_base):
    if len(t) < 30:
        return {'label': label, 'n': int(len(t)), 'note': '樣本過少'}
    r = t['ret'].to_numpy()
    wins = r[r > 0]
    losses = r[r <= 0]
    # 期望值以交易日為單位近似：每筆平均報酬 / 平均持有天數
    return dict(label=label, n=int(len(r)),
                mean=float(r.mean()), median=float(np.median(r)),
                trim=trimmed(r),
                win=float((r > 0).mean()),
                avg_win=float(wins.mean()) if len(wins) else 0.0,
                avg_loss=float(losses.mean()) if len(losses) else 0.0,
                payoff=float(wins.mean() / abs(losses.mean())) if len(losses) and len(wins) else np.nan,
                expectancy=float(r.mean()),
                bars=float(t['bars'].mean()),
                big_loss=float((r <= -0.20).mean()),
                hold_base=hold_base)


def main():
    stocks = load_stocks()
    print(f'母體 {len(stocks)} 檔', flush=True)

    # ================= A. 海龜／唐奇安：真實進出場 =================
    print('\n模擬唐奇安突破（真實進出場，非固定持有期）...', flush=True)
    t20 = turtle_sim(stocks, 20, 10)
    print(f'  20 進 / 10 出：{len(t20):,} 筆交易', flush=True)
    t55 = turtle_sim(stocks, 55, 20)
    print(f'  55 進 / 20 出：{len(t55):,} 筆交易', flush=True)

    # 對照：隨機持有相同天數的報酬（用同一批股票、同分布持有期）
    rng = np.random.default_rng(7)
    rnd = []
    for sym, d in stocks.items():
        close = d['close'].to_numpy(float)
        n = len(close)
        if n < 300:
            continue
        for _ in range(max(1, n // 400)):
            i = int(rng.integers(60, n - 60))
            h = 25   # 接近唐奇安的平均持有期
            if i + h < n:
                rnd.append(close[i + h] / close[i] - 1)
    rnd = np.array(rnd)

    turtle = dict(
        t20=trade_stats(t20, '20 日突破進、10 日低點出', float(t20['bars'].mean())),
        t55=trade_stats(t55, '55 日突破進、20 日低點出', float(t55['bars'].mean())),
        random=dict(n=int(len(rnd)), mean=float(rnd.mean()), median=float(np.median(rnd)),
                    trim=trimmed(rnd), win=float((rnd > 0).mean()),
                    avg_win=float(rnd[rnd > 0].mean()),
                    avg_loss=float(rnd[rnd <= 0].mean()),
                    payoff=float(rnd[rnd > 0].mean() / abs(rnd[rnd <= 0].mean())),
                    big_loss=float((rnd <= -0.20).mean())),
    )

    # ================= B. 市場寬度背馳（用真實指數）=================
    frames = []
    for sym, d in stocks.items():
        close = d['close'].to_numpy(float)
        if len(close) < 400:
            continue
        s = pd.Series(close)
        ma50 = s.rolling(50).mean().to_numpy()
        ma200 = s.rolling(200).mean().to_numpy()
        frames.append(pd.DataFrame({
            'date': d['date'].to_numpy(),
            'above50': (close > ma50) & ~np.isnan(ma50),
            'above200': (close > ma200) & ~np.isnan(ma200),
            'has50': ~np.isnan(ma50), 'has200': ~np.isnan(ma200),
        }))
    A = pd.concat(frames, ignore_index=True)
    g = A.groupby('date')
    pct50 = g.apply(lambda x: x.loc[x.has50, 'above50'].mean(), include_groups=False)
    pct200 = g.apply(lambda x: x.loc[x.has200, 'above200'].mean(), include_groups=False)
    breadth = pd.DataFrame({'pct50': pct50, 'pct200': pct200}).dropna()
    breadth = breadth[breadth.pct200 > 0]
    print(f'\n市場寬度：{len(breadth):,} 個交易日', flush=True)

    breadth_res = {}
    for name, path in (('SPY', 'data_yahoo/SPY_30y.csv'), ('QQQ', 'data_yahoo/QQQ_30y.csv')):
        p = os.path.join(HERE, path)
        if not os.path.exists(p):
            continue
        ix = load_index(p).set_index('date')['close']
        j = breadth.join(ix.rename('idx'), how='inner').dropna()
        if len(j) < 300:
            print(f'  {name}: 對齊後只有 {len(j)} 天，跳過'); continue

        idx = j['idx']
        hi60 = idx >= idx.rolling(60).max()
        w50 = j['pct50'].rolling(20).mean()
        w200 = j['pct200'].rolling(20).mean()
        w200_hi = w200 >= w200.rolling(60).max()

        diverge = hi60 & (~w200_hi) & (w200 > 0)
        sync = hi60 & w200_hi

        rec = dict(n_days=int(len(j)), n_div=int(diverge.sum()), n_sync=int(sync.sum()),
                   pct200_min=float(j['pct200'].min()), pct200_max=float(j['pct200'].max()),
                   div={}, sync={}, diff={})
        for lab, mask in (('div', diverge), ('sync', sync)):
            for h in (10, 20, 60):
                fwd = (idx[mask].shift(-h) / idx[mask] - 1).dropna().to_numpy()
                if len(fwd) < 20:
                    continue
                rec[lab][h] = dict(trim=trimmed(fwd), median=float(np.median(fwd)),
                                   pos=float((fwd > 0).mean()),
                                   big_loss=float((fwd <= -0.10).mean()),
                                   n=int(len(fwd)), t=float(nwt(fwd, h)))
        for h in (10, 20, 60):
            if h not in rec['div'] or h not in rec['sync']:
                continue
            a = (idx[diverge].shift(-h) / idx[diverge] - 1).dropna().to_numpy()
            b = (idx[sync].shift(-h) / idx[sync] - 1).dropna().to_numpy()
            rng2 = np.random.default_rng(31)
            boot = np.array([rng2.choice(a, len(a), True).mean()
                             - rng2.choice(b, len(b), True).mean() for _ in range(1000)])
            rec['diff'][h] = dict(diff=float(a.mean() - b.mean()),
                                  ci=[float(np.quantile(boot, .025)),
                                      float(np.quantile(boot, .975))],
                                  n_div=int(len(a)), n_sync=int(len(b)))
        breadth_res[name] = rec

    # ================= C. EP1 量價突破（修正 min_periods）=================
    print('\nEP1 量價突破（修正 rolling min_periods）...', flush=True)
    rows = []
    for sym, d in stocks.items():
        close = d['close'].to_numpy(float)
        high = d['high'].to_numpy(float)
        vol = d['volume'].to_numpy(float)
        n = len(close)
        if n < 400:
            continue
        pc = np.r_[np.nan, close[:-1]]
        dr = close / pc - 1
        v20 = pd.Series(vol).rolling(20).mean().to_numpy()
        ma50 = pd.Series(close).rolling(50).mean().to_numpy()
        ma50u = np.r_[np.full(10, np.nan), (ma50[10:] > ma50[:-10]).astype(float)]
        up = (close > ma50) & (ma50u == 1)

        big = (dr >= 0.05) & (vol >= 2 * v20)
        # ★ 修正：min_periods=1，否則稀有事件一律 NaN
        bc_hi = pd.Series(np.where(big, high, np.nan)).rolling(10, min_periods=1).max().shift(1).to_numpy()
        low_vol = vol < v20
        ep1 = (close > bc_hi) & low_vol & up
        rows.append(pd.DataFrame({
            'symbol': sym, 'date': d['date'].to_numpy(), 'ep1': ep1,
            'big_any': big,
        }))
    E = pd.concat(rows, ignore_index=True)

    # EP1 的前瞻報酬要重新算，屬於本檔
    frames = []
    for sym, d in stocks.items():
        close = d['close'].to_numpy(float)
        if len(close) < 400:
            continue
        dd = pd.DataFrame({'symbol': sym, 'date': d['date'].to_numpy()})
        for h in (5, 10, 20, 60):
            f = np.full(len(close), np.nan)
            f[:-h] = close[h:] / close[:-h] - 1
            dd[f'f{h}'] = f
        frames.append(dd)
    F = pd.concat(frames, ignore_index=True)
    M = E.merge(F, on=['symbol', 'date'], how='left')
    ep1_mask = M['ep1'].fillna(False).to_numpy()
    n_ep1 = int(ep1_mask.sum())
    print(f'  EP1 訊號 {n_ep1:,} 次', flush=True)

    ep1_res = {'n': n_ep1, 'by_h': {}}
    upf = pd.DataFrame({'date': M['date'], 'up': True})
    for h in (5, 10, 20, 60):
        v = M.loc[ep1_mask, f'f{h}'].dropna().to_numpy()
        allv = M[f'f{h}'].dropna().to_numpy()
        if len(v) >= 40:
            ep1_res['by_h'][h] = dict(n=int(len(v)), trim=trimmed(v),
                                      median=float(np.median(v)),
                                      pos=float((v > 0).mean()),
                                      big_loss=float((v <= -0.20).mean()),
                                      all_trim=trimmed(allv))

    out = dict(turtle=turtle, breadth=breadth_res, ep1=ep1_res,
               n_stocks=len(stocks))
    json.dump(out, open(f'{OUT}/batch3b.json', 'w'), ensure_ascii=False, indent=1,
              default=str)

    # ================= 列印 =================
    print('\n' + '=' * 74)
    print('【A. 海龜／唐奇安：真實進出場模擬】')
    print(f"{'策略':<26}{'筆數':>8}{'平均':>9}{'勝率':>7}{'賺賠比':>8}{'持有天':>7}")
    for k in ('t20', 't55'):
        v = turtle[k]
        if 'note' in v:
            print(f"  {v['label']}: {v['note']}"); continue
        print(f"  {v['label']:<24}{v['n']:>8,}{v['mean']*100:>+8.2f}%"
              f"{v['win']*100:>6.1f}%{v['payoff']:>8.2f}{v['bars']:>7.0f}")
    rv = turtle['random']
    print(f"  {'隨機持有同期（對照）':<24}{rv['n']:>8,}{rv['mean']*100:>+8.2f}%"
          f"{rv['win']*100:>6.1f}%{rv['payoff']:>8.2f}{25:>7.0f}")

    print('\n' + '=' * 74)
    print('【B. 市場寬度背馳】他的主張：背馳出現要減倉')
    for name, rec in breadth_res.items():
        print(f"\n  ── {name}（{rec['n_days']:,} 天，寬度 200MA {rec['pct200_min']*100:.0f}%~{rec['pct200_max']*100:.0f}%）──")
        print(f"    背馳 {rec['n_div']} 天 / 同步創高 {rec['n_sync']} 天")
        print(f"    {'持有期':<8}{'背馳後':>10}{'同步創高後':>12}{'差值':>10}{'CI':>22}")
        for h in (10, 20, 60):
            if h not in rec['diff']:
                continue
            dv = rec['diff'][h]
            d1 = rec['div'][h]['trim']
            s1 = rec['sync'][h]['trim']
            print(f"    {h:>2}日{'':<4}{d1*100:>+9.2f}%{s1*100:>+11.2f}%"
                  f"{dv['diff']*100:>+9.2f}%   [{dv['ci'][0]*100:+.2f},{dv['ci'][1]*100:+.2f}]")

    print('\n' + '=' * 74)
    print('【C. EP1 量價突破】')
    if n_ep1 == 0:
        print('  仍為 0，需再檢查')
    else:
        for h, v in ep1_res['by_h'].items():
            print(f"  {h:>2}日: {v['trim']*100:+.2f}%  (n={v['n']:,}, 上漲 {v['pos']*100:.0f}%, "
                  f"全樣本 {v['all_trim']*100:+.2f}%)")

    print('\n已寫 batch3b.json')


if __name__ == '__main__':
    main()
