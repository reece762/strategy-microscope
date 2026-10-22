#!/usr/bin/env python3
"""第一梯：三支已取得完整逐字稿的策略，用美股/加密貨幣資料重跑。

1. NO5JOVjDO_I  MACD 頂底背離（他測 A 股全部 12 萬次）
2. 08lNIsMgylY  量比（大師說 2.5–5 買進、>5 離場）
3. mUMyXYxpL24  跳空必回補（他測上證 813 次跳空）

規則一律取影片逐字稿的原文，不自行補完；做不到的部分明確標示為未測。
"""
import glob, io, json, os, zipfile
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
DAILY = os.path.join(os.path.dirname(HERE), 'data_daily')
OUT = os.path.join(HERE, 'out')
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------- data
def load_fred(sid):
    df = pd.read_csv(os.path.join(DATA, f'{sid}.csv'))
    df.columns = ['date', 'close']
    df['date'] = pd.to_datetime(df['date'])
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    return df.dropna().set_index('date').sort_index()


def load_etf(sym):
    df = pd.read_csv(os.path.join(DAILY, f'{sym}.csv'))
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
    for c in ['open', 'high', 'low', 'close', 'volume']:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df.set_index('date').sort_index()


def load_binance(sym):
    frames = []
    for f in sorted(glob.glob(os.path.join(DATA, f'{sym}-1d-*.zip'))):
        with zipfile.ZipFile(f) as z:
            raw = z.read(z.namelist()[0]).decode()
        d = pd.read_csv(io.StringIO(raw), header=None).iloc[:, :6]
        d.columns = ['ts', 'open', 'high', 'low', 'close', 'volume']
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    ts = pd.to_numeric(d['ts'], errors='coerce')
    ts = ts.where(ts < 1e14, ts / 1000.0)      # microseconds -> milliseconds
    d['date'] = pd.to_datetime(ts, unit='ms')
    for c in ['open', 'high', 'low', 'close', 'volume']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    return d.dropna(subset=['date', 'close']).drop_duplicates('date').set_index('date').sort_index()


def tstat(x):
    """t statistics for mean != 0; nan-safe on tiny samples."""
    x = pd.Series(x).dropna().astype(float)
    if len(x) < 2 or x.std(ddof=1) == 0:
        return 0.0
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))))



def base_stats(close, horizons=(1, 5, 10, 20)):
    """Unconditional forward return for the SAME market and horizon.

    Without this, a 20-day mean of +0.8% looks like an up-edge on an index that
    simply drifts upward. Every signal stat is reported as excess over this.
    """
    out = {}
    for h in horizons:
        fwd = (close.shift(-h) / close - 1).dropna()
        if len(fwd) < 30:
            continue
        out[h] = dict(mean=float(fwd.mean()), hit_up=float((fwd > 0).mean()),
                      hit_dn=float((fwd < 0).mean()), sd=float(fwd.std(ddof=1)),
                      n=int(len(fwd)))
    return out


def excess_t(sig_vals, base_mean, sd):
    """t statistic for (signal mean - unconditional mean), using the pooled sd.

    Using the full-sample sd as the reference avoids overstating significance
    when the signal subset is small.
    """
    v = pd.Series(sig_vals).dropna().astype(float)
    if len(v) < 3 or not sd or sd <= 0:
        return 0.0
    return float((v.mean() - base_mean) / (sd / np.sqrt(len(v))))


# ------------------------------------------------- 1. MACD divergence
def macd_diff(close, fast=12, slow=26):
    ef = close.ewm(span=fast, adjust=False).mean()
    es = close.ewm(span=slow, adjust=False).mean()
    return ef - es


def swing_points(close, look=1, confirm=30):
    """影片的定義（逐字稿原文）：
       1) 三根 K 線中間那根收盤價高於左右各 look 根 -> 局部高點
       2) 該高點必須創過去 confirm 日新高
       3) 該高點必須高於前一個高點（同理低點反向）
    回傳局部高點/低點的索引。
    """
    c = close.values
    n = len(c)
    highs, lows = [], []
    for i in range(look, n - look):
        seg = c[i - look:i + look + 1]
        if c[i] == seg.max() and c[i] > c[i - 1] and c[i] > c[i + 1]:
            if i - confirm >= 0 and c[i] > c[i - confirm:i].max():
                highs.append(i)
        if c[i] == seg.min() and c[i] < c[i - 1] and c[i] < c[i + 1]:
            if i - confirm >= 0 and c[i] < c[i - confirm:i].min():
                lows.append(i)
    return highs, lows


def divergence(close, top=True):
    """回傳背離確認日的索引（高點/低點當日收盤確認，隔日開盤進場）。"""
    dif = macd_diff(close)
    highs, lows = swing_points(close)
    pts = highs if top else lows
    idx, conf = [], []
    prev = None
    for i in pts:
        if prev is not None:
            price_new = close.iloc[i] > close.iloc[prev] if top else close.iloc[i] < close.iloc[prev]
            if not price_new:
                prev = i
                continue
            dif_weaker = dif.iloc[i] < dif.iloc[prev] if top else dif.iloc[i] > dif.iloc[prev]
            if dif_weaker:
                idx.append(i)
                conf.append(prev)
        prev = i
    return idx, conf, dif


def run_macd_divergence():
    """頂背離看後續下跌、底背離看後續上漲；同時做一個只做多/空倉的機械策略。

    影片的做法：出現底背離買入、出現頂背離平倉，重複訊號忽略。
    """
    out = {}
    markets = {
        '納斯達克綜合指數': load_fred('NASDAQCOM'),
        'SPY': load_etf('SPY'),
        'QQQ': load_etf('QQQ'),
        'BTC': load_binance('BTCUSDT'),
    }
    for name, df in markets.items():
        close = df['close']
        res = {}
        for top in (True, False):
            idx, conf, dif = divergence(close, top=top)
            key = 'top' if top else 'bottom'
            rows = []
            raw = {1: [], 5: [], 20: []}
            for i in idx:
                if i + 21 >= len(close):
                    continue
                entry = df['open'].iloc[i + 1] if 'open' in df else close.iloc[i + 1]
                if not np.isfinite(entry) or entry <= 0:
                    continue
                rec = {'date': close.index[i].strftime('%Y-%m-%d')}
                for h in (1, 5, 20):
                    j = i + 1 + h
                    if j < len(close):
                        # 一律存「真實報酬」。方向判斷留到統計時才套用一次；
                        # 先前在這裡先翻轉、後面又跟原始基準比，方向錯了兩次、
                        # 顯著性也跟著算錯。
                        rec[f'r{h}'] = float(close.iloc[j] / entry - 1)
                    raw[h].append(rec.get(f'r{h}'))
                rows.append(rec)
            d = pd.DataFrame(rows)
            dirn = -1 if top else 1          # 頂背離預期下跌，取負號後應為正
            stat = {'n': len(idx), 'dirn': dirn}
            for h in (1, 5, 20):
                col = d.get(f'r{h}')
                if col is not None and col.notna().sum() > 1:
                    stat[f'r{h}_mean'] = float(col.mean())            # 真實報酬
                    stat[f'r{h}_hit'] = float((col > 0).mean())       # 真實上漲率
                    stat[f'r{h}_t'] = tstat(col)
                    stat[f'r{h}_dir_mean'] = float(dirn * col.mean())  # 正數＝符合宣稱
                    stat[f'r{h}_dir_t'] = tstat(dirn * col)
            stat['_raw'] = raw
            res[key] = stat

        # 機械策略：底背離做多、頂背離出場，其餘空倉
        longs, shorts = [], []
        b_idx, _, _ = divergence(close, top=False)
        t_idx, _, _ = divergence(close, top=True)
        pos = pd.Series(0.0, index=close.index)
        cur = 0.0
        for i in range(len(close)):
            if i in set(b_idx):
                cur = 1.0
            elif i in set(t_idx):
                cur = 0.0
            pos.iloc[i] = cur
        # 訊號當日收盤確認 -> 隔日才生效
        held = pos.shift(1).fillna(0.0)
        ret = close.pct_change().fillna(0.0)
        cost = held.diff().abs().fillna(0.0) * 0.0001
        strat = (1 + held * ret - cost).cumprod()
        bh = (1 + ret).cumprod()
        yrs = (close.index[-1] - close.index[0]).days / 365.25

        def perf(eq):
            r = eq.pct_change().dropna()
            ppy = 252 if name != 'BTC' else 365
            return dict(
                cagr=float(eq.iloc[-1] ** (1 / yrs) - 1),
                mdd=float((eq.values / np.maximum.accumulate(eq.values) - 1).min()),
                sharpe=float(r.mean() / r.std() * np.sqrt(ppy)) if r.std() > 0 else 0.0,
            )

        base = base_stats(close)
        res['base'] = base
        for key in ('top', 'bottom'):
            blk = res[key]
            dirn = blk.get('dirn', 1)
            for h in (1, 5, 20):
                bm = base.get(h, {}).get('mean')
                bsd = base.get(h, {}).get('sd')
                if bm is None or blk.get(f'r{h}_mean') is None:
                    continue
                # 真實超額：正數＝跑贏同期無條件平均
                blk[f'r{h}_excess'] = blk[f'r{h}_mean'] - bm
                # 方向調整後超額：頂背離若真的轉弱，這個值應為「負」
                blk[f'r{h}_excess_dir'] = dirn * (blk[f'r{h}_mean'] - bm)
                blk[f'r{h}_excess_t'] = excess_t(
                    [dirn * v for v in (blk.get('_raw', {}).get(h) or []) if v is not None],
                    dirn * bm, bsd)
        res['strategy'] = perf(strat)
        res['bh'] = perf(bh)
        res['exposure'] = float(held.mean())
        # 同曝險、不擇時基準：固定槓桿買進持有
        e = res['exposure']
        if e > 0:
            em = (1 + held.mean() * ret).cumprod()
            # 逐日用固定曝險係數，避免名目漂移
            em = (1 + e * ret).cumprod()
            res['em'] = perf(em)
        res['years'] = yrs
        res['start'] = close.index[0].strftime('%Y-%m-%d')
        res['end'] = close.index[-1].strftime('%Y-%m-%d')
        out[name] = res
    return out


# --------------------------------------------------------- 2. 量比
def volume_ratio(df, win=5):
    """量比 = 當日成交量 / 前 win 日平均成交量（影片：過去五個交易日均量）。"""
    v = df['volume'].astype(float)
    return v / v.shift(1).rolling(win).mean()


def run_volume_ratio():
    """大師規則：2.5 < 量比 < 5 買進；量比 > 5 賣出。
    影片另測了搭配換手率的版本，但 ETF 沒有流通股數，無法誠實計算換手率，
    因此這裡只測量比單獨使用的版本，並在輸出中標示未測項。
    """
    out = {}
    for name, sym in (('SPY', 'SPY'), ('QQQ', 'QQQ')):
        df = load_etf(sym)
        vr = volume_ratio(df)
        close = df['close']
        fwd = {}
        for h in (1, 3, 5, 10):
            fwd[h] = close.shift(-h) / close - 1

        buy = (vr > 2.5) & (vr < 5)
        sell = vr > 5
        stats = {'n_buy': int(buy.sum()), 'n_sell': int(sell.sum())}
        for h in (1, 3, 5, 10):
            b = fwd[h][buy].dropna()
            s = fwd[h][sell].dropna()
            stats[f'buy_h{h}_mean'] = float(b.mean()) if len(b) else None
            stats[f'buy_h{h}_hit'] = float((b > 0).mean()) if len(b) else None
            stats[f'buy_h{h}_t'] = tstat(b)
            stats[f'sell_h{h}_mean'] = float(s.mean()) if len(s) else None
            stats[f'sell_h{h}_downhit'] = float((s < 0).mean()) if len(s) else None
            stats[f'sell_h{h}_t'] = tstat(s)

        # 機械策略：訊號觸發隔日開盤買、持有 2 日賣（影片實測用持有兩天）
        sig = buy.shift(1).fillna(False)
        hold = pd.Series(0.0, index=close.index)
        hold[sig] = 1.0
        hold = hold.rolling(2, min_periods=1).max()   # 持有兩天
        ret = close.pct_change().fillna(0.0)
        cost = hold.diff().abs().fillna(0.0) * 0.0001
        strat = (1 + hold * ret - cost).cumprod()
        bh = (1 + ret).cumprod()
        yrs = (close.index[-1] - close.index[0]).days / 365.25

        def perf(eq):
            r = eq.pct_change().dropna()
            return dict(cagr=float(eq.iloc[-1] ** (1 / yrs) - 1),
                        mdd=float((eq.values / np.maximum.accumulate(eq.values) - 1).min()),
                        sharpe=float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0)

        e = float(hold.mean())
        stats['strategy'] = perf(strat)
        stats['bh'] = perf(bh)
        stats['exposure'] = e
        stats['em'] = perf((1 + e * ret).cumprod())
        stats['years'] = yrs
        stats['start'] = close.index[0].strftime('%Y-%m-%d')
        stats['end'] = close.index[-1].strftime('%Y-%m-%d')
        out[name] = stats
    out['_not_tested'] = '換手率結合版（ETF 無流通股數，無法計算換手率）'
    return out


# ------------------------------------------------------- 3. 跳空回補
def run_gaps():
    """向上跳空：當日最低 > 前日最高。向下跳空：當日最高 < 前日最低。
    回補：向上跳空後，後續任一日最低 <= 前日最高（跌回跳空前）；向下反向。
    """
    out = {}
    for name, df in (('SPY', load_etf('SPY')), ('QQQ', load_etf('QQQ')),
                     ('BTC', load_binance('BTCUSDT')), ('ETH', load_binance('ETHUSDT'))):
        h, l, c = df['high'], df['low'], df['close']
        prev_h, prev_l, prev_c = h.shift(1), l.shift(1), c.shift(1)

        up = l > prev_h
        dn = h < prev_l

        recs = []
        for direction, mask, target in (('up', up, prev_h), ('down', dn, prev_l)):
            for i in np.where(mask.fillna(False).values)[0]:
                if i + 1 >= len(df):
                    continue
                # 缺口大小：相對前日收盤
                gap = (df['open'].iloc[i] / prev_c.iloc[i] - 1)
                filled_days = None
                if direction == 'up':
                    future = l.iloc[i + 1:]
                    hit = future[future <= target.iloc[i]]
                else:
                    future = h.iloc[i + 1:]
                    hit = future[future >= target.iloc[i]]
                if len(hit):
                    filled_days = int((hit.index[0] - df.index[i]).days)
                recs.append({'dir': direction, 'date': df.index[i].strftime('%Y-%m-%d'),
                             'gap': float(gap), 'filled': filled_days is not None,
                             'days': filled_days})

        d = pd.DataFrame(recs)
        if d.empty:
            out[name] = {'n': 0}
            continue
        filled = d[d['filled']]
        stat = {
            'n': int(len(d)),
            'n_up': int((d['dir'] == 'up').sum()),
            'n_down': int((d['dir'] == 'down').sum()),
            'fill_rate': float(d['filled'].mean()),
            'fill_rate_up': float(d[d['dir'] == 'up']['filled'].mean()),
            'fill_rate_down': float(d[d['dir'] == 'down']['filled'].mean()),
            'median_days': float(filled['days'].median()) if len(filled) else None,
            'mean_days': float(filled['days'].mean()) if len(filled) else None,
            'pct_within_10d': float((filled['days'] <= 10).mean()) if len(filled) else None,
        }
        # 依缺口幅度分組（影片問：跳空越大越難回補嗎）
        d['absgap'] = d['gap'].abs()
        bands = [(0, 0.005), (0.005, 0.01), (0.01, 0.05), (0.05, 1.0)]
        by_gap = []
        for lo, hi in bands:
            sub = d[(d['absgap'] >= lo) & (d['absgap'] < hi)]
            if len(sub) >= 5:
                by_gap.append({'lo': lo, 'hi': hi, 'n': int(len(sub)),
                               'fill_rate': float(sub['filled'].mean())})
        stat['by_gap'] = by_gap
        stat['years'] = (df.index[-1] - df.index[0]).days / 365.25
        out[name] = stat
    return out



# ------------------------------------------------------- 4. KDJ 金叉死叉
def kdj(df, n=9, m1=3, m2=3):
    """影片標準公式：RSV=(收-9日低)/(9日高-9日低)*100
       K = 2/3*前K + 1/3*RSV, D = 2/3*前D + 1/3*K, J = 3K-2D
    """
    low_n = df['low'].rolling(n, min_periods=n).min()
    high_n = df['high'].rolling(n, min_periods=n).max()
    rng = (high_n - low_n).replace(0, np.nan)
    rsv = (df['close'] - low_n) / rng * 100
    k = rsv.ewm(alpha=1 / m1, adjust=False).mean()
    d = k.ewm(alpha=1 / m2, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


def run_kdj():
    """金叉：K 上穿 D。死叉：K 下穿 D。
    進階版（影片第二個測試）：只有 D<20 的金叉才算買、D>80 的死叉才算賣。
    輸出看訊號後 1/5/10/20 日的方向命中率與平均報酬，以及機械策略績效。
    """
    out = {}
    mk = {
        '納斯達克綜合指數': load_fred('NASDAQCOM'),
        'SPY': load_etf('SPY'),
        'QQQ': load_etf('QQQ'),
        'BTC': load_binance('BTCUSDT'),
    }
    skipped = []
    for name, df in mk.items():
        # KDJ 需要最高/最低價；FRED 指數只有收盤價，硬用收盤價代替高低價
        # 會讓 RSV 變成另一條指標，因此這裡直接跳過並在報告中註明。
        if not {'high', 'low'}.issubset(df.columns):
            skipped.append(name)
            continue
        k, d, j = kdj(df)
        close = df['close']
        cross_up = (k > d) & (k.shift(1) <= d.shift(1))
        cross_dn = (k < d) & (k.shift(1) >= d.shift(1))
        res = {'n_gold': int(cross_up.fillna(False).sum()),
               'n_dead': int(cross_dn.fillna(False).sum())}

        def stat(mask, label):
            rec = {}
            for h in (1, 5, 10, 20):
                fwd = close.shift(-h) / close - 1
                v = fwd[mask.fillna(False)].dropna()
                if len(v) < 5:
                    continue
                if label == 'gold':
                    rec[f'h{h}_hit'] = float((v > 0).mean())
                    rec[f'h{h}_mean'] = float(v.mean())
                else:
                    rec[f'h{h}_hit'] = float((v < 0).mean())
                    rec[f'h{h}_mean'] = float(-v.mean())   # 正數＝符合下跌預期
                rec[f'h{h}_t'] = tstat(v if label == 'gold' else -v)
            rec['n'] = int(mask.fillna(False).sum())
            return rec

        base = base_stats(close, horizons=(1, 5, 10, 20))
        res['base'] = base
        res['gold'] = stat(cross_up, 'gold')
        res['dead'] = stat(cross_dn, 'dead')
        # 勝率必須對照「無條件漲率」判讀：SPY 20 日無條件上漲率本身就有 69%，
        # 金叉後 68.7% 看起來很高，其實等於沒有優勢。
        for key, dirn in (('gold', 1), ('dead', -1)):
            blk = res[key]
            for h in (1, 5, 10, 20):
                bm = base.get(h, {}).get('mean')
                if bm is not None and blk.get(f'h{h}_mean') is not None:
                    blk[f'h{h}_excess'] = blk[f'h{h}_mean'] - dirn * bm
                bh = base.get(h, {}).get('hit_up' if dirn == 1 else 'hit_dn')
                if bh is not None and blk.get(f'h{h}_hit') is not None:
                    blk[f'h{h}_hit_excess'] = blk[f'h{h}_hit'] - bh
        # 進階：低位金叉 / 高位死叉
        res['gold_low'] = stat(cross_up & (d < 20), 'gold')
        res['dead_high'] = stat(cross_dn & (d > 80), 'dead')

        # 機械策略：金叉買、死叉賣，隔日生效
        pos = pd.Series(np.nan, index=close.index)
        pos[cross_up.fillna(False)] = 1.0
        pos[cross_dn.fillna(False)] = 0.0
        pos = pos.ffill().fillna(0.0)
        held = pos.shift(1).fillna(0.0)
        ret = close.pct_change().fillna(0.0)
        cost = held.diff().abs().fillna(0.0) * 0.0001
        strat = (1 + held * ret - cost).cumprod()
        bh = (1 + ret).cumprod()
        yrs = (close.index[-1] - close.index[0]).days / 365.25
        ppy = 365 if name == 'BTC' else 252

        def perf(eq):
            r = eq.pct_change().dropna()
            return dict(cagr=float(eq.iloc[-1] ** (1 / yrs) - 1),
                        mdd=float((eq.values / np.maximum.accumulate(eq.values) - 1).min()),
                        sharpe=float(r.mean() / r.std() * np.sqrt(ppy)) if r.std() > 0 else 0.0)

        e = float(held.mean())
        res['strategy'] = perf(strat)
        res['bh'] = perf(bh)
        res['exposure'] = e
        res['em'] = perf((1 + e * ret).cumprod())
        res['years'] = yrs
        res['start'] = close.index[0].strftime('%Y-%m-%d')
        res['end'] = close.index[-1].strftime('%Y-%m-%d')
        out[name] = res
    if skipped:
        out['_skipped'] = f"無最高/最低價，未測：{', '.join(skipped)}（KDJ 需要 H/L，僅有收盤價的指數無法誠實計算）"
    return out



# --------------------------------------------------------- 5. 月份效應
def run_month_effect():
    """影片說法：A 股「五窮六絕七翻身」；他實測 2005-2022 後改口為
    「二月最好、六月最差」，並用「二月持有、其他月份空手」驗證。

    這裡用同一口徑測美股與加密貨幣：每月平均報酬、月勝率、以及
    「只持有某月」的機械策略。判讀重點是同月對全樣本平均的超額。
    """
    out = {}
    mk = {
        '納斯達克綜合指數（55 年）': load_fred('NASDAQCOM'),
        'SPY': load_etf('SPY'),
        'QQQ': load_etf('QQQ'),
        'BTC': load_binance('BTCUSDT'),
    }
    for name, df in mk.items():
        close = df['close']
        # 月底收盤 -> 月報酬
        m = close.resample('ME').last()
        mret = m.pct_change().dropna()
        if len(mret) < 24:
            out[name] = {'n_months': int(len(mret))}
            continue
        tbl = {}
        for mo in range(1, 13):
            sub = mret[mret.index.month == mo]
            if len(sub) == 0:
                continue
            tbl[mo] = dict(
                n=int(len(sub)),
                mean=float(sub.mean()),
                hit=float((sub > 0).mean()),
                t=tstat(sub),
            )
        allmean = float(mret.mean())
        # 「只持有某月」的年度化淨值
        only = {}
        for mo in (2, 5, 6, 7):
            sel = mret[mret.index.month == mo]
            if len(sel) >= 5:
                eq = (1 + sel).prod()
                yrs = len(sel) / 12.0
                only[mo] = dict(n=int(len(sel)), total=float(eq - 1),
                                cagr=float(eq ** (1 / yrs) - 1) if yrs > 0 and eq > 0 else None,
                                mean=float(sel.mean()),
                                excess=float(sel.mean() - allmean),
                                t=tstat(sel))
        out[name] = dict(n_months=int(len(mret)), all_mean=allmean,
                         all_hit=float((mret > 0).mean()),
                         months=tbl, only=only,
                         start=mret.index[0].strftime('%Y-%m'), end=mret.index[-1].strftime('%Y-%m'))
    return out



# --------------------------------------------- 6. 幣圈交易日內分佈
def load_hourly(sym):
    """1 小時 K 線（Binance 月度封存）。回傳 UTC 索引的 OHLCV。"""
    frames = []
    # 1 小時封存放在上層 data/（與日線分開），xbx/data 只有日線與 FRED。
    pats = [os.path.join(DATA, f'{sym}-1h-*.zip'),
            os.path.join(os.path.dirname(HERE), 'data', f'{sym}-1h-*.zip')]
    files = sorted({f for pat in pats for f in glob.glob(pat)})
    for f in files:
        with zipfile.ZipFile(f) as z:
            raw = z.read(z.namelist()[0]).decode()
        d = pd.read_csv(io.StringIO(raw), header=None).iloc[:, :6]
        d.columns = ['ts', 'open', 'high', 'low', 'close', 'volume']
        frames.append(d)
    if not frames:
        return None
    d = pd.concat(frames, ignore_index=True)
    ts = pd.to_numeric(d['ts'], errors='coerce')
    ts = ts.where(ts < 1e14, ts / 1000.0)
    d['dt'] = pd.to_datetime(ts, unit='ms', utc=True)
    for c in ['open', 'high', 'low', 'close', 'volume']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    return (d.dropna(subset=['dt', 'close']).drop_duplicates('dt')
             .set_index('dt').sort_index())


def run_intraday_hours():
    """影片主張：幣安成交量集中在「美國時段」（北京 20:00–01:00），
    且早期（2018）較平均、近年越來越偏美國時段；北京 08:00 異常高（量化換倉）。

    這裡用 BTC 全部 1 小時資料逐年檢驗：每小時平均成交量佔比、以及每小時
    平均報酬。判讀重點是「佔比」而非絕對量，因為絕對量會被幣價與整體活躍度污染。
    """
    out = {}
    for sym, label in (('BTCUSDT', 'BTC'), ('ETHUSDT', 'ETH')):
        h = load_hourly(sym)
        if h is None or len(h) < 5000:
            out[label] = {'n': 0}
            continue
        h = h.copy()
        h['hour_utc'] = h.index.hour
        h['hour_bj'] = (h.index.hour + 8) % 24      # 北京時間
        h['year'] = h.index.year
        ret = h['close'].pct_change()

        def profile(df_idx, vol, r):
            tot = vol.sum()
            rows = []
            for hh in range(24):
                m = df_idx == hh
                v = vol[m]
                rr = r[m].dropna()
                rows.append(dict(hour=int(hh),
                                 vol_share=float(v.sum() / tot) if tot > 0 else None,
                                 n=int(m.sum()),
                                 ret_mean=float(rr.mean()) if len(rr) else None,
                                 ret_t=tstat(rr) if len(rr) > 10 else None))
            return rows

        allprof = profile(h['hour_bj'].values, h['volume'], ret)

        # 美國時段（北京 20:00–01:00，即 UTC 12:00–17:00）vs 中國時段（北京 08:00–17:00）
        bj = h['hour_bj']
        us_mask = bj.isin([20, 21, 22, 23, 0, 1])
        cn_mask = bj.isin([8, 9, 10, 11, 13, 14, 15, 16])   # 含 12:00 午休的近似
        us_share = float(h.loc[us_mask, 'volume'].sum() / h['volume'].sum())
        cn_share = float(h.loc[cn_mask, 'volume'].sum() / h['volume'].sum())
        # 時段數不同 -> 換算成「每小時平均佔比」才可比
        us_per_hour = us_share / 6
        cn_per_hour = cn_share / len([8, 9, 10, 11, 13, 14, 15, 16])
        ratio = us_per_hour / cn_per_hour if cn_per_hour > 0 else None

        yearly = {}
        for y in sorted(h['year'].unique()):
            sub = h[h['year'] == y]
            if len(sub) < 2000:
                continue
            tot = sub['volume'].sum()
            us = float(sub.loc[sub['hour_bj'].isin([20, 21, 22, 23, 0, 1]), 'volume'].sum() / tot)
            yearly[int(y)] = dict(n=int(len(sub)), us_share=us)

        # 北京 08:00 是否異常（影片的量化換倉猜測）
        bj8_share = float(h.loc[bj == 8, 'volume'].sum() / h['volume'].sum())
        uniform = 1 / 24

        out[label] = dict(
            n=int(len(h)),
            start=h.index[0].strftime('%Y-%m-%d'), end=h.index[-1].strftime('%Y-%m-%d'),
            profile=allprof,
            us_share=us_share, cn_share=cn_share,
            us_per_hour=us_per_hour, cn_per_hour=cn_per_hour,
            us_cn_ratio=ratio,
            bj8_share=bj8_share, uniform_share=uniform,
            yearly=yearly,
        )
    return out



# ------------------------------------------------- 7. 大小盤輪動
def run_size_rotation(lookbacks=(20, 10, 60)):
    """影片規則（逐字稿原文）：
       每天收盤算「大盤指數」與「小盤指數」過去 N 天漲跌幅，
       隔天持有漲得多的那一個。N 預設 20（他說 20 在 A 股很神奇）。
       改良版：若兩者過去 N 天都是負的，隔天改為空倉。

    A 股用滬深 300（大）與創業板（小）。美股對應：道瓊斯或 SPY 為大、
    IWM（羅素 2000 ETF）為小。他的 ETF 手續費 0.61‱、免印花稅；
    這裡統一用每趟 1 bp 以與其他卡片可比。
    """
    big = load_etf('SPY')['close']
    small = load_etf('IWM')['close']
    df = pd.DataFrame({'big': big, 'small': small}).dropna()
    out = {}
    for N in lookbacks:
        mom_big = df['big'] / df['big'].shift(N) - 1
        mom_small = df['small'] / df['small'].shift(N) - 1
        pick_big = mom_big >= mom_small

        for variant in ('plain', 'cash_when_both_down'):
            # 隔日持倉：訊號在 t 收盤產生，t+1 生效
            w_big = pick_big.astype(float)
            w_small = (~pick_big).astype(float)
            if variant == 'cash_when_both_down':
                both_dn = (mom_big < 0) & (mom_small < 0)
                w_big = w_big.where(~both_dn, 0.0)
                w_small = w_small.where(~both_dn, 0.0)
            w_big = w_big.shift(1).fillna(0.0)
            w_small = w_small.shift(1).fillna(0.0)

            rb = df['big'].pct_change().fillna(0.0)
            rs = df['small'].pct_change().fillna(0.0)
            turn = (w_big.diff().abs().fillna(0.0) + w_small.diff().abs().fillna(0.0))
            ret = w_big * rb + w_small * rs - turn * 0.0001
            eq = (1 + ret).cumprod()

            yrs = (df.index[-1] - df.index[0]).days / 365.25
            r = eq.pct_change().dropna()
            mdd = float((eq.values / np.maximum.accumulate(eq.values) - 1).min())
            key = f'N{N}_{variant}'
            out[key] = dict(
                cagr=float(eq.iloc[-1] ** (1 / yrs) - 1),
                mdd=mdd,
                sharpe=float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0,
                multiple=float(eq.iloc[-1]),
                exposure=float((w_big + w_small).mean()),
                switches=int((turn > 0).sum()),
                years=yrs,
            )

    # 兩邊買進持有的參照
    for lbl, ser in (('big_bh', df['big']), ('small_bh', df['small'])):
        eq = ser / ser.iloc[0]
        yrs = (df.index[-1] - df.index[0]).days / 365.25
        r = eq.pct_change().dropna()
        out[lbl] = dict(cagr=float(eq.iloc[-1] ** (1 / yrs) - 1),
                        mdd=float((eq.values / np.maximum.accumulate(eq.values) - 1).min()),
                        multiple=float(eq.iloc[-1]), years=yrs)
    out['_meta'] = dict(start=df.index[0].strftime('%Y-%m-%d'),
                        end=df.index[-1].strftime('%Y-%m-%d'),
                        note='美股對應：大盤=SPY、小盤=IWM（羅素 2000）')
    return out


# ------------------------------------------------------------- main
def main():
    print('=== 1. MACD 頂底背離 ===')
    md = run_macd_divergence()
    for k, v in md.items():
        print(f"  {k:20s} 頂背離 n={v['top']['n']:5d} r20={v['top'].get('r20_mean',0)*100:6.2f}% "
              f"t={v['top'].get('r20_t',0):5.2f} | 底背離 n={v['bottom']['n']:5d} "
              f"r20={v['bottom'].get('r20_mean',0)*100:6.2f}% t={v['bottom'].get('r20_t',0):5.2f}")
        print(f"   策略年化 {v['strategy']['cagr']*100:6.2f}% 回撤 {v['strategy']['mdd']*100:6.2f}% "
              f"| 買進持有 {v['bh']['cagr']*100:6.2f}% 曝險 {v['exposure']*100:4.1f}% "
              f"| 同曝險 {v['em']['cagr']*100:6.2f}%")

    print('=== 2. 量比 ===')
    vr = run_volume_ratio()
    for k in ('SPY', 'QQQ'):
        v = vr[k]
        print(f"  {k}: 買訊 {v['n_buy']} 賣訊 {v['n_sell']} | "
              f"買後1日 {v['buy_h1_mean']*100:5.2f}% hit {v['buy_h1_hit']*100:4.1f}% t={v['buy_h1_t']:5.2f}")
        print(f"   策略年化 {v['strategy']['cagr']*100:6.2f}% 回撤 {v['strategy']['mdd']*100:6.2f}% "
              f"曝險 {v['exposure']*100:4.1f}% | 買進持有 {v['bh']['cagr']*100:6.2f}% "
              f"| 同曝險 {v['em']['cagr']*100:6.2f}%")

    print('=== 3. 跳空回補 ===')
    gp = run_gaps()
    for k, v in gp.items():
        if not v.get('n'):
            print(f"  {k}: 無跳空（7x24 連續交易，日線幾乎必然重疊）")
            continue
        md_ = v.get('median_days')
        pw = v.get('pct_within_10d')
        print(f"  {k}: 跳空 {v['n']} 次 (上{v['n_up']}/下{v['n_down']}) "
              f"回補率 {v['fill_rate']*100:5.1f}% "
              f"中位 {(f'{md_:.0f}' if md_ is not None else 'n/a')} 天 "
              f"10日內 {(f'{pw*100:.1f}%' if pw is not None else 'n/a')}")

    print('=== 4. KDJ 金叉死叉 ===')
    kj = run_kdj()
    for k_, v in kj.items():
        if not isinstance(v, dict):
            print('  ' + str(v))
            continue
        g20 = v['gold'].get('h20_hit')
        g20m = v['gold'].get('h20_mean')
        d20 = v['dead'].get('h20_hit')
        print(f"  {k_:20s} 金叉 {v['n_gold']:5d} 死叉 {v['n_dead']:5d} | "
              f"金叉後20日漲率 {(f'{g20*100:.1f}%' if g20 else 'n/a')} "
              f"平均 {(f'{g20m*100:.2f}%' if g20m is not None else 'n/a')} | "
              f"死叉後20日跌率 {(f'{d20*100:.1f}%' if d20 else 'n/a')}")
        ex = v['gold'].get('h20_excess'); hex_ = v['gold'].get('h20_hit_excess')
        bup = v['base'].get(20, {}).get('hit_up')
        print(f"   金叉後20日：超額報酬 {ex*100 if ex is not None else float('nan'):6.2f}% | "
              f"漲率超額 {hex_*100 if hex_ is not None else float('nan'):6.2f}pp "
              f"（無條件漲率 {bup*100 if bup else float('nan'):5.1f}%）")
        print(f"   策略年化 {v['strategy']['cagr']*100:6.2f}% 回撤 {v['strategy']['mdd']*100:6.2f}% "
              f"曝險 {v['exposure']*100:4.1f}% | 買進持有 {v['bh']['cagr']*100:6.2f}% "
              f"| 同曝險 {v['em']['cagr']*100:6.2f}%")

    print('=== 5. 月份效應 ===')
    me = run_month_effect()
    for k_, v in me.items():
        if 'months' not in v:
            print(f"  {k_}: 樣本不足 ({v.get('n_months')} 個月)")
            continue
        print(f"  {k_}: {v['start']}..{v['end']} 共 {v['n_months']} 個月，全樣本月漲率 {v['all_hit']*100:.1f}%")
        for mo in (2, 5, 6, 7):
            mm = v['months'].get(mo); oo = v['only'].get(mo)
            if not mm:
                continue
            print(f"    {mo:2d}月 n={mm['n']:3d} 平均 {mm['mean']*100:6.2f}% 漲率 {mm['hit']*100:5.1f}% "
                  f"t={mm['t']:5.2f}" + (f" | 只持有此月 累積 {oo['total']*100:7.1f}% 超額 {oo['excess']*100:5.2f}%"
                                          if oo else ''))

    print('=== 6. 幣圈日內時段 ===')
    ih = run_intraday_hours()
    for k_, v in ih.items():
        if not v.get('n'):
            print(f"  {k_}: 資料不足")
            continue
        print(f"  {k_}: {v['start']}..{v['end']} {v['n']} 小時")
        print(f"    美國時段(北京20-01) 佔比 {v['us_share']*100:5.1f}% / 每小時均值 {v['us_per_hour']*100:4.2f}%")
        print(f"    中國時段(北京08-16) 佔比 {v['cn_share']*100:5.1f}% / 每小時均值 {v['cn_per_hour']*100:4.2f}%")
        print(f"    每小時密度比 美/中 = {v['us_cn_ratio']:.2f}x（均勻分布應為 1.00）")
        print(f"    北京08點佔比 {v['bj8_share']*100:4.2f}%（均勻為 {v['uniform_share']*100:4.2f}%）")
        ys = v['yearly']
        if ys:
            print('    逐年美國時段佔比: ' + ' '.join(f"{y}:{d['us_share']*100:.1f}%" for y, d in sorted(ys.items())))

    print('=== 7. 大小盤輪動 ===')
    rot = run_size_rotation()
    meta = rot.pop('_meta')
    print(f"  {meta['start']}..{meta['end']}  {meta['note']}")
    for k_, v in sorted(rot.items()):
        if 'cagr' not in v:
            continue
        print(f"  {k_:26s} 年化 {v['cagr']*100:6.2f}% 回撤 {v['mdd']*100:6.1f}% "
              f"曝險 {v.get('exposure',1)*100:5.1f}% 淨值 {v['multiple']:6.2f}x"
              + (f" 換手 {v['switches']} 次" if v.get('switches') else ''))

    json.dump({'macd': md, 'vr': vr, 'gap': gp, 'kdj': kj, 'month': me,
               'intraday': ih, 'rotation': rot},
              open(os.path.join(OUT, 'tier1.json'), 'w'), ensure_ascii=False, indent=1, default=str)
    print('written out/tier1.json')


if __name__ == '__main__':
    main()
