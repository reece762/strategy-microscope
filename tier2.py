#!/usr/bin/env python3
"""第二梯：馬克維茨月配置 + 比特幣均線定投。

規則全部取自兩支影片的逐字稿：

axLvMf4qBBk（馬克維茨）
  每月最後一個交易日，用過去 6 個月的歷史資料算出各資產的
  期望報酬、變異數、相關係數，代進模型算出權重，持有整個下個月。
  他用的五個標的：中證 1000、標普 500、恒生、原油、中債全債。
  美股對應：SPY（美股）、EEM（新興市場）、TLT（長債）、GLD（黃金）、
            DBC（大宗商品）、USO（原油）。

5RzvGxU_2Ak（比特幣定投）
  普通定投：每週四投固定金額。
  均線定投：偏離 MA(D) 的幅度每超過一個 X，定投率就加減一個 Y。
  cancel=True 時允許定投率為負（即賣出）。

兩支的共同陷阱：定投的「總市值÷總投入」是資金加權的比率，不是年化報酬；
在多頭期間會系統性偏高。這裡同時報告 IRR 與時間加權報酬，避免誤讀。
"""
import glob, io, json, os, zipfile
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data')
DAILY = os.path.join(os.path.dirname(HERE), 'data_daily')
OUT = os.path.join(HERE, 'out')
os.makedirs(OUT, exist_ok=True)


# ------------------------------------------------------------------ data
def load_etf(sym):
    df = pd.read_csv(os.path.join(DAILY, f'{sym}.csv'))
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
    for c in ['open', 'high', 'low', 'close', 'volume']:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df.set_index('date').sort_index()


def load_btc():
    frames = []
    for f in sorted(glob.glob(os.path.join(DATA, 'BTCUSDT-1d-*.zip'))):
        with zipfile.ZipFile(f) as z:
            raw = z.read(z.namelist()[0]).decode()
        d = pd.read_csv(io.StringIO(raw), header=None).iloc[:, :6]
        d.columns = ['ts', 'open', 'high', 'low', 'close', 'volume']
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    ts = pd.to_numeric(d['ts'], errors='coerce')
    ts = ts.where(ts < 1e14, ts / 1000.0)
    d['date'] = pd.to_datetime(ts, unit='ms').dt.normalize()
    for c in ['open', 'high', 'low', 'close', 'volume']:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d.dropna(subset=['close']).drop_duplicates('date')
    return d.set_index('date').sort_index()['close']


# ------------------------------------------------------- 1. 馬克維茨
def _max_sharpe(mu, cov, rf=0.0, iters=4000, seed=7):
    """long-only、權重和為 1、不允許放空。
    用隨機搜尋 + 局部精修，比 scipy 的 SLSQP 更不容易卡在邊界。
    維度只有 6，這樣做足夠且可重現（固定亂數種子）。"""
    n = len(mu)
    rng = np.random.default_rng(seed)
    best_w, best_s = None, -np.inf
    for _ in range(iters):
        w = rng.dirichlet(np.ones(n))
        r = w @ mu
        v = np.sqrt(w @ cov @ w)
        s = (r - rf) / v if v > 0 else -np.inf
        if s > best_s:
            best_s, best_w = s, w
    # 局部精修：在最佳點附近做小擾動
    w = best_w.copy()
    step = 0.05
    for _ in range(600):
        cand = w + rng.normal(0, step, n)
        cand = np.clip(cand, 0, None)
        ssum = cand.sum()
        if ssum <= 0:
            continue
        cand /= ssum
        r = cand @ mu
        v = np.sqrt(cand @ cov @ cand)
        s = (r - rf) / v if v > 0 else -np.inf
        if s > best_s:
            best_s, best_w = s, cand
        else:
            step *= 0.995
    return best_w, best_s


def _min_var(cov):
    n = len(cov)
    rng = np.random.default_rng(11)
    best_w, best_v = None, np.inf
    for _ in range(4000):
        w = rng.dirichlet(np.ones(n))
        v = w @ cov @ w
        if v < best_v:
            best_v, best_w = v, w
    return best_w


def run_markowitz(syms=('SPY', 'EEM', 'TLT', 'GLD', 'DBC', 'IWM'),
                  lookback_m=6, methods=('max_sharpe', 'min_var', 'equal')):
    px = pd.DataFrame({s: load_etf(s)['close'] for s in syms}).dropna()
    rets = px.pct_change().dropna()
    # 每月最後一個交易日為再平衡日
    month_end = rets.index.to_series().groupby([rets.index.year, rets.index.month]).max()

    out = {m: {} for m in methods}
    weight_log = {m: [] for m in methods}

    for m in methods:
        eq = [1.0]
        dates = [rets.index[0]]
        prev_w = np.repeat(1.0 / len(syms), len(syms))
        for idx, rday in enumerate(month_end.values):
            if idx == 0:
                continue
            hist = rets.loc[:rday].tail(lookback_m * 21)
            if len(hist) < lookback_m * 15:
                continue
            mu = hist.mean().values * 252
            cov = hist.cov().values * 252
            if m == 'max_sharpe':
                w, _ = _max_sharpe(mu, cov)
            elif m == 'min_var':
                w = _min_var(cov)
            else:
                w = np.repeat(1.0 / len(syms), len(syms))
            weight_log[m].append(dict(date=str(pd.Timestamp(rday).date()),
                                      w={s: round(float(x), 4) for s, x in zip(syms, w)}))
            # 持有到下一個再平衡日
            # 持有到「下一個」再平衡日，不是同一天
            nxt = month_end.values[idx + 1] if idx + 1 < len(month_end) else rets.index[-1]
            seg = rets.loc[pd.Timestamp(rday):pd.Timestamp(nxt)].iloc[1:]
            for _, row in seg.iterrows():
                r = float(np.dot(w, row.values))
                # 每月再平衡一次，換手成本以 1 bp 計（只算變動部分）
                turnover = float(np.abs(w - prev_w).sum())
                r -= turnover * 0.0001
                eq.append(eq[-1] * (1 + r))
                dates.append(row.name)
            prev_w = w

        e = pd.Series(eq, index=pd.DatetimeIndex(dates)).groupby(level=0).last()
        yrs = (e.index[-1] - e.index[0]).days / 365.25
        r = e.pct_change().dropna()
        out[m] = dict(
            cagr=float(e.iloc[-1] ** (1 / yrs) - 1),
            mdd=float((e.values / np.maximum.accumulate(e.values) - 1).min()),
            vol=float(r.std() * np.sqrt(252)),
            sharpe=float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0,
            multiple=float(e.iloc[-1]),
            years=yrs,
            start=str(e.index[0].date()), end=str(e.index[-1].date()),
        )

    # 等權作為基準，外加單一 SPY 買進持有
    spy = px['SPY'] / px['SPY'].iloc[0]
    yrs = (spy.index[-1] - spy.index[0]).days / 365.25
    out['spy_bh'] = dict(
        cagr=float(spy.iloc[-1] ** (1 / yrs) - 1),
        mdd=float((spy.values / np.maximum.accumulate(spy.values) - 1).min()),
        vol=float(spy.pct_change().std() * np.sqrt(252)),
        multiple=float(spy.iloc[-1]), years=yrs,
        start=str(spy.index[0].date()), end=str(spy.index[-1].date()))

    # 六十/四十股債，常見對照
    eq = 1.0
    r6040 = 0.6 * rets['SPY'] + 0.4 * rets['TLT']
    e6040 = (1 + r6040).cumprod()
    out['spy_tlt_60_40'] = dict(
        cagr=float(e6040.iloc[-1] ** (1 / yrs) - 1),
        mdd=float((e6040.values / np.maximum.accumulate(e6040.values) - 1).min()),
        vol=float(r6040.std() * np.sqrt(252)),
        multiple=float(e6040.iloc[-1]), years=yrs,
        start=str(e6040.index[0].date()), end=str(e6040.index[-1].date()))

    out['_meta'] = dict(symbols=list(syms), lookback_m=lookback_m,
                        n_rebalance=len(weight_log['max_sharpe']),
                        note='美股對應：他的中證1000→IWM、標普500→SPY、恒生→EEM、原油→USO/DBC、中債→TLT')
    return out, weight_log


# --------------------------------------------------------- 2. 定投
def _irr(cashflows, guess_lo=-0.9, guess_hi=1.0):
    """年化內部報酬率。cashflows = [(date, amount)]，投入為負、終值為正。

    有買有賣時現金流正負號會多次翻轉，此時 IRR 有多重解，bisection 會撿到
    數學上成立但經濟上無意義的根（實測曾得到 755%）。所以先數正負號翻轉
    次數，超過一次就回報 None，由上層改報「淨投入 + 期末市值 + 倍數」。
    """
    if not cashflows:
        return None
    signs = [1 if a > 0 else (-1 if a < 0 else 0) for _, a in cashflows]
    flips = sum(1 for i in range(1, len(signs)) if signs[i] and signs[i - 1]
                and signs[i] != signs[i - 1])
    if flips > 1:
        return None
    t0 = cashflows[0][0]
    yrs = np.array([(d - t0).days / 365.25 for d, _ in cashflows])
    amt = np.array([a for _, a in cashflows], dtype=float)

    def npv(rt):
        return np.sum(amt / (1 + rt) ** np.maximum(yrs, 1e-9))

    lo, hi = guess_lo, guess_hi
    f_lo, f_hi = npv(lo), npv(hi)
    if f_lo * f_hi > 0:
        return None
    for _ in range(200):
        mid = (lo + hi) / 2
        f_mid = npv(mid)
        if abs(f_mid) < 1e-6:
            return float(mid)
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return float((lo + hi) / 2)


def run_dca(ma_days_list=(250, 600), X_list=(0.10, 0.03), Y_list=(0.10, 0.45),
            base=1000.0, weekday=3, start='2017-12-21', allow_sell=False):
    """weekday: 0=週一 … 3=週四。定投率 = 1 ∓ k*Y，k = floor(|偏離|/X)。"""
    px = load_btc()
    px = px.loc[start:]
    if len(px) < 300:
        return {}, []

    res = {}
    curves = {}

    # --- 普通定投（基準）---
    def plain():
        eq_units = 0.0
        invested = 0.0
        rows = []
        for d, p in px.items():
            if d.weekday() == weekday:
                units = base / p
                eq_units += units
                invested += base
                rows.append((d, p, invested, eq_units * p))
        return rows

    rows = plain()
    if rows:
        val = rows[-1][3]
        inv = rows[-1][2]
        cf = [(d, -base) for d, _, _, _ in rows] + [(rows[-1][0], val)]
        _i = _irr(cf)
        res['plain_DCA'] = dict(
            invested=inv, value=val, profit=val - inv,
            multiple=val / inv if inv else None,
            irr=_i, irr_note=('' if _i is not None else '不適用'),
            n_buys=len(rows),
            coins=rows[-1][3] / rows[-1][1])
        curves['plain_DCA'] = rows

    # --- 均線定投 ---
    full = load_btc()
    for D in ma_days_list:
        ma = full.rolling(D).mean()
        for X in X_list:
            for Y in Y_list:
                if (D, X, Y) not in ((250, 0.10, 0.10), (600, 0.03, 0.45)):
                    continue
                for cancel in (False, True):
                    units = 0.0
                    invested = 0.0
                    rows2 = []
                    for d, p in px.items():
                        if d not in ma.index or pd.isna(ma.loc[d]):
                            continue
                        if d.weekday() != weekday:
                            continue
                        dev = p / ma.loc[d] - 1
                        # 他的表：偏離 0–10% → 定投率 0.9（降一級），所以級距向上取整。
                        # 原文「最多呢變成一加上六倍的 Y」→ 級數上限 6。
                        k = min(int(np.ceil(abs(dev) / X - 1e-12)), 6)
                        rate = 1 - k * Y if dev >= 0 else 1 + k * Y
                        if cancel:
                            rate = rate            # 允許負值＝賣出
                        else:
                            rate = max(rate, 0.0)  # 不允許賣出
                        amt = base * rate
                        if amt >= 0:
                            units += amt / p
                        else:
                            # 賣出：以當時幣價折回，但最多賣到剩餘為零
                            sell_units = min(units, -amt / p)
                            units -= sell_units
                            amt = -sell_units * p
                        invested += amt
                        rows2.append((d, p, invested, units * p, rate, dev))
                    if not rows2:
                        continue
                    val = rows2[-1][3]
                    inv = rows2[-1][2]
                    key = f'MA{D}_X{int(X*100)}_Y{Y}_{"sell" if cancel else "buyonly"}'
                    cf = [(rows2[0][0], -rows2[0][2])] + \
                         [(rows2[i][0], -(rows2[i][2] - rows2[i-1][2])) for i in range(1, len(rows2))]
                    cf.append((rows2[-1][0], val))
                    _irr_v = _irr(cf)
                    res[key] = dict(
                        invested=inv, value=val, profit=val - inv,
                        multiple=(val / inv if inv and inv > 1 else None),
                        irr=_irr_v,
                        irr_note=('買賣交錯，內部報酬率有多重解，不適用' if _irr_v is None else ''),
                        n_buys=sum(1 for r in rows2 if r[4] > 0),
                        n_sells=sum(1 for r in rows2 if r[4] < 0),
                        n_skip=sum(1 for r in rows2 if abs(r[4]) < 1e-9),
                        coins=rows2[-1][3] / rows2[-1][1],
                    )
                    curves[key] = rows2
    return res, curves



# ------------------------------------------------- 3. BTC/ETH 輪動
def run_btc_eth_rotation(lookbacks=(20, 10, 60)):
    """影片 YhEz95Vmm14 的規則：
       每天「北京時間 08:00」（= 格林威治 0 點）當收盤，算 BTC 與 ETH
       最近 20 天漲跌幅，隔天持有漲得多的那個。
       手續費雙邊 0.25%（他自己說「非常高」，實際更低）。
       改良版：兩者過去 N 天都跌就空倉。

       他的回測期間是 2017-01 至 2021-03，宣稱原始 765 倍、改良 1103 倍。
       我們手上資料從 2017-08 起，所以直接比倍數沒有意義——同時報告
       同期間 BTC 單獨持有與 ETH 單獨持有的倍數，才是可比的參照。
    """
    def load(sym):
        frames = []
        for f in sorted(glob.glob(os.path.join(DATA, f'{sym}-1d-*.zip'))):
            with zipfile.ZipFile(f) as z:
                raw = z.read(z.namelist()[0]).decode()
            d = pd.read_csv(io.StringIO(raw), header=None).iloc[:, :6]
            d.columns = ['ts', 'open', 'high', 'low', 'close', 'volume']
            frames.append(d)
        d = pd.concat(frames, ignore_index=True)
        ts = pd.to_numeric(d['ts'], errors='coerce')
        ts = ts.where(ts < 1e14, ts / 1000.0)
        d['date'] = pd.to_datetime(ts, unit='ms').dt.normalize()
        d['close'] = pd.to_numeric(d['close'], errors='coerce')
        d = d.dropna(subset=['close']).drop_duplicates('date')
        return d.set_index('date').sort_index()['close']

    btc, eth = load('BTCUSDT'), load('ETHUSDT')
    px = pd.DataFrame({'BTC': btc, 'ETH': eth}).dropna()
    rets = px.pct_change().fillna(0.0)
    out = {}
    cost = 0.0025           # 雙邊 0.25%，與他一致
    for N in lookbacks:
        mb = px['BTC'] / px['BTC'].shift(N) - 1
        ms = px['ETH'] / px['ETH'].shift(N) - 1
        pick_btc = mb >= ms
        for variant in ('plain', 'cash_when_both_down'):
            wb = pick_btc.astype(float)
            we = (~pick_btc).astype(float)
            if variant == 'cash_when_both_down':
                both_dn = (mb < 0) & (ms < 0)
                wb = wb.where(~both_dn, 0.0)
                we = we.where(~both_dn, 0.0)
            wb = wb.shift(1).fillna(0.0)
            we = we.shift(1).fillna(0.0)
            turn = wb.diff().abs().fillna(0.0) + we.diff().abs().fillna(0.0)
            r = wb * rets['BTC'] + we * rets['ETH'] - turn * cost
            eq = (1 + r).cumprod()
            yrs = (px.index[-1] - px.index[0]).days / 365.25
            dd = eq.values / np.maximum.accumulate(eq.values) - 1
            out[f'N{N}_{variant}'] = dict(
                multiple=float(eq.iloc[-1]),
                cagr=float(eq.iloc[-1] ** (1 / yrs) - 1),
                mdd=float(dd.min()),
                exposure=float((wb + we).mean()),
                trades=int((turn > 0).sum()),
                years=yrs,
                btc_days=int((wb > 0).sum()), eth_days=int((we > 0).sum()),
                cash_days=int(((wb + we) == 0).sum()),
            )
    for lbl, ser in (('btc_bh', px['BTC']), ('eth_bh', px['ETH'])):
        eq = ser / ser.iloc[0]
        yrs = (px.index[-1] - px.index[0]).days / 365.25
        out[lbl] = dict(multiple=float(eq.iloc[-1]),
                        cagr=float(eq.iloc[-1] ** (1 / yrs) - 1),
                        mdd=float((eq.values / np.maximum.accumulate(eq.values) - 1).min()),
                        years=yrs)
    out['_meta'] = dict(start=str(px.index[0].date()), end=str(px.index[-1].date()),
                        note='日線以 UTC 00:00 切分，即北京時間 08:00；成本雙邊各 0.25%')
    return out


def main():
    print('=== 1. 馬克維茨月度配置 ===')
    mk, wlog = run_markowitz()
    meta = mk.pop('_meta')
    print(f"  {meta['note']}")
    for k, v in mk.items():
        print(f"  {k:16s} 年化 {v['cagr']*100:6.2f}% 回撤 {v['mdd']*100:6.1f}% "
              f"波動 {v['vol']*100:5.1f}% 淨值 {v['multiple']:5.2f}x "
              f"{v['start']}..{v['end']}")

    print('\n=== 2. 比特幣定投 ===')
    dca, dc = run_dca()
    for k, v in dca.items():
        print(f"  {k:24s} 投入 {v['invested']:9,.0f} 市值 {v['value']:9,.0f} "
              f"倍數 {v['multiple'] if v['multiple'] else 0:5.2f}x "
              f"IRR {v['irr']*100 if v['irr'] is not None else float('nan'):6.2f}% "
              f"買{v.get('n_buys',0)} 賣{v.get('n_sells',0)} 停{v.get('n_skip',0)}")

    print('\n=== 3. BTC/ETH 輪動 ===')
    rot = run_btc_eth_rotation()
    rmeta = rot.pop('_meta')
    print(f"  {rmeta['start']}..{rmeta['end']}  {rmeta['note']}")
    for k, v in sorted(rot.items()):
        print(f"  {k:28s} 淨值 {v['multiple']:8.1f}x 年化 {v['cagr']*100:7.1f}% "
              f"回撤 {v['mdd']*100:6.1f}% 曝險 {v.get('exposure',1)*100:5.1f}%"
              + (f" 換手 {v['trades']} 次" if v.get('trades') else ''))

    json.dump({'markowitz': mk, 'weights': wlog, 'dca': dca,
               'rotation': rot, 'rotation_meta': rmeta},
              open(os.path.join(OUT, 'tier2.json'), 'w'), ensure_ascii=False, indent=1)
    print('\nwritten out/tier2.json')


if __name__ == '__main__':
    main()
