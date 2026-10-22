#!/usr/bin/env python3
"""「阿尔法量化价格行为」三張卡片的資料計算。

站上既有規矩：只持倉一部分時間的策略，必須跟「同曝險、不擇時」的基準比，
不能直接跟買進持有比——否則比較的其實只是「有持倉 vs 沒持倉」。
（見 robust.py 的 exposure_matched）
"""
import glob
import io
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, 'out')
DAILY = os.path.join(os.path.dirname(HERE), 'data_daily')
DATA = os.path.join(HERE, 'data')

from probe_alpha_pa import (STRATS, buyhold, ema, macd_series, rsi, sma,  # noqa
                            bollinger, votes_series)


def load_etf(sym):
    df = pd.read_csv(os.path.join(DAILY, f'{sym}.csv'))
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
    for c in ['open', 'high', 'low', 'close', 'volume']:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df.set_index('date').sort_index().dropna(subset=['close'])


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
    return (d.dropna(subset=['close']).drop_duplicates('date')
             .set_index('date').sort_index())


def perf(eq, per_year):
    eq = pd.Series(eq).dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = float(eq.iloc[-1] ** (1 / yrs) - 1) if eq.iloc[-1] > 0 else -1.0
    mdd = float((eq.values / np.maximum.accumulate(eq.values) - 1).min())
    r = eq.pct_change().dropna()
    sharpe = float(r.mean() / r.std() * np.sqrt(per_year)) if r.std() > 0 else 0.0
    return dict(cagr=cagr, mdd=mdd, sharpe=sharpe,
                calmar=float(cagr / abs(mdd)) if mdd else 0.0,
                multiple=float(eq.iloc[-1]))


def run(df, pos, cost_bp, per_year):
    """pos: 0/1 序列。回傳 策略、同曝險基準、買進持有 三者績效。"""
    r = df['close'].pct_change().fillna(0)
    pos = pd.Series(pos, index=df.index).astype(float).fillna(0)
    # 關鍵：訊號是用「當日收盤」算出來的，所以只能賺「隔日」的報酬。
    # 不 shift 就等於用今天的收盤決定部位、再賺今天的漲跌 —— 前視偏差。
    # （probe_alpha_pa.backtest 用 pos[:-1] 對 ret，效果相同；這裡漏了，導致
    #   唐奇安 SPY 從 -1.24% 變成 +27.91%、BTC 變成 +192%。）
    pos = pos.shift(1).fillna(0)

    turn_s = pos.diff().abs().fillna(pos.iloc[0])
    strat = (1 + pos * r - turn_s * cost_bp / 1e4).cumprod()

    L = float(pos.mean())
    lev = pd.Series(L, index=r.index)
    turn_b = lev.diff().abs().fillna(0)
    em = (1 + lev * r - turn_b * cost_bp / 1e4).cumprod()

    bh = (1 + r).cumprod()

    # 策略日報酬 vs 同曝險日報酬的差，做 t 檢定
    rs = pos * r - turn_s * cost_bp / 1e4
    re = lev * r - turn_b * cost_bp / 1e4
    d = (rs - re).dropna()
    t = float(d.mean() / d.std() * np.sqrt(len(d))) if d.std() > 0 else 0.0

    return dict(
        exposure=L,
        **{f'strat_{k}': v for k, v in perf(strat, per_year).items()},
        **{f'em_{k}': v for k, v in perf(em, per_year).items()},
        **{f'bh_{k}': v for k, v in perf(bh, per_year).items()},
        excess_t=t,
        d_excess=float(perf(strat, per_year)['cagr'] - perf(em, per_year)['cagr']),
        equity_strat=strat, equity_em=em, equity_bh=bh,
    )


def main():
    sets = {
        'SPY': (load_etf('SPY'), 252, 5.0),
        'QQQ': (load_etf('QQQ'), 252, 5.0),
        'IWM': (load_etf('IWM'), 252, 5.0),
        'BTC': (load_btc(), 365, 25.0),
    }

    rows = {}
    print('=' * 84)
    print('六策略：策略 / 同曝險不擇時 / 買進持有')
    print('=' * 84)
    for sym, (df, py, cost) in sets.items():
        if sym == 'IWM':
            continue                    # IWM 只用在支撐阻力那張
        print(f'\n--- {sym}（成本 {cost:.0f}bp）---')
        print(f"  {'策略':22s} {'曝險':>6s} {'策略年化':>9s} {'同曝險':>9s} "
              f"{'買持':>8s} {'超額':>8s} {'t值':>6s}")
        rows[sym] = {}
        for name, fn in STRATS.items():
            pos = fn(df).values
            r = run(df, pos, cost, py)
            rows[sym][name] = {k: v for k, v in r.items() if not k.startswith('equity')}
            print(f"  {name:22s} {r['exposure']*100:5.1f}% "
                  f"{r['strat_cagr']*100:8.2f}% {r['em_cagr']*100:8.2f}% "
                  f"{r['bh_cagr']*100:7.2f}% {r['d_excess']*100:+7.2f}% "
                  f"{r['excess_t']:+6.2f}")
            if name == '唐奇安突破(20)':
                rows[sym]['_donchian_curve'] = dict(
                    strat=[round(float(x), 4) for x in r['equity_strat']],
                    em=[round(float(x), 4) for x in r['equity_em']],
                    bh=[round(float(x), 4) for x in r['equity_bh']],
                )

    # ---------------- 五票制 PA 引擎 ----------------
    print('\n' + '=' * 84)
    print('五票制 PA 引擎：原樣（bug）vs 修正')
    print('=' * 84)
    pa = {}
    for sym, (df, py, cost) in sets.items():
        if sym == 'IWM':
            continue
        pa[sym] = {}
        for lbl, fix in (('orig', False), ('fix', True)):
            pos = (votes_series(df, fix_trend_bar=fix) >= 3).astype(int).values
            r = run(df, pos, cost, py)
            s = votes_series(df, fix_trend_bar=fix).dropna()
            pa[sym][lbl] = {
                **{k: v for k, v in r.items() if not k.startswith('equity')},
                'pct_long': float((s >= 3).mean()),
                'pct_short': float((s <= -3).mean()),
                'pct_neutral': float((s.abs() < 3).mean()),
                's_max': int(s.max()), 's_min': int(s.min()),
                'entries': int(((pos == 1) & (np.concatenate([[0], pos[:-1]]) == 0)).sum()),
            }
            x = pa[sym][lbl]
            print(f"  {sym:4s} {lbl:4s} 看多{x['pct_long']*100:5.1f}% 看空{x['pct_short']*100:5.1f}% "
                  f"曝險{x['exposure']*100:5.1f}% 年化{x['strat_cagr']*100:6.2f}% "
                  f"同曝險{x['em_cagr']*100:6.2f}% 進場{x['entries']:4d} t={x['excess_t']:+.2f}")

    # 每年進場次數（原樣）——用來說「十年只進場 15 次」
    for sym, (df, py, cost) in sets.items():
        if sym == 'IWM':
            continue
        pos = (votes_series(df, fix_trend_bar=False) >= 3).astype(int).values
        ent = (pos == 1) & (np.concatenate([[0], pos[:-1]]) == 0)
        pa[sym]['orig']['entries_by_year'] = {
            str(y): int(ent[pd.Series(df.index).dt.year.values == y].sum())
            for y in sorted(set(df.index.year))
        }

    # ---------------- 支撐阻力：波動分位配對 ----------------
    print('\n' + '=' * 84)
    print('支撐／阻力守住率（控制波動率）')
    print('=' * 84)
    from probe_alpha_volctl import build as sr_build

    sr = {}
    for sym in ('SPY', 'QQQ', 'IWM'):
        df = load_etf(sym)
        ev, base = sr_build(df)
        dec = []
        for q in range(10):
            lo_q, hi_q = q / 10, (q + 1) / 10
            be = base[(base['volpct'] >= lo_q) & (base['volpct'] < hi_q)]
            ee = ev[(ev['volpct'] >= lo_q) & (ev['volpct'] < hi_q)]
            if ee.empty or be.empty:
                continue
            sup = ee[ee['kind'] == 'L']
            res = ee[ee['kind'] == 'H']
            dec.append(dict(
                q=f'{lo_q*100:.0f}-{hi_q*100:.0f}',
                n=int(len(ee)),
                sup_n=int(len(sup)), res_n=int(len(res)),
                sup_hold=float(1 - sup['broke'].mean()) if len(sup) else None,
                res_hold=float(1 - res['broke'].mean()) if len(res) else None,
                ctrl_hold_dn=float(1 - be['dn'].mean()),
                ctrl_hold_up=float(1 - be['up'].mean()),
            ))
        sup = ev[ev['kind'] == 'L']
        res = ev[ev['kind'] == 'H']
        sr[sym] = dict(
            n_events=int(len(ev)), n_sup=int(len(sup)), n_res=int(len(res)),
            sup_hold=float(1 - sup['broke'].mean()) if len(sup) else None,
            res_hold=float(1 - res['broke'].mean()) if len(res) else None,
            deciles=dec,
        )
        # 加權對照：用同波動分位的對照率，按事件數加權
        w_dn = [(d['ctrl_hold_dn'], d['sup_n']) for d in dec if d['sup_n']]
        w_up = [(d['ctrl_hold_up'], d['res_n']) for d in dec if d['res_n']]
        ctrl_dn = sum(a * b for a, b in w_dn) / sum(b for _, b in w_dn)
        ctrl_up = sum(a * b for a, b in w_up) / sum(b for _, b in w_up)
        sr[sym]['ctrl_dn'] = float(ctrl_dn)
        sr[sym]['ctrl_up'] = float(ctrl_up)
        sr[sym]['d_sup'] = float(sr[sym]['sup_hold'] - ctrl_dn)
        sr[sym]['d_res'] = float(sr[sym]['res_hold'] - ctrl_up)
        # 十個分位中，差額為正/為負的個數
        ds = [d['sup_hold'] - d['ctrl_hold_dn'] for d in dec
              if d['sup_hold'] is not None]
        dr = [d['res_hold'] - d['ctrl_hold_up'] for d in dec
              if d['res_hold'] is not None]
        sr[sym]['sup_pos'] = int(sum(1 for x in ds if x > 0))
        sr[sym]['res_pos'] = int(sum(1 for x in dr if x > 0))
        print(f"  {sym}: 支撐 {sr[sym]['sup_hold']*100:.1f}% vs 對照 {ctrl_dn*100:.1f}% "
              f"→ {sr[sym]['d_sup']*100:+.1f}pp（{sr[sym]['sup_pos']}/{len(ds)} 檔為正）；"
              f"壓力 {sr[sym]['res_hold']*100:.1f}% vs {ctrl_up*100:.1f}% "
              f"→ {sr[sym]['d_res']*100:+.1f}pp（{sr[sym]['res_pos']}/{len(dr)} 檔為正）")

    # 依「歷史觸及次數」分組（他的排序主因子）
    print('\n  依歷史觸及次數分組的守住率（他的排序主因子）:')
    touch = {}
    from probe_alpha_sr import evaluate as sr_eval
    for sym in ('SPY', 'QQQ', 'IWM'):
        df = load_etf(sym)
        r = sr_eval(df)
        if r.empty:
            continue
        r['q'] = pd.qcut(r['n_prior'], 3, labels=['少', '中', '多'], duplicates='drop')
        touch[sym] = {str(q): dict(n=int(len(g)),
                                    hold=float(1 - g['broke'].mean()))
                      for q, g in r.groupby('q', observed=True)}
        print(f"    {sym}: " + '  '.join(f"{k} {v['hold']*100:.1f}%({v['n']})"
                                         for k, v in touch[sym].items()))

    # ---------------- meta ----------------
    meta = {s: dict(start=str(sets[s][0].index[0].date()),
                    end=str(sets[s][0].index[-1].date()),
                    n=int(len(sets[s][0])), cost_bp=sets[s][2],
                    per_year=sets[s][1])
            for s in ('SPY', 'QQQ', 'BTC')}
    meta['IWM'] = dict(start=str(load_etf('IWM').index[0].date()),
                       end=str(load_etf('IWM').index[-1].date()),
                       n=int(len(load_etf('IWM'))), cost_bp=5.0, per_year=252)

    out = dict(meta=meta, six=rows, pa=pa, sr=sr, touch=touch)
    p = os.path.join(OUT, 'alpha.json')
    json.dump(out, open(p, 'w'), ensure_ascii=False, indent=1)
    print(f'\nwrote {p} ({os.path.getsize(p)} bytes)')


if __name__ == '__main__':
    main()
