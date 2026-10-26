#!/usr/bin/env python3
"""純動能版本（52 週高 + 均線多頭 + 收盤突破）再深挖。

為什麼要這樣做
--------------
上一輪發現：拿掉「弱勢市場」濾網後，這個純動能版本有年化 +3.92%、每筆 +0.08%，
t 值只有 0.99（很薄，但不顯著為負）。現在要問：
**加上什麼條件，能把它推到有用的水準？**

最重要的紀律：防過度優化
------------------------
如果在整段 2006–2025 上反覆調參數，一定能找到漂亮數字，但那多半是假象。
做法：
  前半段 2006-01 ~ 2015-12 = 篩選期（可以看，可以挑）
  後半段 2016-01 ~ 2025-12 = 檢驗期（只准看成績，不准回頭改參數）
先看篩選期，選出最好的設定。然後只跑一次檢驗期。檢驗期不好就是不好。

同時記錄我試了幾個變體（多重檢定的坦白揭露）。
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'data_opend')
OUT = os.path.join(HERE, 'out')
os.makedirs(OUT, exist_ok=True)

COST = 1e-4
MAX_POS = 5
WEIGHT = 0.20


def load():
    files = sorted(glob.glob(os.path.join(DATA, '*.csv')))
    out = {}
    for f in files:
        sym = os.path.basename(f)[:-4]
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        if 'time_key' not in df.columns:
            continue
        df['date'] = pd.to_datetime(df['time_key']).dt.normalize()
        df = df.drop_duplicates('date').set_index('date').sort_index()
        cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(c in df.columns for c in cols):
            continue
        d = df[cols].astype(float).dropna(subset=['open', 'high', 'low', 'close'])
        if len(d) < 800:
            continue
        out[sym] = d
    return out


def matrices(data):
    """把所有標的對齊到同一條日期軸，並預先算好指標。"""
    idx = None
    for d in data.values():
        idx = d.index if idx is None else idx.union(d.index)
    idx = idx.sort_values()
    syms = list(data)

    def M(col):
        return pd.DataFrame({s: data[s][col] for s in syms}).reindex(idx).to_numpy()

    C, O, L, H = M('close'), M('open'), M('low'), M('high')
    V = M('volume')

    def sma(a, n):
        return pd.DataFrame(a).rolling(n).mean().to_numpy()

    def roll_max(a, n):
        return pd.DataFrame(a).rolling(n).max().to_numpy()

    def roll_max_prev(a, n):
        """前 n 日最高，**不含當日**——關卡必須這樣算，
        否則「收盤突破」永遠不成立（收盤不可能高於當日最高）。"""
        return pd.DataFrame(a).shift(1).rolling(n).max().to_numpy()

    ma5, ma10, ma20 = sma(C, 5), sma(C, 10), sma(C, 20)
    ma50, ma200 = sma(C, 50), sma(C, 200)
    hi252 = roll_max(H, 252)
    res20 = roll_max_prev(H, 20)
    res60 = roll_max_prev(H, 60)
    res120 = roll_max_prev(H, 120)

    # ATR14
    tr = np.maximum(H - L, np.maximum(np.abs(H - np.roll(C, 1)), np.abs(L - np.roll(C, 1))))
    tr[0] = H[0] - L[0]
    atr = pd.DataFrame(tr).rolling(14).mean().to_numpy()

    vol20 = pd.DataFrame(V).rolling(20).mean().to_numpy()
    ret20 = np.full_like(C, np.nan)
    ret20[20:] = C[20:] / C[:-20] - 1

    return dict(idx=idx, syms=syms, C=C, O=O, L=L, H=H, V=V,
                ma5=ma5, ma10=ma10, ma20=ma20, ma50=ma50, ma200=ma200,
                hi252=hi252, res20=res20, res60=res60, res120=res120,
                atr=atr, vol20=vol20, ret20=ret20)


def engine(m, sig, *, max_pos=MAX_POS, weight=WEIGHT,
           stop_mode='daylow', atr_mult=1.5, exit_ma='ma20',
           rank_key='ret20', date_from=None, date_to=None):
    """sig: bool 矩陣（訊號日）。進場＝隔天開盤。"""
    idx = m['idx']
    C, O, L = m['C'], m['O'], m['L']
    EMA = m[exit_ma]
    RK = m[rank_key] if rank_key else None
    n, nsym = C.shape

    lo = 300 if date_from is None else max(300, int(idx.searchsorted(pd.Timestamp(date_from))))
    hi = n if date_to is None else int(idx.searchsorted(pd.Timestamp(date_to)))

    in_pos = np.zeros(nsym, dtype=bool)
    entry_p = np.full(nsym, np.nan)
    stop_p = np.full(nsym, np.nan)
    trades, pnl_rows = [], []

    for i in range(lo, min(hi, n - 1)):
        day_pnl = 0.0
        # ---- 先處理出場（用前一日收盤對均線，今日開盤執行） ----
        if in_pos.any():
            for j in np.flatnonzero(in_pos):
                px_prev = C[i - 1, j]
                if not np.isfinite(px_prev):
                    continue
                if L[i, j] <= stop_p[j]:
                    r = stop_p[j] / entry_p[j] - 1 - COST * 2
                    day_pnl += weight * (stop_p[j] / px_prev - 1)
                    trades.append(dict(sym=m['syms'][j], exit=idx[i], ret=r, why='停損'))
                    in_pos[j] = False
                    continue
                if np.isfinite(EMA[i - 1, j]) and C[i - 1, j] < EMA[i - 1, j]:
                    xp = O[i, j] if np.isfinite(O[i, j]) else px_prev
                    r = xp / entry_p[j] - 1 - COST * 2
                    day_pnl += weight * (xp / px_prev - 1)
                    trades.append(dict(sym=m['syms'][j], exit=idx[i], ret=r, why='跌破均線'))
                    in_pos[j] = False
                    continue
                day_pnl += weight * (C[i, j] / px_prev - 1) if np.isfinite(C[i, j]) else 0.0

        # ---- 再處理進場（昨日訊號 → 今日開盤） ----
        slots = max_pos - int(in_pos.sum())
        if slots > 0 and i - 1 >= 0:
            cand = np.flatnonzero(sig[i - 1] & ~in_pos)
            if len(cand):
                px = O[i, cand]
                ok = np.isfinite(px)
                cand, px = cand[ok], px[ok]
                if stop_mode == 'atr':
                    win = atr_mult * m['atr'][i - 1, cand]
                    stp = px - win
                    good = np.isfinite(win) & (win > 0) & (stp > 0) & (px > stp)
                else:
                    stp = L[i - 1, cand]
                    good = px > stp
                cand, px, stp = cand[good], px[good], stp[good]
                if len(cand):
                    if RK is not None:
                        order = np.argsort(-np.nan_to_num(RK[i - 1, cand], nan=-9e9))
                        cand, px, stp = cand[order], px[order], stp[order]
                    for j, p, s0 in zip(cand[:slots], px[:slots], stp[:slots]):
                        in_pos[j] = True
                        entry_p[j] = p
                        stop_p[j] = s0
                        # 進場當天的日內損益（開盤→收盤）
                        if np.isfinite(C[i, j]):
                            day_pnl += weight * (C[i, j] / p - 1)
        pnl_rows.append((idx[i], day_pnl))

    P = pd.DataFrame(pnl_rows, columns=['date', 'pnl']).set_index('date')
    eq = (1 + P['pnl']).cumprod()
    return dict(eq=eq, pnl=P['pnl'], trades=pd.DataFrame(trades))


def stats(r, label, bench=None):
    T = r['trades']
    eq = r['eq']
    if len(eq) < 30:
        print(f'  {label:<34} 資料不足')
        return None
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    mdd = float((eq / eq.cummax() - 1).min())
    out = dict(label=label, cagr=cagr, mdd=mdd, n=len(T), years=yrs,
               win=float('nan'), avg=float('nan'), t=float('nan'))
    if len(T) >= 5:
        w = T['ret'] > 0
        t = T['ret'].mean() / (T['ret'].std() / np.sqrt(len(T)))
        out.update(win=float(w.mean()), avg=float(T['ret'].mean()), t=float(t))
    bs = ''
    if bench is not None and len(bench) > 10:
        b = bench.reindex(eq.index).fillna(0)
        byrs = (b.index[-1] - b.index[0]).days / 365.25
        bc = (1 + b).cumprod().iloc[-1] ** (1 / byrs) - 1
        out['bench'] = float(bc)
        bs = f'  QQQ {bc*100:6.2f}%'
    print(f'  {label:<34} 年化 {cagr*100:6.2f}%  回撤 {mdd*100:6.0f}%  '
          f'{out["n"]:>4} 筆  勝率 {out["win"]*100:5.1f}%  '
          f'每筆 {out["avg"]*100:+.3f}%  t={out["t"]:5.2f}{bs}')
    return out


def main():
    data = load()
    m = matrices(data)
    idx = m['idx']
    qqq = data.get('QQQ')
    bench_all = qqq['close'].pct_change() if qqq is not None else None

    # 市場狀態（QQQ 是否在 200 日線之上）
    if qqq is not None:
        qc = qqq['close'].reindex(idx)
        q_above200 = (qc > qc.rolling(200).mean()).to_numpy()
    else:
        q_above200 = np.ones(len(idx), dtype=bool)

    base = ((m['C'] >= 0.95 * m['hi252'])
            & (m['ma10'] > m['ma20'])
            & (pd.DataFrame(m['ma10']).diff().to_numpy() > 0)
            & (pd.DataFrame(m['ma20']).diff().to_numpy() > 0)
            & (m['C'] > m['res60']))

    volf = m['V'] > 1.5 * m['vol20']
    above200 = np.repeat(q_above200[:, None], base.shape[1], axis=1)

    SPLIT = '2016-01-01'
    TRAIN = ('2006-01-01', '2015-12-31')
    TEST = ('2016-01-01', '2025-12-31')

    variants = [
        ('V0 基準（52週高+均線+突破）', base, {}),
        ('V1 +量能 1.5 倍均量', base & volf, {}),
        ('V2 +大盤在 200 日線上', base & above200, {}),
        ('V3 停利改 5 日線', base, dict(exit_ma='ma5')),
        ('V4 停利改 50 日線', base, dict(exit_ma='ma50')),
        ('V5 停損改 1.5×ATR', base, dict(stop_mode='atr', atr_mult=1.5)),
        ('V6 停損改 2.5×ATR', base, dict(stop_mode='atr', atr_mult=2.5)),
        ('V7 壓力改前 20 日高', (m['C'] >= 0.95 * m['hi252']) & (m['ma10'] > m['ma20'])
         & (m['C'] > m['res20']), {}),
        ('V8 10 檔 × 10% 權重', base, dict(max_pos=10, weight=0.10)),
        ('V9 依距離 52 週高排名', base, dict(rank_key='hi252')),
        ('V10 量能+大盤多頭（V1&V2）', base & volf & above200, {}),
    ]

    print('=' * 104)
    print('純動能版本深挖：52 週高 + 均線多頭 + 收盤突破')
    print('=' * 104)
    print(f'  標的 {len(m["syms"])} 支   {idx[300].date()} ~ {idx[-1].date()}')
    print(f'  篩選期 {TRAIN[0]} ~ {TRAIN[1]}    檢驗期 {TEST[0]} ~ {TEST[1]}')
    print(f'  （檢驗期的參數不准回頭改，這是防過度優化的唯一辦法）\n')

    print('---- 篩選期（可以挑）----')
    res_train = {}
    for lab, sig, kw in variants:
        r = engine(m, sig, date_from=TRAIN[0], date_to=TRAIN[1], **kw)
        res_train[lab] = stats(r, lab, bench_all)

    print('\n---- 檢驗期（只准看成績）----')
    res_test = {}
    for lab, sig, kw in variants:
        r = engine(m, sig, date_from=TEST[0], date_to=TEST[1], **kw)
        res_test[lab] = stats(r, lab, bench_all)

    # 依篩選期年化排序，看最好的那個在檢驗期如何
    print('\n' + '=' * 104)
    print('★ 過度優化檢查：篩選期最好的前三名，在檢驗期是否還行')
    print('=' * 104)
    ok = [(k, v) for k, v in res_train.items() if v and np.isfinite(v.get('cagr', np.nan))]
    ok.sort(key=lambda x: -x[1]['cagr'])
    for k, v in ok[:3]:
        vt = res_test.get(k) or {}
        tc = vt.get('cagr', float('nan'))
        drop = (tc - v['cagr']) * 100
        print(f'  {k:<34} 篩選 {v["cagr"]*100:6.2f}%  →  檢驗 {tc*100:6.2f}%   '
              f'（落差 {drop:+.2f}pp）')

    print('\n' + '=' * 104)
    print('★ 全期間（僅供參考，不作為結論依據）')
    print('=' * 104)
    best_lab = ok[0][0] if ok else variants[0][0]
    best_kw = dict(next(kw for lab, _, kw in variants if lab == best_lab))
    r_full = engine(m, dict((lab, sig) for lab, sig, _ in variants)[best_lab],
                    date_from='2006-01-01', date_to='2025-12-31', **best_kw)
    stats(r_full, f'{best_lab}（全期間）', bench_all)

    for lab, sig, kw in variants:
        if lab == best_lab:
            r = engine(m, sig, date_from='2006-01-01', date_to='2025-12-31', **kw)
            r['trades'].to_csv(os.path.join(OUT, 'smb_momentum_best_trades.csv'), index=False)
            r['eq'].to_frame('equity').to_csv(os.path.join(OUT, 'smb_momentum_best_equity.csv'))
            print(f'\n  逐筆交易 → out/smb_momentum_best_trades.csv（{len(r["trades"])} 筆）')
            break

    print(f'\n  本輪共試 {len(variants)} 個變體（多重檢定的坦白揭露：試越多，'
          f'越好運氣就越可能）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
