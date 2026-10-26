#!/usr/bin/env python3
"""Rayner 布林通道回檔系統——用富途 OpenD 抓的股票池，照影片原話的規則回測。

規則（全部出自他影片原話，非我的推測）
--------------------------------------
1. 股價在 200 日均線之上（順勢）
2. 收盤跌破布林下軌（20 期、2.5 標準差）＝ 回檔定義
3. 隔天掛 -3% 限價買單（收盤價 × 0.97），當日最低價觸及才成交
4. 出場：2 期 RSI 穿越 50 以上，或滿 10 個交易日
5. 出場在隔天開盤執行
6. 最多 5 檔、每檔 20% 資金（不融資）
7. 候選過多時按 150 日漲幅排序，取最強 5 檔

為什麼重做
----------
先前的版本用 40 支股票，平均曝險只有 1.7%，年化 1.13%。那不能算檢驗——
他明說在 Russell 10000（數千支）裡篩選，股票池太小就填不滿 5 個部位，
低曝險會自動壓低報酬。現在用 331 支、19 年資料重跑。

資料紀律
--------
只用 data_opend/（OpenD 的 qfq＝拆股＋股利都調整）。不與 data_yahoo/ 混接：
Yahoo 的 close 一般只調拆股，兩者在接縫處會造假跳空（實測中位 +18%）。
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

COST_BP = 1.0            # 單邊成本 1 個基點


def rsi(series, n):
    """Wilder 的 RSI。2 期就是他用的參數。"""
    d = series.diff()
    up = d.clip(lower=0)
    dn = (-d).clip(lower=0)
    au = up.ewm(alpha=1 / n, adjust=False).mean()
    ad = dn.ewm(alpha=1 / n, adjust=False).mean()
    rs = au / ad.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.fillna(50.0)


def load():
    """讀 data_opend/*.csv，回傳 {代碼: DataFrame(open/high/low/close)}。"""
    files = sorted(glob.glob(os.path.join(DATA, '*.csv')))
    data = {}
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
        need = ['open', 'high', 'low', 'close']
        if not all(c in df.columns for c in need):
            continue
        d = df[need].astype(float).dropna()
        if len(d) < 800:          # 少於約 3 年不納入
            continue
        data[sym] = d
    return data


# ---------------------------------------------------------------- 單一標的
def single(df, ma_n=200, bb_n=20, k=2.5, limit_pct=0.03,
           rsi_n=2, rsi_exit=50.0, max_hold=10, use_limit=True):
    c = df['close']
    o = df['open']
    lo = df['low']
    ma = c.rolling(ma_n).mean()
    mid = c.rolling(bb_n).mean()
    sd = c.rolling(bb_n).std()
    lower = mid - k * sd
    r = rsi(c, rsi_n)

    Ca, Oa, La = c.to_numpy(), o.to_numpy(), lo.to_numpy()
    MAa, LOa, Ra = ma.to_numpy(), lower.to_numpy(), r.to_numpy()
    n = len(Ca)

    trades = []
    i = ma_n
    while i < n - 2:
        if not (np.isfinite(MAa[i]) and np.isfinite(LOa[i])
                and Ca[i] > MAa[i] and Ca[i] < LOa[i]):
            i += 1
            continue

        if use_limit:
            lim = Ca[i] * (1 - limit_pct)
            if not (np.isfinite(La[i + 1]) and La[i + 1] <= lim):
                i += 1
                continue
            ep = min(lim, Oa[i + 1]) if np.isfinite(Oa[i + 1]) else lim
        else:
            ep = Oa[i + 1]

        ei = i + 1
        xi = None
        for j in range(ei + 1, min(ei + max_hold + 1, n)):
            if np.isfinite(Ra[j]) and Ra[j] > rsi_exit:
                xi = j + 1 if j + 1 < n else n - 1
                break
        if xi is None:
            xi = min(ei + max_hold, n - 1)
        xp = Oa[xi] if np.isfinite(Oa[xi]) else Ca[xi]

        trades.append(dict(entry=df.index[ei], exit=df.index[xi],
                           ret=xp / ep - 1 - COST_BP / 1e4 * 2,
                           days=xi - ei))
        i = xi
    return pd.DataFrame(trades)


def show_single(t, df, label):
    if t.empty:
        print(f'\n{label}: 無交易')
        return None
    w = t['ret'] > 0
    yrs = (df.index[-1] - df.index[0]).days / 365.25
    eq = float((1 + t['ret']).prod())
    cagr = eq ** (1 / yrs) - 1
    expo = t['days'].sum() / len(df)
    ret = df['close'].pct_change().fillna(0)
    em = float((1 + ret * expo).prod()) ** (1 / yrs) - 1
    aw = float(t['ret'][w].mean()) if w.any() else float('nan')
    al = float(t['ret'][~w].mean()) if (~w).any() else float('nan')
    tstat = float(t['ret'].mean() / (t['ret'].std() / np.sqrt(len(t)))) if len(t) > 2 else float('nan')
    print(f'\n{label}')
    print(f'  交易 {len(t):>4} 筆   勝率 {w.mean()*100:5.2f}%   平均 {t["ret"].mean()*100:+.2f}%')
    print(f'  平均獲利 {aw*100:+.2f}%  平均虧損 {al*100:+.2f}%  賺賠比 {abs(aw/al):.2f}')
    print(f'  總報酬 {eq*100-100:>8.1f}%   年化 {cagr*100:6.2f}%   曝險 {expo*100:5.1f}%')
    print(f'  同曝險基準 {em*100:5.2f}%   逐筆 t 值 {tstat:.2f}')
    return dict(n=len(t), win=w.mean(), total=eq - 1, cagr=cagr, expo=expo)


# ---------------------------------------------------------------- 多股票
def portfolio(data, ma_n=200, bb_n=20, k=2.5, limit_pct=0.03,
              rsi_n=2, rsi_exit=50.0, max_hold=10, roc_n=150,
              max_pos=5, weight=0.20):
    syms = sorted(data)
    idx = pd.DatetimeIndex(sorted(set().union(*[set(d.index) for d in data.values()])))
    C = pd.DataFrame({s: data[s]['close'] for s in syms}).reindex(idx)
    O = pd.DataFrame({s: data[s]['open'] for s in syms}).reindex(idx)
    L = pd.DataFrame({s: data[s]['low'] for s in syms}).reindex(idx)

    MA = C.rolling(ma_n).mean()
    MID = C.rolling(bb_n).mean()
    SD = C.rolling(bb_n).std()
    LOWB = MID - k * SD
    ROC = C / C.shift(roc_n) - 1
    R = pd.DataFrame({s: rsi(C[s], rsi_n) for s in syms})

    Ca, Oa, La = C.to_numpy(), O.to_numpy(), L.to_numpy()
    MAa, LOa, ROCa, Ra = MA.to_numpy(), LOWB.to_numpy(), ROC.to_numpy(), R.to_numpy()

    n, m = C.shape
    warm = max(ma_n, roc_n) + 2
    in_pos = np.zeros(m, dtype=bool)
    entry_i = np.full(m, -1, dtype=int)
    entry_p = np.full(m, np.nan)
    pos = np.zeros((n, m))
    trades = []

    for i in range(warm, n - 1):
        # ---- 出場：今日收盤判斷，明日開盤執行 ----
        if in_pos.any():
            held = i - entry_i
            hit = (Ra[i] > rsi_exit) | (held >= max_hold)
            for j in np.flatnonzero(in_pos & hit):
                xi = i + 1
                xp = Oa[xi, j]
                if np.isfinite(xp) and np.isfinite(entry_p[j]):
                    trades.append(dict(sym=syms[j], entry=idx[entry_i[j]],
                                       exit=idx[xi],
                                       ret=xp / entry_p[j] - 1 - COST_BP / 1e4 * 2,
                                       days=xi - entry_i[j]))
                in_pos[j] = False
                entry_i[j] = -1
                entry_p[j] = np.nan

        # ---- 進場：收盤跌破下軌，明日掛 -3% 限價 ----
        sig = (~in_pos) & np.isfinite(MAa[i]) & np.isfinite(LOa[i]) \
            & (Ca[i] > MAa[i]) & (Ca[i] < LOa[i])
        slots = max_pos - int(in_pos.sum())
        if slots > 0 and sig.any():
            lim = Ca[i] * (1 - limit_pct)
            filled = np.isfinite(La[i + 1]) & (La[i + 1] <= lim)
            cand = np.flatnonzero(sig & filled)
            if len(cand):
                order = cand[np.argsort(-np.nan_to_num(ROCa[i, cand], nan=-9.0))]
                for j in order[:slots]:
                    px = min(lim[j], Oa[i + 1, j]) if np.isfinite(Oa[i + 1, j]) else lim[j]
                    in_pos[j] = True
                    entry_i[j] = i + 1
                    entry_p[j] = px

        pos[i + 1] = in_pos.astype(float) * weight

    P = pd.DataFrame(pos, index=idx, columns=syms)
    rets = C.pct_change().fillna(0.0)
    h = P.shift(1).fillna(0.0)
    turn = h.diff().abs().sum(axis=1).fillna(0.0)
    pnl = (h * rets).sum(axis=1) - turn * COST_BP / 1e4
    eq = (1 + pnl).cumprod()

    sl = idx[warm:]
    eqs = eq.loc[sl]
    expo_s = h.sum(axis=1).loc[sl]
    bench = rets.mean(axis=1).loc[sl]

    def cagr_of(e):
        e = e.dropna()
        if len(e) < 2 or e.iloc[0] <= 0:
            return float('nan')
        y = (e.index[-1] - e.index[0]).days / 365.25
        return float((e.iloc[-1] / e.iloc[0]) ** (1 / y) - 1)

    eqmax = np.maximum.accumulate(eqs.to_numpy())
    return dict(eq=eqs, pnl=pnl.loc[sl], trades=pd.DataFrame(trades),
                cagr=cagr_of(eqs),
                em_cagr=cagr_of((1 + bench * expo_s.mean()).cumprod()),
                bh_cagr=cagr_of((1 + bench).cumprod()),
                expo=float(expo_s.mean()),
                mdd=float((eqs.to_numpy() / eqmax - 1).min()),
                years=(eqs.index[-1] - eqs.index[0]).days / 365.25,
                n_trades=len(trades), n_syms=len(syms))


def report(r, label):
    T = r['trades']
    print(f'\n{label}')
    print(f'  股票數 {r["n_syms"]}   期間 {r["years"]:.1f} 年')
    if T.empty:
        print('  無交易')
        return
    w = T['ret'] > 0
    aw = T['ret'][w].mean() if w.any() else np.nan
    al = T['ret'][~w].mean() if (~w).any() else np.nan
    tstat = T['ret'].mean() / (T['ret'].std() / np.sqrt(len(T)))
    print(f'  交易 {len(T):>5} 筆（每年 {len(T)/r["years"]:.0f}）  勝率 {w.mean()*100:5.2f}%  平均 {T["ret"].mean()*100:+.2f}%')
    print(f'  平均獲利 {aw*100:+.2f}%   平均虧損 {al*100:+.2f}%   賺賠比 {abs(aw/al):.2f}')
    print(f'  總報酬 {r["eq"].iloc[-1]*100-100:>10.0f}%   年化 {r["cagr"]*100:6.2f}%')
    print(f'  最大回撤 {r["mdd"]*100:6.1f}%   平均曝險 {r["expo"]*100:5.1f}%')
    print(f'  同曝險基準 {r["em_cagr"]*100:6.2f}%   等權買進持有 {r["bh_cagr"]*100:6.2f}%')
    print(f'  逐筆 t 值 {tstat:.2f}')


def main():
    data = load()
    print('=' * 80)
    print('Rayner 布林通道回檔系統（影片原話規則）— OpenD 資料')
    print('=' * 80)
    print(f'  載入 {len(data)} 支，資料起點中位數 '
          f'{pd.Series([d.index[0] for d in data.values()]).median().date()}')

    # --- 單一標的對照 ---
    for b in ('SPY', 'QQQ'):
        if b in data:
            show_single(single(data[b]), data[b], f'{b}（單一標的）')

    # --- 主測試：照他原話 ---
    print('\n' + '=' * 80)
    print('主測試：照影片原話（20期/2.5SD、-3% 限價、RSI2>50 或 10 日、5 檔各 20%）')
    print('=' * 80)
    base = portfolio(data)
    report(base, '★ 原話規則')
    print(f'  宣稱：+2834% / 25 年')
    base['trades'].to_csv(os.path.join(OUT, 'opend_bb_trades.csv'), index=False)
    base['eq'].to_frame('equity').to_csv(os.path.join(OUT, 'opend_bb_equity.csv'))

    # --- 只改一個細節，看哪個關鍵 ---
    print('\n' + '=' * 80)
    print('敏感度：每次只改一個細節')
    print('=' * 80)
    variants = [
        ('不用限價（隔天開盤市價買）', dict(limit_pct=0.0)),
        ('限價改 -1%', dict(limit_pct=0.01)),
        ('限價改 -5%', dict(limit_pct=0.05)),
        ('布林改 2.0 標準差', dict(k=2.0)),
        ('出場改 RSI2>70', dict(rsi_exit=70.0)),
        ('最長持有改 20 日', dict(max_hold=20)),
    ]
    rows = []
    for name, kw in variants:
        try:
            v = portfolio(data, **kw)
            report(v, name)
            rows.append((name, v['cagr'], v['expo'], v['n_trades']))
        except Exception as e:
            print(f'\n{name}: 失敗 {type(e).__name__} {e}')

    if rows:
        print('\n' + '=' * 80)
        print('彙總（年化 / 曝險 / 筆數）')
        print('=' * 80)
        for name, cg, ex, nt in [('★ 原話規則', base['cagr'], base['expo'], base['n_trades'])] + rows:
            print(f'  {name:<30} 年化 {cg*100:6.2f}%   曝險 {ex*100:5.1f}%   {nt:>5} 筆')


if __name__ == '__main__':
    sys.exit(main())
