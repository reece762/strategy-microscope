#!/usr/bin/env python3
"""Rayner 布林通道回檔系統——用「影片原話」的規則重跑。

為什麼要重做
------------
我第一版（ray_more.py）是用他網站文章的「原則」自己操作化的：
  20 期布林、2 個標準差、跌破下軌就隔天開盤市價買、10 期 RSI > 40 出場。
那得到 +43.5% / 25 年，對比他宣稱 +2834%，差 65 倍。

但字幕抓到手之後，發現他影片裡講的參數跟我的假設不一樣。他的原話：
  「We're using a 20-day moving average and 2.5 standard deviation.」
  「We're going to place a 3% buy limit order.」
  「We will sell when the 2-day RSI crosses above 50 or after 10 trading days.」
  「20% of your capital and a maximum of five positions.」
  「rank our stocks according to their rate of change ... over the last 150 days」
  「exit at the next day open」

關鍵差異：他是「收盤跌破下軌後，隔天掛低 3% 的限價單」——
也就是要等股價再跌 3% 才買，買在恐慌處。我原本用市價買，進場價高得多。

所以這支程式照他原話重跑，並同時列出「只改一個參數」的對照，
讓「哪個細節造成差異」自己現形。
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from yahoo import fetch  # noqa: E402
from ray_rsi import rsi  # noqa: E402

STOCKS = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'JPM', 'JNJ', 'PG',
          'XOM', 'WMT', 'HD', 'KO', 'PEP', 'COST', 'MCD', 'DIS', 'INTC', 'CSCO',
          'VZ', 'T', 'PFE', 'MRK', 'ABT', 'CVX', 'BA', 'CAT', 'GE', 'F', 'GM',
          'ORCL', 'IBM', 'TXN', 'QCOM', 'AMD', 'MU', 'ADBE', 'CRM', 'NFLX', 'PYPL']


def tstat(x):
    x = pd.Series(x).dropna()
    if len(x) < 3 or x.std() == 0:
        return float('nan')
    return float(x.mean() / (x.std() / np.sqrt(len(x))))


# ---------------------------------------------------------------- 單一標的
def bb_single(df, ma_n=200, bb_n=20, k=2.5, limit_pct=0.03,
              rsi_n=2, rsi_exit=50.0, max_hold=10, cost_bp=1.0,
              use_limit=True):
    c = df['close'].to_numpy()
    o = df['open'].to_numpy()
    lo = df['low'].to_numpy()
    ma = df['close'].rolling(ma_n).mean().to_numpy()
    mid = df['close'].rolling(bb_n).mean()
    sd = df['close'].rolling(bb_n).std()
    lower = (mid - k * sd).to_numpy()
    r = rsi(df['close'], rsi_n).to_numpy()
    n = len(c)

    trades = []
    i = ma_n
    while i < n - 2:
        ok = (np.isfinite(ma[i]) and np.isfinite(lower[i])
              and c[i] > ma[i] and c[i] < lower[i])
        if not ok:
            i += 1
            continue

        if use_limit:
            lim = c[i] * (1 - limit_pct)
            if not (np.isfinite(lo[i + 1]) and lo[i + 1] <= lim):
                i += 1                      # 沒成交，訂單作廢
                continue
            ep = min(lim, o[i + 1]) if np.isfinite(o[i + 1]) else lim
        else:
            ep = o[i + 1]

        ei = i + 1
        xi = None
        for j in range(ei + 1, min(ei + max_hold + 1, n)):
            if np.isfinite(r[j]) and r[j] > rsi_exit:
                xi = j + 1 if j + 1 < n else n - 1
                break
        if xi is None:
            xi = min(ei + max_hold, n - 1)
        xp = o[xi] if np.isfinite(o[xi]) else c[xi]

        trades.append(dict(entry=df.index[ei], exit=df.index[xi],
                           ret=xp / ep - 1 - cost_bp / 1e4 * 2,
                           days=(df.index[xi] - df.index[ei]).days))
        i = xi
    return pd.DataFrame(trades)


def perfs(t, df, label, claimed=''):
    if t.empty:
        print(f'{label}: 無交易')
        return None
    w = t['ret'] > 0
    yrs = (df.index[-1] - df.index[0]).days / 365.25
    held = t['days'].sum()
    expo = held / ((df.index[-1] - df.index[0]).days)
    # 逐筆複利淨值
    eq = float((1 + t['ret']).prod())
    cagr = eq ** (1 / yrs) - 1
    ret = df['close'].pct_change().fillna(0)
    em = float((1 + ret * expo).prod()) ** (1 / yrs) - 1
    win = float(w.mean())
    avg = float(t['ret'].mean())
    aw = float(t['ret'][w].mean()) if w.any() else float('nan')
    al = float(t['ret'][~w].mean()) if (~w).any() else float('nan')
    print(f'\n{label}')
    print(f'  交易 {len(t):>4} 筆（每年 {len(t)/yrs:.1f}）  勝率 {win*100:5.2f}%  '
          f'平均 {avg*100:+.2f}%')
    print(f'  平均獲利 {aw*100:+.2f}%  平均虧損 {al*100:+.2f}%  '
          f'賺賠比 {abs(aw/al):.2f}')
    print(f'  總報酬（複利）{eq*100-100:>8.1f}%   年化 {cagr*100:6.2f}%   '
          f'曝險 {expo*100:.1f}%   同曝險基準 {em*100:.2f}%')
    print(f'  逐筆 t 值 {tstat(t["ret"]):.2f}')
    if claimed:
        print(f'  宣稱：{claimed}')
    return dict(n=len(t), win=win, avg=avg, total=eq - 1, cagr=cagr,
                expo=expo, em=em, t=tstat(t['ret']))


# ---------------------------------------------------------------- 多股票
def bb_portfolio(data, ma_n=200, bb_n=20, k=2.5, limit_pct=0.03,
                 rsi_n=2, rsi_exit=50.0, max_hold=10, roc_n=150,
                 max_pos=5, weight=0.20, cost_bp=1.0):
    """照他原話：最多 5 檔、每檔 20%、按 150 日漲幅排序取最強。"""
    idx = pd.DatetimeIndex(sorted(set().union(*[set(d.index) for d in data.values()])))
    cols = list(data)
    C = pd.DataFrame({s: d['close'] for s, d in data.items()}).reindex(idx)
    O = pd.DataFrame({s: d['open'] for s, d in data.items()}).reindex(idx)
    L = pd.DataFrame({s: d['low'] for s, d in data.items()}).reindex(idx)

    MA = C.rolling(ma_n).mean()
    MID = C.rolling(bb_n).mean()
    SD = C.rolling(bb_n).std()
    LOWB = MID - k * SD
    ROC = C / C.shift(roc_n) - 1
    R = pd.DataFrame({s: rsi(C[s], rsi_n) for s in cols})

    n, m = C.shape
    state = np.zeros(m)          # 1=持倉中
    held = np.zeros(m)           # 已持有天數
    ent_i = np.zeros(m, dtype=int)
    ent_p = np.full(m, np.nan)
    pos = np.zeros((n, m))
    trades = []

    # 預先算 numpy
    Oa, La, Ca = O.to_numpy(), L.to_numpy(), C.to_numpy()
    MAa, LOWa, ROCa, Ra = MA.to_numpy(), LOWB.to_numpy(), ROC.to_numpy(), R.to_numpy()

    for i in range(ma_n + roc_n, n - 1):
        # ---- 出場（今日收盤判斷，明日開盤執行）----
        for j in range(m):
            if state[j] != 1:
                continue
            held[j] += 1
            if (np.isfinite(Ra[i, j]) and Ra[i, j] > rsi_exit) or held[j] >= max_hold:
                xi = i + 1
                xp = Oa[xi, j]
                if np.isfinite(xp) and np.isfinite(ent_p[j]):
                    trades.append(dict(sym=cols[j], entry=idx[ent_i[j]],
                                       exit=idx[xi], ret=xp / ent_p[j] - 1 - cost_bp / 1e4 * 2))
                state[j] = 0.0
                held[j] = 0

        # ---- 進場：收盤跌破下軌 → 明日掛 -3% 限價 ----
        cand = []
        for j in range(m):
            if state[j] == 1:
                continue
            c = Ca[i, j]
            if not np.isfinite(c):
                continue
            if np.isfinite(MAa[i, j]) and np.isfinite(LOWa[i, j]) \
                    and c > MAa[i, j] and c < LOWa[i, j]:
                lim = c * (1 - limit_pct)
                if np.isfinite(La[i + 1, j]) and La[i + 1, j] <= lim:
                    fill = min(lim, Oa[i + 1, j]) if np.isfinite(Oa[i + 1, j]) else lim
                    cand.append((j, ROCa[i, j] if np.isfinite(ROCa[i, j]) else -9, fill))

        cand.sort(key=lambda x: -x[1])
        slots = max_pos - int(state.sum())
        for j, _, fill in cand[:max(0, slots)]:
            state[j] = 1.0
            held[j] = 0
            ent_i[j] = i + 1
            ent_p[j] = fill

        pos[i + 1] = state * weight

    P = pd.DataFrame(pos, index=idx, columns=cols)
    rets = C.pct_change().fillna(0.0)
    h = P.shift(1).fillna(0.0)
    turn = h.diff().abs().sum(axis=1).fillna(0.0)
    pnl = (h * rets).sum(axis=1) - turn * cost_bp / 1e4
    eq = (1 + pnl).cumprod()

    warm = ma_n + roc_n
    eqs = eq.loc[idx[warm]:]
    expo = h.sum(axis=1).loc[idx[warm]:]
    bench = rets.mean(axis=1).loc[idx[warm]:]
    em = (1 + bench * expo.mean()).cumprod()
    bh = (1 + bench).cumprod()

    def _c(e):
        e = e.dropna()
        if len(e) < 2 or e.iloc[0] <= 0:
            return float('nan')
        yy = (e.index[-1] - e.index[0]).days / 365.25
        return float((e.iloc[-1] / e.iloc[0]) ** (1 / yy) - 1)

    return dict(eq=eqs, pnl=pnl.loc[idx[warm]:], trades=pd.DataFrame(trades),
                cagr=_c(eqs), em_cagr=_c(em), bh_cagr=_c(bh),
                expo=float(expo.mean()),
                mdd=float((eqs.to_numpy() / np.maximum.accumulate(eqs.to_numpy()) - 1).min()),
                years=(eqs.index[-1] - eqs.index[0]).days / 365.25)


def main():
    print('=' * 84)
    print('Rayner 布林通道回檔：用影片原話的規則')
    print('  宣稱：+2834% / 25 年')
    print('=' * 84)

    spy = fetch('SPY', '30y')
    print('\n--- 單一標的 SPY（25 年）---')
    base = bb_single(spy)
    perfs(base, spy, '★ 他原話的規則（20期/2.5SD、-3% 限價、RSI2>50 或 10 日出場）',
          '+2834% / 25 年')
    base.to_csv(os.path.join(HERE, 'out', 'bb_exact_spy.csv'), index=False)

    print('\n--- 只改一個參數，看哪個細節關鍵 ---')
    variants = [
        ('市價進場（不用 -3% 限價）', dict(use_limit=False)),
        ('布林改 2.0 標準差', dict(k=2.0)),
        ('用 RSI2>70 出場', dict(rsi_exit=70)),
        ('用 RSI10>40 出場', dict(rsi_n=10, rsi_exit=40)),
        ('限價改 -1%', dict(limit_pct=0.01)),
        ('限價改 -5%', dict(limit_pct=0.05)),
        ('最長持有 20 日', dict(max_hold=20)),
    ]
    rows = []
    for name, kw in variants:
        t = bb_single(spy, **kw)
        st = perfs(t, spy, name)
        if st:
            rows.append((name, st))

    print('\n' + '=' * 84)
    print('多股票版（40 支美股，最多 5 檔、每檔 20%、按 150 日漲幅排序）')
    print('=' * 84)
    data = {}
    for s in STOCKS:
        try:
            df = fetch(s, '30y')
            if len(df) > 700:
                data[s] = df
        except Exception:
            pass
    print(f'  可用 {len(data)} 支')
    r = bb_portfolio(data)
    T = r['trades']
    w = T['ret'] > 0
    print(f'\n  交易 {len(T)} 筆   勝率 {w.mean()*100:.2f}%   '
          f'平均 {T["ret"].mean()*100:+.2f}%')
    print(f'  平均獲利 {T["ret"][w].mean()*100:+.2f}%  '
          f'平均虧損 {T["ret"][~w].mean()*100:+.2f}%  '
          f'賺賠比 {abs(T["ret"][w].mean()/T["ret"][~w].mean()):.2f}')
    print(f'  年化 {r["cagr"]*100:.2f}%   總報酬 {r["eq"].iloc[-1]*100-100:.0f}%')
    print(f'  回撤 {r["mdd"]*100:.1f}%   平均曝險 {r["expo"]*100:.1f}%')
    print(f'  同曝險基準 {r["em_cagr"]*100:.2f}%   等權買進持有 {r["bh_cagr"]*100:.2f}%')
    print(f'  逐筆 t 值 {tstat(T["ret"]):.2f}')
    print(f'  宣稱：+2834% / 25 年')
    T.to_csv(os.path.join(HERE, 'out', 'bb_exact_multi.csv'), index=False)


if __name__ == '__main__':
    main()
