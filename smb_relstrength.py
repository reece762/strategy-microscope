#!/usr/bin/env python3
"""SMB「相對強度突破」策略——照影片原話量化回測。

他的核心主張（反直覺）
--------------------
大家等紅盤日才看得出誰強。但那時所有人都知道了，行情已經走完。
**真正的強勢要在市場下跌時才看得見**——大盤跌 2~3%、收在低點、
連續幾天反彈都被賣掉，那才是「大資金在吃貨」留下的足跡。

規則（全部出自 L7F9yIf8CeE 原話）
--------------------------------
步驟一：需要弱勢市場（大盤跌 2~3%、收低，或連續數日反彈被賣）
步驟二：找不會跌的。三個條件都要成立
   (a) 拒絕跟跌：大盤大跌它持平／小跌／最好收紅，而且**不只一天**
   (b) 位於 52 週或歷史新高附近
   (c) 10 日均線在 20 日均線之上，且兩條都向上
步驟三：排名，不是買進（相對強度只是警示）
觸發：**收盤**突破清楚的高一階壓力關卡（不是盤中戳破）
停損：突破當天的低點
出場：跌破正在跟隨的 5 日或 10 日均線；上漲途中遇跳空／極端延伸就分批獲利
額外濾網：財報日曆（財報落在持有期內就放棄）

量化對應（他的話是主觀的，這裡逐條寫明我怎麼翻譯）
------------------------------------------------
「大盤弱」        → QQQ W 日報酬 <= -MKT_WEAK
「拒絕跟跌」      → 同期間 個股報酬 − QQQ 報酬 >= REL_OUT，
                    且個股報酬 >= STOCK_FLOOR（沒有跟著崩）
「52 週高附近」    → 收盤 >= 0.95 × 過去 252 日最高
「均線多頭」      → SMA10 > SMA20，且兩條斜率皆 > 0
「清楚的壓力關卡」 → 前 N 日最高（不含當日），N 預設 60
「收盤突破」      → 當日收盤 > 該關卡

未能量化的部分（誠實揭露）
------------------------
- 「清楚」的關卡是人工判斷，我用前 N 日高點代替
- 「獲利了結到極端與跳空」需要盤感，我只做均線移動停利
- **財報日曆我沒有資料**，所以這條濾網完全沒套用——他的 DataDog 例子證明
  這條很重要（省掉一次把漲幅全吐回去的災難），我的結果因此偏樂觀
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

COST = 1e-4                    # 單邊 1bp

# 參數
MKT_WEAK = 0.015               # 大盤 W 日跌 1.5% 以上算弱
WIN = 10                       # 觀察窗（10 個交易日 ≈ 兩週）
REL_OUT = 0.04                 # 個股要比大盤強 4 個百分點
STOCK_FLOOR = -0.02            # 個股同期間不能跌超過 2%（沒跟著崩）
NEAR_HIGH = 0.95               # 距 52 週高 5% 以內
RES_N = 60                     # 壓力關卡＝前 60 日最高
EXIT_MA = 10                   # 移動停利用 10 日線（他多數例子跟 5 或 10）
MAX_POS = 5
WEIGHT = 0.20


def load():
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
        if len(d) < 800:
            continue
        data[sym] = d
    return data


def build(px, mkt):
    """把單一標的算成特徵表，並標出訊號日。"""
    c, o, h, l = px['close'], px['open'], px['high'], px['low']
    ma10 = c.rolling(10).mean()
    ma20 = c.rolling(20).mean()
    hi252 = h.rolling(252).max()
    res = h.shift(1).rolling(RES_N).max()      # 前 N 日最高（不含當日）
    mkt_w = mkt['close'].pct_change(WIN)       # 大盤 W 日報酬
    stk_w = c.pct_change(WIN)

    trend = (ma10 > ma20) & (ma10.diff() > 0) & (ma20.diff() > 0)
    near = c >= NEAR_HIGH * hi252
    strong = (stk_w - mkt_w.reindex(c.index).fillna(0) >= REL_OUT) & (stk_w >= STOCK_FLOOR)
    weak = mkt_w.reindex(c.index).fillna(0) <= -MKT_WEAK
    brk = c > res

    sig = weak & strong & near & trend & brk
    return dict(close=c, open=o, low=l, high=h,
                ma10=ma10, ma20=ma20, res=res,
                exit_ma=c.rolling(EXIT_MA).mean(),
                sig=sig, weak=weak, strong=strong, near=near,
                trend=trend, brk=brk)


def simulate(feat, use_filters=True):
    """單一標的逐筆交易。進場＝訊號日隔天開盤；停損＝訊號日低點；
    出場＝收盤跌破 EXIT_MA（隔天開盤執行）。"""
    df = pd.DataFrame(feat)
    n = len(df)
    C = df['close'].to_numpy()
    O = df['open'].to_numpy()
    L = df['low'].to_numpy()
    EMA = df['exit_ma'].to_numpy()
    S = df['sig'].to_numpy()
    RES = df['res'].to_numpy()

    trades = []
    i = 300
    while i < n - 2:
        if not S[i] or not np.isfinite(RES[i]):
            i += 1
            continue
        ei = i + 1
        if not np.isfinite(O[ei]):
            i += 1
            continue
        entry = O[ei]
        stop = L[i]                       # 突破當天低點
        if entry <= stop:                 # 開盤就跌破停損 → 不進場
            i += 1
            continue

        xi, xp, why = None, None, None
        for j in range(ei, min(n, len(C))):
            if L[j] <= stop:
                xi, xp, why = j, stop, '停損'
                break
            if j > ei and np.isfinite(EMA[j - 1]) and C[j] < EMA[j - 1]:
                xi = j + 1 if j + 1 < n else j
                xp = O[xi] if np.isfinite(O[xi]) else C[j]
                why = '跌破均線'
                break
        if xi is None:
            xi, xp, why = n - 1, C[n - 1], '期末'
        trades.append(dict(entry=df.index[ei], exit=df.index[xi],
                           entry_px=entry, exit_px=xp,
                           ret=xp / entry - 1 - COST * 2,
                           stop_pct=entry / stop - 1,
                           days=xi - ei, why=why))
        i = xi + 1                        # 出場後才找下一個
    return pd.DataFrame(trades)


def portfolio(data, feats, mkt_ret, max_pos=MAX_POS, weight=WEIGHT):
    """多標的組合：訊號日隔天開盤進場，最多 max_pos 檔。"""
    idx = None
    for f in feats.values():
        s = f['close'].index
        idx = s if idx is None else idx.union(s)
    idx = idx.sort_values()

    C = pd.DataFrame({k: v['close'] for k, v in feats.items()}).reindex(idx)
    O = pd.DataFrame({k: v['open'] for k, v in feats.items()}).reindex(idx)
    L = pd.DataFrame({k: v['low'] for k, v in feats.items()}).reindex(idx)
    EMA = pd.DataFrame({k: v['exit_ma'] for k, v in feats.items()}).reindex(idx)
    SIG = pd.DataFrame({k: v['sig'] for k, v in feats.items()}).reindex(idx).fillna(False).astype(bool)
    RESD = pd.DataFrame({k: v['res'] for k, v in feats.items()}).reindex(idx)

    Ca, Oa, La, Ea = C.to_numpy(), O.to_numpy(), L.to_numpy(), EMA.to_numpy()
    Sa, Ra = SIG.to_numpy(), RESD.to_numpy()
    n, m = C.shape
    syms = list(feats)

    in_pos = np.zeros(m, dtype=bool)
    entry_i = np.full(m, -1, dtype=int)
    entry_p = np.full(m, np.nan)
    stop_p = np.full(m, np.nan)
    pos = np.zeros((n, m))
    trades = []

    for i in range(300, n - 1):
        # 出場判斷（用前一日收盤 vs 均線；今日開盤執行）
        if in_pos.any():
            for j in np.flatnonzero(in_pos):
                if La[i, j] <= stop_p[j]:
                    xp = stop_p[j]
                    if np.isfinite(Ca[entry_i[j], j]):
                        trades.append(dict(sym=syms[j], entry=idx[entry_i[j]],
                                           exit=idx[i], ret=xp / entry_p[j] - 1 - COST * 2,
                                           days=i - entry_i[j], why='停損'))
                    in_pos[j] = False
                    continue
                prev = i - 1
                if prev >= 0 and np.isfinite(Ea[prev, j]) and Ca[prev, j] < Ea[prev, j]:
                    xp = Oa[i, j] if np.isfinite(Oa[i, j]) else Ca[prev, j]
                    trades.append(dict(sym=syms[j], entry=idx[entry_i[j]],
                                       exit=idx[i], ret=xp / entry_p[j] - 1 - COST * 2,
                                       days=i - entry_i[j], why='跌破均線'))
                    in_pos[j] = False

        # 進場（昨日訊號 → 今日開盤）
        prev = i - 1
        slots = max_pos - int(in_pos.sum())
        if slots > 0 and prev >= 0:
            cand = np.flatnonzero(Sa[prev] & ~in_pos)
            if len(cand):
                px = Oa[i, cand]
                ok = np.isfinite(px) & np.isfinite(Ra[prev, cand])
                cand, px = cand[ok], px[ok]
                if len(cand):
                    stp = La[prev, cand]
                    good = px > stp
                    cand, px, stp = cand[good], px[good], stp[good]
                    for j, p, s0 in zip(cand[:slots], px[:slots], stp[:slots]):
                        in_pos[j] = True
                        entry_i[j] = i
                        entry_p[j] = p
                        stop_p[j] = s0

        pos[i] = in_pos.astype(float) * weight

    P = pd.DataFrame(pos, index=idx, columns=syms)
    rets = C.pct_change().fillna(0.0)
    h = P.shift(1).fillna(0.0)
    turn = h.diff().abs().sum(axis=1).fillna(0.0)
    pnl = (h * rets).sum(axis=1) - turn * COST / 1e4
    eq = (1 + pnl).cumprod()

    sl = idx[300:]
    eqs = eq.loc[sl]
    expo = h.sum(axis=1).loc[sl]
    bench = mkt_ret.reindex(sl).fillna(0.0)
    eqmax = np.maximum.accumulate(eqs.to_numpy())

    def cagr_of(e):
        e = e.dropna()
        if len(e) < 2 or e.iloc[0] <= 0:
            return float('nan')
        y = (e.index[-1] - e.index[0]).days / 365.25
        return float((e.iloc[-1] / e.iloc[0]) ** (1 / y) - 1)

    return dict(eq=eqs, pnl=pnl.loc[sl], trades=pd.DataFrame(trades),
                cagr=cagr_of(eqs),
                em_cagr=cagr_of((1 + bench * expo.mean()).cumprod()),
                bh_cagr=cagr_of((1 + bench).cumprod()),
                expo=float(expo.mean()),
                mdd=float((eqs.to_numpy() / eqmax - 1).min()),
                years=(eqs.index[-1] - eqs.index[0]).days / 365.25)


def report(r, label):
    T = r['trades']
    print(f'\n{label}')
    print(f'  期間 {r["years"]:.1f} 年   交易 {len(T)} 筆（每年 {len(T)/r["years"]:.0f}）')
    if T.empty or len(T) < 3:
        print('  交易太少，無法評估')
        return
    w = T['ret'] > 0
    aw = T['ret'][w].mean() if w.any() else np.nan
    al = T['ret'][~w].mean() if (~w).any() else np.nan
    t = T['ret'].mean() / (T['ret'].std() / np.sqrt(len(T)))
    print(f'  勝率 {w.mean()*100:5.2f}%   每筆平均 {T["ret"].mean()*100:+.2f}%   '
          f'平均獲利 {aw*100:+.2f}% / 虧損 {al*100:+.2f}%   賺賠比 {abs(aw/al):.2f}')
    print(f'  平均持有 {T["days"].mean():.0f} 天   逐筆 t 值 {t:.2f}')
    print(f'  總報酬 {r["eq"].iloc[-1]*100-100:>8.0f}%   年化 {r["cagr"]*100:6.2f}%')
    print(f'  最大回撤 {r["mdd"]*100:6.1f}%   平均曝險 {r["expo"]*100:5.1f}%')
    print(f'  同曝險基準 {r["em_cagr"]*100:6.2f}%   QQQ 買進持有 {r["bh_cagr"]*100:6.2f}%')
    if 'why' in T.columns:
        print('  出場原因: ' + ', '.join(f'{k} {v}' for k, v in T['why'].value_counts().items()))


def main():
    data = load()
    if 'QQQ' not in data:
        print('缺 QQQ')
        return 1
    mkt = data['QQQ']
    mkt_ret = mkt['close'].pct_change()

    print('=' * 84)
    print('SMB「相對強度突破」——弱勢市場裡找不會跌的')
    print('=' * 84)
    print(f'  標的 {len(data)} 支   基準 QQQ')
    print(f'  參數：弱勢＝QQQ {WIN} 日跌 {MKT_WEAK*100:.1f}%↑   超額強度 +{REL_OUT*100:.0f}pp')
    print(f'        距 52 週高 {NEAR_HIGH*100:.0f}% 內   壓力＝前 {RES_N} 日高   停利均線 {EXIT_MA} 日')

    feats = {s: build(d, mkt) for s, d in data.items() if s != 'QQQ'}

    # 訊號總量（單檔層面）
    tot = sum(int(np.nansum(f['sig'].to_numpy())) for f in feats.values())
    print(f'\n  原始訊號總數（未考慮資金上限）: {tot} 個')

    r = portfolio(data, feats, mkt_ret)
    report(r, '★ 原話規則（弱勢市場 + 抗跌 + 52週高 + 均線多頭 + 收盤突破）')

    # ---- 最關鍵的對照：拿掉紅盤日濾網 ----
    print('\n' + '=' * 84)
    print('★ 核心檢驗：紅盤日濾網到底有沒有用？（他的整個主張就在這裡）')
    print('=' * 84)
    print('  做法：其他條件全部保留，只把「必須是弱勢市場」拿掉。')

    feats2 = {}
    for s, d in data.items():
        if s == 'QQQ':
            continue
        f = build(d, mkt)
        # 只拿掉「必須是弱勢市場」，抗跌、52週高、均線、突破全部保留
        f['sig'] = f['strong'] & f['near'] & f['trend'] & f['brk']
        feats2[s] = f
    r2 = portfolio(data, feats2, mkt_ret)
    report(r2, '對照：拿掉「弱勢市場」濾網（抗跌+52週高+均線+突破全保留）')

    # ---- 只留弱勢市場，但去掉抗跌 ----
    feats3 = {}
    for s, d in data.items():
        if s == 'QQQ':
            continue
        f = build(d, mkt)
        f['sig'] = f['weak'] & f['near'] & f['trend'] & f['brk']
        feats3[s] = f
    r3 = portfolio(data, feats3, mkt_ret)
    report(r3, '對照：只留弱勢市場，去掉「抗跌」條件')

    # ---- 敏感度 ----
    print('\n' + '=' * 84)
    print('敏感度')
    print('=' * 84)
    for lab, kw in [
        ('壓力改前 20 日高', dict(RES_N=20)),
        ('壓力改前 120 日高', dict(RES_N=120)),
        ('停利均線改 5 日', dict(EXIT_MA=5)),
        ('停利均線改 20 日', dict(EXIT_MA=20)),
        ('弱勢門檻改 1.0%', dict(MKT_WEAK=0.010)),
        ('弱勢門檻改 2.5%', dict(MKT_WEAK=0.025)),
        ('距高改 3% 內', dict(NEAR_HIGH=0.97)),
        ('觀察窗改 5 日', dict(WIN=5)),
        ('觀察窗改 20 日', dict(WIN=20)),
    ]:
        g = globals()
        old = {k: g[k] for k in kw}
        g.update(kw)
        try:
            ff = {s: build(d, mkt) for s, d in data.items() if s != 'QQQ'}
            rr = portfolio(data, ff, mkt_ret)
            T = rr['trades']
            if len(T) >= 5:
                w = T['ret'] > 0
                print(f'  {lab:<22} 年化 {rr["cagr"]*100:6.2f}%  曝險 {rr["expo"]*100:5.1f}%  '
                      f'{len(T):>4} 筆  勝率 {w.mean()*100:5.1f}%  '
                      f'每筆 {T["ret"].mean()*100:+.2f}%  回撤 {rr["mdd"]*100:5.1f}%')
            else:
                print(f'  {lab:<22} 交易僅 {len(T)} 筆')
        except Exception as e:
            print(f'  {lab:<22} 失敗 {type(e).__name__} {e}')
        finally:
            g.update(old)

    r['trades'].to_csv(os.path.join(OUT, 'smb_rs_trades.csv'), index=False)
    r['eq'].to_frame('equity').to_csv(os.path.join(OUT, 'smb_rs_equity.csv'))
    print(f'\n逐筆交易 → out/smb_rs_trades.csv')
    return 0


if __name__ == '__main__':
    sys.exit(main())
