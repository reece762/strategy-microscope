#!/usr/bin/env python3
"""RTA《交易之路》第二期的說法，用 BTC 日線實際回測。

來源：2026-10-25 取得的完整逐字稿（2 小時 26 分）。這一期是三組學生報告、
院長逐頁挑錯，其中有幾個主張可以檢驗，這裡逐條跑。

要檢驗的五件事：
  1. 主要趨勢確認：MA120 用「拐頭」還是「破線」？學生聲稱四年約 8 倍。
  2. MA60 能不能當主要趨勢分界線？（院長說不行，要用 MA120）
  3. 次要運動的回撤真的是前波段的 1/3~2/3 嗎？（院長說常只到 0.236）
  4. 短暫趨勢的波動是 0.5% 還是 5%？
  5. 主要趨勢持續多久？（一年多，還是一個季度？）

紀律（沿用本專案既有做法）：
  - 信號只用當日收盤，部位次日生效（無前視偏差）
  - 扣交易成本：加密貨幣每趟 5 bp
  - 帳面績效之外，另做「剔除最極端 5% 日」檢驗
    → 被剔除的日子以 0 報酬計，時間軸保留；不是把日子刪掉重算，
      否則等於宣稱那些天不存在，會嚴重高估年化。
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from btc import load_btc, perf  # noqa: E402

OUT = os.path.join(HERE, 'out')
COST_BP = 5.0


# ---------------------------------------------------------------- 均線訊號
def ma_signals(d, n):
    """回傳 ma, above(收盤在線上), up(拐頭向上), dn(拐頭向下)。全部用收盤價。"""
    ma = d['close'].rolling(n, min_periods=n).mean()
    above = d['close'] > ma
    up = (ma > ma.shift(1)) & (ma.shift(1) <= ma.shift(2))
    dn = (ma < ma.shift(1)) & (ma.shift(1) >= ma.shift(2))
    return ma, above, up, dn


def run_position(close, pos, cost_bp=COST_BP):
    """pos[t] 是 t 日收盤決定的目標部位，t+1 日才持有。"""
    pos = pos.shift(1).fillna(0.0)
    ret = close.pct_change().fillna(0.0)
    turn = pos.diff().abs().fillna(pos.abs())
    pnl = pos * ret - turn * cost_bp / 10000.0
    eq = (1 + pnl).cumprod()
    return eq, pos, pnl


def build_variants(d):
    close = d['close']
    ma, above, up, dn = ma_signals(d, 120)
    out = {}

    # V1 破線進出：線上做多、線下做空
    pos1 = pd.Series(np.where(above, 1.0, -1.0), index=d.index)
    pos1[ma.isna()] = 0.0
    out['破線進出'] = pos1

    # V2 拐頭開始 / 破線結束
    pos2 = pd.Series(0.0, index=d.index)
    state = 0.0
    for i in range(len(d)):
        if pd.isna(ma.iloc[i]):
            continue
        if state == 0.0:
            state = 1.0 if up.iloc[i] else (-1.0 if dn.iloc[i] else 0.0)
        elif state > 0 and close.iloc[i] < ma.iloc[i]:
            state = -1.0 if dn.iloc[i] else 0.0
        elif state < 0 and close.iloc[i] > ma.iloc[i]:
            state = 1.0 if up.iloc[i] else 0.0
        pos2.iloc[i] = state
    out['拐頭開始破線結束'] = pos2

    # V3 拐頭進出
    pos3 = pd.Series(0.0, index=d.index)
    state = 0.0
    for i in range(len(d)):
        if pd.isna(ma.iloc[i]):
            continue
        if up.iloc[i]:
            state = 1.0
        elif dn.iloc[i]:
            state = -1.0
        pos3.iloc[i] = state
    out['拐頭進出'] = pos3

    out['買進持有'] = pd.Series(1.0, index=d.index)
    return out


def honest_checks(pnl):
    """剔除最極端 5% 日後的年化。被剔除的日子以 0 報酬計，時間軸保留。"""
    r = pnl.dropna()
    if len(r) < 100:
        return {}
    n_cut = max(1, int(len(r) * 0.05))
    yrs = (r.index[-1] - r.index[0]).days / 365.25

    def cagr(mask):
        s = r.copy()
        s[mask] = 0.0                      # 歸零，不是刪除
        return float((1 + s).prod() ** (1 / yrs) - 1) if yrs > 0 else 0.0

    return {
        '原始年化': cagr(pd.Index([])),
        '剔除最差5%日_年化': cagr(r.nsmallest(n_cut).index),
        '剔除最佳5%日_年化': cagr(r.nlargest(n_cut).index),
        '正報酬日佔比': float((r > 0).mean()),
        '日報酬t值': float(r.mean() / r.std() * np.sqrt(len(r))) if r.std() > 0 else 0.0,
    }


# ---------------------------------------------------------------- 波段
def zigzag(close, threshold):
    """回傳轉折點索引（高低交替）。threshold 是最小反轉幅度。"""
    v = close.values
    if len(v) < 10:
        return []
    piv = [0]
    direction = 1
    last = v[0]
    for i in range(1, len(v)):
        if direction == 1:
            if v[i] > last:
                last = v[i]
                piv[-1] = i
            elif v[i] <= last * (1 - threshold):
                piv.append(i)
                direction = -1
                last = v[i]
        else:
            if v[i] < last:
                last = v[i]
                piv[-1] = i
            elif v[i] >= last * (1 + threshold):
                piv.append(i)
                direction = 1
                last = v[i]
    return piv


def retracements(close, threshold):
    """三段一組：後一段回撤 / 前一段推進。

    只保留「回撤小於推進 3 倍」的樣本；超過者代表那不是回撤而是新趨勢段。
    """
    v = close.values
    piv = zigzag(close, threshold)
    out = []
    for k in range(2, len(piv)):
        a, b, c = piv[k - 2], piv[k - 1], piv[k]
        impulse = abs(v[b] - v[a])
        pull = abs(v[c] - v[b])
        if impulse > 0:
            r = pull / impulse
            if 0 < r < 3:
                out.append(float(r))
    return out


def atr_pct(d, n=14):
    h, l, c = d['high'], d['low'], d['close']
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return (tr.rolling(n).mean() / c).dropna()


def regime_runs(close, n=120):
    """價格連續在 MA120 上方或下方的期間長度（天）。

    這比「均線斜率」穩健：斜率每天都會變號，切出來的段長會碎成幾天，
    那不是趨勢，是雜訊。價格與均線的相對位置才是可持倉的狀態。
    """
    ma = close.rolling(n, min_periods=n).mean()
    above = (close > ma).dropna()
    if len(above) == 0:
        return []
    runs = []
    cur = 1
    prev = above.iloc[0]
    for x in above.iloc[1:]:
        if x == prev:
            cur += 1
        else:
            runs.append(cur)
            prev = x
            cur = 1
    runs.append(cur)
    return runs


def main():
    d = load_btc()
    close = d['close']
    print(f'資料：{d.index[0]:%Y-%m-%d} ~ {d.index[-1]:%Y-%m-%d}  {len(d)} 天')
    print(f'BTC 由 {close.iloc[0]:,.0f} 到 {close.iloc[-1]:,.0f}\n')
    res = {'期間': f'{d.index[0]:%Y-%m-%d} ~ {d.index[-1]:%Y-%m-%d}', '天數': len(d)}

    # ---------------- 測試一
    variants = build_variants(d)
    print('=' * 80)
    print('測試一：MA120 主要趨勢確認方式')
    print('=' * 80)
    print(f'{"方式":<18}{"年化":>9}{"最大回撤":>11}{"總倍數":>9}{"夏普":>8}')
    res['確認方式'] = {}
    pnls = {}
    for name, pos in variants.items():
        eq, _, pnl = run_position(close, pos)
        m = perf(eq)
        pnls[name] = pnl
        res['確認方式'][name] = {**{k: float(v) for k, v in m.items()}}
        print(f'{name:<18}{m["cagr"]*100:>8.1f}%{m["mdd"]*100:>10.1f}%'
              f'{m["multiple"]:>8.2f}{m["sharpe"]:>8.2f}')

    print('\n--- 誠實檢查（被剔除的日子以 0 報酬計，時間軸保留）---')
    print(f'{"方式":<18}{"原始":>9}{"剔最差5%":>11}{"剔最佳5%":>11}{"日t值":>8}')
    for name in variants:
        h = honest_checks(pnls[name])
        if not h:
            continue
        res['確認方式'][name].update({k: float(v) for k, v in h.items()})
        print(f'{name:<18}{h["原始年化"]*100:>8.1f}%{h["剔除最差5%日_年化"]*100:>10.1f}%'
              f'{h["剔除最佳5%日_年化"]*100:>10.1f}%{h["日報酬t值"]:>8.2f}')

    # 學生聲稱的區間：2018-08-29 ~ 2022-08-19
    seg = d.loc['2018-08-29':'2022-08-19']
    print(f'\n--- 學生聲稱的四年區間 {seg.index[0]:%Y-%m-%d} ~ {seg.index[-1]:%Y-%m-%d} ---')
    if len(seg) > 200:
        sc = seg['close']
        for name, pos in variants.items():
            eq, _, _ = run_position(sc, pos.loc[seg.index])
            m = perf(eq)
            print(f'  {name:<18} 倍數 {m["multiple"]:>6.2f}  年化 {m["cagr"]*100:>6.1f}%')
        res['學生四年區間'] = {name: float(perf(run_position(sc, pos.loc[seg.index])[0])['multiple'])
                          for name, pos in variants.items()}

    # ---------------- 測試二
    print('\n' + '=' * 80)
    print('測試二：MA60 能不能當主要趨勢分界線')
    print('=' * 80)
    res['MA比較'] = {}
    for n in (60, 120):
        ma, above, up, dn = ma_signals(d, n)
        pos = pd.Series(np.where(above, 1.0, -1.0), index=d.index)
        pos[ma.isna()] = 0.0
        eq, _, _ = run_position(close, pos)
        m = perf(eq)
        flips = int((pos.diff().abs() > 0).sum())
        res['MA比較'][f'MA{n}'] = {**{k: float(v) for k, v in m.items()}, '翻轉次數': flips}
        print(f'  MA{n:<4} 年化 {m["cagr"]*100:>7.1f}%  回撤 {m["mdd"]*100:>7.1f}%  '
              f'倍數 {m["multiple"]:>6.2f}  翻轉 {flips:>4} 次')

    ma14 = close.rolling(14).mean()
    ma28 = close.rolling(28).mean()
    ma60 = close.rolling(60).mean()
    ma120 = close.rolling(120).mean()
    for yr in ('2022', '2021'):
        s = slice(f'{yr}-01-01', f'{yr}-12-31')
        seg = d.loc[s]
        if len(seg) < 50:
            continue
        c = seg['close']
        below60 = (c < ma60.loc[seg.index]).sum()
        below120 = (c < ma120.loc[seg.index]).sum()
        bear_arr = ((ma14 < ma28) & (ma28 < ma120)).loc[seg.index]
        print(f'  {yr}：收盤在 MA60 下方 {below60}/{len(seg)} 天、'
              f'MA120 下方 {below120}/{len(seg)} 天、空頭排列 {int(bear_arr.sum())}/{len(seg)} 天')

    # ---------------- 測試三（含門檻敏感度）
    print('\n' + '=' * 80)
    print('測試三：次要運動的回撤幅度分布（院長說 1/3~2/3 太粗略）')
    print('=' * 80)
    res['回撤'] = {}
    print(f'{"反轉門檻":<10}{"樣本":>6}{"中位數":>9}{"1/3~2/3":>10}{"<=0.236":>10}{">2/3":>8}')
    for th in (0.15, 0.20, 0.25, 0.30):
        rr = retracements(close, th)
        if not rr:
            continue
        s = pd.Series(rr)
        mid = ((s >= 1 / 3) & (s <= 2 / 3)).mean()
        low = (s <= 0.236).mean()
        high = (s > 2 / 3).mean()
        res['回撤'][f'{int(th*100)}%'] = {
            '樣本': int(len(s)), '中位數': float(s.median()),
            '介於三分之一到三分之二': float(mid), '小於等於0.236': float(low), '大於三分之二': float(high),
            '第25百分位': float(s.quantile(0.25)), '第75百分位': float(s.quantile(0.75)),
        }
        print(f'{int(th*100):>3}%      {len(s):>6}{s.median()*100:>8.1f}%{mid*100:>9.1f}%'
              f'{low*100:>9.1f}%{high*100:>7.1f}%')

    # ---------------- 測試四
    print('\n' + '=' * 80)
    print('測試四：短暫趨勢的波動幅度（0.5% 還是 5%？）')
    print('=' * 80)
    av = atr_pct(d, 14)
    print(f'  日線 ATR(14) 佔收盤價：中位數 {av.median()*100:.2f}%  平均 {av.mean()*100:.2f}%')
    for q in (0.25, 0.50, 0.75):
        print(f'    第 {int(q*100):>2} 百分位：{av.quantile(q)*100:>5.2f}%')
    res['短暫波動'] = {'ATR14中位數': float(av.median()), 'ATR14平均': float(av.mean())}
    for k in (3, 5, 10):
        rng = (d['high'].rolling(k).max() / d['low'].rolling(k).min() - 1).dropna()
        res['短暫波動'][f'{k}日區間中位數'] = float(rng.median())
        print(f'  {k:>2} 日高低區間：中位數 {rng.median()*100:>5.2f}%  '
              f'第75百分位 {rng.quantile(0.75)*100:>5.2f}%')

    # ---------------- 測試五
    print('\n' + '=' * 80)
    print('測試五：主要趨勢持續多久（一年多，還是一個季度？）')
    print('=' * 80)
    runs = regime_runs(close, 120)
    if runs:
        s = pd.Series(runs)
        print(f'  以「價格在 MA120 同側」切段，共 {len(s)} 段（單位：天／季）')
        for q in (0.25, 0.50, 0.75, 0.90):
            print(f'    第 {int(q*100):>2} 百分位：{s.quantile(q):>5.0f} 天（{s.quantile(q)/91.3:>4.1f} 季）')
        print(f'  中位數 {s.median():.0f} 天（{s.median()/91.3:.1f} 季）  平均 {s.mean():.0f} 天')
        print(f'  超過一年（365 天）的比率：{(s > 365).mean()*100:.1f}%')
        print(f'  超過一季（91 天）的比率：{(s > 91).mean()*100:.1f}%')
        res['趨勢長度'] = {
            '中位數天': float(s.median()), '平均天': float(s.mean()),
            '超過一年比率': float((s > 365).mean()), '超過一季比率': float((s > 91).mean()),
            '樣本數': int(len(s)),
        }

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'rta.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f'\n結果已存 out/rta.json')


if __name__ == '__main__':
    main()
