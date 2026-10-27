#!/usr/bin/env python3
"""英文頻道選股影片：10 條篩選器的回測。

## 影片與主張

"YouTube 英文頻道 bowinvest" 的《How I Pick Stocks: Investing for Beginners》。
作者說他用 10 個指標建立篩選器，把「數百家公司縮到幾家」放進觀察名單。
他的原話：「I'm going to show you 10 metrics that I personally use to screen stocks.」
他也明確說：「this stock screener and the stocks it's going to show us today
doesn't mean that it's a buy signal」——**他自己就說了這不是買進訊號，是研究起點。**

10 條規則（他的設定值）：
  1. 5 年價格漲幅 > 50%
  2. trailing PE 介於 10~50
  3. forward PE 介於 5~30
  4. PEG < 2
  5. EPS > 0
  6. 營收 5 年 CAGR > 20%
  7. 淨利 5 年 CAGR > 20%
  8. 淨利率 > 10%
  9. 自由現金流 > 0
 10. 流動比 > 1.5

## 我能測幾條（誠實清單）

  可測（7 條）：1、2、4、5、6、7、8
  替代（1 條）：9 — 資料庫沒有資本支出科目，用「營業現金流 > 0」代替並標明
  不可測（2 條）：3 — 沒有分析師預估（forward PE 需要未來 12 個月預估 EPS）
                 10 — 沒有流動資產／流動負債科目
  **所以我測的是 7 條完整 + 1 條替代，等於原規則的 8 成，缺的兩條都偏「保守/防禦」
  （forward PE 要求成長、流動比要求償債能力），漏掉它們會讓選出的股票更寬鬆。**

## 方法：當時點（point-in-time）篩選，不偷看未來

關鍵在 `filed_date`：每個年度財報只在「申報日」之後才看得到。
每個再平衡日，只用 `filed_date <= 當日` 的資料算指標。
同一個 period_end 若被重述，取**最早**的 filed_date（首次公布），避免用後見之明。

## 對照組設計（三個層次）

  1. 全部股票等權（同期、同樣 12 個月）——看他選的股票有沒有比「全都買」好
  2. **相同規模的隨機組合**（抽 k 檔、抽 500 次）——關鍵：
     選出 14 檔的組合，變異本來就大。要證明「選股有效」，
     必須贏過「隨機選同樣檔數」的分佈，而不是只贏過全體平均。
  3. 各條篩選器單獨使用——看哪一條真正有貢獻

## 已知限制

  - **存活者偏差**：569 檔是「今天還在、且有 XBRL 資料」的公司，
    倒閉下市者不在母體。這對「成長股篩選」是高估。
  - 年度再平衡 → 獨立觀測只有十幾個年份，統計力有限。
  - 未計交易成本（年度換倉一次，成本相對小）。
"""
import glob
import json
import os

import numpy as np
import pandas as pd

STOCK = '/root/.openclaw/stock_data'
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')

REV_TAGS = ('Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax', 'SalesRevenueNet')
NI_TAGS = ('NetIncomeLoss',)
EPS_TAGS = ('EarningsPerShareDiluted', 'EarningsPerShareBasic')
OCF_TAGS = ('NetCashProvidedByUsedInOperatingActivities',)


def load_prices():
    px = {}
    for f in sorted(glob.glob(f'{STOCK}/live/*.csv') + glob.glob(f'{STOCK}/extra/*.csv')):
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        d.columns = [c.lower() for c in d.columns]
        if not {'date', 'close'}.issubset(d.columns) or len(d) < 300:
            continue
        d['date'] = pd.to_datetime(d['date'])
        d = d.dropna(subset=['close']).sort_values('date')
        px[os.path.basename(f).replace('.csv', '')] = d.set_index('date')['close']
    return px


def annual_series(d, tags):
    """抽年報：期間 300~430 天；同一 period_end 取最早 filed_date。"""
    x = d[d['tag'].isin(tags)].copy()
    if x.empty:
        return None
    for c in ('period_end', 'period_start', 'filed_date'):
        x[c] = pd.to_datetime(x[c], errors='coerce')
    x = x.dropna(subset=['period_end', 'filed_date', 'value'])
    if x.empty:
        return None
    x['dur'] = (x['period_end'] - x['period_start']).dt.days
    x = x[(x['dur'] >= 300) & (x['dur'] <= 430)]
    if x.empty:
        return None
    x = x.sort_values('filed_date').drop_duplicates(subset=['period_end'], keep='first')
    return x.sort_values('period_end')[['period_end', 'filed_date', 'value']]


def load_fundamentals():
    out = {}
    for f in sorted(glob.glob(f'{STOCK}/fundamentals/edgar/*.csv')):
        sym = os.path.basename(f).replace('.csv', '')
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        if 'tag' not in d.columns:
            continue
        rec = {}
        for key, tags in (('rev', REV_TAGS), ('ni', NI_TAGS),
                          ('eps', EPS_TAGS), ('ocf', OCF_TAGS)):
            s = annual_series(d, tags)
            if s is not None and len(s) >= 2:
                rec[key] = s
        if rec:
            out[sym] = rec
    return out


def visible(series, asof):
    """當時點可見：filed_date <= asof"""
    if series is None:
        return None
    v = series[series['filed_date'] <= asof]
    return v if len(v) else None


def cagr(series, asof, min_span=3.0, max_span=7.0):
    """以當時點可見的年報算年化成長率，回傳 (cagr, span_years, latest_value)。

    ★ 修正「離一天」錯誤：3 個日曆年若不含閏日只有 1095 天
      （1095/365.25 = 2.9979 年），原本 `span < 3.0` 會把它剔除，
      導致每逢閏年時點（2016/2020/2024）可用股票數暴跌。
      改用「天數」比較，3 年 = 1080 天（留 15 天寬容）。
    """
    v = visible(series, asof)
    if v is None or len(v) < 2:
        return (np.nan, np.nan, np.nan)
    latest = v.iloc[-1]
    # 找最接近「min_span 年前」的那一筆（以天數比較，避免閏年誤差）
    target = latest['period_end'] - pd.Timedelta(days=int(365 * min_span))
    cand = v[v['period_end'] <= target]
    if cand.empty:
        return (np.nan, np.nan, float(latest['value']))
    base = cand.iloc[-1]
    days = (latest['period_end'] - base['period_end']).days
    span = days / 365.25
    if days < 365 * min_span - 15 or span > max_span:
        return (np.nan, span, float(latest['value']))
    if base['value'] <= 0 or latest['value'] <= 0:
        # 虧損轉虧損或負值：CAGR 無意義
        return (np.nan, span, float(latest['value']))
    g = (latest['value'] / base['value']) ** (1.0 / span) - 1.0
    return (g, span, float(latest['value']))


def price_at(series, asof):
    if series is None:
        return np.nan
    s = series[series.index <= asof]
    return float(s.iloc[-1]) if len(s) else np.nan


def fwd_return(series, asof, days=365):
    """從 asof 之後第一個交易日收盤，持有約 days 天。"""
    if series is None:
        return np.nan
    s = series[series.index >= asof]
    if len(s) < 2:
        return np.nan
    p0 = float(s.iloc[0])
    target = s.index[0] + pd.Timedelta(days=days)
    s2 = series[series.index >= target]
    if len(s2) == 0:
        return np.nan
    return float(s2.iloc[0]) / p0 - 1.0


def screen_row(sym, asof, px, fund):
    """在 asof 這天，算出這個股票所有指標與各條是否通過。"""
    f = fund.get(sym)
    p = px.get(sym)
    if f is None or p is None:
        return None
    price = price_at(p, asof)
    if not np.isfinite(price):
        return None

    rev = f.get('rev'); ni = f.get('ni'); eps = f.get('eps'); ocf = f.get('ocf')

    # 1. 5 年價格漲幅 > 50%
    p5 = price_at(p, asof - pd.Timedelta(days=int(365 * 5)))
    chg5 = price / p5 - 1 if (np.isfinite(p5) and p5 > 0) else np.nan

    # EPS（最近年度，當時點可見）
    eps_v = visible(eps, asof)
    eps_last = float(eps_v.iloc[-1]['value']) if (eps_v is not None and len(eps_v)) else np.nan

    # PE = price / 年度 EPS
    pe = price / eps_last if (np.isfinite(eps_last) and eps_last > 0) else np.nan

    # 6/7/8 成長率與利潤率
    rev_g, rev_span, rev_last = cagr(rev, asof)
    ni_g, ni_span, ni_last = cagr(ni, asof)

    # 淨利率 = 最近年度淨利 / 營收
    margin = (ni_last / rev_last) if (np.isfinite(ni_last) and np.isfinite(rev_last)
                                      and rev_last > 0) else np.nan

    # PEG = PE / (淨利成長率 %)
    peg = (pe / (ni_g * 100)) if (np.isfinite(pe) and np.isfinite(ni_g) and ni_g > 0) else np.nan

    # 9. 自由現金流 > 0 → 用營業現金流代替（無 CapEx 科目）
    ocf_v = visible(ocf, asof)
    ocf_last = float(ocf_v.iloc[-1]['value']) if (ocf_v is not None and len(ocf_v)) else np.nan

    ok = {
        'f1_px50': bool(np.isfinite(chg5) and chg5 > 0.50),
        'f2_pe': bool(np.isfinite(pe) and 10 <= pe <= 50),
        'f4_peg': bool(np.isfinite(peg) and peg < 2),
        'f5_eps': bool(np.isfinite(eps_last) and eps_last > 0),
        'f6_revg': bool(np.isfinite(rev_g) and rev_g > 0.20),
        'f7_nig': bool(np.isfinite(ni_g) and ni_g > 0.20),
        'f8_margin': bool(np.isfinite(margin) and margin > 0.10),
        'f9_ocf': bool(np.isfinite(ocf_last) and ocf_last > 0),
    }
    # ★ 關鍵：區分「不符條件」與「資料不足」
    # 若核心指標大量缺失（NaN），這檔股票應該被排除，
    # 而不是被算成「通過 0 條」——否則資料殘缺股會被誤讀成
    # 「什麼都不符合反而表現最好」。
    n_avail = int(sum(np.isfinite([chg5, pe, eps_last, rev_g, ni_g, margin])) )
    data_ok = bool(n_avail >= 5)      # 六個核心指標至少要有五個

    return dict(symbol=sym, asof=asof, price=price,
                chg5=chg5, pe=pe, peg=peg, eps=eps_last,
                rev_g=rev_g, ni_g=ni_g, margin=margin, ocf=ocf_last,
                n_avail=n_avail, data_ok=data_ok,
                n_pass=int(sum(ok.values())) if data_ok else -1, **ok)


def main():
    print('載入價格...', flush=True)
    px = load_prices()
    print(f'  {len(px)} 檔', flush=True)
    print('載入財報（抽年報）...', flush=True)
    fund = load_fundamentals()
    print(f'  {len(fund)} 檔', flush=True)

    syms = sorted(set(px) & set(fund))
    print(f'  價格 ∩ 財報 = {len(syms)} 檔', flush=True)

    # 再平衡日：每年 6/30（多數 10-K 已在 3 月前申報，留緩衝）
    rebals = [pd.Timestamp(f'{y}-06-30') for y in range(2011, 2026)]
    rows = []
    for i, D in enumerate(rebals):
        n = 0
        for sym in syms:
            r = screen_row(sym, D, px, fund)
            if r is None:
                continue
            r['fwd12'] = fwd_return(px.get(sym), D, 365)
            if not np.isfinite(r['fwd12']):
                continue
            rows.append(r)
            n += 1
        print(f'  {D.date()}  可用 {n} 檔', flush=True)

    A_all = pd.DataFrame(rows)
    A = A_all[A_all['data_ok']].copy()
    print(f'\n總觀測 {len(A_all):,}（股票×年度）', flush=True)
    print(f'  資料充足 {len(A):,}　資料不足 {len(A_all)-len(A):,}（後者排除，'
          f'不計入「通過 0 條」）', flush=True)

    FILTERS = ['f1_px50', 'f2_pe', 'f4_peg', 'f5_eps', 'f6_revg', 'f7_nig', 'f8_margin', 'f9_ocf']

    # ---------- 主測試：全部通過 ----------
    A['pass_all'] = A[FILTERS].all(axis=1)
    res = {}
    yearly = []
    for D, g in A.groupby('asof'):
        sel = g[g.pass_all]
        k = len(sel)
        allr = g['fwd12'].mean()
        if k == 0:
            yearly.append(dict(asof=str(D.date()), n_univ=len(g), k=0,
                               ret=np.nan, univ=allr, rand_mean=np.nan,
                               rand_p05=np.nan, rand_p95=np.nan, pctile=np.nan))
            continue
        r = sel['fwd12'].mean()
        # 隨機抽同樣檔數，500 次
        rng = np.random.default_rng(2026)
        uni = g['fwd12'].to_numpy()
        draws = np.array([rng.choice(uni, k, replace=False).mean() for _ in range(500)]) \
            if k < len(uni) else np.array([uni.mean()])
        pct = float((draws < r).mean())
        yearly.append(dict(asof=str(D.date()), n_univ=len(g), k=k, ret=float(r),
                           univ=float(allr), rand_mean=float(draws.mean()),
                           rand_p05=float(np.quantile(draws, .05)),
                           rand_p95=float(np.quantile(draws, .95)),
                           pctile=pct))

    Y = pd.DataFrame(yearly)
    valid = Y.dropna(subset=['ret'])
    res['screen_all'] = dict(
        n_years=int(len(valid)),
        mean_ret=float(valid['ret'].mean()),
        mean_univ=float(valid['univ'].mean()),
        excess=float((valid['ret'] - valid['univ']).mean()),
        beat_univ=int((valid['ret'] > valid['univ']).sum()),
        mean_rand=float(valid['rand_mean'].mean()),
        excess_vs_rand=float((valid['ret'] - valid['rand_mean']).mean()),
        beat_rand=int((valid['ret'] > valid['rand_mean']).sum()),
        median_pctile=float(valid['pctile'].median()),
        avg_k=float(valid['k'].mean()),
        yearly=yearly)

    # ---------- 各篩選器單獨 ----------
    per_filter = {}
    for f in FILTERS:
        sub = A[A[f]]
        if len(sub) < 50:
            per_filter[f] = dict(n=int(len(sub)), note='樣本過少')
            continue
        y = sub.groupby('asof')['fwd12'].mean()
        base = A.groupby('asof')['fwd12'].mean()
        both = pd.concat([y.rename('s'), base.rename('b')], axis=1).dropna()
        per_filter[f] = dict(
            n=int(len(sub)), n_years=int(len(both)),
            mean=float(sub['fwd12'].mean()),
            excess=float((both.s - both.b).mean()),
            beat=int((both.s > both.b).sum()),
            pos=float((sub['fwd12'] > 0).mean()))

    # ---------- 篩選嚴格度 vs 表現（通過越多條越好？） ----------
    by_npass = {}
    for k, g in A.groupby('n_pass'):
        if k < 0 or len(g) < 50:
            continue
        by_npass[int(k)] = dict(n=int(len(g)), mean=float(g['fwd12'].mean()),
                                pos=float((g['fwd12'] > 0).mean()))

    # ---------- 全部通過的子集 vs 全體，逐年明細 ----------
    out = dict(meta=dict(n_stocks=len(syms), n_obs=int(len(A)),
                         years=[str(x.date()) for x in rebals],
                         filters_testable=FILTERS,
                         filters_missing=['f3_fwd_pe（無分析師預估）',
                                          'f10_current_ratio（無流動資產/負債科目）'],
                         f9_note='自由現金流以營業現金流代替（無資本支出科目）'),
               screen_all=res['screen_all'],
               per_filter=per_filter,
               by_npass=by_npass,
               yearly=yearly)
    json.dump(out, open(f'{OUT}/en_screen.json', 'w'), ensure_ascii=False, indent=1,
              default=str)

    # ---------- 列印 ----------
    s = res['screen_all']
    print('\n' + '=' * 70)
    print('【10 條篩選器（實測 8 條）年度再平衡】')
    print(f"  有效年份 {s['n_years']}，平均選出 {s['avg_k']:.1f} 檔")
    print(f"  篩選組合平均 12 個月報酬  {s['mean_ret']*100:+.2f}%")
    print(f"  全體等權平均              {s['mean_univ']*100:+.2f}%")
    print(f"  超額（vs 全體）           {s['excess']*100:+.2f}pp   勝過 {s['beat_univ']}/{s['n_years']} 年")
    print(f"  隨機同檔數組合平均        {s['mean_rand']*100:+.2f}%")
    print(f"  超額（vs 隨機）           {s['excess_vs_rand']*100:+.2f}pp   勝過 {s['beat_rand']}/{s['n_years']} 年")
    print(f"  中位百分位（在隨機分佈中） {s['median_pctile']*100:.0f}%")

    print('\n【各條篩選器單獨使用】')
    print(f"  {'篩選器':<12}{'樣本':>9}{'平均':>9}{'超額':>9}{'勝過基準':>9}")
    for f, v in per_filter.items():
        if 'note' in v:
            print(f"  {f:<12}{v['n']:>9}  ({v['note']})"); continue
        print(f"  {f:<12}{v['n']:>9,}{v['mean']*100:>+8.2f}%{v['excess']*100:>+8.2f}pp"
              f"{v['beat']:>6}/{v['n_years']}")

    print('\n【通過條數 vs 表現】')
    for k in sorted(by_npass):
        v = by_npass[k]
        print(f"  通過 {k} 條: n={v['n']:>6,}  平均 {v['mean']*100:>+7.2f}%  上漲率 {v['pos']*100:.0f}%")

    print('\n【逐年】')
    for y in yearly:
        if y['k'] == 0:
            print(f"  {y['asof']}: 選出 0 檔"); continue
        print(f"  {y['asof']}: 母體 {y['n_univ']:>3} → 選 {y['k']:>2} 檔  "
              f"報酬 {y['ret']*100:>+7.2f}%  全體 {y['univ']*100:>+7.2f}%  "
              f"隨機中位 {y['rand_mean']*100:>+7.2f}%  百分位 {y['pctile']*100:>3.0f}%")

    print('\n已寫 en_screen.json')


if __name__ == '__main__':
    main()
