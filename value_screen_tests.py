#!/usr/bin/env python3
"""兩個英文頻道的選股方法回測。

## 方法一：Phil Town《Rule #1》的「Big Five」（影片 n1FAWFOq_bs）

他主張：一家好公司必須在「過去 10 年」的以下五項平均都 ≥10%：
  1. ROIC（投入資本報酬率）
  2. 營收成長率
  3. 每股盈餘成長率
  4. 股東權益（帳面價值）成長率
  5. 自由現金流成長率
外加一個負債測試：長期負債 ÷ 自由現金流 ≤ 3 年。
他的原話：「a healthy business should average at least 10% ROIC over the past 10 years」、
「town wants averages of 10% per year over 10 years as a baseline」、
「If a company could pay off its long-term debt in three years or less from free cash flow,
the debt is acceptable」。

## 方法二：Everything Money 的「8 支柱」（影片 1u6qvel9XnM）

Paul 與 Seth 公開的 8 條，全部有明確門檻（原話節錄在卡片內）：
  1. 5 年本益比 < 22.5（市值 ÷ 過去 5 年淨利總和）
  2. 5 年 ROIC > 9%（5 年現金流 ÷（總負債＋股東權益））
  3. 5 年營收成長（今天 > 五年前）
  4. 5 年淨利成長（今天 > 五年前）
  5. 流通股數沒有增加（回購）
  6. 長期負債 ÷ 5 年自由現金流 < 5
  7. 5 年現金流成長
  8. 市值 ÷ 5 年自由現金流 < 20

## ★ 資料限制（必須揭露，且會影響結論解讀）

我掃過 569 檔的全部 XBRL 標籤，**總共只有 16 種**，缺三類關鍵科目：

  (a) **沒有資本支出（CapEx）** → 無法算真正的自由現金流（FCF = 營業現金流 − 資本支出）。
      **兩套方法都大量使用 FCF**：Phil Town 的第 5 項與負債測試、
      Everything Money 的第 6、7、8 項。
      我用「營業現金流（OCF）」代替 FCF。
      **偏誤方向：OCF > FCF，所以「負債/FCF」與「市值/FCF」都會偏好看（更容易通過），
      而「FCF 成長率」與 OCF 成長率方向大致一致。**

  (b) **沒有流動資產／流動負債** → 無法算流動比（上一批已揭露）。

  (c) **沒有所得稅費用** → 無法算稅後 ROIC（NOPAT）。
      我用「營業利益 ÷ 投入資本」＝**稅前 ROIC**。
      **偏誤方向：稅前 ROIC 約為稅後 1.27 倍（以 21% 稅率計），
      所以 10% 的門檻實際上只要求稅後約 7.9%——比 Phil Town 的原意寬鬆。**
      投入資本以「股東權益 + 長期負債」近似。

  (d) 沒有分析師預估 → 兩套方法都不用預估值，故不受影響（這點比上一批好）。

## 測法（與上一批一致，可直接比較）

年度再平衡、當時點財報（只用 filed_date 已公布的）、
對照組為「全體等權」與「隨機抽同樣檔數」。
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
EQ_TAGS = ('StockholdersEquity',)
OI_TAGS = ('OperatingIncomeLoss',)
ASSET_TAGS = ('Assets',)
LTD_TAGS = ('LongTermDebtNoncurrent',)
SH_TAGS = ('CommonStockSharesOutstanding', 'EntityCommonStockSharesOutstanding')
LIAB_TAGS = ('Liabilities',)


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
    """抽年報序列。★ 必須處理 XBRL 的兩種事實型態：

      期間值（duration）——營收、淨利、營業現金流、EPS：有 period_start，
          取期間 300~430 天者。
      時點值（instant）——股東權益、總資產、長期負債、流通股數：**沒有 period_start**，
          取 form 為 10-K 者（避免季末雜訊）。

    ★ 這是本輪的實質錯誤：原本只處理期間值，把所有時點值的資產負債表科目
      全部當成「沒有有效年報」剔除，導致兩套方法完全無法計算（0 筆可用）。
    """
    x = d[d['tag'].isin(tags)].copy()
    if x.empty:
        return None
    for c in ('period_end', 'period_start', 'filed_date'):
        x[c] = pd.to_datetime(x[c], errors='coerce')
    x = x.dropna(subset=['period_end', 'filed_date', 'value'])
    if x.empty:
        return None

    has_ps = x['period_start'].notna()
    parts = []

    dur = x[has_ps].copy()
    if len(dur):
        dur['d'] = (dur['period_end'] - dur['period_start']).dt.days
        dur = dur[(dur['d'] >= 300) & (dur['d'] <= 430)]
        if len(dur):
            parts.append(dur)

    inst = x[~has_ps].copy()
    if len(inst):
        if 'form' in inst.columns:
            inst = inst[inst['form'].astype(str).str.startswith('10-K')]
        if len(inst):
            parts.append(inst)

    if not parts:
        return None
    y = pd.concat(parts)
    if y.empty:
        return None
    y = y.sort_values('filed_date').drop_duplicates(subset=['period_end'], keep='first')
    return y.sort_values('period_end')[['period_end', 'filed_date', 'value']]


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
        for key, tags in (('rev', REV_TAGS), ('ni', NI_TAGS), ('eps', EPS_TAGS),
                          ('ocf', OCF_TAGS), ('eq', EQ_TAGS), ('oi', OI_TAGS),
                          ('assets', ASSET_TAGS), ('ltd', LTD_TAGS),
                          ('sh', SH_TAGS), ('liab', LIAB_TAGS)):
            s = annual_series(d, tags)
            if s is not None and len(s) >= 2:
                rec[key] = s
        if rec:
            out[sym] = rec
    return out


def vis(series, asof):
    if series is None:
        return None
    v = series[series['filed_date'] <= asof]
    return v if len(v) else None


def growth(series, asof, years):
    """回傳 (年化成長率, 實際年距, 最新值)。以天數判斷，避免閏年離一天誤差。"""
    v = vis(series, asof)
    if v is None or len(v) < 2:
        return (np.nan, np.nan, np.nan)
    latest = v.iloc[-1]
    target = latest['period_end'] - pd.Timedelta(days=int(365.25 * years))
    cand = v[v['period_end'] <= target]
    if cand.empty:
        return (np.nan, np.nan, float(latest['value']))
    base = cand.iloc[-1]
    days = (latest['period_end'] - base['period_end']).days
    span = days / 365.25
    if days < 365.25 * years - 20:          # 留 20 天寬容，避免閏年差一天
        return (np.nan, span, float(latest['value']))
    b, l = float(base['value']), float(latest['value'])
    if b <= 0 or l <= 0:
        return (np.nan, span, l)
    return ((l / b) ** (1.0 / span) - 1.0, span, l)


def avg_ratio(series, asof, denom_series, years):
    """多年平均比率：sum(分子) / sum(分母)，用最近 years 個年度。"""
    a = vis(series, asof); b = vis(denom_series, asof)
    if a is None or b is None or len(a) < 2 or len(b) < 2:
        return np.nan
    a = a.tail(years); b = b.tail(years)
    m = pd.merge(a[['period_end', 'value']], b[['period_end', 'value']],
                 on='period_end', suffixes=('_a', '_b'))
    if len(m) < 2:
        return np.nan
    da, db = m['value_a'].sum(), m['value_b'].sum()
    if db <= 0:
        return np.nan
    return float(da / db)


def last_val(series, asof):
    v = vis(series, asof)
    if v is None or len(v) == 0:
        return np.nan
    return float(v.iloc[-1]['value'])


def price_at(s, asof):
    if s is None:
        return np.nan
    x = s[s.index <= asof]
    return float(x.iloc[-1]) if len(x) else np.nan


def fwd(series, asof, days=365):
    if series is None:
        return np.nan
    s = series[series.index >= asof]
    if len(s) < 2:
        return np.nan
    p0 = float(s.iloc[0])
    tgt = s.index[0] + pd.Timedelta(days=days)
    s2 = series[series.index >= tgt]
    if len(s2) == 0:
        return np.nan
    return float(s2.iloc[0]) / p0 - 1.0


def compute(sym, asof, px, fund):
    f = fund.get(sym); p = px.get(sym)
    if f is None or p is None:
        return None
    price = price_at(p, asof)
    if not np.isfinite(price):
        return None

    # --- 成長率（10 年為主，不足則記錄實際年距）---
    rev10, rev_span, rev_last = growth(f.get('rev'), asof, 10)
    eps10, eps_span, eps_last = growth(f.get('eps'), asof, 10)
    eq10, eq_span, eq_last = growth(f.get('eq'), asof, 10)
    ocf10, ocf_span, ocf_last = growth(f.get('ocf'), asof, 10)

    # --- ROIC（稅前，近似）---
    oi_sum = np.nan; ic = np.nan
    oi = vis(f.get('oi'), asof); eq = vis(f.get('eq'), asof)
    if oi is not None and eq is not None:
        oi_sum = float(oi.tail(10)['value'].sum())
        eq_last_v = float(eq.iloc[-1]['value'])
        ltd = last_val(f.get('ltd'), asof)
        ltd = ltd if np.isfinite(ltd) else 0.0
        ic = eq_last_v + ltd
    roic_pre = (oi_sum / 10.0) / ic if (np.isfinite(oi_sum) and np.isfinite(ic) and ic > 0) else np.nan

    # --- 5 年 PE（價格 ÷ 5 年 EPS 總和）---
    e = vis(f.get('eps'), asof)
    pe5 = np.nan
    if e is not None and len(e) >= 3:
        s5 = float(e.tail(5)['value'].sum())
        if s5 > 0:
            pe5 = price / s5

    # --- 5 年 ROIC（Everything Money：5 年現金流 ÷（負債＋權益））---
    ocf5sum = np.nan
    o = vis(f.get('ocf'), asof)
    if o is not None and len(o) >= 3:
        ocf5sum = float(o.tail(5)['value'].sum())
    debt_eq = np.nan
    liab = last_val(f.get('liab'), asof); eqv = last_val(f.get('eq'), asof)
    if np.isfinite(liab) and np.isfinite(eqv):
        debt_eq = liab + eqv
    roic5 = (ocf5sum / debt_eq) if (np.isfinite(ocf5sum) and np.isfinite(debt_eq)
                                    and debt_eq > 0) else np.nan

    # --- 5 年成長（簡單比較今昔）---
    def chg5(series):
        v = vis(series, asof)
        if v is None:
            return np.nan
        latest = v.iloc[-1]
        target = latest['period_end'] - pd.Timedelta(days=int(365.25 * 5))
        cand = v[v['period_end'] <= target]
        if cand.empty:
            return np.nan
        b = float(cand.iloc[-1]['value']); l = float(latest['value'])
        if b <= 0:
            return np.nan
        return l / b - 1.0
    rev_c5 = chg5(f.get('rev'))
    ni_c5 = chg5(f.get('ni'))
    ocf_c5 = chg5(f.get('ocf'))

    # --- 流通股數變化（5 年）---
    sh_chg = chg5(f.get('sh'))

    # --- 負債 /（5 年 OCF 年均）---
    ltd_now = last_val(f.get('ltd'), asof)
    debt_fcf = np.nan
    if np.isfinite(ltd_now) and np.isfinite(ocf5sum) and ocf5sum > 0:
        debt_fcf = ltd_now / (ocf5sum / 5.0)

    # --- 市值 / 5 年 OCF ---
    shares = last_val(f.get('sh'), asof)
    mcap = price * shares if (np.isfinite(shares) and shares > 0) else np.nan
    mcap_ocf = (mcap / ocf5sum) if (np.isfinite(mcap) and np.isfinite(ocf5sum)
                                    and ocf5sum > 0) else np.nan

    # ============ Phil Town Big Five ============
    th = 0.10
    ok_pt = {
        'pt1_roic': bool(np.isfinite(roic_pre) and roic_pre >= th),
        'pt2_sales': bool(np.isfinite(rev10) and rev10 >= th),
        'pt3_eps': bool(np.isfinite(eps10) and eps10 >= th),
        'pt4_equity': bool(np.isfinite(eq10) and eq10 >= th),
        'pt5_fcf': bool(np.isfinite(ocf10) and ocf10 >= th),
    }
    # 負債測試為附加條件（他說是 follow-up，非必要）
    ok_pt['pt6_debt'] = bool(np.isfinite(debt_fcf) and debt_fcf <= 3.0)

    have_pt = sum(np.isfinite(x) for x in (roic_pre, rev10, eps10, eq10, ocf10))
    pt_data_ok = have_pt >= 4

    # ============ Everything Money 8 支柱 ============
    ok_em = {
        'em1_pe5': bool(np.isfinite(pe5) and pe5 < 22.5),
        'em2_roic5': bool(np.isfinite(roic5) and roic5 > 0.09),
        'em3_rev': bool(np.isfinite(rev_c5) and rev_c5 > 0),
        'em4_ni': bool(np.isfinite(ni_c5) and ni_c5 > 0),
        'em5_shares': bool(np.isfinite(sh_chg) and sh_chg <= 0.02),
        'em6_debt': bool(np.isfinite(debt_fcf) and debt_fcf < 5.0),
        'em7_ocf': bool(np.isfinite(ocf_c5) and ocf_c5 > 0),
        'em8_mcap': bool(np.isfinite(mcap_ocf) and mcap_ocf < 20.0),
    }
    have_em = sum(np.isfinite(x) for x in (pe5, roic5, rev_c5, ni_c5, sh_chg,
                                           debt_fcf, ocf_c5, mcap_ocf))
    em_data_ok = have_em >= 7

    row = dict(symbol=sym, asof=asof, price=price,
               roic_pre=roic_pre, rev10=rev10, eps10=eps10, eq10=eq10, ocf10=ocf10,
               pe5=pe5, roic5=roic5, rev_c5=rev_c5, ni_c5=ni_c5, sh_chg=sh_chg,
               debt_fcf=debt_fcf, ocf_c5=ocf_c5, mcap_ocf=mcap_ocf,
               pt_data_ok=pt_data_ok, em_data_ok=em_data_ok,
               pt_pass=sum(ok_pt.values()), em_pass=sum(ok_em.values()))
    row.update(ok_pt); row.update(ok_em)
    return row


def stats_for(A, flag, label, seed=7):
    """對某個通過旗標算年度組合表現（含隨機同檔數對照）。"""
    yearly = []
    for D, g in A.groupby('asof'):
        sel = g[g[flag]]
        k = len(sel)
        uni = g['fwd12'].to_numpy()
        if k == 0 or len(uni) < k:
            continue
        r = float(sel['fwd12'].mean())
        rng = np.random.default_rng(seed)
        draws = np.array([rng.choice(uni, k, replace=False).mean() for _ in range(500)])
        yearly.append(dict(asof=str(D.date()), k=k, ret=r,
                           univ=float(uni.mean()),
                           rand=float(draws.mean()),
                           pct=float((draws < r).mean())))
    if not yearly:
        return dict(label=label, n_years=0)
    Y = pd.DataFrame(yearly)
    ex = (Y['ret'] - Y['rand']).to_numpy()
    t = ex.mean() / (ex.std(ddof=1) / np.sqrt(len(ex))) if len(ex) > 2 else np.nan
    return dict(label=label, n_years=int(len(Y)),
                mean_ret=float(Y['ret'].mean()),
                mean_univ=float(Y['univ'].mean()),
                mean_rand=float(Y['rand'].mean()),
                excess_rand=float(ex.mean()),
                excess_univ=float((Y['ret'] - Y['univ']).mean()),
                beat_rand=int((ex > 0).sum()),
                t=float(t),
                avg_k=float(Y['k'].mean()),
                total_picks=int(Y['k'].sum()),
                median_pct=float(Y['pct'].median()),
                yearly=yearly)


def per_flag(A, flags):
    out = {}
    base = A.groupby('asof')['fwd12'].mean()
    for f in flags:
        sub = A[A[f]]
        if len(sub) < 50:
            out[f] = dict(n=int(len(sub)), note='樣本過少')
            continue
        y = sub.groupby('asof')['fwd12'].mean()
        b = pd.concat([y.rename('s'), base.rename('b')], axis=1).dropna()
        out[f] = dict(n=int(len(sub)), n_years=int(len(b)),
                      mean=float(sub['fwd12'].mean()),
                      excess=float((b.s - b.b).mean()),
                      beat=int((b.s > b.b).sum()),
                      pos=float((sub['fwd12'] > 0).mean()))
    return out


def main():
    print('載入價格...', flush=True)
    px = load_prices()
    print(f'  {len(px)} 檔', flush=True)
    print('載入財報...', flush=True)
    fund = load_fundamentals()
    print(f'  {len(fund)} 檔', flush=True)
    syms = sorted(set(px) & set(fund))
    print(f'  交集 {len(syms)} 檔', flush=True)

    rebals = [pd.Timestamp(f'{y}-06-30') for y in range(2011, 2026)]
    rows = []
    for D in rebals:
        n = 0
        for sym in syms:
            r = compute(sym, D, px, fund)
            if r is None:
                continue
            r['fwd12'] = fwd(px.get(sym), D, 365)
            if not np.isfinite(r['fwd12']):
                continue
            rows.append(r)
            n += 1
        print(f'  {D.date()}  可用 {n} 檔', flush=True)

    A_all = pd.DataFrame(rows)
    PT = ['pt1_roic', 'pt2_sales', 'pt3_eps', 'pt4_equity', 'pt5_fcf']
    EM = ['em1_pe5', 'em2_roic5', 'em3_rev', 'em4_ni', 'em5_shares',
          'em6_debt', 'em7_ocf', 'em8_mcap']

    A = A_all[A_all['pt_data_ok'] | A_all['em_data_ok']].copy()
    A['pt_all'] = A[PT].all(axis=1) & A['pt_data_ok']
    A['em_all'] = A[EM].all(axis=1) & A['em_data_ok']
    # 兩套都通過
    A['both_all'] = A['pt_all'] & A['em_all']

    print(f'\n總觀測 {len(A_all):,}　資料可用 {len(A):,}', flush=True)

    res = dict(
        meta=dict(n_stocks=len(syms), n_obs=int(len(A)),
                  n_obs_all=int(len(A_all)),
                  years=[str(x.date()) for x in rebals]),
        phil_town=stats_for(A, 'pt_all', 'Phil Town Big Five（5 條全過）'),
        everything_money=stats_for(A, 'em_all', 'Everything Money 8 支柱（8 條全過）'),
        both=stats_for(A, 'both_all', '兩套都通過'),
        pt_flags=per_flag(A, PT),
        em_flags=per_flag(A, EM),
        pt_by_npass={int(k): dict(n=int(len(g)), mean=float(g['fwd12'].mean()),
                                  pos=float((g['fwd12'] > 0).mean()))
                     for k, g in A[A['pt_data_ok']].groupby('pt_pass') if len(g) >= 30},
        em_by_npass={int(k): dict(n=int(len(g)), mean=float(g['fwd12'].mean()),
                                  pos=float((g['fwd12'] > 0).mean()))
                     for k, g in A[A['em_data_ok']].groupby('em_pass') if len(g) >= 30},
    )
    json.dump(res, open(f'{OUT}/value_screen.json', 'w'), ensure_ascii=False,
              indent=1, default=str)

    for key in ('phil_town', 'everything_money', 'both'):
        s = res[key]
        print('\n' + '=' * 72)
        print(f"【{s['label']}】")
        if s['n_years'] == 0:
            print('  無有效年份'); continue
        print(f"  有效年份 {s['n_years']}，平均選出 {s['avg_k']:.1f} 檔，"
              f"共 {s['total_picks']} 個選中觀測")
        print(f"  組合報酬 {s['mean_ret']*100:+.2f}%　全體 {s['mean_univ']*100:+.2f}%　"
              f"隨機 {s['mean_rand']*100:+.2f}%")
        print(f"  超額 vs 隨機 {s['excess_rand']*100:+.2f}pp "
              f"(t={s['t']:+.2f}, 勝 {s['beat_rand']}/{s['n_years']})　"
              f"vs 全體 {s['excess_univ']*100:+.2f}pp")
        print(f"  中位百分位 {s['median_pct']*100:.0f}%")

    for key, label in (('pt_flags', 'Phil Town 各條'), ('em_flags', 'Everything Money 各柱')):
        print(f'\n【{label}單獨使用】')
        for f, v in res[key].items():
            if 'note' in v:
                print(f'  {f:<16} n={v["n"]:>6}  ({v["note"]})')
            else:
                print(f'  {f:<16} n={v["n"]:>6,}  平均 {v["mean"]*100:>+7.2f}%  '
                      f'超額 {v["excess"]*100:>+6.2f}pp  勝 {v["beat"]}/{v["n_years"]}')

    print('\n已寫 value_screen.json')


if __name__ == '__main__':
    main()
