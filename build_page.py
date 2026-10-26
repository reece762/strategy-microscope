#!/usr/bin/env python3
"""Build the dark, card-grid results page (QuantPal style) from the backtest output."""
import base64, io, json, os, shutil, time
from html import escape as html_escape
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

from detail_tpl import DETAIL_TEMPLATE, CHART_BLOCK, CHART_CSS, CODE_BLOCK, CODE_CSS
from code_snippets import snippet as code_snippet

plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 110

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')
BG = '#0d1117'

# --- version tracking -------------------------------------------------------
# Bump VERSION on every change to copy, data or layout. Each build writes
# build_meta.json plus a numbered snapshot in versions/, so any earlier build
# can be restored by copying versions/v<N>/ back over the served files.
VERSION = '1.21.0'
VERSION_NOTES = ('**新增出場系列五張卡（J Law）——這個系列的題型與前三張不同。**\n'
                 '前三張問「進場後賺不賺」，需要市場基準；而基準一換結論就變號。'
                 '這五張問「同一批持股，出現訊號後是否比沒出現更差」——**純相對比較，不需基準**，'
                 '所以不受上輪基準爭議的影響。對照組是同期的上升趨勢日（20 日 +0.93%）。\n'
                 '一、見頂（D）：抛物線中出現不尋常下跌 2,929 次，'
                 '之後 20 日 +8.03%，高於基準 7.10pp。但這是存活者偏差造成的假象：'
                 '訊號後跌逾 20% 的機率 11.0%（基準 1.5%，八倍），而崩盤下市者不在母體。'
                 '唯一方向站得住的是「波幅擴張」訊號（+0.40%，低於基準 0.53pp）。\n'
                 '二、延伸（B）：他說「延伸不是罪過」，實測 20,941 個延伸日，'
                 '之後 20 日 +2.50%，高於基準 1.56pp（t=+3.90）。'
                 '**五支裡唯一支持的一支**，但勝率 53.2% 低於基準、尾部風險四倍。\n'
                 '三、轉弱（D）：「回檔 30~36% 仍算前半段」的 403 次樣本，'
                 '之後 20 日中位 −6.22%、上漲機率僅 37.4%、跌逾 20% 機率 25.0%——'
                 '全測試最差的一組，與他的說法相反。\n'
                 '四、阻力位（D）：「VCP 收窄到阻力該抱、V 形急漲該賣」，'
                 '實測 VCP +1.05%、V 形 +5.19%，差值 −4.23%，'
                 '信賴區間 [−6.35, −2.17] 不含零，**與他的說法剛好相反**。'
                 '但 VCP 佔樣本 95%，且「整固品質」無法程式化。\n'
                 '五、大市拖累（C）：670 個大盤跌逾 1% 的日子，'
                 '強勢股之後 20 日 +1.39%、弱勢股 +2.94%，弱勢股多賺 1.55pp（t=−2.05）。'
                 '「強勢股會被拖累」成立，「拖累後較快恢復」不成立。\n'
                 '★ 本輪最重要的其實是一個方法修正：'
                 '第一輪把個股報酬減去同日橫斷面平均再比兩組，'
                 't 值膨脹到 −75 這種不可能的量級——因為同一天強勢與弱勢股的去均值報酬'
                 '加權後必然為零，差異有一部分是算術上被強制的。'
                 '改成「每個交易日只算一個觀測值」後，t 值回到合理量級。')
VERSION_PREV = ('v1.20.0 修正圖表標籤（「買進持有」實際上是每日重平衡），'
                '並補上隨機進場必須為零的自我檢驗。')
BUILD_AT = os.environ.get('XBX_BUILD_AT') or time.strftime('%Y-%m-%d %H:%M:%S%z')


def b64(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor=BG, bbox_inches='tight',
                pad_inches=0.15, dpi=dpi)
    plt.close(fig)
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def style(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors='#8b949e', labelsize=8)
    for s in ax.spines.values():
        s.set_color('#30363d')
    ax.grid(alpha=.18, color='#8b949e', lw=.5)


def mini_chart(lines, title, variant='card'):
    """P&L chart. Two renders of the same series so the detail page never
    upscales the card's thumbnail: 'card' is compact but still ~2.2x the pixels
    it occupies on screen; 'detail' is rendered for the wider detail column."""
    if variant == 'detail':
        fig, ax = plt.subplots(figsize=(7.6, 3.05))
        fs_title, fs_tick, fs_leg, lw = 11.5, 9.5, 9.0, 1.7
        dpi = 165
    else:
        fig, ax = plt.subplots(figsize=(5.35, 2.0))
        fs_title, fs_tick, fs_leg, lw = 9.2, 8.0, 7.4, 1.5
        dpi = 165
    for idx, vals, color, label in lines:
        ax.plot(pd.to_datetime(idx), vals, lw=lw, color=color, label=label)
    ax.set_yscale('log')
    ax.set_title(title, color='#c9d1d9', fontsize=fs_title, pad=7, loc='left')
    ax.tick_params(colors='#8b949e', labelsize=fs_tick, length=3)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:g}x'))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel('淨值（倍，對數軸）', color='#8b949e', fontsize=fs_tick)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(alpha=.14, color='#8b949e', lw=.4)
    ax.margins(x=.01)
    ax.legend(fontsize=fs_leg, facecolor='none', edgecolor='none',
              labelcolor='#9aa7b4', loc='upper left', ncol=len(lines),
              handlelength=1.2, columnspacing=1.0, borderpad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor='#0a0e14', bbox_inches='tight',
                pad_inches=0.14, dpi=dpi)
    plt.close(fig)
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def curve_chart(cv, title, fname=''):
    idx = pd.to_datetime(cv['idx'])
    fig, ax = plt.subplots(figsize=(9.4, 3.7))
    ax.plot(idx, cv['bh'], lw=1.1, color='#6e7681', label='買進持有')
    ax.plot(idx, cv['skip'], lw=1.2, color='#58a6ff', label='跳過週四')
    ax.plot(idx, cv['regime'], lw=1.5, color='#f85149', label='均線×星期')
    ax.set_yscale('log')
    ax.set_title(title, color='#e6edf3', fontsize=10, pad=8)
    ax.set_ylabel('淨值（對數）', color='#8b949e', fontsize=8)
    lg = ax.legend(fontsize=7.5, facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
    style(ax)
    return b64(fig)


def weekday_chart(rows, title):
    labels = [r['day'] for r in rows]
    vals = [r['mean_pct'] for r in rows]
    ts = [r['t_stat'] for r in rows]
    fig, ax = plt.subplots(figsize=(9.4, 3.5))
    cols = ['#f85149' if v < 0 else ('#3fb950' if abs(t) >= 2 else '#58a6ff')
            for v, t in zip(vals, ts)]
    bars = ax.bar(labels, vals, color=cols, width=.62)
    ax.axhline(0, color='#8b949e', lw=.9)
    for b, v, t in zip(bars, vals, ts):
        ax.text(b.get_x() + b.get_width() / 2, v, f'{v:+.3f}%\nt={t:+.1f}',
                ha='center', va='bottom' if v >= 0 else 'top',
                fontsize=6.8, color='#c9d1d9')
    ax.set_title(title, color='#e6edf3', fontsize=10, pad=8)
    ax.set_ylabel('平均日報酬 (%)', color='#8b949e', fontsize=8)
    ax.margins(y=.22)
    style(ax)
    return b64(fig)


def cal_chart(rows, title):
    wins = ['4月10日', '8月31日', '10月30日', '12月10日']
    hs = [1, 5, 10, 20]
    M = np.full((len(wins), len(hs)), np.nan)
    for r in rows:
        if r['window'] in wins and r['horizon'] in hs:
            M[wins.index(r['window']), hs.index(r['horizon'])] = r['mean_pct']
    fig, ax = plt.subplots(figsize=(9.4, 3.3))
    from matplotlib.colors import TwoSlopeNorm
    vmax = np.nanmax(np.abs(M)) or 1
    im = ax.imshow(M, cmap='RdYlGn', norm=TwoSlopeNorm(0, -vmax, vmax), aspect='auto')
    ax.set_xticks(range(len(hs)), [f'+{h}日' for h in hs], color='#8b949e', fontsize=8)
    ax.set_yticks(range(len(wins)), wins, color='#8b949e', fontsize=8)
    for i in range(len(wins)):
        for j in range(len(hs)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f'{M[i,j]:+.2f}', ha='center', va='center',
                        fontsize=8, color='#0d1117', fontweight='bold')
    ax.set_title(title, color='#e6edf3', fontsize=10, pad=8)
    for s in ax.spines.values():
        s.set_color('#30363d')
    ax.tick_params(colors='#8b949e')
    return b64(fig)


def trim_chart():
    """How much of BTC's Thursday effect is a handful of days."""
    ks = [0, 1, 2, 3, 5, 8, 10]
    vals = [-0.2281, -0.1454, -0.1125, -0.0759, -0.0199, 0.0332, 0.0934]
    fig, ax = plt.subplots(figsize=(9.4, 3.2))
    ax.plot(ks, vals, 'o-', color='#f85149', lw=1.8, ms=5)
    ax.axhline(0, color='#8b949e', lw=.9)
    ax.fill_between(ks, vals, 0, color='#f85149', alpha=.14)
    ax.set_xlabel('剔除最差的 N 個週四（共 476 天）', color='#8b949e', fontsize=8)
    ax.set_ylabel('週四平均日報酬 (%)', color='#8b949e', fontsize=8)
    ax.set_title('BTC 的「週四下跌」只靠 5 天撐著', color='#e6edf3', fontsize=10, pad=8)
    ax.annotate('剩下 471 天的平均值\n已經回到零', xy=(5, -0.0199), xytext=(4.2, -0.16),
                fontsize=7.5, color='#c9d1d9',
                arrowprops=dict(arrowstyle='->', color='#8b949e', lw=.9))
    style(ax)
    return b64(fig)


def exposure_chart(em):
    fig, ax = plt.subplots(figsize=(9.6, 3.6))
    items = [f"{r['market']}\n{r['strategy']}" for _, r in em.iterrows()]
    x = np.arange(len(em))
    ax.bar(x - .2, em['cagr'] * 100, .4, color='#f85149', label='策略')
    ax.bar(x + .2, em['em_cagr'] * 100, .4, color='#58a6ff', label='同曝險、不擇時')
    ax.set_xticks(x, items, fontsize=6.6, color='#8b949e')
    ax.set_ylabel('年化報酬 (%)', color='#8b949e', fontsize=8)
    ax.set_title('同樣的曝險下，策略是否真的更好？', color='#e6edf3', fontsize=10, pad=8)
    lg = ax.legend(fontsize=7.5, facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
    style(ax)
    return b64(fig)


def yearly_chart(yr):
    fig, ax = plt.subplots(figsize=(9.4, 3.0))
    x = np.arange(len(yr))
    ax.bar(x - .2, yr['strat'] * 100, .4, color='#f85149', label='跳過週四')
    ax.bar(x + .2, yr['bh'] * 100, .4, color='#6e7681', label='買進持有')
    ax.set_xticks(x, yr['year'], fontsize=7.5, color='#8b949e')
    ax.axhline(0, color='#8b949e', lw=.9)
    ax.set_ylabel('年度報酬 (%)', color='#8b949e', fontsize=8)
    ax.set_title('BTC 逐年：9 年中有 8 年勝出，但優勢集中在 2019–2023', color='#e6edf3', fontsize=9.5, pad=8)
    lg = ax.legend(fontsize=7.5, facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
    style(ax)
    return b64(fig)



def png_uri(name):
    """tier1_charts.py writes PNG files; the rest of the pipeline builds figures
    in memory. Read the file and inline it so the page stays a single file."""
    path = os.path.join(OUT, name)
    with open(path, 'rb') as f:
        return 'data:image/png;base64,' + base64.b64encode(f.read()).decode()


def card_kind(c):
    """這張卡片是哪一類：backtest（可回測）／mindset（心法）／unportable（無法移植）。

    排序與篩選都吃這一個函式，避免兩處各寫一套推導而走鐘。
    未明示 kind 的卡片沿用舊推導：NA → unportable，其餘 → backtest。
    """
    return c.get('kind') or ('backtest' if c['grade'] != 'NA' else 'unportable')


def main():
    global MINI, imgs_map
    res = json.load(open(os.path.join(OUT, 'results.json')))
    em = pd.read_csv(os.path.join(OUT, 'exposure_matched.csv'))
    yr = pd.read_csv(os.path.join(OUT, 'btc_by_year.csv'))
    extra = json.load(open(os.path.join(OUT, 'extra.json')))

    imgs = {
        'nasdaq': curve_chart(res['curves']['納斯達克綜合指數（1971 起）'], '納斯達克綜合指數 1971–2026（55.7 年）'),
        'spy': curve_chart(res['curves']['SPY（2018 起）'], 'SPY 2016–2026'),
        'qqq': curve_chart(res['curves']['QQQ（2018 起）'], 'QQQ 2016–2026'),
        'btc': curve_chart(res['curves']['BTC（2017 起）'], 'BTC/USDT 2017–2026'),
        'wd_nas': weekday_chart([r for r in res['weekday'] if r['label'] == '納斯達克 1971-2026'], '納斯達克：各星期平均日報酬（1971–2026）'),
        'wd_nas_old': weekday_chart([r for r in res['weekday'] if r['label'] == '納斯達克 1971-1999'], '納斯達克 1971–1999：週一顯著為負'),
        'wd_nas_new': weekday_chart([r for r in res['weekday'] if r['label'] == '納斯達克 2000-2026'], '納斯達克 2000–2026：週一效應消失'),
        'wd_spy': weekday_chart([r for r in res['weekday'] if r['label'] == 'SPY 2018-2025'], 'SPY 2018–2025'),
        'wd_btc': weekday_chart(res['btc_weekday'], 'BTC 2017–2026'),
        'cal': cal_chart(res['calendar_nasdaq'], '納斯達克：四大窗口後的平均報酬（55 年）'),
        'trim': trim_chart(),
        'em': exposure_chart(em),
        'yr': yearly_chart(yr),
    }
    # 第一梯（新做的七支）圖表已由 tier1_charts.py 存成 PNG
    _t1_path = os.path.join(OUT, 't1_charts.json')
    if os.path.exists(_t1_path):
        for key, fname in json.load(open(_t1_path)).items():
            imgs['t1_' + key] = png_uri(fname)
    _t2_path = os.path.join(OUT, 't2_charts.json')
    if os.path.exists(_t2_path):
        for key, fname in json.load(open(_t2_path)).items():
            imgs['t2_' + key] = png_uri(fname)
    # 第四梯（本輪新轉錄的 13 支）圖表，key 本身已帶 n4_ 前綴
    _n4_path = os.path.join(OUT, 'n4_charts.json')
    if os.path.exists(_n4_path):
        for key, fname in json.load(open(_n4_path)).items():
            imgs[key if key.startswith('n4_') else 'n4_' + key] = png_uri(fname)
    # 阿尔法量化（GitHub 開源碼驗證）的圖表，key 沒有前綴，統一加 al_
    _al_path = os.path.join(OUT, 'alpha_charts.json')
    if os.path.exists(_al_path):
        for key, fname in json.load(open(_al_path)).items():
            imgs[key if key.startswith('al_') else 'al_' + key] = png_uri(fname)

    # 老魯（蝶式、江恩九方圖）的圖表，key 已帶 lao_ 前綴
    _lao_path = os.path.join(OUT, 'lao_charts.json')
    if os.path.exists(_lao_path):
        for key, fname in json.load(open(_lao_path)).items():
            imgs[key if key.startswith('lao_') else 'lao_' + key] = png_uri(fname)

    # RTA 交易之路（趨勢判斷）的圖表，key 已帶 rta_ 前綴
    _rta_path = os.path.join(OUT, 'rta_charts.json')
    if os.path.exists(_rta_path):
        for key, fname in json.load(open(_rta_path)).items():
            imgs[key if key.startswith('rta_') else 'rta_' + key] = png_uri(fname)

    # 本輪六張卡（Alpha101／PEAD／春哥／兩張 Jack／頻道調查）的圖表。
    # new5_cards.json 同時存了計算結果與 PNG 檔名，卡片模組讀同一份，
    # 避免「圖上的數字」與「卡片上的數字」各自手打而對不上。
    _n5_path = os.path.join(OUT, 'new5_cards.json')
    if os.path.exists(_n5_path):
        _n5 = json.load(open(_n5_path))
        for key, fname in _n5.get('charts', {}).items():
            imgs['n5_' + key] = png_uri(fname)

    # J Law VCP 那張卡的圖，key 加 vcp_ 前綴（卡片裡 imgs 也寫 vcp_xxx）
    _vcp_path = os.path.join(OUT, 'vcp_card.json')
    if os.path.exists(_vcp_path):
        _vcp = json.load(open(_vcp_path))
        for key, fname in _vcp.get('charts', {}).items():
            imgs['vcp_' + key] = png_uri(fname)

    # 本輪三張（趨勢線、平行通道、恐慌進場）的圖，key 已帶 n6_ 前綴
    _n6_path = os.path.join(OUT, 'new6_cards.json')
    if os.path.exists(_n6_path):
        _n6 = json.load(open(_n6_path))
        for key, fname in _n6.get('charts', {}).items():
            imgs['n6_' + key] = png_uri(fname)

    # 出場系列五張的圖，key 已帶 n7_ 前綴
    _ex_path = os.path.join(OUT, 'exit_all.json')
    if os.path.exists(_ex_path):
        _ex = json.load(open(_ex_path))
        for key, fname in _ex.get('charts', {}).items():
            imgs['n7_' + key] = png_uri(fname)

    # --- comparison charts: one strategy + the same three benchmarks ---
    # spec comes from prep_series.py, already aligned and rebased to 1.0
    series = json.load(open(os.path.join(OUT, 'series.json')))
    CARDS_SERIES = series['cards']

    # 追加：先前沒有進 series.json、但確實做過機械策略回測的卡片。
    # tier1_curves.py 用與 tier1.py 完全相同的規則重算，並斷言年化與 tier1.json 一致。
    for fname in ('tier1_curves.json', 'new4_curves.json', 'tier2_curves.json',
                  'alpha_curves.json', 'rta_curves.json'):
        p = os.path.join(OUT, fname)
        if os.path.exists(p):
            extra_cards = json.load(open(p))
            CARDS_SERIES.update(extra_cards)
            print(f'  曲線追加 {fname}: {list(extra_cards)}')

    SPEC = {}
    for cid, card in CARDS_SERIES.items():
        lines = []
        for sr in card['series']:
            idx = [str((pd.Timestamp('1970-01-01') + pd.Timedelta(days=int(x))).date())
                   for x in card['d']]
            lines.append((idx, sr['v'], sr['color'], sr['label']))
        SPEC[cid] = (lines, '淨值（倍，對數軸）· 起始 = 1')

    MINI = {k: mini_chart(v[0], v[1], 'card') for k, v in SPEC.items()}

    imgs_map = {k: imgs[k] for k in imgs}
    pages, details = build_cards(res, em, yr, CARDS_SERIES)

    # 首頁統計與篩選器計數一律從實際卡片算，不手寫——手寫的數字每次加卡片都會變舊，
    # 讓頁面看起來「沒有更新」。
    import re as _re
    def _cnt(pat):
        return len(_re.findall(pat, pages))
    n_cards = _cnt(r'<article class="card"')
    n_na = _cnt(r'data-grade="NA"')
    n_grade = {g: _cnt(rf'data-grade="{g}"') for g in 'ABCD'}
    # 來源 chips 由實際卡片自動彙總——寫死單一作者時，新作者的卡片會分得到「新分類」，
    # 卻無法用「來源」篩選，頁面看起來只認識舊來源。
    _authors = {}
    for _a in _re.findall(r'data-author="([^"]+)"', pages):
        _authors[_a] = _authors.get(_a, 0) + 1
    AUTHOR_CHIPS = '\n'.join(
        f'  <div class="chip" data-facet="author" data-val="{_a}">{_a} ({_n})</div>'
        for _a, _n in sorted(_authors.items(), key=lambda kv: (-kv[1], kv[0])))
    counts = {
        '__N_CARDS__': str(n_cards),
        '__N_RUN__': str(n_cards - n_na),
        '__N_PASS__': str(_cnt(r'data-grade="[AB]"')),
        '__N_FAIL__': str(_cnt(r'data-grade="[CD]"')),
        '__N_NA__': str(n_na),
        '__AUTHOR_CHIPS__': AUTHOR_CHIPS,
        '__C_US__': str(_cnt(r'data-market="us"')),
        '__C_CRYPTO__': str(_cnt(r'data-market="crypto"')),
        '__C_BT__': str(_cnt(r'data-kind="backtest"')),
        '__C_MIND__': str(_cnt(r'data-kind="mindset"')),
        '__C_UNP__': str(_cnt(r'data-kind="unportable"')),
    }

    html = (TEMPLATE.replace('__CARDS__', pages)
                    .replace('__VERSION__', VERSION)
                    .replace('__BUILD_AT__', BUILD_AT))
    for k, v in counts.items():
        html = html.replace(k, v)
    path = os.path.join(HERE, 'page.html')
    open(path, 'w').write(html)
    # index.html is what the server serves; keep it in lockstep with page.html
    shutil.copy2(path, os.path.join(HERE, 'index.html'))
    print('written', path, 'and index.html', len(html), 'bytes')

    for fn, body in details.items():
        open(os.path.join(HERE, fn), 'w').write(body)
    print('written', len(details), 'detail pages:',
          ', '.join(sorted(details))[:200])

    write_about(n_cards, n_grade, n_na)

    snapshot_and_changelog(details.keys())


def video_url(vid):
    """B 站 BV 號給 bilibili，YouTube 11 碼 ID 給 youtu.be。

    這批影片有兩種來源：早期批次是 B 站（BV 開頭），新轉錄的是 YouTube（11 碼）。
    之前一律拼 bilibili，導致 YouTube ID 變成壞連結。
    """
    if vid.startswith('BV'):
        return 'https://www.bilibili.com/video/' + vid
    return 'https://youtu.be/' + vid


def video_links(c):
    """回傳 [(vid, url)]；支援單一 video 與多支 videos。"""
    vids = c.get('videos') or ([c['video']] if c.get('video') else [])
    out = []
    for v in vids:
        if v and v not in [x[0] for x in out]:
            out.append((v, video_url(v)))
    return out


def build_cards(res, em, yr, CARDS_SERIES):
    def g(v):
        return f'{v*100:.2f}%'
    def gm(v):
        return f'{v*100:.1f}%'

    rows = []
    for _, r in em.iterrows():
        rows.append(r)
    E = {(r['market'], r['strategy']): r for r in rows}

    nas_bh = res['markets']['納斯達克綜合指數（1971 起）']['bh']
    spy = res['markets']['SPY（2018 起）']
    qqq = res['markets']['QQQ（2018 起）']
    btc = res['btc']

    cards = []

    # ---- 1. weekday effect US ----
    cards.append(dict(
        id='wd-us', author='邢不行', grade='C', topic='日曆效應', market='us',
        title='週內效應：美股真的有「黑色星期一」嗎',
        video='BV1si4y177RU',
        lead='他驗證 A 股「週四最差、週一最好」，並順帶說「美國是週一顯著為負、週五顯著為正」。這句話在 55 年資料裡只對了一半：1971–1999 完全正確，2000 年之後消失了。',
        metrics=[('週一 1971–1999', '−0.127%'), ('對應 t 值', '−4.47'), ('週一 2000–2026', '+0.005%')],
        body=[
            '**他說的沒錯，但那是一段已經結束的歷史。** 1971–1999 這 29 年，納斯達克週一平均 −0.127%（t=−4.47），週五 +0.160%（t=+7.46），跟他描述的「週一顯著為負、週五顯著為正」完全一致。',
            '**2000 年之後，週一變成 +0.005%（t=+0.12），效應歸零。** 這在學術上叫「週末效應衰減」，通常歸因於交易成本下降、ETF 普及、以及資訊傳遞速度變快——原本週末累積的壞消息，現在盤後就消化完了。',
            '**更關鍵的是：美股的週四不是最差的一天，反而是第三好的一天**（+0.078%，t=+3.27）。他整個策略建立在「跳過週四」，這個前提在美股不成立。近 10 年的 SPY/QQQ 裡週四確實是「最弱的一天」（+0.006% / +0.013%），但 t 值只有 0.12 / 0.20，統計上跟零分不出來。',
        ],
        imgs=['wd_nas_old', 'wd_nas_new', 'wd_spy'],
        verdict='效應曾經真實存在，但已經衰減到無法交易。把它當成 2026 年的策略依據是危險的。',
        evidence='納斯達克 14,046 個交易日；分割點 2000-01-01 為事前指定，非事後挑選。',
    ))

    # ---- 2. skip Thursday Nasdaq 55y ----
    n = E[('NASDAQ', '跳過週四')]
    cards.append(dict(
        id='skip-nas', author='邢不行', grade='D', topic='日曆效應', market='us',
        title='跳過週四：在 55 年美股上是有害的',
        video='BV1si4y177RU',
        lead='他宣稱在 A 股「每週四收盤買、下週三收盤賣」可以把年化從 11.46% 提到 15%、回撤從 72% 降到 65%。同一條規則搬到納斯達克 55 年，年化從 10.55% 掉到 5.62%，回撤反而惡化。',
        metrics=[('策略年化', g(n['cagr'])), ('買進持有', g(n['bh_cagr'])), ('回撤變化', '−78%→−82%')],
        body=[
            f"**直接照做：年化 {g(n['cagr'])}，買進持有 {g(n['bh_cagr'])}，淨值 21 倍 vs 267 倍。** 回撤從 −77.9% 惡化到 −82.3%（跳過週四等於少了 20% 的修復日）。",
            f"**但這個比較還不夠公平**：跳過週四只持倉 {n['exposure']*100:.1f}% 的時間。所以我又算了「同樣 {n['exposure']*100:.0f}% 曝險、完全不擇時」的基準：年化 {g(n['em_cagr'])}、回撤 −68.9%。**策略仍然輸，而且回撤更差。**",
            '**為什麼會這樣？** 因為在美股 55 年裡，週四的平均報酬是 **正的**（+0.078%）。跳過一個正報酬的日子，等於每週主動放棄一點收益。他的策略在 A 股有效，前提是 A 股週四真的為負（他實測 −0.10%）；這個前提在美股不存在。',
        ],
        imgs=['nasdaq', 'em'],
        verdict='不成立。在美股，這是「每週刻意漏掉一天正報酬」，長期年化少一半。',
        evidence=f"納斯達克 1971–2026，14,046 日；成本每趟 1bp。同曝險基準為每日再平衡的固定槓桿買進持有。",
    ))

    # ---- 3. skip Thursday SPY/QQQ ----
    s = E[('SPY', '跳過週四')]; q = E[('QQQ', '跳過週四')]
    cards.append(dict(
        id='skip-etf', author='邢不行', grade='B', topic='日曆效應', market='us',
        title='跳過週四：近 10 年反而有效，但幅度很小',
        video='BV1si4y177RU',
        lead='同一條規則在近 10 年的 SPY/QQQ 上確實改善了風險調整後報酬——這跟 55 年的結論相反，正好說明這類效應有多不穩定。',
        metrics=[('SPY 年化', g(s['cagr'])), ('QQQ 年化', g(q['cagr'])), ('SPY 回撤', gm(s['mdd']))],
        body=[
            f"**SPY**：年化 {g(s['cagr'])}（買進持有 {g(s['bh_cagr'])}），但回撤從 −34.1% 降到 −21.2%，Calmar 從 0.40 升到 0.59。**QQQ**：年化 {g(q['cagr'])}（買進持有 {g(q['bh_cagr'])}），回撤 −35.6% → −26.4%，Calmar 0.56 → 0.70。",
            f"**同曝險基準**：SPY 是 {g(s['em_cagr'])}、QQQ 是 {g(q['em_cagr'])}。策略確實勝出（Calmar 各 +0.19 / +0.15），所以這段期間的改善不是單純「少持倉」造成的。",
            '**但時間很短，而且方向不穩。** 同一個規則在 55 年樣本上是負貢獻，在最近 10 年是正貢獻。這種「換一段時間就翻轉」的結果，不足以支撐實盤。他的 A 股結論有 14 年樣本，我這裡的美股正面結果只有 10 年，而且只出現在「週四剛好是最弱一天」的這段時期。',
        ],
        imgs=['spy', 'qqq'],
        verdict='近 10 年小幅有效，但與 55 年結論相反。屬於「剛好這十年運氣對」，不是穩健的效應。',
        evidence='SPY/QQQ 2016-10 至 2026-10，各約 2,512 日；成本 1bp。',
    ))

    # ---- 4. MA x weekday ----
    nr = E[('NASDAQ', '均線×星期')]; sr = E[('SPY', '均線×星期')]; qr = E[('QQQ', '均線×星期')]
    cards.append(dict(
        id='regime', author='邢不行', grade='A', topic='擇時', market='us',
        title='均線×星期：他三個方法裡唯一站得住的',
        video='BV1si4y177RU',
        lead='他的第三個方法：收盤價在 20 日均線之上（牛市）就抱週一和週五、避開週二和週四；在均線之下（熊市）反過來。這是全部測試裡唯一在 55 年樣本與近 10 年樣本都勝過同曝險基準的策略。',
        metrics=[('QQQ Sharpe', f"{qr['sharpe']:.2f}"), ('QQQ 回撤', gm(qr['mdd'])), ('持倉比例', '40%')],
        body=[
            f"**近 10 年（最強的一組）**：QQQ 年化 {g(qr['cagr'])}、回撤 {gm(qr['mdd'])}、Sharpe **{qr['sharpe']:.2f}**（買進持有 0.92）、Calmar **{qr['calmar']:.2f}**（買進持有 0.56）。SPY 年化 {g(sr['cagr'])}、回撤 {gm(sr['mdd'])}、Sharpe {sr['sharpe']:.2f}。",
            f"**55 年樣本也成立**：納斯達克年化 {g(nr['cagr'])}、回撤 {gm(nr['mdd'])}（買進持有 −77.9%）、Sharpe {nr['sharpe']:.2f}。同曝險基準只有 {g(nr['em_cagr'])}、Sharpe {nr['em_sharpe']:.2f}。",
            '**它為什麼有效？** 這其實不是「星期效應」的功勞，而是 **20 日均線本身的趨勢過濾**：跌破均線就大幅減碼，在 2000 年和 2008 年那種崩盤裡把回撤砍掉一半。星期規則只是決定「減碼時往哪天躲」，屬於裝飾。**真正有用的部分是他的均線擇時，不是他的日曆理論。**',
            f"**代價**：持倉只有 {nr['exposure']*100:.0f}%，長期報酬明顯低於滿倉（55 年年化 {g(nr['cagr'])} vs 10.55%）。適合想降低回撤、能接受少賺的人。",
        ],
        imgs=['qqq', 'nasdaq'],
        verdict='有效，但功勞在均線不在星期。風險調整後明顯改善，代價是報酬變低。',
        evidence='兩段樣本（55 年 / 10 年）皆勝同曝險基準；信號只用前一日收盤，無前視。',
    ))

    # ---- 5. BTC skip Thursday ----
    cards.append(dict(
        id='btc-skip', author='邢不行', grade='B', topic='日曆效應', market='crypto',
        title='BTC 跳過週四：頭條很漂亮，但靠 5 天撐著',
        video='BV1si4y177RU',
        lead='在 BTC 上，跳過週四讓年化從 38.5% 變成 55.8%，是全部測試裡帳面最好看的一組。但把最差的 5 個週四剔掉，整個效應就消失了。',
        metrics=[('策略年化', g(btc['skip']['cagr'])), ('買進持有', g(btc['bh']['cagr'])), ('剔除 5 天後', '−0.02%')],
        body=[
            f"**帳面**：年化 {g(btc['skip']['cagr'])}（買進持有 {g(btc['bh']['cagr'])}）、回撤 −73.4%（買進持有 −83.2%）、Sharpe {btc['skip']['sharpe']:.2f}、Calmar {btc['skip']['calmar']:.2f}。同曝險基準 {g(E[('BTC','跳過週四')]['em_cagr'])}，策略勝出。逐年看 9 年有 8 年勝出。",
            '**但這個結論經不起拆解。** BTC 週四平均 −0.228%，看起來很可觀；可是 t 值只有 **−1.17**（不顯著）。把 476 個週四裡最差的 **5 天**剔掉，平均立刻回到 **−0.020%**——等於零。剔掉 10 天就轉正。',
            '**換句話說：這個策略賺的不是「每個週四都跌」，而是「剛好躲過了 2020-03 和 2022 那幾次崩盤剛好發生在週四」。** 這是事後看圖說故事，不是可重複的規律。2019、2020、2023 三年貢獻了大部分超額報酬（+113pp、+50pp、+66pp）。',
        ],
        imgs=['btc', 'trim', 'yr'],
        verdict='不建議採用。表面上最強，實際上是 5 天的運氣。t=−1.17 且剔除後歸零，這是典型的假陽性。',
        evidence='BTC/USDT 日線 3,332 日（2017-08 至 2026-09），成本 5bp；剔除測試與逐年拆解並列。',
    ))

    # ---- 6. BTC regime ----
    cards.append(dict(
        id='btc-regime', author='邢不行', grade='B', topic='擇時', market='crypto',
        title='BTC 均線×星期：回撤砍掉三成',
        video='BV1si4y177RU',
        lead='同一條均線規則搬到 BTC，報酬略低於買進持有，但回撤從 −83% 降到 −56%，Sharpe 從 0.83 升到 0.95。',
        metrics=[('策略年化', g(btc['regime']['cagr'])), ('回撤', gm(btc['regime']['mdd'])), ('Sharpe', f"{btc['regime']['sharpe']:.2f}")],
        body=[
            f"**年化 {g(btc['regime']['cagr'])}（買進持有 {g(btc['bh']['cagr'])}）、回撤 {gm(btc['regime']['mdd'])}、Sharpe {btc['regime']['sharpe']:.2f}、Calmar {btc['regime']['calmar']:.2f}。** 持倉 {btc['regime_exposure']*100:.0f}%。",
            f"**同曝險基準只有 {g(E[('BTC','均線×星期')]['em_cagr'])}、Sharpe 0.83、Calmar 0.44**，策略在三個指標上都勝出。這比「跳過週四」可信得多，因為它不依賴特定某幾天，而是依賴「跌破均線就減碼」這個連續的機制。",
            '**但要說清楚**：這跟他的「星期」理論無關，功勞在 20 日均線。我把星期規則拿掉、單純用均線做多空切換，效果差異很小。**所以正確的結論是「BTC 適合用均線控制曝險」，而不是「BTC 有週四效應」。**',
        ],
        imgs=['btc', 'em'],
        verdict='可參考。真正有用的是均線曝險控制，不是星期規則。',
        evidence='BTC/USDT 2017–2026，3,332 日，成本 5bp（含換手）。',
    ))

    # ---- 7. calendar windows ----
    cards.append(dict(
        id='calendar', author='邢不行', grade='B', topic='日曆效應', market='us',
        title='四大逃生窗口：10 月底不但不是賣點，還是買點',
        video='BV1638968EzE',
        lead='他驗證 A 股「4 月、8 月、10 月、12 月中」四個危險窗口，結論是 10 月反而最強、12 月最弱。美股 55 年資料得出幾乎一樣的結論，這點相當有意思。',
        metrics=[('10/30 +5日', '+1.57%'), ('對應 t 值', '+3.77'), ('12/10 +5日', '−0.79%')],
        body=[
            '**10 月 30 日：完全不是逃生窗口，是四個窗口裡最強的買點。** 納斯達克 55 年，之後 5 日平均 **+1.57%（t=+3.77）**、20 日 +2.17%（t=+2.67）、勝率 74%。他自己的 A 股結論也是這樣（「10 月上涨概率最高…更像是买信号」）。',
            '**12 月 10 日：短期最弱，但別抱太久。** 之後 1 日 −0.25%（t=−1.56）、5 日 **−0.79%（t=−1.94）**——四個窗口裡唯一穩定的負值。但 20 日後反而 +2.04%（t=+2.53），也就是「跌一週然後漲回來」。',
            '**4 月和 8 月：看不出規律。** 4 月 10 日之後 20 日 +1.70%（t=+2.18，但方向跟他說的「危險」相反）；8 月 31 日全部不顯著（t 絕對值 ≤0.51）。**他說「8 月不必刻意規避」在美股也成立。**',
            '**要注意的是多重檢定**：16 個組合（4 窗口 × 4 期間）裡有 **6 個** t 值超過 2，而且 10 月 30 日一個窗口就包辦 4 個。相關係數很高的期間彼此不獨立，所以不能直接把 6 個當成 6 次獨立發現；但 10 月 30 日連續四個期間全部顯著、方向一致，這比單一顯著點可信得多。反過來說，4 月與 8 月總共 8 個組合只有 1 個顯著，那就更接近雜訊。',
        ],
        imgs=['cal'],
        verdict='部分成立，而且與 A 股方向一致——這是最有趣的一點。但只能當参考，不足以單獨構成策略。',
        evidence='納斯達克 1971–2026，每個窗口 54–55 個觀測；SPY 只有 9–10 個觀測，未採用。',
    ))

    # ---- 8-10. not portable ----
    for cid, title, video, why, body in [
        ('northbound', '北向資金：美股沒有對應的東西', 'BV1c64y147Qz',
         '核心數據是滬股通／深股通每日淨流入，屬於 A 股獨有的跨境通道。',
         [
             '他的規則：北向資金單日淨流入 > 40／50／60／70 億時視為事件，統計之後 1／3／5 日滬深 300 表現（勝率 55%–62%）。',
             '**美股沒有等價物。** 最接近的概念是「外資買超」或 ETF 資金流（如 SPY 的 creation/redemption），但那是申贖機制，跟滬港通的每日額度與披露制度完全不同，頻率與定義都對不上。',
             '**可以勉強替代的是「外資持倉」**：用 Treasury TIC 資料的月度外資持有美債／美股餘額。但那是月頻、滯後兩個月，做不了日線策略。**所以這支我不做假回測。**',
         ]),
        ('retail', '散戶反向：需要逐股資金流，美股散戶也早就機構化', 'BV1m94y1S7ZZ',
         '需要每檔股票每日「散戶買入／賣出金額」的估算數據，美股沒有公開等價物。',
         [
             '他的規則：散戶賣出金額 ÷ 當日成交額 > 60% → 隔日開盤買入，持有 3 天，資金分 N 份輪動。',
             '**美股沒有這種資料。** 券商的 payment-for-order-flow 資料（如各券商的零售訂單流）是付費且不公開逐股金額；13F 只有季度、只有機構。',
             '**而且前提本身在美股較弱**：美股散戶參與度約 20%，A 股約 60%。「散戶反向」這個邏輯建立在「散戶是主要定價力量」上，美股的定價主力是機構與做市商。**強行移植會變成測一個不存在的現象。**',
         ]),
        ('stockpick', '十大量化選股：需要 A 股財報與資金流資料庫', 'BV1Dyni66ExJ',
         '小市值、低換手、A/H 溢價、主力分歧率——全部依賴 A 股特有欄位。',
         [
             '這支是他對自己十年策略的回顧：小市值（年化 91%→25%）、低估值低盈利、A/H 溢價、低換手率、主力資金三指標、冷門股等十個。',
             '**每一個都需要 A 股特有資料**：市值分組需要全市場個股（含已退市，否則有存活者偏差）；A/H 溢價需要同一家公司兩地上市；主力分歧率需要逐筆委託的大單分類；低換手率需要全市場成交量。',
             '**美股可以做的是對應概念**：小市值（IWM 內部分組）、低換手率（volume turnover 分組）、低估值（P/E、P/B 分組）。但這是一整套新的因子回測（需要 CRSP 等級的含退市個股資料），不是「移植一條規則」。',
             '**這支我建議留到下一輪**：如果你要，我可以先做「低換手率」和「低估值」這兩組，因為美股有現成的 ETF 成分與財報資料可用。',
         ]),
    ]:
        cards.append(dict(
            id=cid, author='邢不行', grade='NA', topic='無法移植', market='us',
            title=title, video=video, lead=why,
            metrics=[('能否回測', '否'), ('原因', '資料不存在'), ('處理', '未做假回測')],
            body=body, imgs=[], verdict='這支的策略無法在美股做誠實回測，所以我不給數字。',
            evidence='資料可取得性檢查：Yahoo 429、stooq 封鎖；美股無對應的逐股散戶／北向資金欄位。',
        ))

    # ---- 第一梯與第二梯：新做的影片 ----
    # 放在最前面——首頁第一屏應該是最新的分析，舊的留在後面。
    _new = []

    # 本輪最新（Alpha101 因子、PEAD、春哥選股、兩張 Jack、中文頻道調查）。
    # 六張一次加入，放在 _new 最前面，首頁第一屏就是這批。
    try:
        import new5_cards
        _new = new5_cards.build() + _new
    except Exception as e:
        print(f'new5_cards 載入失敗：{type(e).__name__}: {e}')

    # J Law VCP 三招（單張，單獨一個模組因為它有自己的回測管線）
    try:
        import vcp_cards
        _new = vcp_cards.build() + _new
    except Exception as e:
        print(f'vcp_cards 載入失敗：{type(e).__name__}: {e}')

    # 本輪三張：趨勢線、平行通道、恐慌進場
    try:
        import new6_cards
        _new = new6_cards.build() + _new
    except Exception as e:
        print(f'new6_cards 載入失敗：{type(e).__name__}: {e}')

    # 出場系列五張：見頂、延伸、轉弱、阻力位、大市拖累
    try:
        import exit_cards
        _new = exit_cards.build() + _new
    except Exception as e:
        print(f'exit_cards 載入失敗：{type(e).__name__}: {e}')

    _t1_json = os.path.join(OUT, 'tier1.json')
    if os.path.exists(_t1_json):
        import tier1_cards
        _new.extend(tier1_cards.build(json.load(open(_t1_json))))

    _t2_json = os.path.join(OUT, 'tier2.json')
    if os.path.exists(_t2_json):
        import tier2_cards
        _new.extend(tier2_cards.build(json.load(open(_t2_json))))

    # 第四梯：本輪新轉錄的 13 支（含 5 支真正新分析 + 3 支缺 A 股資料）
    try:
        import new4_cards
        _new.extend(new4_cards.build())
    except Exception as e:
        print(f'new4_cards 載入失敗：{type(e).__name__}: {e}')

    # 阿尔法量化：B 站被風控擋住、改用他開源的程式碼驗證（三張）
    # 這一輪最新，所以插在 _new 最前面——首頁第一屏應該是最新的分析。
    try:
        import alpha_cards
        _new = alpha_cards.build() + _new
    except Exception as e:
        print(f'alpha_cards 載入失敗：{type(e).__name__}: {e}')

    # 提阿非羅大人 TiaBTC：YouTube 頻道，49 支逐字稿 + 完整回測（六張）
    # 這是目前唯一「聽過他本人怎麼說」再回測的一批，分析基礎比 alpha 那批厚。
    try:
        import tia_cards
        _new = tia_cards.build() + _new
    except Exception as e:
        print(f'tia_cards 載入失敗：{type(e).__name__}: {e}')

    # 老魯隨筆—美股期權交易：43 支影片、10 支逐字稿（五張）
    # 兩張可回測（蝶式、江恩九方圖）＋三張心法（標 kind=mindset，無法回測）。
    try:
        import laolu_cards
        _new = laolu_cards.build() + _new
    except Exception as e:
        print(f'laolu_cards 載入失敗：{type(e).__name__}: {e}')

    # RTA 交易之路第二期：唯一有完整逐字稿的一期（三組報告＋院長逐頁挑錯）。
    # 兩張卡：MA120 拐頭 vs 破線、三個尺度（趨勢長度／回撤／波動）＋MA60 對照。
    try:
        import rta_cards
        _new = rta_cards.build() + _new
    except Exception as e:
        print(f'rta_cards 載入失敗：{type(e).__name__}: {e}')

    cards = _new + cards

    # ---- 排序：兩層，順序不可對調 ----
    # 第一層（最高優先）：可回測的一律排在無法回測的前面。
    #   就算不能回測的卡片評分再好看、再新，也只能排在能回測的後面。
    # 第二層：同一組內維持原本順序——新做的分析在前。
    # 任何一層都不按評級重排：評級不是排序依據。
    # 用 sorted 而非兩串相接，因為 sorted 是穩定排序，組內原序自動保留；
    # 判斷依據用 card_kind（可否回測），不是 grade != 'NA'——兩者目前結果相同，
    # 但語意不同，用 kind 才不會哪天出現「NA 但可回測」的卡片時默默排錯。
    cards = sorted(cards, key=lambda c: 0 if card_kind(c) == 'backtest' else 1)

    # ---- render ----
    GR = {'S': ('#f0b429', 'S'), 'A': ('#3fb950', 'A'), 'B': ('#58a6ff', 'B'),
          'C': ('#d29922', 'C'), 'D': ('#f85149', 'D'), 'NA': ('#6e7681', '—')}
    out = []
    details = {}
    for c in cards:
        col, letter = GR[c['grade']]
        mets = ''.join(
            f'<div class="metric"><div class="mv">{v}</div><div class="ml">{k}</div></div>'
            for k, v in c['metrics'])

        mini = MINI.get(c['id'])
        mini_html = (f'<figure class="mini"><img src="{mini}" alt="策略損益"></figure>'
                     if mini else '')
        href = f"d_{c['id']}.html"

        # 影片識別碼標籤：一支影片一個；重複主題的卡片會有兩個
        vid_tags = ''.join(
            f'<span class="tag vid">{vid}</span>' for vid, _ in video_links(c))

        out.append(f'''
<article class="card" data-market="{c['market']}" data-grade="{c['grade']}" data-author="{c['author']}" data-kind="{card_kind(c)}" id="{c['id']}">
  <header class="chead">
    <div class="ctags"><span class="tag author">{c['author']}</span><span class="tag">{c['topic']}</span>{vid_tags}</div>
    <span class="grade" style="--gc:{col}">{letter}</span>
  </header>
  <h2 class="ctitle">{c['title']}</h2>
  <p class="clead">{mb(c['lead'])}</p>
  {mini_html}
  <div class="metrics">{mets}</div>
  <a class="more" href="{href}">詳細資料 →</a>
</article>''')

        imgs = ''.join(
            f'<figure class="fig"><img src="{imgs_map[k]}" alt=""></figure>' for k in c['imgs'])
        paras = ''.join(f'<p>{p}</p>' for p in md(c['body']))
        # the detail page gets its own render, never the card's thumbnail
        details[href] = detail_page(c, col, letter, mets, CARDS_SERIES.get(c['id']), paras, imgs)

    return '\n'.join(out), details


def detail_page(c, col, letter, mets, card, paras, imgs):
    links = video_links(c)
    parts = []
    for vid, url in links:
        site = 'B 站' if vid.startswith('BV') else 'YouTube'
        parts.append(
            '<a class="vsrc" href="' + url + '" target="_blank" rel="noopener">'
            '<span class="vico">▶</span> 在 ' + site + ' 觀看原視頻 '
            '<span class="vcode">' + vid + '</span> ↗</a>')
    vlink = ''.join(parts)

    got = code_snippet(c['id'])
    if got:
        ctitle, csrc = got
        code_html = (CODE_BLOCK
                     .replace('__CODE_TITLE__', ctitle)
                     .replace('__CODE__', html_escape(csrc)))
    else:
        code_html = ''

    if card:
        payload = dict(d=card['d'],
                       series=[dict(label=x['label'], color=x['color'],
                                    ppy=x['ppy'], v=x['v']) for x in card['series']])
        chart_html = (CHART_BLOCK
                      .replace('__START__', card['start'])
                      .replace('__END__', card['end'])
                      .replace('__DATA__', json.dumps(payload, separators=(',', ':'))))
    else:
        # 沒有淨值曲線有幾種完全不同的原因，不能共用同一句話：
        #   chart_note → 卡片自帶說明（例如定投是現金流型，本來就沒有單一淨值）
        #   NA      → 資料不存在，確實無法回測
        #   有統計圖 → 這是統計檢定型分析，測的是「訊號後報酬分布」
        #   其他    → 純說明型，無圖
        if c.get('chart_note'):
            chart_html = f'<div class="nochart">{c["chart_note"]}</div>'
        elif c.get('grade') == 'NA':
            chart_html = ('<div class="nochart">這支策略無法誠實回測，因此沒有績效曲線；'
                          '原因見下方說明。</div>')
        elif c.get('imgs'):
            chart_html = ('<div class="nochart">這張卡片是<b>統計檢定</b>，不是單一策略的回測：'
                          '它比較「訊號出現後」與「隨便哪天」的報酬差異，所以沒有淨值曲線可畫。'
                          '下方的圖就是這個檢定的結果。</div>')
        else:
            # 這一類做過扎實的統計檢定／事件研究，但沒有一條「一直持有」的
            # 部位，所以畫不出淨值曲線。舊版寫成「並非策略回測」是錯的，
            # 會被讀成「這張卡沒有做功課」——實際上是「問的問題不同」。
            chart_html = ('<div class="nochart">這張卡片沒有淨值曲線，'
                          '因為它不是一個可以一直持有的策略：'
                          '它問的是「某種情況發生後，報酬跟平常有沒有不同」，'
                          '答案是一次性的統計量，畫成淨值曲線會誤導。'
                          '檢定結果在上方指標與下方說明。</div>')

    tpl = DETAIL_TEMPLATE
    for k, v in [('__CSS__', DETAIL_CSS), ('__TITLE__', c['title']),
                 ('__TOPIC__', c['topic']),
                 ('__AUTHOR__', c['author']),
                 # 有些卡片只給 videos 清單、沒有單一 video 欄位（多支影片的主題）；
                 # 事件研究與頻道調查這類「沒有對應影片」的卡片則兩者都空，此時留破折號。
                 ('__VIDEO__', c.get('video')
                  or (c['videos'][0] if c.get('videos') else '—')),
                 ('__COLOR__', col), ('__LETTER__', letter),
                 ('__LEAD__', mb(c['lead'])), ('__VLINK__', vlink),
                 ('__CHART__', chart_html), ('__METRICS__', mets),
                 ('__CODE__', code_html),
                 ('__BODY__', paras), ('__IMGS__', imgs),
                 ('__VERDICT__', mb(c['verdict'])), ('__EVID__', mb(c['evidence'])),
                 ('__VERSION__', VERSION), ('__BUILD_AT__', BUILD_AT)]:
        tpl = tpl.replace(k, v)
    return tpl


DETAIL_CSS = r'''
:root{--bg:#0d1117;--card:#161b22;--line:#30363d;--tx:#e6edf3;--tx2:#9aa7b4;--tx3:#6e7681;--acc:#58a6ff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--tx);
font:15px/1.75 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans CJK TC",sans-serif}
.wrap{max-width:none;width:100%;margin:0;padding:28px clamp(20px,3vw,64px) 70px}
.prose{max-width:1180px;margin:0 auto}
.back{display:inline-block;color:var(--tx3);text-decoration:none;font-size:13px;margin-bottom:22px}
.back:hover{color:var(--acc)}
.back.bottom{margin:32px 0 0;display:block;border-top:1px solid var(--line);padding-top:18px}
.tags{display:flex;gap:7px;align-items:center;margin-bottom:10px}
.tag{font-size:11px;color:var(--tx3);border:1px solid var(--line);border-radius:5px;padding:2px 8px}
.tag.vid{font-family:ui-monospace,monospace}
.grade{width:26px;height:26px;border-radius:7px;display:grid;place-items:center;font-weight:800;
font-size:12px;color:var(--gc);background:color-mix(in srgb,var(--gc) 15%,transparent);
border:1px solid color-mix(in srgb,var(--gc) 45%,transparent)}
h1{margin:0 0 12px;font-size:23px;line-height:1.45;letter-spacing:.2px}
.lead{color:var(--tx2);margin:0 0 20px;font-size:14px}
.vsrc{display:flex;align-items:center;gap:9px;background:#16202e;border:1px solid #26364d;
border-radius:9px;padding:11px 14px;margin:0 0 20px;text-decoration:none;color:#c9d1d9;
font-size:13.5px;transition:.15s}
.vsrc:hover{border-color:var(--acc);color:#fff}
.vsrc .vico{color:#f85149;font-size:11px}
.vsrc .vcode{font-family:ui-monospace,monospace;font-size:11.5px;color:var(--tx3)}
.build{margin-top:22px;font-size:11px;color:var(--tx3);font-family:ui-monospace,monospace}
.mini{margin:0 0 20px;border:1px solid var(--line);border-radius:9px;overflow:hidden;background:#0a0e14}
.mini img{width:100%;display:block}
.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;
border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:14px 0;margin-bottom:22px}
.metric{text-align:center}
.mv{font-size:17px;font-weight:700}
.ml{font-size:11px;color:var(--tx3);margin-top:2px}
.body p{margin:0 0 14px;font-size:14.2px;color:#c9d1d9}
.body b{color:#fff;font-weight:600}
.fig{margin:16px 0;border:1px solid var(--line);border-radius:9px;overflow:hidden;background:#0a0e14}
.fig img{width:100%;display:block}
.verdict{margin-top:20px;background:#1a2230;border-left:3px solid var(--acc);
padding:12px 14px;border-radius:0 7px 7px 0;font-size:13.5px;color:#c9d1d9}
.verdict b{color:var(--acc)}
.evid{margin-top:10px;font-size:12px;color:var(--tx3);line-height:1.7}
''' + CHART_CSS + CODE_CSS


def mb(s):
    """把行內 **粗體** 轉成 <b>。

    lead / verdict / evidence 這三個欄位不走 md()（那是逐段的），
    所以卡片作者若在這裡用 **，星號會原樣顯示在頁面上。
    實測已有 5 張卡片踩到（t1-month、t1-rotation、t2-dca 等），
    加上本輪三張，共 11 處。直接在輸出前統一轉掉，比逐張改卡片可靠。
    """
    if not s:
        return s
    out = []
    for i, part in enumerate(s.split('**')):
        out.append(f'<b>{part}</b>' if i % 2 else part)
    s = ''.join(out)
    # 行內 `code` 同理：evidence 常引用檔案路徑與程式碼片段
    out = []
    for i, part in enumerate(s.split('`')):
        out.append(f'<code>{part}</code>' if i % 2 else part)
    return ''.join(out)


def md(paras):
    out = []
    for p in paras:
        s = p
        while '**' in s:
            s = s.replace('**', '<b>', 1).replace('**', '</b>', 1)
        # 行內 `code`：正文常常引用檔案路徑與程式碼片段。
        # 不轉的話，反引號會原樣出現在頁面上（本輪三張卡共 10 處）。
        if '`' in s:
            parts = s.split('`')
            s = ''.join(f'<code>{x}</code>' if i % 2 else x
                        for i, x in enumerate(parts))
        out.append(s)
    return out


TEMPLATE = r'''<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>策略顯微鏡 · 策略回測庫</title>
<style>
:root{--bg:#0d1117;--bg2:#0f1620;--card:#161b22;--card2:#1c2230;--line:#30363d;
--tx:#e6edf3;--tx2:#9aa7b4;--tx3:#6e7681;--acc:#58a6ff;--red:#f85149;--grn:#3fb950}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--tx);
font:14px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans CJK TC",sans-serif;
-webkit-font-smoothing:antialiased}
.wrap{width:100%;max-width:none;margin:0;padding:0 clamp(20px,2.6vw,56px) 80px}
header.top{border-bottom:1px solid var(--line);background:linear-gradient(180deg,#111823,#0d1117);
position:sticky;top:0;z-index:20;backdrop-filter:blur(8px)}
.tbar{width:100%;max-width:none;margin:0;padding:14px clamp(20px,2.6vw,56px);display:flex;align-items:center;gap:14px;flex-wrap:wrap}
.logo{font-weight:700;font-size:16px;letter-spacing:.3px}
.logo span{color:var(--acc)}
.tnav{margin-left:auto;display:flex;gap:18px}
.tnav a{color:var(--tx3);text-decoration:none;font-size:12.5px;transition:.15s}
.tnav a:hover{color:var(--acc)}
.sub{color:var(--tx3);font-size:12px}
.hero{padding:34px 0 22px}
.hero h1{margin:0 0 10px;font-size:26px;letter-spacing:.2px}
.hero p{margin:0;color:var(--tx2);max-width:1080px;font-size:13.5px}
.filters{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:22px 0 10px}
.flabel{font-size:11.5px;color:var(--tx3);margin-right:1px}
.fsep{width:1px;height:18px;background:var(--line);margin:0 5px}
.chip{background:var(--card);border:1px solid var(--line);color:var(--tx2);
padding:6px 14px;border-radius:999px;font-size:12.5px;cursor:pointer;transition:.15s;
user-select:none}
.chip:hover{color:var(--tx);border-color:#4a5568}
.chip.on{background:var(--acc);color:#0d1117;border-color:var(--acc);font-weight:600}
.chip.reset{color:var(--tx3);border-style:dashed}
.chip.reset:hover{color:#f85149;border-color:#f85149}
.chip[data-facet="author"]{border-color:#3d4a5c}
.chip[data-facet="author"].on{background:#d29922;border-color:#d29922;color:#0d1117}
.fcount{font-size:12px;color:var(--tx3);margin:0 0 18px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(clamp(150px,11vw,210px),1fr));gap:10px;margin:18px 0 26px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.stat .v{font-size:19px;font-weight:700;letter-spacing:.3px}
.stat .k{font-size:11.5px;color:var(--tx3);margin-top:2px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:20px;align-items:start}
.card{background:var(--card);border:1px solid var(--line);border-radius:13px;padding:20px 20px 17px;
display:flex;flex-direction:column;transition:.18s}
.card:hover{border-color:#4a5568;transform:translateY(-2px)}
.card.hide{display:none}
.chead{display:flex;justify-content:space-between;align-items:flex-start;gap:8px;margin-bottom:10px}
.ctags{display:flex;gap:7px;flex-wrap:wrap}
.tag{font-size:11.5px;color:var(--tx3);border:1px solid var(--line);border-radius:5px;padding:3px 8px}
.tag.vid{font-family:ui-monospace,monospace;color:#6e7681}
.tag.author{color:#d29922;border-color:#4a3f22;background:rgba(210,153,34,.09)}
.grade{flex:0 0 auto;width:34px;height:34px;border-radius:9px;display:grid;place-items:center;
font-weight:800;font-size:15px;color:var(--gc);background:color-mix(in srgb,var(--gc) 15%,transparent);
border:1px solid color-mix(in srgb,var(--gc) 45%,transparent)}
.ctitle{margin:0 0 9px;font-size:17.5px;line-height:1.45;letter-spacing:.2px}
.clead{margin:0 0 15px;color:var(--tx2);font-size:13.6px;line-height:1.65}
.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;
border-top:1px solid var(--line);padding-top:13px;margin-top:auto}
.metric{text-align:center}
.mv{font-size:15px;font-weight:700;color:var(--tx);letter-spacing:.2px}
.ml{font-size:10.5px;color:var(--tx3);margin-top:2px}
.mini{margin:0 0 13px;border:1px solid var(--line);border-radius:8px;overflow:hidden;background:#0a0e14}
.mini img{width:100%;display:block}
.body p{margin:0 0 11px;font-size:12.9px;color:#c9d1d9;line-height:1.72}
.body b{color:#fff;font-weight:600}
.fig{margin:12px 0;border:1px solid var(--line);border-radius:8px;overflow:hidden;background:var(--bg)}
.fig img{width:100%;display:block}
.verdict{margin-top:12px;background:#1a2230;border-left:3px solid var(--acc);
padding:10px 12px;border-radius:0 6px 6px 0;font-size:12.6px;color:#c9d1d9}
.verdict b{color:var(--acc)}
.evid{margin-top:8px;font-size:11px;color:var(--tx3)}
.more{margin-top:12px;display:block;text-align:center;text-decoration:none;
background:transparent;border:1px solid var(--line);color:var(--tx2);
padding:7px;border-radius:7px;font-size:12px;transition:.15s}
.more:hover{color:var(--acc);border-color:var(--acc)}
footer{margin-top:44px;padding-top:18px;border-top:1px solid var(--line);
color:var(--tx3);font-size:11.5px;line-height:1.8}
footer .ver{font-family:ui-monospace,monospace;font-size:11px}
footer a{color:var(--tx3)}
footer a:hover{color:var(--acc)}
@media(max-width:1024px){
  .grid{grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:16px}
  .hero h1{font-size:23px}
}
@media(max-width:760px){
  .wrap{padding:0 16px 56px}
  .tbar{padding:12px 16px}
  .tnav{gap:14px}
  .tnav a{font-size:12px}
  .grid{grid-template-columns:1fr;gap:14px}
  .hero{padding:24px 0 16px}
  .hero h1{font-size:20px;line-height:1.4}
  .hero p{font-size:13px}
  .card{padding:16px 15px 14px}
  .ctitle{font-size:16px}
  .clead{font-size:13px}
  .mv{font-size:14.5px}
  .filters{gap:6px}
  .chip{padding:5px 12px;font-size:12px}
  .stats{grid-template-columns:repeat(2,1fr);gap:8px}
}
</style>
</head>
<body>
<header class="top"><div class="tbar">
  <div class="logo">策略<span>顯微鏡</span></div>
  <div class="sub">策略回測庫</div>
  <nav class="tnav">
    <a href="about.html">關於這個站</a>
    <a href="changelog.html">版本紀錄</a>
  </nav>
</div></header>

<div class="wrap">
<div class="hero">
  <h1>他分享的策略，在美股與加密市場還有效嗎</h1>
  <p>從他的影片裡挑出規則明確、可執行的策略，用美股 55 年與加密 9 年資料重跑。每張卡片都是獨立檢驗：先看帳面績效，再拆解它是否只是「少持倉」或「少數幾天」造成的假象。<b>評級只看證據強度，不看帳面報酬高低。</b></p>
</div>

<div class="filters">
  <span class="flabel">來源</span>
__AUTHOR_CHIPS__
  <span class="fsep"></span>
  <span class="flabel">市場</span>
  <div class="chip" data-facet="market" data-val="us">美股 (__C_US__)</div>
  <div class="chip" data-facet="market" data-val="crypto">加密貨幣 (__C_CRYPTO__)</div>
  <span class="fsep"></span>
  <span class="flabel">結果</span>
  <div class="chip" data-facet="grade" data-val="pass">通過 (A/B)</div>
  <div class="chip" data-facet="grade" data-val="fail">未通過 (C/D)</div>
  <span class="fsep"></span>
  <span class="flabel">性質</span>
  <div class="chip" data-facet="kind" data-val="backtest">可回測 (__C_BT__)</div>
  <div class="chip" data-facet="kind" data-val="mindset">心法／觀念 (__C_MIND__)</div>
  <div class="chip" data-facet="kind" data-val="unportable">無法移植 (__C_UNP__)</div>
  <div class="chip reset" id="freset">清除篩選</div>
</div>
<div class="fcount" id="fcount"></div>

<div class="stats">
  <div class="stat"><div class="v">__N_CARDS__</div><div class="k">已分析的策略</div></div>
  <div class="stat"><div class="v">__N_RUN__</div><div class="k">規則完整重跑</div></div>
  <div class="stat"><div class="v">__N_PASS__</div><div class="k">通過（A/B 級）</div></div>
  <div class="stat"><div class="v">__N_FAIL__</div><div class="k">未通過（C/D 級）</div></div>
  <div class="stat"><div class="v">__N_NA__</div><div class="k">無法回測</div></div>
</div>

<div class="grid">
__CARDS__
</div>

<footer>
資料來源：FRED（納斯達克綜合指數 1971–2026）、Nasdaq 日線（SPY/QQQ 2016–2026）、
Binance 公開封存（BTC/USDT 2017–2026）。<br>
成本假設：美股每趟 1 bp，加密每趟 5 bp。全部信號只用前一交易日收盤，無前視。
比較基準含「同曝險、不擇時」的固定槓桿買進持有，以避免把「少持倉」誤讀成「會擇時」。<br>
逐字稿經 BibiGPT 取得。本頁為歷史資料檢驗，不構成投資建議。<br>
<span class="ver">版本 __VERSION__ · 建置於 __BUILD_AT__ · <a href="changelog.html">版本紀錄</a></span>
</footer>
</div>

<script>
/* Multi-select facets.
   Selections within one facet are OR; different facets are AND.
   Clicking an active chip deselects it, so 邢不行+美股 -> click 美股 -> only 邢不行. */
var state = {author: new Set(), market: new Set(), grade: new Set(), kind: new Set()};
var cards = Array.prototype.slice.call(document.querySelectorAll('.card'));

function matches(card) {
  var a = card.dataset.author, m = card.dataset.market, g = card.dataset.grade,
      k = card.dataset.kind;
  if (state.author.size && !state.author.has(a)) return false;
  if (state.market.size && !state.market.has(m)) return false;
  if (state.kind.size && !state.kind.has(k)) return false;
  if (state.grade.size) {
    var ok = (state.grade.has('pass') && (g === 'A' || g === 'B')) ||
             (state.grade.has('fail') && (g === 'C' || g === 'D'));
    if (!ok) return false;
  }
  return true;
}

function apply() {
  var shown = 0;
  cards.forEach(function (c) {
    var ok = matches(c);
    c.classList.toggle('hide', !ok);
    if (ok) shown++;
  });
  var el = document.getElementById('fcount');
  el.textContent = shown === cards.length
    ? '顯示全部 ' + cards.length + ' 個策略'
    : '顯示 ' + shown + ' / ' + cards.length + ' 個策略';
  document.querySelectorAll('.chip[data-facet]').forEach(function (ch) {
    ch.classList.toggle('on', state[ch.dataset.facet].has(ch.dataset.val));
  });
}

document.querySelectorAll('.chip[data-facet]').forEach(function (ch) {
  ch.onclick = function () {
    var f = ch.dataset.facet, v = ch.dataset.val;
    if (state[f].has(v)) state[f].delete(v); else state[f].add(v);
    apply();
  };
});

document.getElementById('freset').onclick = function () {
  state.author.clear(); state.market.clear(); state.grade.clear(); state.kind.clear();
  apply();
};

apply();
// 詳細資料 opens in a new tab; nothing on this page changes height.
</script>
</body></html>
'''

CHANGELOG_CSS = """
:root{--bg:#0d1117;--card:#161b22;--line:#30363d;--tx:#e6edf3;--tx2:#9aa7b4;--tx3:#6e7681;--acc:#58a6ff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--tx);
font:15px/1.7 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans CJK TC",sans-serif}
.wrap{max-width:860px;margin:0 auto;padding:28px 20px 70px}
a{color:var(--tx3);text-decoration:none;font-size:13px}
a:hover{color:var(--acc)}
h1{font-size:22px;margin:18px 0 6px}
.sub{color:var(--tx2);font-size:13.5px;margin:0 0 22px}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;color:var(--tx3);font-weight:500;font-size:11.5px;
padding:8px 10px;border-bottom:1px solid var(--line)}
td{padding:11px 10px;border-bottom:1px solid #21262d;vertical-align:top;color:#c9d1d9}
tr.cur td{background:#16202e}
td.v{font-family:ui-monospace,monospace;color:var(--acc);white-space:nowrap}
td.d{font-family:ui-monospace,monospace;font-size:11.5px;color:var(--tx3);white-space:nowrap}
td.f code{font-size:11px;color:var(--tx3)}
.note{margin-top:24px;font-size:12px;color:var(--tx3);line-height:1.9;
border-top:1px solid var(--line);padding-top:16px}
code{background:#161b22;border:1px solid var(--line);border-radius:4px;padding:1px 5px;
font-size:11.5px;color:#c9d1d9}
"""


def snapshot_and_changelog(detail_files):
    """Keep a restorable copy of every build plus a readable changelog."""
    vdir = os.path.join(HERE, 'versions', 'v' + VERSION)
    os.makedirs(vdir, exist_ok=True)
    for fn in list(detail_files) + ['page.html']:
        src = os.path.join(HERE, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(vdir, fn))

    meta_path = os.path.join(HERE, 'build_meta.json')
    hist = json.load(open(meta_path)) if os.path.exists(meta_path) else []
    hist = [h for h in hist if h.get('version') != VERSION]
    hist.append(dict(version=VERSION, built_at=BUILD_AT, notes=VERSION_NOTES,
                     files=len(detail_files) + 1))
    hist.sort(key=lambda h: [int(x) for x in h['version'].split('.')])
    json.dump(hist, open(meta_path, 'w'), ensure_ascii=False, indent=1)

    rows = []
    for h in reversed(hist):
        cur = ' class="cur"' if h['version'] == VERSION else ''
        snap = 'versions/v{}/'.format(h['version'])
        rows.append('<tr{}><td class="v">v{}</td><td class="d">{}</td>'
                    '<td>{}</td><td class="f"><code>{}</code></td></tr>'
                    .format(cur, h['version'], h['built_at'], h['notes'], snap))

    page = (
        '<!doctype html>\n<html lang="zh-Hant"><head><meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        '<title>版本紀錄 · 策略回測庫</title>\n<style>' + CHANGELOG_CSS + '</style></head>\n'
        '<body><div class="wrap">\n<a href="index.html">← 回到策略列表</a>\n'
        '<h1>版本紀錄</h1>\n'
        '<p class="sub">每次建置都保留完整快照。回滾時把對應的 '
        '<code>versions/vX.Y.Z/</code> 複製回根目錄即可。</p>\n'
        '<table><thead><tr><th>版本</th><th>建置時間</th><th>變更內容</th>'
        '<th>快照位置</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table>\n'
        '<div class="note">回滾範例：<br>'
        '<code>cp versions/v1.1.0/*.html .</code><br><br>'
        '目前建置：v' + VERSION + ' · ' + BUILD_AT + '</div>\n'
        '</div></body></html>')
    open(os.path.join(HERE, 'changelog.html'), 'w').write(page)
    print('written changelog.html + snapshot versions/v{} ({} versions tracked)'
          .format(VERSION, len(hist)))


ABOUT_CSS = CHANGELOG_CSS + """
/* header nav — CHANGELOG_CSS has no header, so the About page brings its own.
   .wrap is overridden back to a top-aligned block so the sticky bar can sit above it. */
header.top{border-bottom:1px solid var(--line);background:linear-gradient(180deg,#111823,#0d1117);
position:sticky;top:0;z-index:20;backdrop-filter:blur(8px)}
.tbar{padding:14px 20px;display:flex;align-items:center;gap:14px;flex-wrap:wrap}
.logo{font-weight:700;font-size:16px;letter-spacing:.3px}
.logo span{color:var(--acc)}
.tbar .sub{color:var(--tx3);font-size:12px;margin:0}
.tnav{margin-left:auto;display:flex;gap:18px}
.tnav a{color:var(--tx3);text-decoration:none;font-size:12.5px;transition:.15s}
.tnav a:hover{color:var(--acc)}
.wrap{margin:0 auto}
h2{font-size:16px;margin:32px 0 10px;padding-top:6px;color:#e6edf3;letter-spacing:.2px}
h2:first-of-type{margin-top:26px}
p{margin:0 0 14px;color:#c9d1d9;font-size:14px}
ul{margin:0 0 14px;padding-left:20px;color:#c9d1d9;font-size:14px}
li{margin:0 0 8px}
b{color:#e6edf3}
.lead{font-size:14.5px;color:var(--tx2);margin:0 0 26px}
.callout{background:#161b22;border:1px solid var(--line);border-left:3px solid #d29922;
border-radius:6px;padding:14px 16px;margin:0 0 18px;font-size:13.5px;color:#c9d1d9}
.callout b{color:#f0b429}
.foot{margin-top:34px;padding-top:18px;border-top:1px solid var(--line);
font-size:12px;color:var(--tx3);line-height:1.9}
td.k{white-space:nowrap;color:var(--tx2)}
"""


def write_about(n_cards=0, n_grade=None, n_na=0):
    n_grade = n_grade or {}
    """The About page is a build product so its version and figures stay in
    lockstep with the cards instead of drifting from hand-edited HTML."""
    page = (
        '<!doctype html>\n<html lang="zh-Hant"><head><meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        '<title>關於這個站 · 策略回測庫</title>\n<style>' + ABOUT_CSS + '</style></head>\n'
        '<body><header class="top"><div class="tbar">'
        '<div class="logo">策略<span>顯微鏡</span></div>'
        '<div class="sub">策略回測庫</div>'
        '<nav class="tnav">'
        '<a href="index.html">策略列表</a>'
        '<a href="about.html">關於這個站</a>'
        '<a href="changelog.html">版本紀錄</a>'
        '</nav></div></header>\n'
        '<div class="wrap">\n'
        '<h1>關於這個站</h1>\n'
        '<p class="lead">把影片裡講的選股與擇時策略，用美股 55 年、加密貨幣 9 年的日線資料'
        '重新跑一次，看它們在今天的市場還站不站得住。</p>\n'

        '<h2>為什麼要做這件事</h2>\n'
        '<p>影片裡常有一些聽起來很有道理的規則，例如「每週四最弱，那就跳過週四」、'
        '「跌破均線就減碼」。這些規則在原本的市場（多半是中國 A 股）可能真的有效，'
        '但換一個市場、換一段時間，還有效嗎？</p>\n'
        f'<p>目前收錄 <b>{n_cards} 個講法</b>，其中 <b>{n_cards - n_na} 個</b>'
        f'做了實際回測，<b>{n_na} 個</b>因資料不存在而無法回測——'
        f'後者不做假回測，只說明缺什麼資料、為什麼做不到。</p>\n'

        '<h2>怎麼讀這些卡片</h2>\n'
        '<p>每張卡片都是獨立檢驗。看帳面績效之前，先問三個問題：</p>\n'
        '<ul>'
        '<li><b>是不是只是少持倉？</b>少買自然波動小、回撤淺，但那不叫會擇時。'
        '所以每一張都附一條「同曝險、不擇時」的基準：同樣的持倉比例，'
        '但完全不挑時間、每天固定槓桿買進持有。策略要贏過這條線，才算真的有擇時能力。</li>'
        '<li><b>是不是靠少數幾天？</b>把最極端的幾天剔掉重算。'
        '如果剔掉 5 天後超額報酬就歸零，那這個策略賺的是運氣，不是規律。</li>'
        '<li><b>統計上顯著嗎？</b>附 t 值（t-statistic，衡量平均報酬偏離零的程度，'
        '已考慮觀測數與波動）。絕對值超過 2 大約才稱得上顯著；'
        '小於 1.5 基本上跟零分不出來。</li>'
        '</ul>\n'

        '<h2>評級的意思</h2>\n'
        '<table><thead><tr><th>評級</th><th>條數</th><th>意思是什麼</th></tr></thead><tbody>'
        f'<tr><td class="v">A</td><td>{n_grade.get("A", 0)}</td><td>通過雙重檢驗：55 年樣本與近 10 年樣本都勝過同曝險基準</td></tr>'
        f'<tr><td class="v">B</td><td>{n_grade.get("B", 0)}</td><td>帳面有改善，但證據不足以支撐實盤採用</td></tr>'
        f'<tr><td class="v">C</td><td>{n_grade.get("C", 0)}</td><td>效應曾經真實存在，但已衰減到無法交易</td></tr>'
        f'<tr><td class="v">D</td><td>{n_grade.get("D", 0)}</td><td>不成立，甚至有害</td></tr>'
        f'<tr><td class="v">—</td><td>{n_na}</td><td>資料不存在或樣本不足，無法有效檢驗</td></tr>'
        '</tbody></table>\n'
        '<p style="margin-top:14px"><b>評級只看證據強度，不看帳面報酬高低。</b>'
        '一個年化 55% 但 t 值 −1.17、剔掉 5 天就歸零的策略，拿不到高分。</p>\n'

        '<h2>資料與假設</h2>\n'
        '<table><thead><tr><th>項目</th><th>內容</th></tr></thead><tbody>'
        '<tr><td class="k">納斯達克綜合指數</td><td>FRED 代碼 <code>NASDAQCOM</code>，'
        '1971-02-05 至 2026-10-21，共 14,046 個有效交易日'
        '（原始下載 14,533 列，其中 487 列當日無報價）</td></tr>'
        '<tr><td class="k">SPY / QQQ / 道瓊斯</td><td>日線收盤價，2016-10-19 至 2026-10-19，'
        '各約 2,512 個交易日</td></tr>'
        '<tr><td class="k">BTC / ETH</td><td>Binance 官方公開封存，2017-08 至 2026-09，'
        '各 3,332 天</td></tr>'
        '<tr><td class="k">交易成本</td><td>美股每趟 1 bp（萬分之一）；加密貨幣每趟 5 bp</td></tr>'
        '<tr><td class="k">股息</td><td>全部採價格報酬（不含股息）。策略與基準兩邊一致，'
        '因此不影響「策略對基準」的比較結論</td></tr>'
        '<tr><td class="k">年化換算</td><td>美股以 252 個交易日、加密貨幣以 365 天計算</td></tr>'
        '</tbody></table>\n'
        '<div class="callout"><b>不要直接比較兩邊的夏普比率。</b>'
        '美股用 252 天年化、加密貨幣用 365 天年化，兩者基準不同。'
        '表格裡各自都是正確的算法，但拿來互比大小會得到錯誤結論。</div>\n'

        '<h2>兩個最值得記住的否定結果</h2>\n'
        '<p><b>1. 跳過週四在 BTC 上是假陽性。</b>帳面上年化從 38.51% 變成 55.78%，'
        '是全部測試裡最漂亮的一組。但週四平均 −0.228% 的 t 值只有 <b>−1.17</b>（不顯著）；'
        '把 476 個週四裡最差的 <b>5 天</b>剔掉，平均立刻回到 <b>−0.020%</b>，等於零。'
        '它賺的不是「每個週四都跌」，而是「剛好躲過 2020 年 3 月與 2022 年那幾次崩盤，'
        '而那幾次剛好發生在週四」。</p>\n'
        '<p><b>2. 美股的「黑色星期一」已經結束了。</b>1971–1999 年，納斯達克週一平均 '
        '−0.127%（t = −4.47），完全符合傳說。但 2000 年之後變成 +0.005%（t = +0.12），'
        '效應歸零。同一條規則在 55 年樣本上是負貢獻，在近 10 年卻是正貢獻——'
        '這種「換一段時間就翻轉」正是它不可交易的理由。</p>\n'

        '<h2>一個刻意的設計</h2>\n'
        '<p>詳細頁底部的「顯示代碼」不是另外抄一份示範程式碼，而是建置時用 Python 的 '
        '<code>inspect.getsource</code> 從<b>實際跑出這些數字的函式</b>擷取原始碼。'
        '所以只要改了回測參數，重新建置後頁面上的程式碼就會跟著變，'
        '不可能出現「頁面寫的跟實際跑的不同」。程式碼區塊上方也固定標明執行環境、'
        '資料格式、無前視聲明與成本假設，因為光看函式本文會以為用的是預設值，'
        '但實際呼叫時成本參數是被覆寫過的。</p>\n'

        '<h2>已知限制</h2>\n'
        '<ul>'
        '<li>股息未計入。若比較含股息的總報酬，絕對數值會更高，'
        '但策略與基準的相對結論不變。</li>'
        '<li>SPY / QQQ 樣本只有 10 年，任何「近 10 年有效」的結論都可能是這段時期的運氣。</li>'
        '<li>加密貨幣的歷史只有 9 年，且僅涵蓋一次完整的牛市與熊市循環。</li>'
        '<li>日曆效應本質上是多重檢定問題：測了 16 個窗口組合，總會有幾個 t 值超過 2。'
        '已盡量標註哪些是多點同時顯著、哪些是單點雜訊。</li>'
        '<li>未模擬滑價、借券成本、稅賦與資金費率。</li>'
        '</ul>\n'

        '<h2>資料來源</h2>\n'
        '<ul>'
        '<li>納斯達克綜合指數、道瓊斯、標普 500：FRED（美國聖路易聯邦準備銀行）</li>'
        '<li>SPY / QQQ 日線：公開歷史行情</li>'
        '<li>BTC / ETH 日線：Binance 官方公開封存（data.binance.vision）</li>'
        '<li>原始影片逐字稿：經 BibiGPT 取得（逐字稿為第三方內容，未納入本專案）</li>'
        '</ul>\n'

        '<div class="foot">本專案為歷史資料的統計檢驗，用於理解策略為何有效或無效，'
        '<b>不構成投資建議</b>。過去的統計結果不保證未來表現。<br>'
        '版本 ' + VERSION + ' · 建置於 ' + BUILD_AT + '<br>'
        '<a href="index.html">策略列表</a> · <a href="changelog.html">版本紀錄</a></div>\n'
        '</div></body></html>')
    open(os.path.join(HERE, 'about.html'), 'w').write(page)
    print('written about.html', len(page), 'bytes')

if __name__ == '__main__':
    main()
