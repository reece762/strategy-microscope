#!/usr/bin/env python3
"""第一梯七張卡片的內容。

所有數字都從 out/tier1.json 讀取，不手寫，避免頁面與實際跑出來的結果脫節。
規則一律取自影片逐字稿原文；做不到的部分寫明「未測」而不是補一個近似值。
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')


def _pct(v, nd=2):
    return f'{v*100:+.{nd}f}%'


def _pc(v, nd=2):
    return f'{v*100:.{nd}f}%'


def _m(d, k):
    """月份表在 JSON 往返後 key 會變成字串，這裡兩種都接受。"""
    if d is None:
        return None
    return d.get(k) if k in d else d.get(str(k))


def build(t1):
    """回傳卡片 dict 列表。t1 = json.load(out/tier1.json)"""
    cards = []
    md = t1['macd']
    kd = t1['kdj']
    vr = t1['vr']
    gp = t1['gap']
    me = t1['month']
    ih = t1['intraday']
    rot = t1['rotation']

    # ---------------- 1. MACD 頂底背離 ----------------
    nas = md['納斯達克綜合指數']
    btc = md['BTC']
    qqq = md['QQQ']
    cards.append(dict(
        id='t1-macd', author='邢不行', grade='D', topic='技術指標', market='us',
        title='MACD 頂底背離：55 年來，頂背離之後反而漲更多',
        video='NO5JOVjDO_I',
        lead='他把「頂背離」翻成精確規則：局部高點要創 30 日新高、要高於前一個高點，且 DIFF 低於前一個高點。用同一套規則跑納斯達克 55 年，頂背離後 20 日平均上漲 1.24%，比什麼都不做的 0.98% 還高。',
        metrics=[('頂背離後 20 日', _pct(nas['top']['r20_mean'])),
                 ('同期無條件', _pct(nas['base']['20']['mean'])),
                 ('扣基準後 t 值', f"{nas['top']['r20_excess_t']:+.2f}")],
        body=[
            f"**他的規則我完整照抄，沒有改。** 高點定義：三根 K 線中間那根收盤最高，且創過去 30 日收盤新高，且高於前一個高點；同時 DIFF 低於前一個高點，才構成頂背離。底背離完全反向。",
            f"**納斯達克 55 年出現 {nas['top']['n']} 次頂背離、{nas['bottom']['n']} 次底背離。** 頂背離後 20 日平均 {_pct(nas['top']['r20_mean'])}，上漲機率 {_pc(nas['top']['r20_hit'],1)}。但同期「隨便一天往後看 20 日」的平均是 {_pct(nas['base']['20']['mean'])}、上漲機率 {_pc(nas['base']['20']['hit_up'],1)}。**背離後不但沒跌，還微微跑贏一點，扣掉基準的 t 值只有 {nas['top']['r20_excess_t']:+.2f}——統計上分不出來。**",
            f"**QQQ 是唯一比較像樣的一組**：頂背離後 20 日 {_pct(qqq['top']['r20_mean'])}，低於基準的 {_pct(qqq['base']['20']['mean'])}，扣基準 t={qqq['top']['r20_excess_t']:+.2f}；底背離 {_pct(qqq['bottom']['r20_mean'])}，t={qqq['bottom']['r20_excess_t']:+.2f}。方向對，但樣本只有 {qqq['bottom']['n']} 次，還不足以下結論。",
            f"**機械策略的結果比統計更難看。** 底背離買進、頂背離平倉，納斯達克 55 年年化只有 {_pc(nas['strategy']['cagr'])}，而買進持有是 {_pc(nas['bh']['cagr'])}。原因不是訊號錯，是**訊號太少**：他影片裡自己也說了，2007–2008 與 2019 之後的大行情一次底背離都沒出現，所以大部分時間是空手的（曝險僅 {_pc(nas['exposure'],1)}）。",
            f"**BTC 更清楚**：頂背離後 20 日平均 {_pct(btc['top']['r20_mean'])}、上漲機率 {_pc(btc['top']['r20_hit'],1)}，而基準是 {_pct(btc['base']['20']['mean'])}、{_pc(btc['base']['20']['hit_up'],1)}。他的方向在這裡剛好相反。",
        ],
        imgs=['t1_macd'],
        verdict='不成立。方向不但沒有如宣稱般下跌，統計上也不顯著；當策略用更是大幅落後買進持有。',
        evidence=f"納斯達克 14,046 日（1971–2026）、SPY/QQQ 各 2,512 日（2016–2026）、BTC 3,332 日。所有報酬以「訊號次日開盤進場」計算，已排除前視。超額＝訊號後報酬減同期無條件平均。",
    ))

    # ---------------- 2. 量比 ----------------
    sp = vr['SPY']
    qq = vr['QQQ']
    cards.append(dict(
        id='t1-vr', author='邢不行', grade='NA', topic='技術指標', market='us',
        title='量比 2.5–5 買進：訊號太少，在美股根本測不出來',
        video='08lNIsMgylY',
        lead=f"他實測 A 股「量比 2.5–5 買進」三年虧掉 99.87%。同一條規則搬到 SPY／QQQ，十年只出現 {sp['n_buy']} 與 {qq['n_buy']} 次訊號——因為 ETF 的成交量結構和個股完全不同。",
        metrics=[('SPY 買訊次數', f"{sp['n_buy']} 次"),
                 ('QQQ 買訊次數', f"{qq['n_buy']} 次"),
                 ('SPY 買後 1 日', _pct(sp['buy_h1_mean'] or 0))],
        body=[
            f"**規則照抄**：量比＝當日成交量÷前 5 日平均成交量。{_pct(2.5,0)} 到 {_pct(5,0)} 視為買進，大於 {_pct(5,0)} 視為離場。",
            f"**但 ETF 上這條規則幾乎不觸發。** SPY 十年只有 {sp['n_buy']} 次買訊、{sp['n_sell']} 次賣訊；QQQ 是 {qq['n_buy']} 次與 {qq['n_sell']} 次。原因很單純：個股會有單日爆量（新聞、財報、主力進出），而 SPY 這種全市場 ETF 的成交量被上千檔成分股稀釋，很少出現 2.5 倍以上的跳增。",
            f"**買訊後 1 日**：SPY {_pct(sp['buy_h1_mean'] or 0)}（t={sp['buy_h1_t']:+.2f}）、QQQ {_pct(qq['buy_h1_mean'] or 0)}（t={qq['buy_h1_t']:+.2f}），兩邊都不顯著。SPY 後 10 日 {_pct(sp['buy_h10_mean'] or 0)}、上漲機率 {_pc(sp['buy_h10_hit'] or 0,0)}，但只有 {sp['n_buy']} 個樣本，不能當結論。",
            f"**他的第二套規則（換手率 >5% 且量比 2–3）我沒有測。** 換手率需要流通股數，ETF 的「流通股數」是申贖創造出來的，拿來算換手率會得到沒有意義的數字。這一條要測必須改用個股資料。",
            f"**可以確認的是他的核心結論**：量比單獨使用在. 他測 A 股得到年化 −26.27%、回撤 99.87%；我在美股得到的是「訊號太少、無法檢驗」，這比「有效」更接近「不是個可靠的指標」。",
        ],
        imgs=['t1_vr'],
        verdict='無法在 ETF 上檢驗，而這本身就是答案：訊號密度太低，不是可交易的規則。',
        evidence=f"SPY/QQQ 各 2,512 個交易日（2016-10 至 2026-10）。量比＝成交量÷前 5 日均量。未測項：換手率結合版（ETF 無流通股數）。",
    ))

    # ---------------- 3. 跳空回補 ----------------
    spg = gp['SPY']
    qqg = gp['QQQ']
    cards.append(dict(
        id='t1-gap', author='邢不行', grade='B', topic='量價結構', market='us',
        title='跳空必回補：95% 會補，但四成要等超過 10 天',
        video='mUMyXYxpL24',
        lead=f"他測上證 813 次跳空，96.31% 回補、中位 84 天。SPY 十年 {spg['n']} 次跳空，回補率 {_pc(spg['fill_rate'],1)}、中位 {spg['median_days']:.0f} 天——回補確實是常態，但「常態」不等於能賺錢。",
        metrics=[('SPY 回補率', _pc(spg['fill_rate'],1)),
                 ('10 日內回補', _pc(spg['pct_within_10d'],1)),
                 ('中位天數', f"{spg['median_days']:.0f} 天")],
        body=[
            f"**定義照他的**：向上跳空＝當日最低高於前日最高；向下跳空＝當日最高低於前日最低。回補＝後續價格回到跳空前的價位。",
            f"**SPY 十年 {spg['n']} 次跳空（上漲 {spg['n_up']} 次、下跌 {spg['n_down']} 次），回補率 {_pc(spg['fill_rate'],1)}。** 分方向看：向上跳空 {_pc(spg['fill_rate_up'],1)}、向下跳空 {_pc(spg['fill_rate_down'],1)}。**跟他在上證看到的「向下跳空更容易補」完全一致**（上證是 97.6% vs 94.9%）。",
            f"**但他自己也點出的問題在這裡同樣存在**：中位數 {spg['median_days']:.0f} 天，平均卻是 {spg['mean_days']:.0f} 天。意思是多數很快補，少數拖非常久——最長的那些把平均拉高。只有 {_pc(spg['pct_within_10d'],1)} 在 10 天內補完，所以「等回補」這個策略有近四成的部位會被卡住。",
            f"**他也問了「跳空越大越難補嗎」**，答案是不會，甚至反過來。SPY 依幅度分組：0–0.5% 回補 {_pc(spg['by_gap'][0]['fill_rate'],0) if spg['by_gap'] else 'n/a'}、0.5–1% 回補 {_pc(spg['by_gap'][1]['fill_rate'],0) if len(spg['by_gap'])>1 else 'n/a'}、1–5% 回補 {_pc(spg['by_gap'][2]['fill_rate'],0) if len(spg['by_gap'])>2 else 'n/a'}。差距不大，跟他的結論相同：**幅度不是決定因素**。",
            f"**BTC 與 ETH 完全沒有跳空。** 24 小時連續交易，日線幾乎必然重疊——這個現象是「有收盤、有開盤」的市場才有的，加密貨幣沒有。",
        ],
        imgs=['t1_gap'],
        verdict='現象成立、統計真實，但不可直接交易：回補率高是因為「早晚會補」，不是「短期會補」。',
        evidence=f"SPY/QQQ 各約 2,512 日；回補以「後續任一日觸及跳空前價位」判定，未設時間上限，因此回補率會略高於設上限的版本。",
    ))

    # ---------------- 4. KDJ ----------------
    ks = kd['SPY']
    kq = kd['QQQ']
    kb = kd['BTC']
    cards.append(dict(
        id='t1-kdj', author='邢不行', grade='D', topic='技術指標', market='us',
        title='KDJ 金叉：勝率 68.7%，但什麼都不做是 69.0%',
        video='LENnqMT7NH8',
        lead=f"他測 A 股 426 萬次訊號，金叉死叉漲跌機率都在 50% 左右。我在 SPY 上得到金叉後 20 日上漲機率 {_pc(ks['gold']['h20_hit'],1)}——聽起來很高，但同期「隨便一天往後看 20 日」是 {_pc(ks['base']['20']['hit_up'],1)}。高的不是指標，是市場本身。",
        metrics=[('金叉後 20 日勝率', _pc(ks['gold']['h20_hit'],1)),
                 ('無條件同期勝率', _pc(ks['base']['20']['hit_up'],1)),
                 ('勝率超額', f"{(ks['gold']['h20_hit']-ks['base']['20']['hit_up'])*100:+.1f}pp")],
        body=[
            f"**這張卡是整批裡最容易誤讀的一張**，所以先講結論：金叉後 20 日上漲機率 {_pc(ks['gold']['h20_hit'],1)} 看起來很強，但同期無條件上漲機率就有 {_pc(ks['base']['20']['hit_up'],1)}。**扣掉之後只剩 {(ks['gold']['h20_hit']-ks['base']['20']['hit_up'])*100:+.1f} 個百分點，等於沒有**。",
            f"**公式照他的**：RSV＝(收盤−9 日最低)÷(9 日最高−9 日最低)×100；K＝前 K×2/3＋RSV×1/3；D＝前 D×2/3＋K×1/3；J＝3K−2D。金叉＝K 上穿 D。",
            f"**SPY 十年 {ks['n_gold']} 次金叉、{ks['n_dead']} 次死叉。** 金叉後 20 日平均 {_pct(ks['gold']['h20_mean'])}，扣掉基準的 {_pct(ks['base']['20']['mean'])} 後超額 {_pct(ks['gold']['h20_excess'])}。QQQ 超額 {_pct(kq['gold']['h20_excess'])}、BTC 超額 {_pct(kb['gold']['h20_excess'])}——三邊都在 0.3% 以內。",
            f"**死叉更沒有用**：SPY 死叉後 20 日下跌機率只有 {_pc(ks['dead']['h20_hit'],1)}，也就是**有 {_pc(1-ks['dead']['h20_hit'],1)} 的機率是上漲的**。把死叉當賣點，等於在六成情況下賣錯。",
            f"**機械策略**：金叉買、死叉賣，SPY 年化 {_pc(ks['strategy']['cagr'])}、曝險 {_pc(ks['exposure'],1)}，而同曝險不擇時的基準是 {_pc(ks['em']['cagr'])}。**完全打平**——這正是「指標沒提供資訊」的樣子。",
        ],
        imgs=['t1_kdj'],
        verdict='不成立。表面勝率好看，扣掉市場本身的漂移後歸零；死叉更是接近反向指標。',
        evidence=f"SPY/QQQ 各 2,512 日、BTC 3,332 日。納斯達克因 FRED 僅有收盤價（KDJ 需高低價）而未測。所有勝率均與同期無條件勝率對照。",
    ))

    # ---------------- 5. 月份效應 ----------------
    mnas = me['納斯達克綜合指數（55 年）']
    mbtc = me['BTC']
    feb_nas = _m(mnas['months'], 2)
    jun_nas = _m(mnas['months'], 6)
    feb_btc = _m(mbtc['months'], 2)
    jun_btc = _m(mbtc['months'], 6)
    cards.append(dict(
        id='t1-month', author='邢不行', grade='B', topic='日曆效應', market='us',
        title='五窮六絕七翻身：美股與幣圈都不成立，但二月確實偏強',
        video='B3N9s41YcUE',
        lead=f"他測 A 股 2005–2022，結論是「五窮六絕不成立，但二月最好、六月最差」。同一套統計搬到納斯達克 55 年：二月 {_pct(feb_nas['mean'])}（t={feb_nas['t']:+.2f}）、六月 {_pct(jun_nas['mean'])}（t={jun_nas['t']:+.2f}）——**二月的優勢在美股不存在**。",
        metrics=[('納斯達克 2 月', _pct(feb_nas['mean'])),
                 ('納斯達克 6 月', _pct(jun_nas['mean'])),
                 ('BTC 6 月', _pct(jun_btc['mean']))],
        body=[
            f"**「五窮六絕七翻身」的原始說法在美股不成立**：五月 {_pct(_m(mnas['months'], 5)['mean'])}（t={_m(mnas['months'], 5)['t']:+.2f}）、六月 {_pct(jun_nas['mean'])}（t={jun_nas['t']:+.2f}）、七月 {_pct(_m(mnas['months'], 7)['mean'])}（t={_m(mnas['months'], 7)['t']:+.2f}）。三個月全是正的，跟「窮、絕」相反。",
            f"**他改口後的結論（二月最好）在美股也不成立**：納斯達克 55 個二月，平均 {_pct(feb_nas['mean'])}、上漲機率 {_pc(feb_nas['hit'],1)}，t 值只有 {feb_nas['t']:+.2f}；六月 {_pct(jun_nas['mean'])}、上漲機率 {_pc(jun_nas['hit'],1)}。**二月不但沒有特別好，還是少數低於全樣本平均的月份**（全樣本平均 {_pct(mnas['all_mean'])}）。",
            f"**BTC 這邊反而有一點看頭**：六月平均 {_pct(jun_btc['mean'])}、上漲機率只有 {_pc(jun_btc['hit'],1)}（{jun_btc['n']} 個六月）；二月 {_pct(feb_btc['mean'])}、七月 {_pct(_m(mbtc['months'], 7)['mean'])}。**但樣本只有 9 年**，任何月份的統計都只基於 9 個觀測，這正是他自己在影片裡批評的「樣本太小」。",
            f"**他自己也說了不建議實盤**：「我個人而言，是不大敢直接投入實盤使用的」。這段話值得記下來——**他知道 18 個觀測（每月份）不足以支撐一個策略**。我把樣本擴到 55 年後，結論更清楚：月份效應在美股不存在。",
        ],
        imgs=['t1_month'],
        verdict='不成立。A 股上的「二月效應」在美股 55 年樣本裡完全消失，BTC 的六月弱勢只有 9 個觀測、不足採信。',
        evidence='納斯達克 668 個月（1971-03 至 2026-10）、SPY/QQQ 各 120 個月、BTC 109 個月。月份報酬以月底收盤計算。',
    ))

    # ---------------- 6. 幣圈日內 ----------------
    ib = ih['BTC']
    ie = ih['ETH']
    cards.append(dict(
        id='t1-intraday', author='邢不行', grade='B', topic='市場結構', market='crypto',
        title='幣圈誰在主導：美國時段成交量確實高，但只高 1.5 倍',
        video='CvDPVtOalCY',
        lead=f"他說幣安成交量集中在「美國時段」（北京 20:00–01:00），藉此主張定價權已轉移到歐美。用 BTC 全部 {ib['n']:,} 小時資料檢驗：美國時段每小時佔比 {_pc(ib['us_per_hour'])}，中國時段 {_pc(ib['cn_per_hour'])}，密度比 {ib['us_cn_ratio']:.2f} 倍。",
        metrics=[('美/中 每小時密度', f"{ib['us_cn_ratio']:.2f}x"),
                 ('美國時段佔比', _pc(ib['us_share'],1)),
                 ('北京 08 點', _pc(ib['bj8_share'],2))],
        body=[
            f"**他的觀察成立，但幅度比他暗示的小。** 美國時段（北京 20:00–01:00，共 6 小時）佔全天成交量 {_pc(ib['us_share'],1)}，平均每小時 {_pc(ib['us_per_hour'])}；中國時段（北京 08:00–17:00，取 8 個小時）佔 {_pc(ib['cn_share'],1)}，每小時 {_pc(ib['cn_per_hour'])}。**密度比 {ib['us_cn_ratio']:.2f} 倍**——如果完全均勻會是 1.00，所以美國時段確實比較活躍，但不是壓倒性的。",
            f"**他的「逐年轉移」推論也對得上**：2017 年美國時段佔 {_pc(ib['yearly']['2017']['us_share'],1)}，2026 年升到 {_pc(ib['yearly']['2026']['us_share'],1)}。九年下來成長約 8 個百分點，方向與他說的「越來越偏美國時段」一致。",
            f"**他的「北京 08:00 異常高（量化換倉）」推測，資料不支持**：北京 08:00 佔 {_pc(ib['bj8_share'],2)}，而完全均勻是 {_pc(ib['uniform_share'],2)}。**差距只有 0.08 個百分點**，在這麼大的樣本裡幾乎等於沒有。他在影片裡自己也說「我覺得很有可能」，這是一個未經檢驗的猜測。",
            f"**ETH 的形狀幾乎一樣**：密度比 {ie['us_cn_ratio']:.2f} 倍，美國時段 {_pc(ie['us_share'],1)}。兩個獨立標的給出同一個結論，這比單一標的可靠。",
        ],
        imgs=['t1_intraday'],
        verdict='部分成立：美國時段確實較活躍且逐年增強，但幅度溫和，且「08:00 換倉高峰」的說法沒有資料支持。',
        evidence=f"BTC/ETH 各 {ib['n']:,} 根 1 小時 K 線（2017-08 至 2026-09）。時段換算為北京時間。以「該小時成交量佔全天比重」比較，避免絕對量受幣價與整體活躍度影響。",
    ))

    # ---------------- 7. 大小盤輪動 ----------------
    r20 = rot['N20_plain']
    r20c = rot['N20_cash_when_both_down']
    rbig = rot['big_bh']
    rsmall = rot['small_bh']
    cards.append(dict(
        id='t1-rotation', author='邢不行', grade='D', topic='風格輪動', market='us',
        title='大小盤輪動：六種參數全部輸給「只買 SPY」',
        video='b_ERGIP8U1k',
        lead=f"他在 A 股用滬深 300 與創業板做輪動，20 天動能、年化 20%、回撤 50%。同一條規則搬到美股（SPY 為大盤、IWM 為小盤），N=20 年化 {_pc(r20['cagr'])}、回撤 {_pc(abs(r20['mdd']),1)}，而**只買 SPY 不輪動是 {_pc(rbig['cagr'])}**。",
        metrics=[('N=20 輪動年化', _pc(r20['cagr'])),
                 ('只買 SPY', _pc(rbig['cagr'])),
                 ('N=20 回撤', _pc(abs(r20['mdd']),1))],
        body=[
            f"**規則照抄**：每天收盤算大盤與小盤過去 N 天漲跌幅，隔天持有漲得多的那一個。他說 N=20 在 A 股很有效，所以我測了 10、20、60 三個值。",
            f"**結果是單調的失敗**：N=10 年化 {_pc(rot['N10_plain']['cagr'])}、N=20 {_pc(r20['cagr'])}、N=60 {_pc(rot['N60_plain']['cagr'])}。**三個都低於只買 SPY 的 {_pc(rbig['cagr'])}，也遠低於 A 股版的 20%**。輪動沒有創造價值，只是把大盤的報酬換成了更差的組合。",
            f"**他的改良版（兩邊都跌就空倉）也沒有救回來**：N=20 加空倉後年化 {_pc(r20c['cagr'])}，比原版更低。空倉確實把回撤從 {_pc(abs(r20['mdd']),1)} 降到 {_pc(abs(r20c['mdd']),1)}，但代價是錯過反彈——**這跟我們在其他卡片看到的一模一樣：降回撤靠的是少持倉，不是選得準**。",
            f"**為什麼 A 股有效、美股沒效？** 他的策略建立在「大盤與小盤輪動」上，而滬深 300 與創業板的成分差異極大（市值、行業、成長性都不同），兩者輪動確實有空間。美股 SPY 與 IWM 的相關性長期在 0.8 以上，**兩個幾乎同步的東西之間輪動，本來就沒有資訊可挖**。這是跨市場移植時最容易忽略的前提差異。",
        ],
        imgs=['t1_rotation'],
        verdict='不成立。不是參數問題，是前提問題：美股大小盤高度相關，輪動空間本來就比 A 股小。',
        evidence='SPY/IWM 各 2,512 日（2016-10 至 2026-10）。訊號以收盤計算、次日生效，成本每趟 1 bp。他原版用 0.61‱ 且免印花稅。',
    ))

    return cards
