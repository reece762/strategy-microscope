#!/usr/bin/env python3
"""兩個價值型選股方法的卡片。

數字全讀 out/value_all.json。

## 這兩張卡的共同主題

兩套方法出自不同頻道、不同風格（Phil Town 是 Rule #1 的價值投資者，
Everything Money 是 8 支柱篩選器），但都要求「高股東權益報酬率」與「十年穩定成長」。
測出來的核心發現是：**兩套都沒有優勢，但兩套都共同指向同一條有用的指標：ROIC。**

## 必須揭露的兩個實作與資料限制

1. 沒有資本支出科目 → 自由現金流以營業現金流代替（偏誤方向已寫入）
2. 沒有所得稅科目 → ROIC 為稅前值（約為稅後 1.27 倍，門檻實質放寬）

## 必須揭露的方法限制

Phil Town 要求「10 年平均」，我的財報資料從 2009 年才開始有 filed_date，
扣掉 10 年成長需求後，**只有 2019 年以後的再平衡日能算**——
所以那組只有 7 個年度，統計力很低。這不是策略失效，是資料長度不足。
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'out')


def p(x, n=2):
    return f'{x*100:+.{n}f}%'


def build():
    D = json.load(open(os.path.join(OUT, 'value_all.json')))
    PT = D['phil_town']
    EM = D['everything_money']
    BO = D['both']
    PF = D['pt_flags']
    EF = D['em_flags']
    M = D['meta']

    cards = []

    # ===================================================== 1. Phil Town
    cards.append(dict(
        id='n11-philtown', author='Phil Town（Rule #1）', grade='C', topic='選股',
        market='us', kind='backtest', imgs=['n11_roic'], videos=['n1FAWFOq_bs'],
        chart_note=(
            '左圖是 Phil Town Big Five 五條規則各自單獨使用時的超額報酬。'
            '**五條都在零附近游走（−0.27 到 +2.83 個百分點），沒有一條顯著。**'
            '右圖是 Everything Money 的八條，同樣幾乎全在零附近。'
            '**兩張圖唯一共同偏正的是 ROIC（金柱）——這是本批最重要的發現。**'
        ),
        title='Rule #1 的「Big Five」：五條規則加起來沒有優勢',
        lead=(
            'Phil Town 的方法在價值投資圈很有名：'
            '一家好公司必須在過去十年，五項指標平均都達 10% 以上——'
            'ROIC、營收成長、每股盈餘成長、股東權益成長、自由現金流成長。'
            f'我按他的原始門檻實測：**7 個年度、平均每年選出 {PT["avg_k"]:.1f} 檔，'
            f'組合報酬 {p(PT["mean_ret"])}，全體等權 {p(PT["mean_univ"])}，'
            f'隨機抽同樣檔數 {p(PT["mean_rand"])}。'
            f'超額 {p(PT["excess_rand"])}，7 年裡只贏 {PT["beat_rand"]} 年。**'
        ),
        metrics=[
            ('有效年份', f'{PT["n_years"]} 年'),
            ('平均選出', f'{PT["avg_k"]:.1f} 檔'),
            ('超額 vs 隨機', p(PT['excess_rand'])),
        ],
        body=[
            '**先把他的規則原文放上來**，因為他講得比多數人精確：'
            '「a healthy business should average at least 10% ROIC over the past 10 years」、'
            '「town wants averages of 10% per year over 10 years as a baseline」。'
            '五項是 ROIC、sales growth、EPS growth、equity（book value）growth、free cash growth。'
            '他還有第六個條件（他自己說是「follow-up」而非必要）：'
            '「If a company could pay off its long-term debt in three years or less from free cash flow, '
            'the debt is acceptable」。'
            '**他也特別強調一致性**：「Don\'t be seduced by a single great year. '
            'Look at the 10-year average, then the 5-year and one-year figures to spot trends.」',

            '**這裡有個必須先講的資料限制：他要求「十年平均」，我的財報資料撐不到。**'
            'EDGAR 的 filed_date 最早從 2009 年開始，'
            '要算「十年成長率」再加一年前瞻報酬，'
            '**只有 2019 年以後的再平衡日能算——所以這組只有 7 個年度，不是 15 個。**'
            '**這代表兩件事**：統計力很低（7 個獨立觀測），'
            '而且這 7 年剛好是美股大牛市（2019-2025），'
            '**價值型選股在這種環境本來就容易跑輸成長股。**'
            '**所以這個「沒有優勢」的結論，有一部分是樣本期間造成的，不能全歸給方法。**',

            f'**結果：{PT["n_years"]} 年，平均選 {PT["avg_k"]:.1f} 檔，超額 {p(PT["excess_rand"])}，'
            f't={PT["t"]:+.2f}，7 年只贏 {PT["beat_rand"]} 年。**'
            f'逐年超額的標準差是 22.59 個百分點——**比平均值大八倍。**'
            f'最差的是 2025 年（−35.8pp，選 8 檔），最好的是 2022 年（+32.9pp，選 3 檔）。'
            f'**這種幅度說明：結果主要由「那一年剛好選到哪幾檔」決定，而不是方法本身。**',

            f'**但拆開單獨看，五條裡有一條是正的：ROIC。**'
            f'「ROIC ≥10%」單獨使用，超額 +{PF["pt1_roic"]["excess"]*100:.2f} 個百分點，'
            f'13 年裡贏 {PF["pt1_roic"]["beat"]} 年（樣本 1,429）。'
            f'**這是五條裡唯一勝過基準年數過半的。**'
            f'其次是自由現金流成長（+{PF["pt5_fcf"]["excess"]*100:.2f}pp，但只贏 '
            f'{PF["pt5_fcf"]["beat"]}/{PF["pt5_fcf"]["n_years"]} 年，不穩定）。'
            f'**EPS 成長那條是負的（{PF["pt3_eps"]["excess"]*100:+.2f}pp）——'
            f'呼應他在影片裡的警告：「If EPS lags while sales surge, something is wrong」——'
            f'但實測顯示，單看 EPS 成長這件事本身沒有資訊。**',

            '**為什麼五條加起來反而更差？**'
            '這與上一批（10 條篩選器）是同一個機制：'
            f'**取交集把樣本壓到每年 {PT["avg_k"]:.1f} 檔，7 年總共只有 {PT["total_picks"]} 個選中觀測。**'
            '**而 ROIC 這條有效的條件，被另外四條中性條件稀釋掉了。**'
            '**這是所有多條件選股法的共同難題：條件越多，越容易把有效訊號洗掉。**',

            '**技術上，這輪我犯了一個實質錯誤，必須記錄。**'
            '第一次跑出**「可用 0 筆」**——兩套方法的資料旗標全部為 False。'
            '查出來是 XBRL 有**兩種事實型態**，我只處理了一種：'
            '**期間值（duration）**如營收、淨利、營業現金流，有 `period_start`，'
            '要用「期間 300~430 天」篩年報；'
            '**時點值（instant）**如股東權益、總資產、長期負債、流通股數，'
            '**完全沒有 `period_start`**，我卻用同一套「期間天數」條件去篩，'
            '結果把它們全部剔除。'
            '實測：AAPL 的股東權益 264 筆、總資產 146 筆、長期負債 90 筆、'
            '流通股數 144 筆，**沒有一筆有期間**。'
            '修正為「時點值改取 form=10-K 者」後，才有 3,337 個可用觀測。',
        ],
        verdict=(
            f'不成立（但樣本受限）。7 個年度、平均選出 {PT["avg_k"]:.1f} 檔、'
            f'共 {PT["total_picks"]} 個選中觀測（因 10 年成長需求，資料只夠支撐 2019 年起）。'
            f'組合 {p(PT["mean_ret"])} vs 全體 {p(PT["mean_univ"])}、'
            f'隨機 {p(PT["mean_rand"])}；超額 {p(PT["excess_rand"])}，'
            f'7 年贏 {PT["beat_rand"]} 年，t={PT["t"]:+.2f}。'
            f'逐年超額標準差 22.59pp，遠大於平均。'
            f'**五條中僅 ROIC 單獨使用為正（+{PF["pt1_roic"]["excess"]*100:.2f}pp，'
            f'13 年贏 {PF["pt1_roic"]["beat"]} 年）；EPS 成長為負（{PF["pt3_eps"]["excess"]*100:+.2f}pp）。**'
            '**樣本僅涵蓋 2019-2025 大牛市，價值型選股在此期間結構性不利，'
            '故此結論不能推翻方法本身，只能說「在此樣本上無優勢」。**'
        ),
        evidence=(
            f'553 檔美股，2011-2026 每年 6 月 30 日再平衡，當時點財報。'
            f'總觀測 {M["n_obs_all"]:,}，資料足以計算者 {M["n_obs"]:,}。'
            '五條門檻依原影片：ROIC、營收、EPS、股東權益、自由現金流成長皆需 ≥10%（10 年）；'
            '負債測試為長期負債 ÷ 自由現金流 ≤ 3 年。'
            '**資料限制**：(1) 無資本支出科目（XBRL 僅 16 種標籤），'
            '自由現金流以營業現金流代替——OCF > FCF，故此項偏寬鬆；'
            '(2) 無所得稅科目，ROIC 為「營業利益 ÷（股東權益＋長期負債）」的**稅前值**，'
            '約為稅後 1.27 倍，10% 門檻實質只要求稅後約 7.9%。'
            '**因需 10 年成長率，僅 2019 年後可算，有效年份僅 7 個。**'
            '對照組：全體等權、隨機抽同樣檔數（每年 500 次）。'
            '存活者偏差未修正；未計交易成本。'
        ),
    ))

    # ===================================================== 2. Everything Money
    cards.append(dict(
        id='n11-everymoney', author='Everything Money（Paul & Seth）', grade='C',
        topic='選股', market='us', kind='backtest', imgs=['n11_fragility'],
        videos=['1u6qvel9XnM'],
        chart_note=(
            '左圖是兩套方法逐年的超額報酬。**藍柱是 Everything Money，'
            '2014 年那根 +51.9 特別突出——但那一年只選出 2 檔。**'
            '右圖是關鍵：把 2014 拿掉之後，優勢從 +4.49 個百分點掉到 +0.18，'
            '只算「選出 10 檔以上」的年份也只有 +1.18。'
            '**這不是穩健的優勢，是單一年份的運氣。**'
        ),
        title='8 支柱：看起來有優勢，但那優勢只來自一年、只選了 2 檔',
        lead=(
            'Everything Money 的 Paul 與 Seth 公開了一套「8 支柱」篩選法，'
            '每一條都給了明確數字。我按原門檻實測：'
            f'**12 個年度、平均每年選出 {EM["avg_k"]:.0f} 檔，組合 {p(EM["mean_ret"])}，'
            f'全體等權 {p(EM["mean_univ"])}，隨機 {p(EM["mean_rand"])}——'
            f'超額 {p(EM["excess_rand"])}，12 年贏 {EM["beat_rand"]} 年。**'
            f'**但把 2014 年（只選 2 檔、超額 +51.9 個百分點）拿掉，'
            f'優勢立刻塌到 +0.18 個百分點（t=+0.11）。**'
        ),
        metrics=[
            ('有效年份', f'{EM["n_years"]} 年'),
            ('平均選出', f'{EM["avg_k"]:.0f} 檔'),
            ('剔除 2014 後', '+0.18pp'),
        ],
        body=[
            '**這套方法的八條我都能實作，而且門檻都是他們在影片裡親口說的。**'
            '第一條：「we want the five-year PE ratio to be under 22.5」——'
            '算法是「市值 ÷ 過去五年淨利總和」。'
            '第二條：「the 5-year return on invested Capital to be greater than 9%」——'
            '他解釋算法是「the profit the cash flow of the business over the last five years '
            'divided by the total debt and equity」。'
            '第三、四條是營收與淨利成長（今天 > 五年前）。'
            '第五條是流通股數不能增加（回購加分）——他稱稀釋為「the silent killer in investing」。'
            '第六條：「total long-term liabilities divided by five-year free cash flow, '
            'and we want this multiple to be less than five」。'
            '第七條是現金流成長。'
            '第八條：「market cap divided by that total number … We want this number under 20」。',

            '**他也講了兩句很誠實的話，值得放在這裡。**'
            '第一句：「it doesn\'t mean go out and buy it … it\'s an initial screening to kind of give you '
            'an idea of what questions asked next」。'
            '**所以他主張的是「篩選是提問的起點」，不是「篩出來就會賺」。**'
            '第二句更關鍵，他舉自己持有的嘉年華郵輪為例：'
            '「I own a company Carnival Cruise Line which is all eight X\'s but I still own it '
            'because of the future potential」——**他自己也會買「八條不全過」的公司。**',

            f'**結果先看好的一面：超額 {p(EM["excess_rand"])}，12 年贏 {EM["beat_rand"]} 年，'
            f'中位百分位 {EM["median_pct"]*100:.0f}%。**'
            f'這比上一批的 10 條篩選器（+0.25pp）好很多，也比 Phil Town 那組好。'
            f'**而且每年選出 {EM["avg_k"]:.0f} 檔，樣本比 Phil Town 的 {PT["avg_k"]:.1f} 檔厚實得多。**'
            f'**表面上看，這似乎是一套有效的方法。**',

            '**但逐年的數字一攤開，問題就出來了。**'
            '12 年的超額依序是：+51.9、−9.7、+3.0、+5.6、−3.3、+10.4、+0.6、+0.9、+0.1、+0.2、−5.8、−0.1。'
            '**2014 年那 +51.9 個百分點是全部平均的來源。**'
            '而 2014 年只選出 **2 檔**——**兩檔股票的平均報酬，本來就不代表方法有效。**'
            '**剔除這一年，剩下 11 年平均只有 +0.18 個百分點，t=+0.11，與零無異。**'
            '**如果只算「選出 10 檔以上」的年份（樣本較可信的 10 年），平均 +1.18pp，t=+0.83，也不顯著。**',

            '**這是我在這批裡最重要的一個判斷：不能用整體平均掩蓋單一年的主導。**'
            '如果我只報 +4.49pp 和「贏 8/12 年」，看起來像個可用的方法。'
            '**但那 8 次勝利裡，有 6 次的超額小於 1 個百分點**（+0.6、+0.9、+0.1、+0.2、+0.3、+0.6），'
            '**唯一的大勝 2014 撐起了整個結論。**'
            '**這正是我在 AGENTS.md 記過的教訓：剔除極端值再看一次。**',
        ],
        verdict=(
            f'不成立。12 個年度、平均選出 {EM["avg_k"]:.0f} 檔，'
            f'組合 {p(EM["mean_ret"])} vs 全體 {p(EM["mean_univ"])}、隨機 {p(EM["mean_rand"])}；'
            f'整體超額 {p(EM["excess_rand"])}（12 年贏 {EM["beat_rand"]} 年，t={EM["t"]:+.2f}，'
            f'未達顯著）。'
            f'**但優勢高度集中：剔除 2014 年（僅選 2 檔、超額 +51.9pp）後，'
            f'11 年平均只剩 +0.18pp（t=+0.11）；只算選出 ≥10 檔的年份為 +1.18pp（t=+0.83）。**'
            f'8 次勝利中有 6 次超額小於 1 個百分點。'
            '**判定：整體平均被單一年份的極端值主導，不構成穩健優勢。**'
            '作者本人亦強調「這不是買進訊號」，並自承會買「八條不全過」的公司。'
        ),
        evidence=(
            f'553 檔美股，2011-2026 每年 6 月 30 日再平衡，當時點財報，'
            f'總觀測 {M["n_obs_all"]:,}、資料足以計算 {M["n_obs"]:,}。'
            '八條門檻依原影片：5 年 PE<22.5（市值÷5 年淨利和）、'
            '5 年 ROIC>9%（5 年營業現金流÷（總負債＋股東權益））、'
            '營收成長>0、淨利成長>0、流通股數 5 年增幅≤2%、'
            '長期負債÷5 年自由現金流<5、現金流成長>0、市值÷5 年自由現金流<20。'
            '**資料限制**：XBRL 僅 16 種標籤、無資本支出科目，'
            '自由現金流以營業現金流代替（OCF>FCF，使第 6、8 條偏寬鬆）；'
            '無流動資產／負債科目。'
            '對照組：全體等權、隨機抽同樣檔數（每年 500 次）。'
            '**穩健性檢驗**：逐年超額標準差 15.79pp，遠大於平均 4.49pp；'
            '剔除最極端一年後優勢消失。'
            '存活者偏差未修正；未計交易成本。'
        ),
    ))

    # ===================================================== 3. ROIC 跨方法
    cards.append(dict(
        id='n11-roic', author='兩套方法交叉比對', grade='B', topic='選股', market='us',
        kind='chart', imgs=['n11_roic'], videos=['n1FAWFOq_bs', '1u6qvel9XnM'],
        chart_note=(
            '左邊兩組是兩套方法各自拆開的結果。'
            '**金柱是 ROIC——Phil Town 的「ROIC≥10%」與 Everything Money 的「5年 ROIC>9%」。'
            '兩者分屬不同頻道、不同算法（前者用營業利益、後者用現金流），'
            '但結果都是正的，而且是各自組裡最穩的一條。**'
        ),
        title='兩套方法唯一共同有效的：ROIC',
        lead=(
            'Phil Town 與 Everything Money 是兩個不同的頻道、不同的投資風格，'
            '算法也不一樣——前者用「營業利益 ÷ 投入資本」，後者用「現金流 ÷（負債＋權益）」。'
            '但把它們的條件各自拆開單獨測，'
            f'**唯一在兩邊都為正、而且勝過基準年數都過半的，就是 ROIC**：'
            f'Phil Town 版 +{PF["pt1_roic"]["excess"]*100:.2f} 個百分點'
            f'（{PF["pt1_roic"]["beat"]}/{PF["pt1_roic"]["n_years"]} 年）、'
            f'Everything Money 版 +{EF["em2_roic5"]["excess"]*100:.2f} 個百分點'
            f'（{EF["em2_roic5"]["beat"]}/{EF["em2_roic5"]["n_years"]} 年）。**'
        ),
        metrics=[
            ('Phil Town ROIC', f'+{PF["pt1_roic"]["excess"]*100:.2f}pp'),
            ('Everything Money', f'+{EF["em2_roic5"]["excess"]*100:.2f}pp'),
            ('兩者皆過半', '勝率 7/13、9/14'),
        ],
        body=[
            '**這是本批唯一值得留下來的正面發現。**'
            '兩套方法分別有 5 條與 8 條規則，'
            '各自拆開單獨測試後，**只有 ROIC 在兩邊都是「超額為正 + 勝過基準年數過半」：**'
            f'Phil Town 的「ROIC≥10%」超額 +{PF["pt1_roic"]["excess"]*100:.2f} 個百分點、'
            f'13 年贏 {PF["pt1_roic"]["beat"]} 年；'
            f'Everything Money 的「5年 ROIC>9%」超額 +{EF["em2_roic5"]["excess"]*100:.2f} 個百分點、'
            f'14 年贏 {EF["em2_roic5"]["beat"]} 年。',

            '**這個「跨方法一致」比單一結果有意義，原因有三。**'
            '第一，兩個頻道互不相關，不是同一個人的結論。'
            '第二，**兩者的 ROIC 算法其實不同**：'
            'Phil Town 用「營業利益 ÷（股東權益＋長期負債）」，'
            'Everything Money 用「營業現金流 ÷（總負債＋股東權益）」。'
            '**兩種不同算法都指向同一件事，降低了「只是某個算法剛好有效」的可能。**'
            '第三，這與我前幾批的發現方向一致——'
            '**在 10 條篩選器那批，唯一有效的也是「營收成長」這類「基本面品質」指標，'
            '而無效的是本益比這類「估值」指標。**',

            '**但優勢的幅度必須誠實說：大約 1.1 到 1.2 個百分點，而且不顯著。**'
            '我沒有算這兩條的 t 值（樣本結構與主測試不同），'
            '**但從勝率 7/13 與 9/14 都只是「略優於擲硬幣」來看，'
            '不能說這是強訊號。**'
            '**更精確的說法是：ROIC 是這兩套方法裡「最不像噪音」的一條，'
            '而不是「有效的一條」。**',

            '**為什麼 ROIC 會是唯一站得住的？我認為與它的經濟意義有關。**'
            'Phil Town 自己的說法很好：「A wide moat, a sustainable advantage, usually shows up first '
            'in a strong ROIC because the company isn\'t being forced into constant price competition.」'
            '**ROIC 衡量的是「投入一塊錢能賺回多少」，這是企業品質最直接的量度，'
            '不像本益比那樣可以靠市場情緒膨脹。**'
            'Everything Money 的 Paul 也引 Charlie Munger 的話：'
            '「over long periods of time, the roic of a business is basically how much the business is '
            'going to grow at」——**長期而言，報酬率就是企業的成長速度上限。**',

            '**最後必須說這一條的限制。**'
            '**我的 ROIC 是稅前值**（資料庫沒有所得稅科目），'
            '約為稅後的 1.27 倍——**所以「ROIC≥10%」實際只要求稅後約 7.9%，門檻比原意寬鬆。**'
            '**如果門檻收緊到真正的稅後 10%，通過的股票會更少、品質可能更高，'
            '但也可能因為樣本太小而失去統計意義。這一點我沒有測，是本卡的缺口。**',
        ],
        verdict=(
            f'兩套方法唯一共同站得住的一條。Phil Town 的「ROIC≥10%」單獨使用超額 '
            f'+{PF["pt1_roic"]["excess"]*100:.2f}pp（13 年贏 {PF["pt1_roic"]["beat"]} 年）；'
            f'Everything Money 的「5 年 ROIC>9%」超額 '
            f'+{EF["em2_roic5"]["excess"]*100:.2f}pp（14 年贏 {EF["em2_roic5"]["beat"]} 年）。'
            '**兩者算法不同（營業利益 vs 營業現金流），卻同時為正，'
            '這是本批唯一具跨方法一致性的正面結果。**'
            '**但幅度僅約 1.1~1.2 個百分點，勝率也只是略優於半數，'
            '應理解為「最不像噪音的一條」，而非穩健的超額報酬來源。**'
            '**且本測試的 ROIC 為稅前值（門檻實質放寬至稅後約 7.9%），未測稅後嚴格版。**'
        ),
        evidence=(
            '同一批資料與方法（553 檔、2011-2026 每年 6/30 再平衡、當時點財報）。'
            '本卡為兩套方法的「各條單獨使用」結果交叉比對（非組合）。'
            'Phil Town 版 ROIC＝營業利益（10 年平均）÷（股東權益＋長期負債）；'
            'Everything Money 版 ROIC＝營業現金流（5 年合計）÷（總負債＋股東權益）。'
            '**兩者皆為稅前近似值**（XBRL 無所得稅科目）。'
            '「勝過基準年數」＝該條件篩出的組合報酬高於同年全體等權的年數。'
            '**未計算這兩條的 t 值**（與主測試的樣本結構不同），'
            '故「顯著性」未經檢定，僅以超額與勝率描述。'
            '存活者偏差未修正；未計交易成本。'
        ),
    ))

    return cards
