# 策略卡片製作標準（v1.0）

適用於 `strategy-microscope`（本目錄）。目的：任何 agent 接手都能產出同格式、同品質、可追溯、重建後不會消失的卡片。

**一句話原則：網站的 HTML 是建置產物，不是手寫稿。**

---

## 一、不可違反的三條

1. **`index.html`、`d_*.html`、`changelog.html` 一律由 `build_page.py` 產生，禁止手動編輯。**
   手改的內容會在下次重建時消失，且無法回溯。
2. **每個數字都要能指向檔案或指令。** 沒跑過的回測不能寫成「已驗證」；推論要標「未驗證」。
3. **產出由另一個人（或另一個 agent）核實。** 實作者不得自我驗收；核實看產物與數字，不看敘述。

---

## 二、標準流程（七步）

### 步驟 1：資料與回測
- 回測程式放在本目錄的 `*.py`（例如 `tier1.py`、`vcp_test.py`）。
- 輸出到 `out/`（不進版控）：`results.json`、圖表 PNG、曲線 JSON。
- 圖表命名要有前綴（`n12_*`、`t1_*`、`lao_*`…），供 `build_page.py` 自動載入。

### 步驟 2：寫卡片模組
建立 `<主題>_cards.py`，提供 `build()` 回傳 list of dict。照 `inst_cards.py` 的格式：

```python
def build():
    return [dict(
        id='opt-rules',            # 唯一、穩定、只用小寫與連字號
        author='期權研究',          # 來源（見第四節標籤規則）
        grade='B',                 # S/A/B/C/D/NA
        topic='選合約',
        market='us',               # us | crypto
        kind='backtest',           # backtest | mindset | unportable
        title='…',
        lead='一到三句，含關鍵數字…',
        metrics=[('名稱','數值'), …],   # 3–4 個
        body=['段落 1（可用 **粗體**）', '段落 2', …],
        imgs=['n12_opt_spread'],   # 對應 out/*.png；沒有圖表時見第五節
        verdict='判定一句話',
        evidence='資料來源、期間、成本、樣本外切分…',
    )]
```

### 步驟 3：掛進建置流程
在 `build_page.py` 的「追加卡片」區塊，照現有模式加：

```python
try:
    import opt_cards
    _new = opt_cards.build() + _new
except Exception as e:
    print(f'opt_cards 載入失敗：{type(e).__name__}: {e}')
```

### 步驟 4：版本與說明
- `VERSION` 升一版（語意版號；內容修正用 patch，新卡片用 minor）。
- `VERSION_NOTES` 寫：新增什麼、關鍵數字、**限制**。
- 不要手改 `changelog.html`；它由建置產生。

### 步驟 5：建置
```bash
python build_page.py
```
產出：`index.html`、`d_<id>.html`、`changelog.html`、`build_meta.json`、`versions/v<VERSION>/`。

### 步驟 6：驗證（缺一不可）
- [ ] 卡片數正確（`<article class="card"` 計數）。
- [ ] 每張新卡的「詳細資料」連結回 200，且標題與卡片一致。
- [ ] 卡片與詳情頁的市場、作者標籤一致（crypto 不得標成期權）。
- [ ] 至少一個數字可由 `out/` 或指令重現。
- [ ] 圖表存在；若刻意沒有圖表，頁面要有說明原因的文字。
- [ ] 推送後等 GitHub Pages 更新，用線上 URL 再檢查一次（不是只看本機）。

### 步驟 7：推送
```bash
git add index.html d_*.html changelog.html build_meta.json *.py
git commit -m "v<VERSION>：<一句話>"
git push origin HEAD
```
`out/` 與 `versions/` 依現行 .gitignore 不進版控；`versions/` 是本機回滾快照。

---

## 三、卡片內容規則

- **結構固定**：標題 → 一句話結論（lead）→ 3–4 個指標 → 正文（先講結論、再講理由、最後講限制）→ 判定 → 資料來源。
- **正文**：結論 → 理由 → 風險 → 下一步。不要流水帳。
- **數字**：一律寫出樣本數、期間、成本假設；百分比要標口徑（價格報酬／總報酬）。
- **限制**：每張卡必須有「限制」段落，寫存活者偏差、樣本長度、成本、多重檢定等。
- **用語**：少用縮寫與自創代號；首次出現的術語要用白話解釋。
- **評級**：
  - S/A：通過穩健性（逐年、剔除極端、成本加倍、樣本外）。
  - B：方向成立但有明確限制。
  - C：方向不成立或只在特定樣本成立。
  - D：與宣稱相反。
  - NA：無法回測（心法／缺資料），必須標 `kind='mindset'` 或 `unportable`。

---

## 四、標籤規則（最常出錯）

- `market`：美股 `us`、加密貨幣 `crypto`。**加密貨幣的卡不得標成期權研究。**
- `author`：寫真正的來源。
  - 期權內容（選合約、賣方策略）→ `期權研究`。
  - AOA（워뇨띠）→ `AOA（워뇨띠）`，market=`crypto`。
  - 機構申報 → `SEC 13F`；影片作者 → 頻道名。
- 同一張卡只能有一個 `author`；跨來源請在 `evidence` 說明。

---

## 五、圖表規則

- 每張量化卡片**至少一張圖**（淨值曲線、分布、分組柱狀圖皆可）。
- 沒有損益序列（例如橫斷面篩選、統計檢定）時：
  - 畫「分布／分組」圖，或
  - 在詳情頁放 `nochart` 說明區塊，寫清楚「為什麼沒有淨值曲線」。
- 圖片由 `build_page.py` 從 `out/*.png` 載入，經 `imgs` 字典注入；**不要在 HTML 手動貼 base64**。
- 圖表標題、座標軸、單位要完整；中文用 Noto Sans CJK。

---

## 六、禁止事項

- 手改 `index.html`／`d_*.html`／`changelog.html`。
- 兩張卡共用同一個詳情頁。
- 把未回測的卡片標成 `kind='backtest'`。
- 用「不知道」「資料不足」帶過而不寫清楚缺什麼。
- 宣稱做過沒做的事；編造數字；事後修改歷史紀錄（要改就加修訂註記）。
- 把憑證、金鑰、個資寫進頁面或程式碼。

---

## 七、核實清單（給非實作者）

1. 產物存在：`d_<id>.html`、卡片在 `index.html`、changelog 有該版本。
2. 數字可重現：挑一個指標，用 `out/` 檔或指令重算，誤差在四捨五入內。
3. 禁區未被動過：`git diff` 只含預期檔案；沒有手改建置產物的痕跡。
4. 標籤正確：market／author／kind 與內容相符。
5. 線上可用：GitHub Pages 的 URL 回 200，內容與本機一致。

---

## 八、例外處理

- 若必須在重建前先上線（例如緊急修正），仍以「改模組 → 重建」為準；手改只作為最後手段，且必須在 `changelog` 註記「手改，待遷移」。
- 若 `build_page.py` 因缺資料無法重建，先補資料或降級卡片（`kind='unportable'`），不要讓網站停在半成品狀態。

---

## 九、排序規則（三層，順序不可對調）

1. **可回測的卡片一律排在無法回測的前面**（即使後者評級更高、更新）。
2. 同一組內，**新分析在前**（沿用原始順序，不按評級重排）。
3. **有圖表的排在沒有圖表的前面**——沒有 `imgs` 的卡片，不論評級，一律排在同組最後。

實作位置：`build_page.py` 的 `cards = sorted(...)`，排序鍵為 `(card_kind, has_images)`，穩定排序保留組內原序。改動排序規則時，三層要一起看，不能只改一層。
