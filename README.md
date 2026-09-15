# 財經新聞情緒分析 Multi-Agent Pipeline

用 LangGraph 串接四個 Agent，自動抓取鉅亨網台股新聞、判讀情緒、生成白話總結，
並用 LLM-as-a-Judge 做自我校正，架構參考自「Self-Correcting Multi-Agent System
for TOEFL Reading Prep」專案的 Retrieval → Generation → Evaluation + 條件邊重試設計。

## 架構

```
scraper -> sentiment -> summary -> judge --(不通過且未達重試上限)--> summary（重試，上限 3 次）
                                       |
                                       +--(通過 或 已達重試上限)--> 結束
```

| Agent | 檔案 | 職責 | 是否呼叫 LLM |
|---|---|---|---|
| Scraper | `cnyes_sentiment_agent/agents/scraper_agent.py` | 抓取鉅亨網公開 API 新聞 | 否 |
| Sentiment | `cnyes_sentiment_agent/agents/sentiment_agent.py` | 逐則新聞打情緒分數（五檔） | 是（GPT-4o） |
| Summary | `cnyes_sentiment_agent/agents/summary_agent.py` | 彙整成白話多空總結 | 是（GPT-4o） |
| Judge | `cnyes_sentiment_agent/agents/judge_agent.py` | 檢查方向一致性、幻覺、格式 | 部分是（幻覺檢查用 GPT-4o，其餘規則式） |

其餘檔案：
- `cnyes_sentiment_agent/state.py`：定義所有 Agent 共用的 `PipelineState`（TypedDict），
  以及五檔情緒與數值互轉的對照表
- `cnyes_sentiment_agent/graph.py`：組裝 LangGraph StateGraph，包含 retry 的條件邊邏輯
- `cnyes_sentiment_agent/agents/_retry_utils.py`：三個會呼叫 GPT-4o 的 Agent 共用的
  API 重試邏輯（Rate Limit 指數退避、暫時性錯誤固定重試）
- `cnyes_sentiment_agent/main.py`：主程式進入點，跑完整條 pipeline 並印出結果、存 debug JSON

## 為什麼這樣設計（工程決策）

**1. 把「確定性邏輯」和「LLM 生成」分開**
Scraper Agent 完全不呼叫 LLM，純粹是 API 呼叫與資料整理，這讓抓新聞這一步
速度快、成本低、結果可重現。只有真正需要語言理解與生成能力的步驟
（情緒判讀、摘要生成、幻覈事實查核）才呼叫 GPT-4o。

**2. Judge 的檢查分三層，只有必要的一層才花錢呼叫 LLM**
- 方向一致性（`_check_direction_consistency`）：規則式關鍵字比對，免費、毫秒級
- 字數檢查（`_check_length`）：規則式，免費
- 幻覺檢查（`_check_hallucination_llm`）：需要理解語意才能做，才呼叫 GPT-4o

這個分層設計讓大部分「總結格式明顯有問題」的案例不需要多花一次 API 呼叫就能被攔截，
只有語意層級的問題才動用 LLM-as-a-Judge。

**3. 幻覺檢查從關鍵字比對升級成 LLM-as-a-Judge 的過程，是這個專案最值得講的部分**
最早期版本用簡單的關鍵字比對（總結命中任一原始標題關鍵字就算過），
但實測發現這種寬鬆比對抓不到「大部分真實、只夾雜一兩個編造公司」的局部幻覺——
例如總結提到的公司名稱在幾十則新聞列表裡完全找不到對應報導，關鍵字比對卻因為
其他部分文字重疊而誤判通過。因此改用 GPT-4o 逐一比對總結中的公司名稱
與原始新聞標題，`temperature=0` 確保事實查核的穩定性。這是「先上線一個簡單版本
發現盲點，再針對盲點升級」的真實迭代過程，不是一開始就設計對。

**4. Retry 迴圈只回到 summary，不回到 sentiment**
`graph.py` 的條件邊設計是：Judge 判定不通過就跳回 Summary Agent 重新生成，
而不是跳回 Sentiment Agent 重新判讀。原因是同樣輸入下，情緒判讀分數本身穩定性高，
真正容易出錯、需要重新生成的是「總結的文字表達」（例如摘要提到查無此新聞的公司名稱，
或總結文字方向與整體檔位矛盾），把重試範圍限縮在真正容易出錯的環節，
可以省下不必要的 API 呼叫。

**5. Few-shot 範例是根據實測資料反覆校準出來的，不是憑感覺寫的**
`sentiment_agent.py` 的 few-shot 範例經過一次基於 25 則人工標註樣本的校準：
實測發現模型有「強度收斂」傾向——完全一致率只有 56%，但放寬到「相鄰檔位也算對」
的容忍標準後，一致率高達 96%。這代表模型「方向判斷得準，但不太敢下大多/大空
這種極端判斷」，即使新聞訊號（例如 EPS 暴衝、爆量漲停）已經很明確也一樣。
針對這個發現，prompt 加入了明確的強度判斷準則與大多/小多、大空/小空的對比範例，
引導模型不要傾向選中庸選項。

## 安裝

```bash
pip install -r requirements.txt
cp .env.example .env   # 填入你自己的 OPENAI_API_KEY
export OPENAI_API_KEY="your-key-here"
```

## 執行

在專案根目錄（本 README 所在目錄，`cnyes_sentiment_agent/` 套件的**上一層**）執行，
因為套件內部使用了相對匯入：

```bash
python -m cnyes_sentiment_agent.main
```

執行完會在終端機印出整體情緒檔位、驗證結果、重試次數、五檔分布統計，
並把完整結果（含每則新聞標題、標籤、判讀理由）存成 `pipeline_output.json`，
方便人工檢查有沒有幻覺、分布是否合理，不用只看終端機截斷輸出。

## 測試

```bash
pytest tests/ -v
```

`tests/test_judge_agent.py` 把 Judge Agent 的三層檢查都寫成正式測試：
- 方向一致性、字數檢查是規則式邏輯，直接測試即可
- 幻覺檢查用 `unittest.mock` 假造 GPT-4o 回應，把原本只是手動跑一次、
  人工目視確認的「2 個對抗性測試案例（乾淨摘要應放行、混入編造公司名稱應攔截）」
  變成可以重複執行、不需要真的呼叫 API、不會因為 LLM 輸出不穩定而 flaky 的測試

## 驗證指標

- **Judge Agent 幻覺偵測**：2/2 對抗性測試案例通過（乾淨總結正確放行、混入編造公司名稱正確攔截並精準指出幻覺內容），現已寫成 `tests/test_judge_agent.py` 中可重複執行的測試
- **端到端流程**：多次真實新聞測試（27-60 則/次）皆順利跑完 scraper → sentiment → summary → judge 全流程，無異常中斷
- Pass rate（大量真實新聞端到端測試）：待累積更多次數後填入
- 情緒判讀與人工標註一致率：完全一致率 56%，相鄰檔位容忍一致率 96%（詳見上方 few-shot 校準說明）

## 已知觀察（待進一步驗證）

- 多次真實新聞測試中，五檔分布持續偏向「小多」（40-58%），「大空」出現次數為 0。
  可能原因：(1) 測試期間恰逢財報季，新聞本身正面居多；(2) 模型判讀存在樂觀傾向。
  尚未透過人工標註小樣本驗證何者為真，是後續要補的一項分析。

## 後續規劃

- [ ] 跑 20-30 則歷史新聞做端到端測試，記錄 pass rate 與平均重試次數
- [ ] 建立人工標註小樣本，驗證「小多」佔比偏高是真實市場反映還是模型偏誤
- [ ] 整理成果，寫展示介面（Streamlit 或簡單網頁）
- [ ] Scraper 目前只抓單一分類（tw_stock），視需求評估是否要支援多分類
