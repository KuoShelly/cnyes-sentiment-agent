"""
Sentiment Agent (LangGraph Node)
===================================
負責：讀取 state["raw_news"]，用 GPT-4o 逐則新聞打情緒分數，
      寫回 state["sentiment_results"]。

這是你在 CMONEY 已經做過的部分（Gemini API 版本），這裡改用 GPT-4o，
架構邏輯完全一樣：few-shot prompt + schema-constrained output（強制 JSON）。

TODO（這是 Day 2 你要親手填的部分，我先把架構和一個範例 prompt 搭好）：
    [ ] 確認 few-shot 範例是否符合你想要的判讀風格（機構語氣 vs 散戶語氣）
    [ ] 決定要不要批次呼叫（一次丟多則新聞 vs 逐則呼叫，影響成本和速度）
    [ ] 補上 OPENAI_API_KEY 環境變數
"""

import json
import os
import time
from typing import List

from openai import OpenAI

from ..state import NewsItemDict, SentimentResultDict, SENTIMENT_LEVELS
from ._retry_utils import call_gpt4o_with_retry

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ---- Few-shot 範例 ----
# v2 更新（根據 25 則人工標註樣本比對後調整）：
#   實測發現模型有「強度收斂」傾向：完全一致率只有 56%，但相鄰容忍一致率高達 96%，
#   代表方向判斷是準的，問題出在不敢下「大多/大空」這種極端判斷，
#   例如「EPS暴衝、爆量漲停」這類訊號明確強烈的新聞，也常被判成溫和的「小多」。
#   因此這版加入明確的強度判斷準則，並補充「小多 vs 大多」「小空 vs 大空」的對比範例，
#   讓模型有更清楚的依據去區分程度，而不是傾向選中庸選項。
SYSTEM_PROMPT = f"""你是一位專業的財經新聞情緒分析師。針對輸入的新聞標題與內容，
判斷其對台股市場的情緒傾向，並嚴格以 JSON 格式輸出，不要有任何額外文字。

情緒檔位只能是以下五個之一：{"、".join(SENTIMENT_LEVELS)}

強度判斷準則（這是你最容易判斷失準的地方，請特別注意）：
- 「大多」/「大空」：新聞包含具體且顯著的量化訊號，例如
  EPS/財測大幅上修或下修、爆量漲停或跌停、目標價大幅調整、公司營運方向出現決定性轉折。
  不要因為「不確定要不要下這麼強的判斷」就迴避大多/大空，只要訊號本身夠具體強烈就應該用。
- 「小多」/「小空」：訊號存在但幅度普通、屬於一般性正面或負面消息，
  沒有強烈的量化佐證，或只是符合預期的常態性更新。
- 「中性」：無明顯方向，或純粹的數據公布、公告類新聞。

輸出格式：
{{
  "level": "<五檔之一>",
  "reason": "<不超過 30 字的判讀理由>",
  "tags": ["<關鍵字1>", "<關鍵字2>"]
}}

範例：
輸入：台積電法說會上修全年財測，AI 需求強勁帶動營收成長
輸出：{{"level": "大多", "reason": "法說會財測上修，AI需求強勁", "tags": ["台積電", "法說會", "財測上修"]}}

輸入：某電子零組件廠第三季營收月增5%，符合市場預期
輸出：{{"level": "小多", "reason": "營收成長但幅度普通，符合預期", "tags": ["營收", "月增"]}}

輸入：某公司Q2業外大進補，EPS暴衝至13元，單日爆量漲停站回年線
輸出：{{"level": "大多", "reason": "EPS暴衝且爆量漲停，訊號強烈明確", "tags": ["EPS暴衝", "爆量漲停"]}}

輸入：央行意外升息半碼，市場擔憂資金成本上升
輸出：{{"level": "小空", "reason": "意外升息，市場擔憂資金成本", "tags": ["央行", "升息", "資金成本"]}}

輸入：某公司EPS預估遭大幅下修，目標價同步大砍
輸出：{{"level": "大空", "reason": "EPS與目標價同步大幅下修，訊號明確偏空", "tags": ["EPS下修", "目標價下修"]}}

輸入：某上市公司公告第三季財務報告，營收與去年同期持平
輸出：{{"level": "中性", "reason": "財報數字持平，無明顯驚喜或利空", "tags": ["財報", "營收持平"]}}
"""


def _score_single_news(news: NewsItemDict, temperature: float = 0.3) -> SentimentResultDict:
    user_prompt = f"標題：{news['title']}\n內容：{news['content'][:500]}"

    response = call_gpt4o_with_retry(
        client,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        print(f"[Sentiment Agent] JSON 解析失敗，新聞：{news['title']}")
        parsed = {"level": "中性", "reason": "解析失敗，預設為中性", "tags": []}

    level = parsed.get("level", "中性")
    if level not in SENTIMENT_LEVELS:
        # 模型偶爾可能吐出不在五檔內的字串，防呆處理，避免下游計算出錯
        print(f"[Sentiment Agent] 回傳檔位「{level}」不在允許範圍內，改設為中性")
        level = "中性"

    return SentimentResultDict(
        news_id=news["news_id"],
        title=news["title"],
        level=level,
        reason=parsed.get("reason", ""),
        tags=parsed.get("tags", []),
    )


def sentiment_node(state: dict) -> dict:
    """LangGraph 節點入口。"""
    raw_news: List[NewsItemDict] = state.get("raw_news", [])

    results: List[SentimentResultDict] = []
    for news in raw_news:
        result = _score_single_news(news)
        results.append(result)
        print(f"[Sentiment Agent] {result['title'][:20]}... -> {result['level']}")
        time.sleep(0.3)  # 輕微節流，避免連續大量新聞時打到 rate limit

    return {"sentiment_results": results}


def score_news_multiple(news: NewsItemDict, n_runs: int = 5, temperature: float = 0.7) -> List[SentimentResultDict]:
    """
    對同一則新聞重複呼叫多次，用來檢查模型輸出穩不穩定（self-consistency）。
    這是驗證用的輔助函式，不影響正式 pipeline 的行為
    （正式流程的 sentiment_node 固定用 temperature=0.3，只呼叫一次）。

    temperature 預設調高到 0.7，是刻意的：temperature 越高，模型的隨機性越大，
    如果連在這種條件下輸出都還很穩定，代表模型對這則新聞真的有把握；
    反過來說，如果稍微提高隨機性就整個亂跳，也能更容易暴露出模糊案例。
    """
    results = []
    for _ in range(n_runs):
        results.append(_score_single_news(news, temperature=temperature))
        time.sleep(0.3)  # 輕微節流，這支腳本短時間內呼叫次數多，特別容易撞到 rate limit
    return results
