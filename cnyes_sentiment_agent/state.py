"""
共享狀態定義
=============
LangGraph 的每個節點（Agent）都會接收這個 state，並回傳「要更新的部分欄位」。
LangGraph 會自動把回傳的欄位 merge 回主 state，所以每個 Agent 不需要知道
其他 Agent 的完整實作，只需要知道自己要讀哪些欄位、寫哪些欄位。

資料流動順序：
    raw_news (Scraper 寫)
        -> sentiment_results (Sentiment Agent 寫)
            -> summary (Summary Agent 寫)
                -> is_valid / judge_feedback (Judge Agent 寫)
                    -> 不合格則 retry_count += 1，回到 Summary Agent 重跑
"""

from typing import TypedDict, List, Dict, Optional


class NewsItemDict(TypedDict):
    news_id: int
    title: str
    content: str
    publish_at: str
    category: str
    keywords: List[str]
    url: str


SENTIMENT_LEVELS = ["大空", "小空", "中性", "小多", "大多"]

# 五檔轉數值，僅供內部彙總計算用（例如多篇新聞的整體走勢），
# 不對外顯示數字，使用者看到的永遠是五檔文字標籤。
LEVEL_TO_VALUE = {"大空": -2, "小空": -1, "中性": 0, "小多": 1, "大多": 2}
VALUE_TO_LEVEL = {v: k for k, v in LEVEL_TO_VALUE.items()}


class SentimentResultDict(TypedDict):
    news_id: int
    title: str
    level: str              # 五檔之一："大空" / "小空" / "中性" / "小多" / "大多"
    reason: str              # 一句話理由，方便 Judge 和人工檢查
    tags: List[str]           # 關鍵字標籤，例如 ["法說會", "財報優於預期"]


class JudgeFeedbackDict(TypedDict):
    is_valid: bool
    issues: List[str]       # 不合格時列出具體問題，例如「摘要提到的公司在原始新聞中未出現」
    checked_at: str


class PipelineState(TypedDict, total=False):
    # ---- 輸入參數 ----
    days_back: int
    max_pages: int
    max_retries: int

    # ---- Scraper Agent 輸出 ----
    raw_news: List[NewsItemDict]

    # ---- Sentiment Agent 輸出 ----
    sentiment_results: List[SentimentResultDict]

    # ---- Summary Agent 輸出 ----
    summary: str
    overall_level: str      # 全部新聞彙總後的整體檔位，給 Judge 檢查一致性用

    # ---- Judge Agent 輸出 ----
    judge_feedback: JudgeFeedbackDict
    is_valid: bool
    retry_count: int
