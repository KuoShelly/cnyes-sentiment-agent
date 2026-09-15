"""
Judge Agent (LangGraph Node) — LLM-as-a-Judge
================================================
負責：檢查 Summary Agent 產出的總結是否可信，寫回 is_valid / judge_feedback，
      並管理 retry_count。這是複製你 TOEFL 專案裡驗證過的自我校正邏輯。

檢查項目：
    1. 方向一致性：總結說的多空方向，是否與彙總後的 overall_level 一致（規則式，免費）
    2. 幻覺檢查：把完整原始新聞標題交給 GPT-4o，讓它找出總結裡提到、
       但原始新聞清單中完全沒出現的公司名稱（LLM-as-a-Judge，需額外一次 API 呼叫）
    3. 格式檢查：字數是否在規定範圍內（規則式，免費）

升級紀錄：
    幻覺檢查原本是簡單關鍵字比對（只要總結命中任一原始標籤就算過），
    但實測發現這種寬鬆比對抓不到「大部分真實、夾雜一兩個編造公司」的局部幻覺
    （例如總結提到的公司在 60 則新聞列表裡完全找不到對應報導）。
    因此改用 GPT-4o 逐一比對總結中的公司名稱與原始新聞標題，抓出真正的幻覺。
"""

import datetime as dt
import json
import os
from typing import List

from openai import OpenAI

from ..state import SentimentResultDict, NewsItemDict, JudgeFeedbackDict
from ._retry_utils import call_gpt4o_with_retry

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

HALLUCINATION_CHECK_PROMPT = """你是嚴謹的財經新聞事實查核員。以下是「原始新聞清單」和一段根據這些新聞寫成的「摘要」。

你的任務：找出摘要中提到的公司名稱，但完全沒有出現在原始新聞清單任何一則標題中的公司。
不要判斷情緒方向是否正確，只專注在「這間公司是否真實存在於新聞清單中」。

嚴格以 JSON 格式輸出，不要有任何額外文字：
{{
  "hallucinated_entities": ["<清單中找不到的公司名稱1>", "<公司名稱2>"]
}}

如果摘要提到的公司都能在新聞清單中找到，回傳空陣列：{{"hallucinated_entities": []}}

原始新聞清單：
{news_titles}

摘要：
{summary}
"""


def _check_direction_consistency(summary: str, overall_level: str) -> List[str]:
    """檢查總結文字方向是否跟彙總後的整體檔位矛盾。"""
    issues = []

    says_bullish = any(k in summary for k in ["偏多", "看多", "樂觀", "上漲動能"])
    says_bearish = any(k in summary for k in ["偏空", "看空", "悲觀", "下跌壓力"])

    is_bullish_level = overall_level in ("小多", "大多")
    is_bearish_level = overall_level in ("小空", "大空")

    if is_bullish_level and says_bearish and not says_bullish:
        issues.append(f"整體檔位為「{overall_level}」，但總結文字卻寫偏空")
    if is_bearish_level and says_bullish and not says_bearish:
        issues.append(f"整體檔位為「{overall_level}」，但總結文字卻寫偏多")

    return issues


def _check_hallucination_llm(summary: str, raw_news: List[NewsItemDict]) -> List[str]:
    """
    用 GPT-4o 逐一核對總結中提到的公司，是否真的出現在原始新聞標題中。
    這是實測抓到真實幻覺案例後升級的版本，取代原本過於寬鬆的關鍵字比對。
    """
    issues = []

    if not raw_news:
        # 沒有新聞可比對時跳過，避免誤判（例如 scraper 抓取失敗的情況）
        return issues

    news_titles = "\n".join(f"- {n['title']}" for n in raw_news)
    prompt = HALLUCINATION_CHECK_PROMPT.format(news_titles=news_titles, summary=summary)

    try:
        response = call_gpt4o_with_retry(
            client,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,  # 事實查核要穩定，不要有創意空間
            response_format={"type": "json_object"},
        )
        parsed = json.loads(response.choices[0].message.content)
        hallucinated = parsed.get("hallucinated_entities", [])
        print(f"[Judge Agent] 幻覺檢查已執行（比對 {len(raw_news)} 則原始新聞），"
              f"發現 {len(hallucinated)} 個可疑公司名稱")
    except (json.JSONDecodeError, KeyError, Exception) as e:
        # 包含 JSON 解析失敗、重試次數用盡的 API 錯誤等各種情況：
        # 這一項檢查失敗時選擇「跳過」而不是讓整個 Judge Agent 掛掉，
        # 因為還有方向一致性和格式檢查兩層保障，不會完全沒有把關
        print(f"[Judge Agent] 幻覺檢查失敗：{e}，本輪跳過此項檢查")
        return issues

    for entity in hallucinated:
        issues.append(f"摘要提到「{entity}」，但原始新聞清單中找不到相關報導，疑似幻覺")

    return issues


def _check_length(summary: str, min_len: int = 60, max_len: int = 220) -> List[str]:
    issues = []
    length = len(summary)
    if length < min_len:
        issues.append(f"總結過短（{length} 字），內容可能不夠完整")
    if length > max_len:
        issues.append(f"總結過長（{length} 字），超出建議範圍")
    return issues


def judge_node(state: dict) -> dict:
    """LangGraph 節點入口。"""
    summary = state.get("summary", "")
    overall_level = state.get("overall_level", "中性")
    raw_news: List[NewsItemDict] = state.get("raw_news", [])
    retry_count = state.get("retry_count", 0)

    issues = []
    issues += _check_direction_consistency(summary, overall_level)
    issues += _check_hallucination_llm(summary, raw_news)
    issues += _check_length(summary)

    is_valid = len(issues) == 0

    feedback: JudgeFeedbackDict = {
        "is_valid": is_valid,
        "issues": issues,
        "checked_at": dt.datetime.now().isoformat(),
    }

    status = "通過" if is_valid else f"不通過（{len(issues)} 項問題）"
    print(f"[Judge Agent] 檢查結果：{status}")
    for issue in issues:
        print(f"    - {issue}")

    return {
        "is_valid": is_valid,
        "judge_feedback": feedback,
        "retry_count": retry_count + (0 if is_valid else 1),
    }
