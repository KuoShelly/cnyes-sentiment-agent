"""
Judge Agent 測試
=================
原本 README 提到「Judge Agent 幻覺偵測：2/2 對抗性測試案例通過」，
但這件事原本是手動跑一次、人工目視檢查結果，沒有寫成可以重複執行的測試。
這裡把它變成兩支正式的 pytest：

    test_hallucination_check_passes_clean_summary   乾淨摘要應該放行
    test_hallucination_check_catches_fake_company    混入編造公司名稱應該被攔截

呼叫 GPT-4o 的部分用 unittest.mock 假造回應，不需要真的打 API、
不需要 OPENAI_API_KEY，任何人 clone 下來都能直接跑，
也不會因為 LLM 輸出不穩定導致測試 flaky。

另外兩個 direction/length 檢查是純規則邏輯，直接測試即可，不需要 mock。
"""

import json
from unittest.mock import MagicMock, patch

from cnyes_sentiment_agent.agents.judge_agent import (
    _check_direction_consistency,
    _check_hallucination_llm,
    _check_length,
)


def _mock_openai_response(payload: dict):
    """組出跟 openai SDK 回傳結構一致的假回應，讓 _check_hallucination_llm 能正常解析。"""
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=json.dumps(payload)))]
    return mock_response


# ---- 方向一致性檢查（規則式，不需要 mock）----

def test_direction_consistency_flags_contradiction():
    issues = _check_direction_consistency("今日新聞整體偏空，市場氣氛悲觀", overall_level="大多")
    assert len(issues) == 1


def test_direction_consistency_accepts_matching_direction():
    issues = _check_direction_consistency("今日新聞整體偏多，法說會財測樂觀上修", overall_level="小多")
    assert issues == []


def test_direction_consistency_neutral_level_not_flagged():
    # 中性檔位不觸發多空矛盾檢查（規則只針對明確偏多/偏空的檔位設計）
    issues = _check_direction_consistency("市場觀望氣氛濃厚", overall_level="中性")
    assert issues == []


# ---- 字數檢查（規則式）----

def test_length_check_flags_too_short():
    assert len(_check_length("太短", min_len=60, max_len=220)) == 1


def test_length_check_accepts_normal_length():
    normal_summary = "今日新聞整體偏多" + "，市場氣氛樂觀" * 15  # 湊到規定字數區間內
    assert _check_length(normal_summary, min_len=60, max_len=220) == []


# ---- 幻覺檢查（LLM-as-a-Judge，mock 掉實際的 GPT-4o 呼叫）----

def test_hallucination_check_passes_clean_summary():
    raw_news = [{"title": "台積電法說會上修全年財測", "news_id": 1}]
    summary = "台積電法說會上修財測，市場氣氛偏多。"

    with patch(
        "cnyes_sentiment_agent.agents.judge_agent.call_gpt4o_with_retry",
        return_value=_mock_openai_response({"hallucinated_entities": []}),
    ):
        issues = _check_hallucination_llm(summary, raw_news)

    assert issues == []


def test_hallucination_check_catches_fake_company():
    raw_news = [{"title": "台積電法說會上修全年財測", "news_id": 1}]
    # 摘要提到了原始新聞清單完全沒出現的「鴻海」
    summary = "台積電與鴻海法說會齊上修財測，市場氣氛偏多。"

    with patch(
        "cnyes_sentiment_agent.agents.judge_agent.call_gpt4o_with_retry",
        return_value=_mock_openai_response({"hallucinated_entities": ["鴻海"]}),
    ):
        issues = _check_hallucination_llm(summary, raw_news)

    assert len(issues) == 1
    assert "鴻海" in issues[0]


def test_hallucination_check_skips_when_no_raw_news():
    # scraper 抓取失敗、raw_news 為空時應該跳過檢查而不是誤判，避免無中生有的假警報
    issues = _check_hallucination_llm("任何摘要內容", raw_news=[])
    assert issues == []
