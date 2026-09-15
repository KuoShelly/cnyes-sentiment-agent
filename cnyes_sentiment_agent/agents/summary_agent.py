"""
Summary Agent (LangGraph Node)
=================================
負責：讀取 state["sentiment_results"]，用 GPT-4o 生成一段白話多空總結，
      並計算 overall_level（五檔彙總結果），寫回 state["summary"] 和 state["overall_level"]。

如果是「重試」（Judge Agent 判定不合格），這個節點會被重新呼叫，
這時候 state["judge_feedback"] 會有上一輪的問題描述，
要把這個 feedback 加進 prompt，讓 GPT-4o 知道哪裡要修正。

TODO（Day 3 你要確認的部分）：
    [ ] 總結的語氣要偏機構分析師，還是更口語化？
    [ ] 是否要限制字數（例如 100-150 字）？
"""

import os
from typing import List

from openai import OpenAI

from ..state import SentimentResultDict, LEVEL_TO_VALUE, VALUE_TO_LEVEL
from ._retry_utils import call_gpt4o_with_retry

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

SYSTEM_PROMPT = """你是一位財經分析師，負責把多則新聞的情緒判讀結果彙整成一段
給投資人看的白話總結。

規則：
1. 總結必須明確指出「今日新聞整體偏多、偏空、或中性」
2. 必須提到 1-2 個主要驅動原因（引用新聞中的關鍵字或標籤）
3. 禁止捏造原始新聞中沒有出現的公司名稱或數字
4. 字數控制在 100-150 字之間，語氣專業但易讀
"""


def _build_user_prompt(sentiment_results: List[SentimentResultDict], overall_level: str, feedback: dict) -> str:
    lines = [f"整體情緒檔位：{overall_level}\n"]
    lines.append("個別新聞判讀：")
    for r in sentiment_results:
        lines.append(f"- [{r['level']}] {r['title']}（理由：{r['reason']}，標籤：{', '.join(r['tags'])}）")

    prompt = "\n".join(lines)

    # 如果是重試，把上一輪 Judge 的問題附上去，讓模型知道要修正什麼
    if feedback and feedback.get("issues"):
        issues_text = "；".join(feedback["issues"])
        prompt += f"\n\n【上一輪總結被退回，原因：{issues_text}。請修正後重新生成】"

    return prompt


def _compute_overall_level(sentiment_results: List[SentimentResultDict]) -> str:
    """
    把多篇新聞的五檔標籤彙總成一個整體檔位。
    做法：五檔轉數值 -> 算平均 -> 四捨五入到最近的整數 -> 轉回檔位文字。
    這只是彙總多筆類別型資料的標準做法，最終仍然只輸出五檔文字，
    不會把數字暴露給使用者或 LLM prompt。
    """
    if not sentiment_results:
        return "中性"

    values = [LEVEL_TO_VALUE[r["level"]] for r in sentiment_results]
    avg_value = sum(values) / len(values)
    rounded = max(-2, min(2, round(avg_value)))
    return VALUE_TO_LEVEL[rounded]


def summary_node(state: dict) -> dict:
    """LangGraph 節點入口。"""
    sentiment_results: List[SentimentResultDict] = state.get("sentiment_results", [])
    overall_level = _compute_overall_level(sentiment_results)

    prev_feedback = state.get("judge_feedback", {})
    user_prompt = _build_user_prompt(sentiment_results, overall_level, prev_feedback)

    response = call_gpt4o_with_retry(
        client,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.5,
    )

    summary = response.choices[0].message.content.strip()
    print(f"[Summary Agent] 總結（整體檔位 {overall_level}）：{summary[:50]}...")

    return {"summary": summary, "overall_level": overall_level}
