"""
共用重試邏輯
=============
sentiment_agent、summary_agent、judge_agent 都會呼叫 GPT-4o，
都可能遇到 Rate Limit（429）或暫時性 API 錯誤，抽成共用函式避免三邊各自維護一份。

策略：
    - Rate Limit：指數退避重試（2秒 -> 4秒 -> 8秒...），最多 5 次
    - 其他暫時性 API 錯誤：固定等 2 秒後重試一次
    - 重試次數用盡就把例外往外丟，讓呼叫端（例如 self_consistency_check.py）
      自行決定要不要跳過這一則、記錄失敗原因後繼續下一則
"""

import time

from openai import RateLimitError, APIError


def call_gpt4o_with_retry(client, messages, temperature, response_format=None, max_retries=5):
    kwargs = {"model": "gpt-4o", "messages": messages, "temperature": temperature}
    if response_format:
        kwargs["response_format"] = response_format

    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            wait_seconds = 2.0 * (2 ** attempt)
            print(f"[Retry] 遇到 Rate Limit，等待 {wait_seconds:.0f} 秒後重試"
                  f"（第 {attempt + 1}/{max_retries} 次）")
            time.sleep(wait_seconds)
        except APIError as e:
            if attempt == max_retries - 1:
                raise
            print(f"[Retry] API 錯誤：{e}，2 秒後重試")
            time.sleep(2.0)
