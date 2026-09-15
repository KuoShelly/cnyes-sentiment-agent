"""
主程式進入點
=============
執行方式：
    python -m cnyes_sentiment_agent.main
"""

import json
import datetime as dt

from .graph import build_graph


def _save_debug_output(final_state: dict, filepath: str = "pipeline_output.json"):
    """
    把完整結果（含每則新聞的標題、標籤、判讀結果）存成 JSON，
    方便人工檢查有沒有幻覺、分布是否合理，而不用只看終端機的截斷輸出。
    """
    debug_data = {
        "run_at": dt.datetime.now().isoformat(),
        "overall_level": final_state.get("overall_level"),
        "is_valid": final_state.get("is_valid"),
        "retry_count": final_state.get("retry_count"),
        "summary": final_state.get("summary"),
        "judge_feedback": final_state.get("judge_feedback"),
        "raw_news": final_state.get("raw_news", []),
        "sentiment_results": final_state.get("sentiment_results", []),
        "all_known_tags": sorted(set(
            tag
            for r in final_state.get("sentiment_results", [])
            for tag in r.get("tags", [])
        )),
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(debug_data, f, ensure_ascii=False, indent=2)

    print(f"\n完整結果已存至 {filepath}，可用來檢查幻覺與分布問題")


def run():
    app = build_graph()

    initial_state = {
        "days_back": 1,
        "max_pages": 2,
        "max_retries": 3,
        "retry_count": 0,
    }

    final_state = app.invoke(initial_state)

    print("\n" + "=" * 50)
    print("最終結果")
    print("=" * 50)
    print(f"整體情緒檔位：{final_state.get('overall_level', '中性')}")
    print(f"是否通過驗證：{final_state.get('is_valid')}")
    print(f"重試次數：{final_state.get('retry_count')}")
    print(f"\n總結內容：\n{final_state.get('summary')}")

    # 印出分布統計，方便直接在終端機看出偏斜問題
    from .state import SENTIMENT_LEVELS
    level_counts = {level: 0 for level in SENTIMENT_LEVELS}
    for r in final_state.get("sentiment_results", []):
        level_counts[r["level"]] = level_counts.get(r["level"], 0) + 1

    print("\n檔位分布：")
    total = len(final_state.get("sentiment_results", [])) or 1
    for level in SENTIMENT_LEVELS:
        count = level_counts[level]
        pct = count / total * 100
        print(f"  {level}：{count} 則（{pct:.0f}%）")

    _save_debug_output(final_state)

    return final_state


if __name__ == "__main__":
    run()
