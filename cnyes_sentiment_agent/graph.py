"""
Graph 組裝
============
把四個 Agent 接成一個 LangGraph StateGraph：

    scraper -> sentiment -> summary -> judge --(不通過且未達重試上限)--> summary
                                          |
                                          +--(通過 或 已達重試上限)--> END

這個「judge 檢查失敗就跳回 summary 重跑」的迴圈，就是複製你 TOEFL 專案裡
「Retrieval → Generation → Evaluation + 條件邊 + 重試上限」的核心設計。

注意：重試迴圈只回到 summary_node，不會回到 sentiment_node，
因為情緒判讀分數本身不太可能出錯（同樣輸入基本上結果穩定），
真正容易出錯、需要重新生成的是「總結的文字表達」。
如果之後發現 sentiment 也常常需要重跑，可以把迴圈往前延伸。
"""

from langgraph.graph import StateGraph, END

from .state import PipelineState
from .agents.scraper_agent import scraper_node
from .agents.sentiment_agent import sentiment_node
from .agents.summary_agent import summary_node
from .agents.judge_agent import judge_node


def _should_retry(state: dict) -> str:
    """條件邊：決定 judge 之後要去 END 還是回 summary 重跑。"""
    is_valid = state.get("is_valid", False)
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    if is_valid:
        return "end"
    if retry_count >= max_retries:
        print(f"[Graph] 已達重試上限（{max_retries} 次），強制結束並保留目前版本")
        return "end"

    print(f"[Graph] 未通過檢查，進行第 {retry_count} 次重試")
    return "retry"


def build_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("scraper", scraper_node)
    graph.add_node("sentiment", sentiment_node)
    graph.add_node("summary", summary_node)
    graph.add_node("judge", judge_node)

    graph.set_entry_point("scraper")

    graph.add_edge("scraper", "sentiment")
    graph.add_edge("sentiment", "summary")
    graph.add_edge("summary", "judge")

    graph.add_conditional_edges(
        "judge",
        _should_retry,
        {
            "retry": "summary",
            "end": END,
        },
    )

    return graph.compile()
