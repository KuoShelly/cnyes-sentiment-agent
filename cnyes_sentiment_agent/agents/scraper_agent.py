"""
Scraper Agent (LangGraph Node)
================================
負責：呼叫鉅亨網公開 API，抓取結構化新聞，寫回 state["raw_news"]。

這個節點是「確定性邏輯」（deterministic），不呼叫 LLM，
所以速度快、成本低、結果穩定可重現 —— 跟你 TOEFL 專案裡
「把 deterministic 邏輯和 LLM 生成分開」的架構原則是一致的。

TODO（Day 1 已完成大半，這裡列出還需要你確認/補完的部分）：
    [ ] 確認 category 參數是否要支援多分類（例如 tw_stock + headline）
    [ ] 視需求加入關鍵字篩選（例如只保留含特定股票代號的新聞）
"""

import requests
import datetime as dt
import time
import json
from typing import List

from ..state import NewsItemDict

CNYES_API = "https://api.cnyes.com/media/api/v1/newslist/category/tw_stock"


def _fetch_raw_news(days_back: int, max_pages: int, limit_per_page: int = 30) -> List[NewsItemDict]:
    end_at = int(dt.datetime.now().timestamp())
    start_at = int((dt.datetime.now() - dt.timedelta(days=days_back)).timestamp())

    all_items: List[NewsItemDict] = []
    seen_ids = set()

    for page in range(1, max_pages + 1):
        params = {"startAt": start_at, "endAt": end_at, "limit": limit_per_page, "page": page}

        try:
            resp = requests.get(CNYES_API, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, json.JSONDecodeError) as e:
            print(f"[Scraper Agent] 第 {page} 頁請求失敗：{e}")
            break

        page_data = data.get("items", {}).get("data", [])
        if not page_data:
            break

        for raw in page_data:
            news_id = raw.get("newsId")
            if news_id in seen_ids or news_id is None:
                continue

            title = (raw.get("title") or "").strip()
            if not title:
                continue

            publish_ts = raw.get("publishAt")
            publish_at = dt.datetime.fromtimestamp(publish_ts).isoformat() if publish_ts else ""
            categories = raw.get("category") or []

            all_items.append(NewsItemDict(
                news_id=news_id,
                title=title,
                content=(raw.get("content") or "").strip(),
                publish_at=publish_at,
                category=categories[0].get("name", "") if categories else "",
                keywords=raw.get("keyword") or [],
                url=f"https://news.cnyes.com/news/id/{news_id}",
            ))
            seen_ids.add(news_id)

        if len(page_data) < limit_per_page:
            break
        time.sleep(0.3)

    return all_items


def scraper_node(state: dict) -> dict:
    """LangGraph 節點入口。輸入完整 state，回傳要更新的欄位。"""
    days_back = state.get("days_back", 1)
    max_pages = state.get("max_pages", 2)

    news = _fetch_raw_news(days_back=days_back, max_pages=max_pages)
    print(f"[Scraper Agent] 抓取到 {len(news)} 則新聞")

    return {"raw_news": news}
