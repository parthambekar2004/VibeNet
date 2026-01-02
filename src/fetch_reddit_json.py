import requests
import pandas as pd
from datetime import datetime, timedelta
import time

HEADERS = {
    "User-Agent": "Mozilla/5.0 (topic-sentiment-project)"
}

BASE_URL = "https://www.reddit.com/search.json"


def fetch_year_top_posts(topic, year, posts_per_month=100, final_limit=1000):
    all_rows = []

    for month in range(1, 13):
        start = datetime(year, month, 1)
        if month == 12:
            end = datetime(year + 1, 1, 1)
        else:
            end = datetime(year, month + 1, 1)

        params = {
            "q": topic,
            "sort": "new",
            "limit": posts_per_month,
            "before": int(end.timestamp())
        }

        response = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=30)

        if response.status_code != 200:
            continue

        data = response.json().get("data", {}).get("children", [])

        for item in data:
            post = item["data"]
            created = datetime.utcfromtimestamp(post["created_utc"])

            if start <= created < end:
                all_rows.append({
                    "date": created.date(),
                    "text": (post.get("title", "") + " " + post.get("selftext", "")).strip(),
                    "upvotes": post.get("score", 0)
                })

        time.sleep(1)  # be polite to Reddit

    if not all_rows:
        raise RuntimeError("No posts collected")

    df = pd.DataFrame(all_rows)

    # 🧠 THIS IS THE KEY PART
    # Sort locally by upvotes → simulate TOP posts
    df = df.sort_values("upvotes", ascending=False)

    df = df.drop_duplicates(subset=["text"])
    df = df.head(final_limit)

    df.to_csv("data/raw/posts.csv", index=False)

    return df



