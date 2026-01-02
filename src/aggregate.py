import pandas as pd

df = pd.read_csv("data/processed/sentiment.csv", parse_dates=["date"])

df["month"] = df["date"].dt.to_period("M").astype(str)
df["year"] = df["date"].dt.year

# Weighted sentiment
df["weighted_score"] = df["score"] * df["upvotes"]

monthly = (
    df.groupby("month")
      .agg(
          sentiment=("score", "mean"),
          weighted_sentiment=("weighted_score", "sum"),
          total_upvotes=("upvotes", "sum"),
          posts=("score", "count")
      )
      .reset_index()
)

monthly["final_weighted_sentiment"] = (
    monthly["weighted_sentiment"] / monthly["total_upvotes"]
)

yearly = (
    df.groupby("year")
      .agg(
          sentiment=("score", "mean"),
          weighted_sentiment=("weighted_score", "sum"),
          total_upvotes=("upvotes", "sum"),
          posts=("score", "count")
      )
      .reset_index()
)

yearly["final_weighted_sentiment"] = (
    yearly["weighted_sentiment"] / yearly["total_upvotes"]
)

monthly.to_csv("data/processed/monthly.csv", index=False)
yearly.to_csv("data/processed/yearly.csv", index=False)

