import sys
import subprocess

from src.fetch_reddit_json import fetch_year_top_posts
from src.finbert_sentiment import run_finbert

# --------------------------------
# Args: topic, year
# --------------------------------
if len(sys.argv) < 3:
    raise ValueError("Usage: python main.py <topic> <year>")

TOPIC = sys.argv[1]
YEAR = int(sys.argv[2])

# --------------------------------
# Config
# --------------------------------
POSTS_PER_MONTH = 100
FINAL_LIMIT = 300   # increase later if needed

print(f"🔍 Topic: {TOPIC}")
print(f"📅 Year: {YEAR}")
print(f"📦 Max posts: {FINAL_LIMIT}")

# --------------------------------
# Fetch Reddit data
# --------------------------------
fetch_year_top_posts(
    topic=TOPIC,
    year=YEAR,
    posts_per_month=POSTS_PER_MONTH,
    final_limit=FINAL_LIMIT
)

# --------------------------------
# Run sentiment + aggregation
# --------------------------------
run_finbert()
subprocess.run([sys.executable, "src/aggregate.py"], check=True)

print("✅ Pipeline completed")

