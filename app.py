import streamlit as st
import pandas as pd
import subprocess
import sys
import os

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="VibeNet - Versatile Topic Sentiment Analysis ",
    layout="wide"
)

from PIL import Image

reddit_logo = Image.open("assets/reddit.png")

st.image(
    reddit_logo,
    width=60
)

# -------------------------------------------------
# Custom CSS (dark dashboard look)
# -------------------------------------------------
st.markdown("""
<style>

/* ---- App background ---- */
.stApp {
    background-color: #000000;
}

/* ---- Main content container ---- */
.block-container {
    background-color: #000000;
    padding-top: 4rem;
    padding-bottom: 0rem;
}

/* ---- Remove Streamlit gradient ---- */
[data-testid="stAppViewContainer"] {
    background: #000000;
}

/* ---- Sidebar (if used) ---- */
[data-testid="stSidebar"] {
    background-color: #000000;
}

/* ---- Cards ---- */
.card {
    background-color: #0a0a0a;
    padding: 18px;
    border-radius: 14px;
    box-shadow: 0 0 0 1px #1f1f1f;
}

/* ---- Titles ---- */
.big-title {
    font-size: 38px;
    font-weight: 700;
    color: #ffffff;
}

.sub-title {
    color: #9aa0a6;
    margin-bottom: 20px;
}

/* ---- Charts background ---- */
[data-testid="stPlotlyChart"],
[data-testid="stChart"] {
    background-color: #000000 !important;
}

/* ---- Dataframes ---- */
[data-testid="stDataFrame"] {
    background-color: #000000;
}

/* ---- Buttons ---- */
button {
    background-color: #111 !important;
    color: white !important;
    border: 1px solid #222 !important;
}

/* ---- Inputs ---- */
input {
    background-color: #111 !important;
    color: white !important;
    border: 1px solid #222 !important;
}

/* ---- Success / info boxes ---- */
[data-testid="stAlert"] {
    background-color: #061f13 !important;
    border-left: 4px solid #1db954;
}

</style>
""", unsafe_allow_html=True)


# -------------------------------------------------
# Header
# -------------------------------------------------
st.markdown('<div class="big-title">VibeNet</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Reddit Topic Sentiment Intelligence</div>', unsafe_allow_html=True)

# -------------------------------------------------
# Topic input
# -------------------------------------------------
from datetime import datetime

current_year = datetime.now().year

col_topic, col_year = st.columns([3, 1])

with col_topic:
    topic = st.text_input(
        "",
        placeholder="Search a topic… (e.g. minecraft, bitcoin, ai stocks)"
    )

with col_year:
    year = st.selectbox(
        "Year",
        options=list(range(current_year, current_year - 6, -1))
    )

analyze = st.button("Analyze")


# -------------------------------------------------
# Run pipeline
# -------------------------------------------------
if analyze:
    if not topic.strip():
        st.warning("Please enter a topic")
        st.stop()

    with st.spinner(f"Running analysis for '{topic}' ({year}) …"):
        try:
            subprocess.run(
                [sys.executable, "main.py", topic, str(year)],
                check=True
            )
        except subprocess.CalledProcessError as e:
            st.error("Pipeline failed. Check terminal logs.")
            st.stop()

    st.success(f"Analysis complete for: {topic} ({year})")

# -------------------------------------------------
# Load data
# -------------------------------------------------
def load(path):
    return pd.read_csv(path) if os.path.exists(path) else None

monthly = load("data/processed/monthly.csv")
yearly = load("data/processed/yearly.csv")
sentiment = load("data/processed/sentiment.csv")

if monthly is None or sentiment is None:
    st.info("Run an analysis to see results")
    st.stop()

# -------------------------------------------------
# Metrics calculation
# -------------------------------------------------
avg_sent = monthly["sentiment"].mean()
weighted_sent = monthly["final_weighted_sentiment"].mean()
mentions = len(sentiment)

def sentiment_label(x):
    if x > 0.15:
        return "🟢 Bullish"
    elif x < -0.15:
        return "🔴 Bearish"
    return "🟡 Neutral"

# -------------------------------------------------
# KPI cards
# -------------------------------------------------
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="card">
        <h4>Overall Signal</h4>
        <h2>{sentiment_label(avg_sent)}</h2>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="card">
        <h4>Avg Sentiment</h4>
        <h2>{avg_sent:.2f}</h2>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="card">
        <h4>Weighted Sentiment</h4>
        <h2>{weighted_sent:.2f}</h2>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="card">
        <h4>Mentions</h4>
        <h2>{mentions}</h2>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# -------------------------------------------------
# Charts
# -------------------------------------------------
c1, c2 = st.columns(2)

# ----------------------------
# Sentiment Trend Card
# ----------------------------
with c1:
    st.markdown("""
    <div class="card">
        <h3>📈 Sentiment Trend</h3>
    """, unsafe_allow_html=True)

    st.line_chart(
        monthly.set_index("month")[["sentiment", "final_weighted_sentiment"]]
    )

    st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------
# Sentiment Distribution Card
# ----------------------------
with c2:
    st.markdown("""
    <div class="card">
        <h3>📊 Sentiment Distribution</h3>
    """, unsafe_allow_html=True)

    st.bar_chart(
        sentiment["sentiment"].value_counts()
    )

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# Interpretation box
# -------------------------------------------------
st.markdown("### Interpretation")

if avg_sent > 0.15:
    st.success(
        "Reddit sentiment is predominantly **positive**. "
        "Discussion tone suggests optimism and favorable perception."
    )
elif avg_sent < -0.15:
    st.error(
        "Reddit sentiment is predominantly **negative**. "
        "Discussion shows concern or bearish outlook."
    )
else:
    st.warning(
        "Reddit discussion is mostly **neutral**. "
        "Content is informational rather than opinion-driven."
    )

# -------------------------------------------------
# Raw data preview
# -------------------------------------------------
with st.expander("🔍 View sample posts"):
    st.dataframe(
        sentiment[["date", "sentiment", "score", "upvotes", "text"]].head(25),
        use_container_width=True
    )
