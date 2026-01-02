# VibeNet - Versatile Topic Sentiment Analysis 

A topic based sentiment analysis dashboard that analyzes Reddit discussions over a selected year using transformer-based NLP models.

### Dashboard Overview
![Dashboard](screenshot/dashboard.png)

## 🔍 What This Project Does
- Enter any topic (e.g. *crypto*, *AI stocks*, *mutual funds*)
- Select a year
- Fetch high-engagement Reddit posts for that year
- Analyze sentiment using **FinBERT**
- Visualize trends, distribution, and insights in a **local dashboard**

## 🧠 Key Features
- Topic-based sentiment
- Mainly focused on financial topics
- Year-wise analysis
- Transformer-based NLP (FinBERT)
- Bias-corrected sentiment logic
- Engagement-weighted sentiment
- Modern dark UI dashboard (Streamlit)

## 🛠 Tech Stack
- Python
- Streamlit
- PyTorch
- HuggingFace Transformers (FinBERT)
- Pandas
- Reddit public JSON endpoints

## 📈 Sentiment Logic
- Positive / Neutral / Negative classification
- Confidence thresholding to reduce false positives
- Neutral detection for questions
- Engagement-weighted aggregation

## 🖥 Running Locally

```bash
# Activate virtual environment
venv\Scripts\activate   # Windows
# source venv/bin/activate  # macOS/Linux

# Run the app
streamlit run app.py
