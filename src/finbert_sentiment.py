import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import re

MODEL = "ProsusAI/finbert"
LABELS = ["negative", "neutral", "positive"]
CONFIDENCE_THRESHOLD = 0.55

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.eval()

# -------------------------
# Helper: detect questions
# -------------------------
def is_question(text: str) -> bool:
    text = text.strip().lower()
    if "?" in text:
        return True
    if re.match(r"^(how|what|which|why|can|should|is|are|need)\b", text):
        return True
    return False


def run_finbert():
    df = pd.read_csv("data/raw/posts.csv")

    sentiments = []
    scores = []

    for text in tqdm(df["text"], desc="FinBERT sentiment"):
        text = str(text).strip()

        # 1️⃣ Force neutral for questions
        if is_question(text):
            sentiments.append("neutral")
            scores.append(0.0)
            continue

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128
        )

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1)[0]

        idx = torch.argmax(probs).item()
        confidence = probs[idx].item()
        label = LABELS[idx]

        # 2️⃣ Confidence threshold → neutral
        if confidence < CONFIDENCE_THRESHOLD:
            sentiments.append("neutral")
            scores.append(0.0)
            continue

        # 3️⃣ Signed sentiment score
        if label == "positive":
            sentiments.append("positive")
            scores.append(confidence)
        elif label == "negative":
            sentiments.append("negative")
            scores.append(-confidence)
        else:
            sentiments.append("neutral")
            scores.append(0.0)

    df["sentiment"] = sentiments
    df["score"] = scores

    df.to_csv("data/processed/sentiment.csv", index=False)

