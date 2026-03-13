---
title: Water Billing Verifier
emoji: 💧
colorFrom: blue
colorTo: cyan
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
---

# Water Billing RAG Verifier

Streamlit + Qdrant Cloud + OpenAI RAG system for verifying BWWB water billing.

## Setup

```bash
pip install -r requirements.txt
```

## Environment Variables

Copy `.env.example` to `.env` and fill in your keys:

```
OPENAI_API_KEY=sk-...
QDRANT_URL=https://xxxx.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key
```

Get Qdrant Cloud credentials free at: https://cloud.qdrant.io

## Usage

### Step 1 — Ingest (run once)

```bash
python ingest.py
```

### Step 2 — Run the app

```bash
streamlit run app.py
```

## Deploying to Hugging Face Spaces

1. Create a new Space (Streamlit SDK)
2. Push this repo
3. Add env vars in Space Settings → Variables and Secrets:
    - `OPENAI_API_KEY`
    - `QDRANT_URL`
    - `QDRANT_API_KEY`

## .gitignore

```
.env
csv/
output/
chroma_db/
```
