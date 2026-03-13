import os, json, re
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="Water Billing Verifier",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; background: #0a0f1e; color: #ffffff; }
.stApp { background: #0a0f1e; }

section[data-testid="stSidebar"] { background: #0d1526 !important; border-right: 1px solid #1e3a5f !important; }
section[data-testid="stSidebar"] * { color: #ffffff !important; }
section[data-testid="stSidebar"] .stButton > button { width: 100% !important; }

[data-testid="stChatMessage"] { background: #0f1e35 !important; border: 1px solid #1e3a5f !important; border-radius: 8px !important; margin-bottom: 8px !important; }
[data-testid="stChatMessage"] * { color: #ffffff !important; }
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) { border: 1px solid #166534 !important; }

[data-testid="stChatInput"] textarea { background: #000000 !important; border: 1px solid #1e6fa8 !important; color: #ffffff !important; font-family: 'IBM Plex Mono', monospace !important; }
[data-testid="stChatInput"] textarea::placeholder { color: #7dd3fc !important; }

[data-testid="stBottom"], section[data-testid="stBottom"], section[data-testid="stBottom"] > div { background: #0a0f1e !important; }

p, li, span, div { color: #ffffff; }
h1, h2, h3 { color: #7dd3fc; }

.metric-card { background: #0d1e35; border: 1px solid #1e4a7a; border-radius: 6px; padding: 12px 16px; margin: 4px 0; font-family: 'IBM Plex Mono', monospace; font-size: 13px; }
.metric-ok  { border-left: 3px solid #22c55e; }
.metric-warn { border-left: 3px solid #f59e0b; }
.metric-bad  { border-left: 3px solid #ef4444; }
.tag { display: inline-block; padding: 2px 8px; border-radius: 3px; font-size: 11px; font-weight: 600; }
.tag-ok   { background: #14532d; color: #4ade80 !important; }
.tag-warn { background: #78350f; color: #fbbf24 !important; }
.tag-bad  { background: #7f1d1d; color: #fca5a5 !important; }

.stButton > button { background: #1e3a5f; color: #7dd3fc !important; border: 1px solid #1e6fa8; border-radius: 4px; }
.stButton > button:hover { background: #1e4a7a; }
code { background: #1e3a5f !important; color: #7dd3fc !important; }

table { color: #ffffff !important; }
th { color: #7dd3fc !important; background: #0d1e35 !important; }
td { color: #ffffff !important; }
tr { border-bottom: 1px solid #1e3a5f !important; }
</style>
""", unsafe_allow_html=True)

# ── Password gate ─────────────────────────────────────────────────────────────
APP_PASSWORD = os.environ.get("APP_PASSWORD", "")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown("# 💧 Water Billing Verifier")
    pwd = st.text_input("Enter password", type="password")
    if st.button("Login"):
        if APP_PASSWORD and pwd == APP_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password")
    st.stop()

# ── Keys from environment ─────────────────────────────────────────────────────
openai_api_key = os.environ.get("OPENAI_API_KEY", "")
qdrant_url     = os.environ.get("QDRANT_URL", "")
qdrant_api_key = os.environ.get("QDRANT_API_KEY", "")
hf_token       = os.environ.get("HF_TOKEN", "")

base_meter = {
    "5/8_INCH": 28.49, "3/4_INCH": 37.78, "1_INCH": 56.87,
    "1 1/2 inch": 102.9, "2_INCH": 161.11, "3_INCH": 340.73,
    "4_INCH": 513.87, "6_INCH": 988.45, "8_INCH": 1556.08, "10_INCH": 2212.0,
}
water_res = {"upto_3": 2.96, "4_to_12": 4.91, "greater_12": 8.58}

COLLECTION_NAME = "water_billing"
HF_DATASET_REPO = "ns00/water-billing-data"

@st.cache_resource
def load_resources():
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    model  = SentenceTransformer("all-MiniLM-L6-v2")
    oai    = OpenAI(api_key=openai_api_key)
    return client, model, oai

@st.cache_resource
def load_dataframe():
    path = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename="verified_water.csv",
        repo_type="dataset",
        token=hf_token,
    )
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    numeric_cols = (
        ["totalBilled", "totalBilledRev", "verification"] +
        [c for c in df.columns if any(x in c for x in ["BilledAmount", "BilledConsumption", "MeterCharge"])]
    )
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def compute_stats(df):
    mismatch_count = int((df["has_mismatch"] == "True").sum()) if "has_mismatch" in df.columns else None
    mismatch_str   = f"{mismatch_count:,}" if mismatch_count is not None else "N/A"
    return f"""
## Dataset Statistics (full dataset)
- Total billing rows: {len(df):,}
- Unique meters: {df["Meter"].nunique():,}
- Unique premises: {df["Premise"].nunique():,}
- Unique PODs: {df["POD"].nunique():,}
- Rate categories: {json.dumps(df["RateCategoryKey"].value_counts().to_dict())}
- Meter sizes: {json.dumps(df["Size"].value_counts().to_dict())}
- Average annual bill: ${df["totalBilled"].mean():,.2f}
- Total billed (all meters): ${df["totalBilled"].sum():,.2f}
- Rows with billing mismatches: {mismatch_str}
"""

AGGREGATE_KEYWORDS = [
    "how many", "count", "total", "average", "avg", "sum", "all meters",
    "all premises", "all pods", "list all", "how much", "percentage",
    "most common", "breakdown", "distribution", "statistics", "stats",
    "dataset", "overall", "across all", "entire",
]

def is_aggregate_query(query):
    return any(kw in query.lower() for kw in AGGREGATE_KEYWORDS)

def answer_aggregate(df, query):
    q = query.lower()
    lines = ["Aggregate query results from full dataset:"]
    if "how many meter" in q or "count meter" in q:
        lines.append(f"Unique meters: {df['Meter'].nunique():,}")
    if "how many premise" in q or "count premise" in q:
        lines.append(f"Unique premises: {df['Premise'].nunique():,}")
    if "how many pod" in q or "count pod" in q:
        lines.append(f"Unique PODs: {df['POD'].nunique():,}")
    if "how many row" in q or "total row" in q or "how many record" in q:
        lines.append(f"Total rows: {len(df):,}")
    if "mismatch" in q:
        if "has_mismatch" in df.columns:
            count = int((df["has_mismatch"] == "True").sum())
            lines.append(f"Rows with billing mismatches: {count:,} ({count/len(df)*100:.1f}%)")
        else:
            lines.append("Mismatch data not available in this dataset.")
    if "average" in q or "avg" in q:
        lines.append(f"Average annual bill: ${df['totalBilled'].mean():,.2f}")
    if "total billed" in q or "sum" in q:
        lines.append(f"Total billed across all meters: ${df['totalBilled'].sum():,.2f}")
    if "size" in q or "meter size" in q:
        lines.append(f"Meter size breakdown: {df['Size'].value_counts().to_dict()}")
    if "rate" in q or "category" in q:
        lines.append(f"Rate category breakdown: {df['RateCategoryKey'].value_counts().to_dict()}")
    if len(lines) == 1:
        lines.append(compute_stats(df))
    return "\n".join(lines)

def build_system_prompt(stats):
    return f"""You are a water billing verification assistant for BWWB (Birmingham Water Works Board).
## Rate Structure
Base meter charges ($/month): {json.dumps(base_meter, indent=2)}
Residential consumption rates ($/CCF): 0 CCF=$0, 1-3 CCF=${water_res['upto_3']}, 4-12 CCF=${water_res['4_to_12']}, >12 CCF=${water_res['greater_12']}
Formula: charge = base_meter[size] + (consumption x rate)
{stats}
## Your Job
1. LOOKUP queries: identify the row, explain monthly calculations, flag mismatches, summarize annual accuracy
2. AGGREGATE queries: answer using dataset statistics only — never guess
January excluded from annual verification (totalBilledRev). Be concise and precise. Use markdown tables.
"""

def retrieve(qdrant, model, query, n=3):
    numbers = re.findall(r"\d{5,}", query)
    if numbers:
        for num in numbers:
            for field in ("Meter", "Premise", "POD"):
                try:
                    results = qdrant.scroll(
                        collection_name=COLLECTION_NAME,
                        scroll_filter=Filter(must=[FieldCondition(key=field, match=MatchValue(value=num))]),
                        limit=n, with_payload=True,
                    )
                    if results[0]:
                        return [h.payload for h in results[0]]
                except Exception:
                    pass
    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=model.encode(query).tolist(),
        limit=n, with_payload=True,
    )
    return [r.payload for r in results.points]

def payload_to_doc(payload):
    lines = [
        f"Meter: {payload.get('Meter','?')}",
        f"Premise: {payload.get('Premise','?')}",
        f"POD: {payload.get('POD','?')}",
        f"Size: {payload.get('Size','?')}",
        f"RateCategory: {payload.get('RateCategoryKey','?')}",
        f"TotalBilled: {payload.get('totalBilled','?')}",
        f"TotalBilledExcludingJanuary: {payload.get('totalBilledRev','?')}",
        f"OverallVerificationRatio: {payload.get('verification','?')}",
        f"ServiceAddress: {payload.get('ServiceAddress','?')}",
    ]
    try:
        monthly = json.loads(payload.get("monthly_json", "{}"))
        lines.append("\nMonthly breakdown:")
        for m, v in monthly.items():
            flag_str = " MISMATCH" if v.get("flag") else ""
            lines.append(f"  {m.capitalize():12s}: billed={v['billed']:.2f}  cons={v['consumption']}  expected={v['expected']:.2f}  ratio={v['ratio']}{flag_str}")
    except Exception:
        pass
    return "\n".join(lines)

def render_monthly_table(monthly_json):
    try:
        monthly = json.loads(monthly_json)
    except Exception:
        return
    st.markdown("#### Monthly Breakdown")
    for m, v in monthly.items():
        ratio = v.get("ratio")
        flag  = v.get("flag", False)
        if v["billed"] == 0 and v["consumption"] == 0:
            css, tag = "metric-card", ""
        elif flag:
            css, tag = "metric-card metric-bad", '<span class="tag tag-bad">MISMATCH</span>'
        elif ratio is not None and abs(ratio - 1.0) < 0.01:
            css, tag = "metric-card metric-ok", '<span class="tag tag-ok">OK</span>'
        else:
            css, tag = "metric-card metric-warn", '<span class="tag tag-warn">CLOSE</span>'
        st.markdown(f"""<div class="{css}"><b>{m.capitalize()}</b> {tag}<br>
            Billed: <b>${v['billed']:.2f}</b> &nbsp;|&nbsp;
            Expected: <b>${v['expected']:.2f}</b> &nbsp;|&nbsp;
            Cons: {v['consumption']} CCF &nbsp;|&nbsp;
            Ratio: {ratio if ratio is not None else 'N/A'}</div>""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Water Billing RAG")
    st.markdown("---")
    st.markdown("**Example queries:**")
    for ex in [
        "Verify meter 15058104",
        "Show billing for premise 6101000075",
        "Is POD 110905 billed correctly?",
        "How many unique meters are there?",
        "How many rows have billing mismatches?",
        "What is the average annual bill?",
    ]:
        if st.button(ex, key=ex):
            st.session_state["inject_query"] = ex
    st.markdown("---")
    if st.button("Clear chat"):
        st.session_state.messages  = []
        st.session_state.last_meta = None
    if st.session_state.get("last_meta"):
        st.markdown("---")
        meta = st.session_state.last_meta
        st.markdown(f"**Meter:** `{meta.get('Meter')}`")
        st.markdown(f"**Premise:** `{meta.get('Premise')}`")
        st.markdown(f"**POD:** `{meta.get('POD')}`")
        st.markdown(f"**Size:** `{meta.get('Size')}`")
        vf = float(meta.get("verification", 0) or 0)
        color = "#22c55e" if 0.95 <= vf <= 1.05 else "#ef4444"
        st.markdown(f"**Verification ratio:** <span style='color:{color};font-weight:700'>{vf:.4f}</span>", unsafe_allow_html=True)
        st.markdown(f"**Total billed:** `${meta.get('totalBilled')}`")
        st.markdown(f"**Total (excl. Jan):** `${meta.get('totalBilledRev')}`")
        render_monthly_table(meta.get("monthly_json", "{}"))

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("# Water Billing Verification")
st.markdown("Ask about any **meter**, **premise**, or **POD** — I'll verify the calculations.")

missing = [k for k, v in {"OPENAI_API_KEY": openai_api_key, "QDRANT_URL": qdrant_url, "QDRANT_API_KEY": qdrant_api_key, "HF_TOKEN": hf_token}.items() if not v]
if missing:
    st.error(f"Missing environment variables: {', '.join(missing)}")
    st.stop()

if "messages"  not in st.session_state: st.session_state.messages  = []
if "last_meta" not in st.session_state: st.session_state.last_meta = None

try:
    qdrant, embed_model, oai_client = load_resources()
    df = load_dataframe()
    stats = compute_stats(df)
    SYSTEM_PROMPT = build_system_prompt(stats)
except Exception as e:
    import traceback
    st.error(traceback.format_exc())
    st.stop()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.session_state.pop("inject_query", None) or st.chat_input("Ask about a meter, premise, or POD...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        if is_aggregate_query(prompt):
            with st.spinner("Querying dataset..."):
                context = answer_aggregate(df, prompt)
        else:
            with st.spinner("Retrieving billing records..."):
                payloads = retrieve(qdrant, embed_model, prompt)
            if not payloads:
                st.markdown("No matching records found.")
                st.session_state.messages.append({"role": "assistant", "content": "No matching records found."})
                st.stop()
            st.session_state.last_meta = payloads[0]
            context = "\n\n---\n\n".join(payload_to_doc(p) for p in payloads)

        messages_for_api = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]],
            {"role": "user", "content": f"Data:\n\n{context}\n\nQuestion: {prompt}"},
        ]
        response_text = ""
        placeholder   = st.empty()
        stream = oai_client.chat.completions.create(
            model="gpt-4o", messages=messages_for_api, stream=True, temperature=0,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            response_text += delta
            placeholder.markdown(response_text + "▌")
        placeholder.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        st.rerun()