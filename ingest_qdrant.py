"""
ingest.py - Load verified_water.csv into Qdrant Cloud
Run once (or re-run to refresh): python ingest.py
"""
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import pandas as pd
import json
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()

base_meter = {
    "5/8_INCH": 28.49, "3/4_INCH": 37.78, "1_INCH": 56.87,
    "1 1/2 inch": 102.9, "2_INCH": 161.11, "3_INCH": 340.73,
    "4_INCH": 513.87, "6_INCH": 988.45, "8_INCH": 1556.08, "10_INCH": 2212.0,
}
water_res = {"upto_3": 2.96, "4_to_12": 4.91, "greater_12": 8.58}

MONTHS           = ["february", "march", "april", "may", "june",
                    "july", "august", "september", "october", "november", "december"]
COLLECTION_NAME  = "water_billing"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"
BATCH_SIZE       = 500

def calc_water_charge(consumption: float, size: str) -> float:
    consumption = float(consumption)
    base = base_meter.get(size, 0)
    if consumption == 0:
        charge = base
    elif consumption <= 3:
        charge = base + (consumption * water_res["upto_3"])
    elif consumption <= 12:
        charge = base + (3 * water_res["upto_3"]) + ((consumption - 3) * water_res["4_to_12"])
    else:
        charge = base + (3 * water_res["upto_3"]) + (9 * water_res["4_to_12"]) + ((consumption - 12) * water_res["greater_12"])
    return round(charge, 4)

def build_monthly_verification(row) -> dict:
    size = str(row.get("Size", "")).strip()
    monthly = {}
    for m in MONTHS:
        billed_col = f"{m.capitalize()}BilledAmount"
        cons_col   = f"{m.capitalize()}BilledConsumption"
        try:
            billed   = float(row.get(billed_col, 0) or 0)
            cons     = float(row.get(cons_col,   0) or 0)
            expected = calc_water_charge(cons, size)
            ratio    = round(billed / expected, 4) if expected > 0 else None
            flag     = abs(ratio - 1.0) > 0.05 if ratio is not None else False
        except Exception:
            billed, cons, expected, ratio, flag = 0, 0, 0, None, False
        monthly[m] = {"billed": billed, "consumption": cons,
                      "expected": expected, "ratio": ratio, "flag": flag}
    return monthly

def check_base_verified(row, meter_charge_cols) -> int:
    """Check if all non-January MeterCharge columns match the base_meter for the row's Size."""
    expected_base = base_meter.get(str(row.get("Size", "")).strip(), None)
    if expected_base is None:
        return 0
    for col in meter_charge_cols:
        try:
            val = float(row.get(col, 0) or 0)
            if round(val, 2) != round(expected_base, 2):
                return 0
        except (ValueError, TypeError):
            return 0
    return 1

def build_document(row, monthly) -> str:
    lines = [
        f"Meter: {row.get('Meter','?')}",
        f"Premise: {row.get('Premise','?')}",
        f"POD: {row.get('POD','?')}",
        f"Size: {row.get('Size','?')}",
        f"RateCategory: {row.get('RateCategoryKey','?')}",
        f"TotalBilled: {row.get('totalBilled','?')}",
        f"TotalBilledExcludingJanuary: {row.get('totalBilledRev','?')}",
        f"OverallVerificationRatio: {row.get('verification','?')}",
        f"BaseVerified: {row.get('BaseVerified','?')}",
        f"ServiceAddress: {row.get('ServiceAddress','?')}",
        "",
        "Monthly breakdown (billed | consumption | expected | ratio | base_ok | flagged):",
    ]
    for m, v in monthly.items():
        flag_str = " MISMATCH" if v["flag"] else ""
        base_ok  = "BASE_OK" if v.get("base_ok") else "BASE_MISMATCH"
        lines.append(
            f"  {m.capitalize():12s}: billed={v['billed']:.2f}  "
            f"cons={v['consumption']}  expected={v['expected']:.2f}  "
            f"ratio={v['ratio']}  {base_ok}{flag_str}"
        )
    return "\n".join(lines)

def ingest(csv_path: str):
    qdrant_url     = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    print("Connecting to Qdrant Cloud...")
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    print(f"Loading embedding model: {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    df = pd.read_csv(csv_path, dtype={"POD": "string"})
    df.columns = [c.strip() for c in df.columns]
    total = len(df)
    print(f"Loaded {total} rows from {csv_path}")

    # Find MeterCharge columns (excluding January)
    meter_charge_cols = [c for c in df.columns if 'metercharge' in c.lower() and 'january' not in c.lower()]
    print(f"MeterCharge columns: {meter_charge_cols}")

    # Month ratio columns from ingest_verify (feb, mar, apr...)
    month_ratio_cols = [c for c in df.columns if c.lower() in
                        ['feb','mar','apr','may','june','july','aug','sep','oct','nov','dec']]
    print(f"Month ratio columns: {month_ratio_cols}")

    # Recreate collection
    if client.collection_exists(COLLECTION_NAME):
        print(f"Deleting existing collection '{COLLECTION_NAME}'...")
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )
    print(f"Created collection '{COLLECTION_NAME}'")

    # Build all docs and metadata
    print("Building documents...")
    all_docs, all_metas = [], []
    for idx, row in df.iterrows():
        monthly    = build_monthly_verification(row)
        size       = str(row.get("Size", "")).strip()
        expected_base = base_meter.get(size, None)

        # Enrich monthly with per-month base check and ratio from CSV
        month_map = {"february":"feb","march":"mar","april":"apr","may":"may",
                     "june":"june","july":"july","august":"aug","september":"sep",
                     "october":"oct","november":"nov","december":"dec"}
        for m in MONTHS:
            short = month_map[m]
            # base_ok: check MeterCharge col for this month
            mc_col = next((c for c in meter_charge_cols if m[:3] in c.lower() or m in c.lower()), None)
            if mc_col and expected_base is not None:
                try:
                    monthly[m]["base_ok"] = round(float(row.get(mc_col, 0) or 0), 2) == round(expected_base, 2)
                except Exception:
                    monthly[m]["base_ok"] = False
            else:
                monthly[m]["base_ok"] = None
            # csv_ratio: the pre-computed ratio from ingest_verify
            if short in df.columns:
                try:
                    monthly[m]["csv_ratio"] = float(row.get(short, None))
                except Exception:
                    monthly[m]["csv_ratio"] = None

        base_verified = check_base_verified(row, meter_charge_cols)
        doc  = build_document(row, monthly)
        meta = {c: str(row[c]) if pd.notna(row[c]) else "" for c in df.columns}
        meta["monthly_json"]  = json.dumps(monthly)
        meta["has_mismatch"]  = str(any(v["flag"] for v in monthly.values()))
        meta["BaseVerified"]  = str(base_verified)

        all_docs.append(doc)
        all_metas.append(meta)

        done = idx + 1
        pct  = done / total
        bar  = int(pct * 40)
        print(f"\r  Building [{('█'*bar)+('░'*(40-bar))}] {done}/{total} ({pct:.0%})", end="", flush=True)
    print()

    # Embed and upload in batches
    print("Embedding and uploading to Qdrant...")
    for i in range(0, total, BATCH_SIZE):
        batch_docs  = all_docs[i:i+BATCH_SIZE]
        batch_metas = all_metas[i:i+BATCH_SIZE]
        embeddings  = model.encode(batch_docs, show_progress_bar=False).tolist()
        points = [
            PointStruct(id=i+j, vector=embeddings[j], payload=batch_metas[j])
            for j in range(len(batch_docs))
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        done = min(i+BATCH_SIZE, total)
        pct  = done / total
        bar  = int(pct * 40)
        print(f"\r  Uploading [{('█'*bar)+('░'*(40-bar))}] {done}/{total} ({pct:.0%})", end="", flush=True)
    print()
    print(f"Ingested {total} rows into Qdrant collection '{COLLECTION_NAME}'")

if __name__ == "__main__":
    ingest("output/verified_water.csv")