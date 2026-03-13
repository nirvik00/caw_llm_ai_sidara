"""
ingest.py - Load verified_water.csv into ChromaDB
Run once (or re-run to refresh): python ingest.py
"""
import pandas as pd
import chromadb
import json
import os

# ── Rate tables ──────────────────────────────────────────────────────────────
base_meter = {
    "5/8_INCH": 28.49, "3/4_INCH": 37.78, "1_INCH": 56.87,
    "1 1/2 inch": 102.9, "2_INCH": 161.11, "3_INCH": 340.73,
    "4_INCH": 513.87, "6_INCH": 988.45, "8_INCH": 1556.08, "10_INCH": 2212.0,
}
water_res = {"upto_3": 2.96, "4_to_12": 4.91, "greater_12": 8.58}

MONTHS = ["february", "march", "april", "may", "june",
          "july", "august", "september", "october", "november", "december"]

def calc_water_charge(consumption: float, size: str) -> float:
    base = base_meter.get(size, 0)
    if consumption == 0:
        rate = 0
    elif consumption <= 3:
        rate = water_res["upto_3"]
    elif consumption <= 12:
        rate = water_res["4_to_12"]
    else:
        rate = water_res["greater_12"]
    return round(base + (consumption * rate), 4)

def build_monthly_verification(row) -> dict:
    """Returns per-month dict with billed, expected, ratio, flag."""
    size = str(row.get("Size", "")).strip()
    monthly = {}
    for m in MONTHS:
        billed_col   = f"{m.capitalize()}BilledAmount"
        cons_col     = f"{m.capitalize()}BilledConsumption"
        try:
            billed = float(row.get(billed_col, 0) or 0)
            cons   = float(row.get(cons_col,   0) or 0)
            expected = calc_water_charge(cons, size)
            ratio    = round(billed / expected, 4) if expected > 0 else None
            flag     = abs(ratio - 1.0) > 0.05 if ratio is not None else False
        except Exception:
            billed, cons, expected, ratio, flag = 0, 0, 0, None, False
        monthly[m] = {
            "billed": billed,
            "consumption": cons,
            "expected": expected,
            "ratio": ratio,
            "flag": flag,
        }
    return monthly

def build_document(row) -> str:
    """Plain-text document that gets embedded."""
    monthly = build_monthly_verification(row)
    lines = [
        f"Meter: {row.get('Meter','?')}",
        f"Premise: {row.get('Premise','?')}",
        f"POD: {row.get('POD','?')}",
        f"Size: {row.get('Size','?')}",
        f"RateCategory: {row.get('RateCategoryKey','?')}",
        f"TotalBilled: {row.get('totalBilled','?')}",
        f"TotalBilledExcludingJanuary: {row.get('totalBilledRev','?')}",
        f"OverallVerificationRatio: {row.get('verification','?')}",
        f"ServiceAddress: {row.get('ServiceAddress','?')}",
        "",
        "Monthly breakdown (billed | consumption | expected | ratio | flagged):",
    ]
    for m, v in monthly.items():
        flag_str = " ⚠ MISMATCH" if v["flag"] else ""
        lines.append(
            f"  {m.capitalize():12s}: billed={v['billed']:.2f}  "
            f"cons={v['consumption']}  expected={v['expected']:.2f}  "
            f"ratio={v['ratio']}{flag_str}"
        )
    return "\n".join(lines)

def ingest(csv_path: str, chroma_path: str = "./chroma_db"):
    df = pd.read_csv(csv_path, dtype={"POD": "string"})
    df.columns = [c.strip() for c in df.columns]

    client = chromadb.PersistentClient(path=chroma_path)
    # Wipe & recreate so re-runs are idempotent
    try:
        client.delete_collection("water_billing")
    except Exception:
        pass
    col = client.create_collection("water_billing")

    docs, metas, ids = [], [], []
    total = len(df)
    print(f"Ingesting {total} rows...")
    for idx, row in df.iterrows():
        monthly = build_monthly_verification(row)
        doc  = build_document(row)
        # Store ALL columns from the CSV row
        meta = {c: str(row[c]) if row[c] is not None else "" for c in df.columns}
        # Add computed verification fields
        meta["monthly_json"] = json.dumps(monthly)
        meta["has_mismatch"] = str(any(v["flag"] for v in monthly.values()))
        docs.append(doc)
        metas.append(meta)
        ids.append(f"row_{idx}")

        # Progress bar
        done = idx + 1
        pct  = done / total
        bar  = int(pct * 40)
        print(f"\r  [{'█' * bar}{'░' * (40 - bar)}] {done}/{total} ({pct:.0%})", end="", flush=True)

    print()  # newline after bar
    # Upload in batches of 5000
    BATCH_SIZE = 5000
    print("Uploading to ChromaDB...")
    for i in range(0, len(docs), BATCH_SIZE):
        col.add(
            documents=docs[i:i + BATCH_SIZE],
            metadatas=metas[i:i + BATCH_SIZE],
            ids=ids[i:i + BATCH_SIZE]
        )
        done = min(i + BATCH_SIZE, len(docs))
        pct  = done / len(docs)
        bar  = int(pct * 40)
        print(f"\r  Uploading [{'█' * bar}{'░' * (40 - bar)}] {done}/{len(docs)} ({pct:.0%})", end="", flush=True)
    print()
    print(f"Ingested {len(docs)} rows into ChromaDB at {chroma_path}")

if __name__ == "__main__":
    ingest("output/verified_water.csv")