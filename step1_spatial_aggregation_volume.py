import os
import sys
from pathlib import Path

import pandas as pd


if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


ROOT_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = ROOT_DIR / "archive (2)"
PROCESSED_DATA_DIR = ROOT_DIR / "olist_processed"

PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

print("[Step1] Loading source tables...")
df_customers = pd.read_csv(RAW_DATA_DIR / "olist_customers_dataset.csv")
df_sellers = pd.read_csv(RAW_DATA_DIR / "olist_sellers_dataset.csv")
df_geo = pd.read_csv(RAW_DATA_DIR / "olist_geolocation_dataset.csv")

print(f"  customers: {len(df_customers):,}")
print(f"  sellers  : {len(df_sellers):,}")
print(f"  geo rows : {len(df_geo):,}")

# Normalize zip prefixes
for col in ["geolocation_zip_code_prefix"]:
    df_geo[col] = df_geo[col].astype(str).str.zfill(5)

for col in ["customer_zip_code_prefix"]:
    df_customers[col] = df_customers[col].astype(str).str.zfill(5)

for col in ["seller_zip_code_prefix"]:
    df_sellers[col] = df_sellers[col].astype(str).str.zfill(5)

# Build 2-digit region ids
df_geo["region_id"] = df_geo["geolocation_zip_code_prefix"].str[:2]
df_customers["region_id"] = df_customers["customer_zip_code_prefix"].str[:2]
df_sellers["region_id"] = df_sellers["seller_zip_code_prefix"].str[:2]

# Clean geo and build centroids
df_geo = df_geo.dropna(subset=["geolocation_lat", "geolocation_lng"])
df_geo = df_geo.drop_duplicates(subset=["geolocation_zip_code_prefix"])

def mode_or_first(series: pd.Series) -> str:
    if series.empty:
        return "NA"
    m = series.mode(dropna=True)
    if not m.empty:
        return str(m.iloc[0])
    return str(series.iloc[0])

print("[Step1] Building micro-region centroids...")
df_regions = (
    df_geo.groupby("region_id", as_index=False)
    .agg(
        lat=("geolocation_lat", "mean"),
        lng=("geolocation_lng", "mean"),
        state=("geolocation_state", mode_or_first),
    )
    .sort_values("region_id")
)

customer_mapping = df_customers[["customer_id", "region_id"]].drop_duplicates()
seller_mapping = df_sellers[["seller_id", "region_id"]].drop_duplicates()

out_regions = PROCESSED_DATA_DIR / "step1_micro_regions_nodes.csv"
out_cust = PROCESSED_DATA_DIR / "step1_customer_mapping.csv"
out_sell = PROCESSED_DATA_DIR / "step1_seller_mapping.csv"

df_regions.to_csv(out_regions, index=False)
customer_mapping.to_csv(out_cust, index=False)
seller_mapping.to_csv(out_sell, index=False)

print("\n[Step1] Done")
print(f"  regions           : {len(df_regions):,}")
print(f"  customer mappings : {len(customer_mapping):,}")
print(f"  seller mappings   : {len(seller_mapping):,}")
print(f"  -> {out_regions}")
print(f"  -> {out_cust}")
print(f"  -> {out_sell}")
