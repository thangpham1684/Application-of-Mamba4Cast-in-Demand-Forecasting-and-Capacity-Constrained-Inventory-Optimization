import sys
from pathlib import Path

import numpy as np
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

VALID_ORDER_STATUS = {"delivered", "shipped", "invoiced", "processing", "approved"}

print("[Step2] Loading order-level source data...")
df_orders = pd.read_csv(RAW_DATA_DIR / "olist_orders_dataset.csv")
df_items = pd.read_csv(RAW_DATA_DIR / "olist_order_items_dataset.csv")
df_products = pd.read_csv(RAW_DATA_DIR / "olist_products_dataset.csv")

customer_mapping = pd.read_csv(PROCESSED_DATA_DIR / "step1_customer_mapping.csv")
seller_mapping = pd.read_csv(PROCESSED_DATA_DIR / "step1_seller_mapping.csv")

print(f"  orders  : {len(df_orders):,}")
print(f"  items   : {len(df_items):,}")
print(f"  products: {len(df_products):,}")

# Keep valid orders with valid timestamps before the data collection cutoff
df_orders = df_orders[df_orders["order_status"].isin(VALID_ORDER_STATUS)].copy()
df_orders["order_purchase_timestamp"] = pd.to_datetime(
    df_orders["order_purchase_timestamp"], errors="coerce"
)
df_orders = df_orders.dropna(subset=["order_purchase_timestamp"]).copy()
df_orders = df_orders[df_orders["order_purchase_timestamp"] < "2018-08-27"].copy()

# Weekly index (Monday)
df_orders["ds"] = (
    df_orders["order_purchase_timestamp"]
    - pd.to_timedelta(df_orders["order_purchase_timestamp"].dt.weekday, unit="D")
).dt.normalize()

# Build product volume (m3) from dimensions
for col in ["product_length_cm", "product_width_cm", "product_height_cm"]:
    df_products[col] = pd.to_numeric(df_products[col], errors="coerce")

df_products["volume_cm3"] = (
    df_products["product_length_cm"]
    * df_products["product_width_cm"]
    * df_products["product_height_cm"]
)

valid_mask = df_products["volume_cm3"].notna() & (df_products["volume_cm3"] > 0)
if valid_mask.any():
    median_volume_cm3 = float(df_products.loc[valid_mask, "volume_cm3"].median())
else:
    median_volume_cm3 = 1000.0

# Fill missing/non-positive with median so each order item still contributes volume
bad_mask = (~df_products["volume_cm3"].notna()) | (df_products["volume_cm3"] <= 0)
df_products.loc[bad_mask, "volume_cm3"] = median_volume_cm3

df_products["item_volume_m3"] = df_products["volume_cm3"] / 1_000_000.0

print(f"[Step2] Product volume median fallback (cm3): {median_volume_cm3:,.2f}")

# Join order items -> orders -> mappings -> products
print("[Step2] Joining tables...")
df_merge = pd.merge(
    df_items,
    df_orders[["order_id", "customer_id", "ds"]],
    on="order_id",
    how="inner",
)

df_merge = pd.merge(df_merge, customer_mapping, on="customer_id", how="inner")
df_merge = df_merge.rename(columns={"region_id": "destination_region"})

df_merge = pd.merge(df_merge, seller_mapping, on="seller_id", how="inner")
df_merge = df_merge.rename(columns={"region_id": "origin_region"})

df_merge = pd.merge(
    df_merge,
    df_products[["product_id", "item_volume_m3"]],
    on="product_id",
    how="left",
)

missing_volume_mask = df_merge["item_volume_m3"].isna()
if missing_volume_mask.any():
    df_merge.loc[missing_volume_mask, "item_volume_m3"] = median_volume_cm3 / 1_000_000.0

# Standard ids
df_merge["origin_region"] = df_merge["origin_region"].astype(str).str.zfill(2)
df_merge["destination_region"] = df_merge["destination_region"].astype(str).str.zfill(2)
df_merge["unique_id"] = df_merge["origin_region"] + "_" + df_merge["destination_region"]

print("[Step2] Aggregating weekly OD volume...")
panel_df = (
    df_merge.groupby(["unique_id", "ds"], as_index=False)
    .agg(
        y=("item_volume_m3", "sum"),
        total_items=("order_item_id", "count"),
    )
    .sort_values(["unique_id", "ds"])
)

out_panel = PROCESSED_DATA_DIR / "step2_temporal_panel_volume.csv"
out_stats = PROCESSED_DATA_DIR / "step2_volume_metadata.csv"

panel_df.to_csv(out_panel, index=False)

meta_df = pd.DataFrame(
    {
        "metric": [
            "rows_panel",
            "active_routes",
            "total_volume_m3",
            "rows_missing_product_volume_filled",
            "median_product_volume_cm3_fallback",
        ],
        "value": [
            len(panel_df),
            panel_df["unique_id"].nunique(),
            float(panel_df["y"].sum()),
            int(missing_volume_mask.sum()),
            float(median_volume_cm3),
        ],
    }
)
meta_df.to_csv(out_stats, index=False)

print("\n[Step2] Done")
print(f"  panel rows    : {len(panel_df):,}")
print(f"  active routes : {panel_df['unique_id'].nunique():,}")
print(f"  total volume  : {panel_df['y'].sum():,.4f} m3")
print(f"  -> {out_panel}")
print(f"  -> {out_stats}")
