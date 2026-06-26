from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "archive (2)"
OUTPUT_DIR = ROOT / "olist_processed" / "raw_normalized"


def _normalize_ids(df: pd.DataFrame) -> pd.DataFrame:
    for col in [c for c in df.columns if c.endswith("_id")]:
        s = df[col].astype("string").str.strip().str.lower()
        s = s.replace({"": pd.NA, "nan": pd.NA, "none": pd.NA})
        df[col] = s
    return df


def _normalize_zip_prefix(df: pd.DataFrame) -> pd.DataFrame:
    for col in [c for c in df.columns if "zip_code_prefix" in c]:
        num = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        df[col] = num.astype("string").str.zfill(5)
    return df


def _normalize_geo_text(df: pd.DataFrame) -> pd.DataFrame:
    for col in [c for c in df.columns if c.endswith("_city")]:
        s = df[col].astype("string").str.strip().str.lower()
        s = s.replace({"": pd.NA, "nan": pd.NA, "none": pd.NA})
        df[col] = s
    for col in [c for c in df.columns if c.endswith("_state")]:
        s = df[col].astype("string").str.strip().str.upper()
        s = s.replace({"": pd.NA, "nan": pd.NA, "none": pd.NA})
        df[col] = s
    return df


def _normalize_datetimes(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            continue
        ts = pd.to_datetime(df[col], errors="coerce")
        df[col] = ts.dt.strftime("%Y-%m-%d %H:%M:%S")
    return df


def _coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _fill_products(df: pd.DataFrame) -> pd.DataFrame:
    if "product_category_name" in df.columns:
        df["product_category_name"] = (
            df["product_category_name"].astype("string").str.strip().str.lower().fillna("unknown")
        )

    numeric_fill_cols = [
        "product_name_lenght",
        "product_description_lenght",
        "product_photos_qty",
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]
    for col in numeric_fill_cols:
        if col in df.columns:
            med = float(df[col].median(skipna=True)) if not pd.isna(df[col].median(skipna=True)) else 0.0
            df[col] = df[col].fillna(med)
    return df


def _rules_for_file(name: str) -> dict[str, list[str]]:
    common_dt = [c for c in [
        "shipping_limit_date",
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
        "review_creation_date",
        "review_answer_timestamp",
    ]]

    numeric_map = {
        "olist_order_items_dataset.csv": ["order_item_id", "price", "freight_value"],
        "olist_order_payments_dataset.csv": ["payment_sequential", "payment_installments", "payment_value"],
        "olist_order_reviews_dataset.csv": ["review_score"],
        "olist_geolocation_dataset.csv": ["geolocation_lat", "geolocation_lng"],
        "olist_products_dataset.csv": [
            "product_name_lenght",
            "product_description_lenght",
            "product_photos_qty",
            "product_weight_g",
            "product_length_cm",
            "product_height_cm",
            "product_width_cm",
        ],
    }

    dt_cols = []
    if name == "olist_order_items_dataset.csv":
        dt_cols = ["shipping_limit_date"]
    elif name == "olist_orders_dataset.csv":
        dt_cols = [
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date",
        ]
    elif name == "olist_order_reviews_dataset.csv":
        dt_cols = ["review_creation_date", "review_answer_timestamp"]

    return {"datetime": dt_cols, "numeric": numeric_map.get(name, []), "_common_dt": common_dt}


def normalize_file(path: Path, output_dir: Path) -> dict:
    df = pd.read_csv(path, low_memory=False)
    rows_in = int(len(df))

    rules = _rules_for_file(path.name)

    df = _normalize_ids(df)
    df = _normalize_zip_prefix(df)
    df = _normalize_geo_text(df)
    df = _coerce_numeric(df, rules["numeric"])
    df = _normalize_datetimes(df, rules["datetime"])

    if path.name == "olist_products_dataset.csv":
        df = _fill_products(df)

    dup_removed = rows_in - int(len(df.drop_duplicates()))
    df = df.drop_duplicates()

    out_path = output_dir / path.name
    df.to_csv(out_path, index=False)

    na_rate = (df.isna().mean() * 100).sort_values(ascending=False)
    top_na = {k: float(v) for k, v in na_rate[na_rate > 0].head(8).items()}

    return {
        "file": path.name,
        "rows_in": rows_in,
        "rows_out": int(len(df)),
        "duplicates_removed": int(dup_removed),
        "top_missing_percent": top_na,
        "output": str(out_path),
    }


def main() -> None:
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(INPUT_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {INPUT_DIR}")

    reports: list[dict] = []
    for path in csv_files:
        print(f"Normalizing {path.name} ...")
        rep = normalize_file(path, OUTPUT_DIR)
        reports.append(rep)
        print(
            f"  rows: {rep['rows_in']:,} -> {rep['rows_out']:,} | "
            f"duplicates_removed: {rep['duplicates_removed']:,}"
        )

    report_path = OUTPUT_DIR / "normalization_report.json"
    report_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(f"\nDone. Report: {report_path}")


if __name__ == "__main__":
    main()
