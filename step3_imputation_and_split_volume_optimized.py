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
PROCESSED_DATA_DIR = ROOT_DIR / "olist_processed"

# Tunable knobs for sparsity reduction (Stricter thresholds for sMAPE < 50%)
MIN_ACTIVE_WEEKS = 32
MIN_DENSITY = 0.50
RECENT_WEEKS = 12
MIN_RECENT_NONZERO = 4
VAL_WEEKS = 4
TEST_WEEKS = 4

INPUT_FILE = PROCESSED_DATA_DIR / "step2_temporal_panel_volume.csv"

print(f"[OptStep3] Loading panel: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)
if df.empty:
    raise RuntimeError("Input panel is empty.")

df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
df["y"] = pd.to_numeric(df["y"], errors="coerce")
df = df.dropna(subset=["unique_id", "ds", "y"]).copy()
df = df[df["y"] >= 0].copy()

# Base time anchors
global_min = df["ds"].min().normalize()
global_max = df["ds"].max().normalize()
recent_start = global_max - pd.Timedelta(weeks=RECENT_WEEKS - 1)

print(f"  date range: {global_min.date()} -> {global_max.date()}")

# Route-level quality stats on non-zero events (step2 already non-zero rows)
route_stats = (
    df.groupby("unique_id", as_index=False)
    .agg(
        first_ds=("ds", "min"),
        last_ds=("ds", "max"),
        active_weeks=("ds", "count"),
        total_volume_m3=("y", "sum"),
    )
)

route_stats["span_weeks"] = ((route_stats["last_ds"] - route_stats["first_ds"]).dt.days // 7) + 1
route_stats["density"] = route_stats["active_weeks"] / route_stats["span_weeks"].clip(lower=1)

recent_counts = (
    df[df["ds"] >= recent_start]
    .groupby("unique_id")
    .size()
    .rename("recent_nonzero")
)
route_stats = route_stats.merge(recent_counts, on="unique_id", how="left")
route_stats["recent_nonzero"] = route_stats["recent_nonzero"].fillna(0).astype(int)

keep = route_stats[
    (route_stats["active_weeks"] >= MIN_ACTIVE_WEEKS)
    & (route_stats["density"] >= MIN_DENSITY)
    & (route_stats["recent_nonzero"] >= MIN_RECENT_NONZERO)
].copy()

kept_uids = set(keep["unique_id"])
print(
    f"[OptStep3] Route filter: {df['unique_id'].nunique():,} -> {len(kept_uids):,} "
    f"(min_active={MIN_ACTIVE_WEEKS}, density>={MIN_DENSITY:.2f}, recent_nonzero>={MIN_RECENT_NONZERO})"
)

if not kept_uids:
    raise RuntimeError("No routes left after optimized filtering. Loosen thresholds.")

df_keep = df[df["unique_id"].isin(kept_uids)].copy()

# Bounded zero-imputation per route: from first active week up to global_max
print("[OptStep3] Performing weekly imputation per route up to global_max...")
parts = []
for uid, g in df_keep.groupby("unique_id"):
    g = g.sort_values("ds")
    first_ds = g["ds"].iloc[0].normalize()
    rng = pd.date_range(start=first_ds, end=global_max, freq="7D")
    tmp = g.set_index("ds")[["y"]].reindex(rng, fill_value=0.0).reset_index()
    tmp.columns = ["ds", "y"]
    tmp.insert(0, "unique_id", uid)
    parts.append(tmp)

df_filled = pd.concat(parts, ignore_index=True)

# Chronological split using global end horizon
test_start = global_max - pd.Timedelta(weeks=TEST_WEEKS - 1)
val_start = test_start - pd.Timedelta(weeks=VAL_WEEKS)

df_train = df_filled[df_filled["ds"] < val_start].copy()
df_val = df_filled[(df_filled["ds"] >= val_start) & (df_filled["ds"] < test_start)].copy()
df_test = df_filled[df_filled["ds"] >= test_start].copy()

# Keep only routes that appear in all 3 splits for stable training/evaluation
t_uids = set(df_train["unique_id"].unique())
v_uids = set(df_val["unique_id"].unique())
te_uids = set(df_test["unique_id"].unique())
final_uids = t_uids & v_uids & te_uids

if not final_uids:
    raise RuntimeError("No common routes across train/val/test after optimized split.")

df_train = df_train[df_train["unique_id"].isin(final_uids)].copy()
df_val = df_val[df_val["unique_id"].isin(final_uids)].copy()
df_test = df_test[df_test["unique_id"].isin(final_uids)].copy()

for dset in [df_train, df_val, df_test]:
    dset["ds"] = pd.to_datetime(dset["ds"]).dt.strftime("%Y-%m-%d")

final_cols = ["unique_id", "ds", "y"]
df_train = df_train[final_cols].sort_values(["unique_id", "ds"])
df_val = df_val[final_cols].sort_values(["unique_id", "ds"])
df_test = df_test[final_cols].sort_values(["unique_id", "ds"])

out_train = PROCESSED_DATA_DIR / "olist_panel_volume_opt_train.csv"
out_val = PROCESSED_DATA_DIR / "olist_panel_volume_opt_val.csv"
out_test = PROCESSED_DATA_DIR / "olist_panel_volume_opt_test.csv"
out_stats = PROCESSED_DATA_DIR / "step3_volume_opt_stats.csv"

df_train.to_csv(out_train, index=False)
df_val.to_csv(out_val, index=False)
df_test.to_csv(out_test, index=False)

stats = pd.DataFrame(
    [
        {"split": "train", "rows": len(df_train), "uids": df_train["unique_id"].nunique(), "zero_ratio": float((df_train["y"] == 0).mean())},
        {"split": "val", "rows": len(df_val), "uids": df_val["unique_id"].nunique(), "zero_ratio": float((df_val["y"] == 0).mean())},
        {"split": "test", "rows": len(df_test), "uids": df_test["unique_id"].nunique(), "zero_ratio": float((df_test["y"] == 0).mean())},
    ]
)
stats.to_csv(out_stats, index=False)

print("\n[OptStep3] Done")
print(f"  common routes: {len(final_uids):,}")
print(f"  train rows: {len(df_train):,} | zero%: {(df_train['y'] == 0).mean() * 100:.2f}")
print(f"  val rows  : {len(df_val):,} | zero%: {(df_val['y'] == 0).mean() * 100:.2f}")
print(f"  test rows : {len(df_test):,} | zero%: {(df_test['y'] == 0).mean() * 100:.2f}")
print(f"  -> {out_train}")
print(f"  -> {out_val}")
print(f"  -> {out_test}")
print(f"  -> {out_stats}")
