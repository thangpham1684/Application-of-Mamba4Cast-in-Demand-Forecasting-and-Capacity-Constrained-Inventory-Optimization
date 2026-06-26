import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_csv_rows(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_rows_per_series(
    rows: List[Dict[str, str]],
    unique_id_col: str,
    time_col: str,
    val_len: int,
    test_len: int,
) -> Dict[str, List[Dict[str, str]]]:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row[unique_id_col]].append(row)

    train_rows: List[Dict[str, str]] = []
    val_rows: List[Dict[str, str]] = []
    test_rows: List[Dict[str, str]] = []

    for uid, series in grouped.items():
        series_sorted = sorted(series, key=lambda r: r[time_col])
        n = len(series_sorted)

        # Ensure at least one train sample when possible.
        split_train_end = max(1, n - val_len - test_len)
        split_val_end = min(n, split_train_end + val_len)

        train_part = series_sorted[:split_train_end]
        val_part = series_sorted[split_train_end:split_val_end]
        test_part = series_sorted[split_val_end:]

        train_rows.extend(train_part)
        val_rows.extend(val_part)
        test_rows.extend(test_part)

    # Sort globally for reproducible files.
    train_rows.sort(key=lambda r: (r[unique_id_col], r[time_col]))
    val_rows.sort(key=lambda r: (r[unique_id_col], r[time_col]))
    test_rows.sort(key=lambda r: (r[unique_id_col], r[time_col]))

    return {"train": train_rows, "val": val_rows, "test": test_rows}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Split a panel CSV into train/val/test CSVs by time order per unique_id."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input panel CSV path (must include unique_id and ds columns).",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("prodhon_panel"),
        help=(
            "Output prefix path. The script writes: <prefix>_train.csv, <prefix>_val.csv, <prefix>_test.csv"
        ),
    )
    parser.add_argument(
        "--unique-id-col",
        type=str,
        default="unique_id",
        help="Unique series identifier column.",
    )
    parser.add_argument(
        "--time-col",
        type=str,
        default="ds",
        help="Time column name. Recommended format: YYYY-MM-DD.",
    )
    parser.add_argument(
        "--val-len",
        type=int,
        default=1,
        help="Number of trailing timesteps per series reserved for validation.",
    )
    parser.add_argument(
        "--test-len",
        type=int,
        default=1,
        help="Number of trailing timesteps per series reserved for test.",
    )

    args = parser.parse_args()

    if args.val_len < 0 or args.test_len < 0:
        raise ValueError("val-len and test-len must be non-negative integers.")

    rows = read_csv_rows(args.input)
    if not rows:
        raise ValueError(f"No rows found in input CSV: {args.input}")

    if args.unique_id_col not in rows[0] or args.time_col not in rows[0]:
        raise KeyError(
            f"Input CSV must contain columns '{args.unique_id_col}' and '{args.time_col}'."
        )

    fieldnames = list(rows[0].keys())
    splits = split_rows_per_series(
        rows=rows,
        unique_id_col=args.unique_id_col,
        time_col=args.time_col,
        val_len=args.val_len,
        test_len=args.test_len,
    )

    train_path = Path(f"{args.output_prefix}_train.csv")
    val_path = Path(f"{args.output_prefix}_val.csv")
    test_path = Path(f"{args.output_prefix}_test.csv")

    write_csv_rows(train_path, splits["train"], fieldnames)
    write_csv_rows(val_path, splits["val"], fieldnames)
    write_csv_rows(test_path, splits["test"], fieldnames)

    print(f"Input rows: {len(rows)}")
    print(f"Train rows: {len(splits['train'])} -> {train_path}")
    print(f"Val rows:   {len(splits['val'])} -> {val_path}")
    print(f"Test rows:  {len(splits['test'])} -> {test_path}")


if __name__ == "__main__":
    main()
