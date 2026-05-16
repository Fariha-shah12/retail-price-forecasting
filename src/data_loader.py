"""
data_loader.py
Loads Walmart and Kroger pricing CSVs from S3, maps products using the
`category` (Walmart) and `term` (Kroger) columns, and returns a daily
average price DataFrame ready for forecasting.
"""

import boto3
import pandas as pd
import gzip
import io
import os

# ── Config ────────────────────────────────────────────────────────────────────
S3_BUCKET      = os.environ.get("S3_BUCKET", "walmart-scraper-data")
WALMART_PREFIX = "walmart/pricing/"
KROGER_PREFIX  = "raw/retailer=kroger/"

# Maps raw category/term values → canonical product name
TERM_TO_PRODUCT = {
    "eggs":         "eggs",
    "egg":          "eggs",
    "olive_oil":    "olive_oil",
    "olive oil":    "olive_oil",
    "paper_towel":  "paper_towels",
    "paper_towels": "paper_towels",
    "paper towels": "paper_towels",
}

# Price caps — filters out bulk/commercial SKUs from Walmart
PRICE_CAPS = {
    "eggs":         15.0,   # no standard retail egg pack costs more than $15
    "olive_oil":    25.0,   # standard 1L bottle
    "paper_towels": 30.0,   # standard 12-pack
}

s3 = boto3.client("s3")


# ── Walmart ───────────────────────────────────────────────────────────────────
def load_walmart() -> pd.DataFrame:
    """
    Reads all Walmart CSVs from S3 under WALMART_PREFIX.
    Uses the `category` column to identify the product.
    Applies price caps to remove bulk/commercial outlier SKUs.
    """
    print("Loading Walmart from S3...")
    paginator = s3.get_paginator("list_objects_v2")
    dfs = []

    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=WALMART_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".csv"):
                continue
            try:
                body = s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
                df   = pd.read_csv(io.BytesIO(body))

                df = df[df["sale_price"].notna() & (df["sale_price"] > 0)].copy()
                if df.empty:
                    continue

                # Map product using `category` column
                df["_product_tmp"] = (
                    df["category"].str.lower().str.strip().map(TERM_TO_PRODUCT)
                )

                # Apply price cap per product to remove bulk/commercial outliers
                for product, cap in PRICE_CAPS.items():
                    mask = (df["_product_tmp"] == product) & (df["sale_price"] > cap)
                    df = df[~mask]

                df["product"] = df["_product_tmp"].fillna("unknown")
                df = df.drop(columns=["_product_tmp"])

                if df.empty:
                    continue

                df["retailer"] = "walmart"
                df["date"]     = pd.to_datetime(df["run_date"], errors="coerce")
                df["price"]    = df["sale_price"].astype(float)
                df["location"] = df.get("location_name", pd.Series("unknown", index=df.index))

                keep = ["date", "retailer", "product", "name", "brand", "price", "discount_pct", "location"]
                dfs.append(df[[c for c in keep if c in df.columns]].rename(columns={"name": "product_name"}))

            except Exception as e:
                print(f"  SKIP {key}: {e}")

    if not dfs:
        print("  WARNING: No Walmart data loaded.")
        return pd.DataFrame()

    out = pd.concat(dfs, ignore_index=True)
    print(f"  Walmart: {len(out):,} rows | products: {out['product'].unique().tolist()}")
    return out


# ── Kroger ────────────────────────────────────────────────────────────────────
def load_kroger() -> pd.DataFrame:
    """
    Reads all Kroger .gz (or .csv) files from S3 under KROGER_PREFIX.
    Uses the `term` column to identify the product.
    """
    print("Loading Kroger from S3...")
    paginator = s3.get_paginator("list_objects_v2")
    dfs = []

    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=KROGER_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not (key.endswith(".gz") or key.endswith(".csv")):
                continue
            try:
                body = s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
                if key.endswith(".gz"):
                    with gzip.open(io.BytesIO(body), "rt") as f:
                        df = pd.read_csv(f)
                else:
                    df = pd.read_csv(io.BytesIO(body))

                df = df[df["effective_price"].notna() & (df["effective_price"] > 0)].copy()
                if df.empty:
                    continue

                # Map product using `term` column
                df["product"] = (
                    df["term"]
                    .str.lower()
                    .str.strip()
                    .map(TERM_TO_PRODUCT)
                    .fillna("unknown")
                )

                df["retailer"] = "kroger"
                df["date"] = (
                    pd.to_datetime(df["snapshot_utc"], utc=True, errors="coerce")
                    .dt.normalize()
                    .dt.tz_localize(None)
                )
                df["price"]    = df["effective_price"].astype(float)
                df["location"] = df.get("city", pd.Series("unknown", index=df.index))

                keep = ["date", "retailer", "product", "description", "brand", "price", "size", "location"]
                dfs.append(df[[c for c in keep if c in df.columns]].rename(columns={"description": "product_name"}))

            except Exception as e:
                print(f"  SKIP {key}: {e}")

    if not dfs:
        print("  WARNING: No Kroger data loaded.")
        return pd.DataFrame()

    out = pd.concat(dfs, ignore_index=True)
    print(f"  Kroger: {len(out):,} rows | products: {out['product'].unique().tolist()}")
    return out


# ── Daily aggregation ─────────────────────────────────────────────────────────
def get_daily_avg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates raw rows to one record per (date, retailer, product).
    Drops any rows where product is 'unknown'.
    """
    df = df[df["product"] != "unknown"].copy()
    return (
        df.groupby(["date", "retailer", "product"])
        .agg(
            avg_price     = ("price",        "mean"),
            min_price     = ("price",        "min"),
            max_price     = ("price",        "max"),
            num_skus      = ("product_name", "nunique"),
            num_locations = ("location",     "nunique"),
        )
        .reset_index()
        .sort_values(["retailer", "product", "date"])
    )


# ── S3 entry point ────────────────────────────────────────────────────────────
def load_all_data() -> pd.DataFrame:
    """Used by forecast_model.py in production (reads from S3)."""
    parts = []

    wdf = load_walmart()
    if not wdf.empty:
        parts.append(wdf)

    kdf = load_kroger()
    if not kdf.empty:
        parts.append(kdf)

    if not parts:
        raise ValueError("No data loaded from S3. Check bucket name and prefixes.")

    combined = pd.concat(parts, ignore_index=True)
    daily    = get_daily_avg(combined)

    print(f"\nDaily records: {len(daily)}")
    print(daily.groupby(["retailer", "product"]).agg(
        days  = ("date", "nunique"),
        from_ = ("date", "min"),
        to_   = ("date", "max"),
    ))
    return daily


# ── Local test entry point ────────────────────────────────────────────────────
def load_all_data_local(walmart_files: list, kroger_files: list) -> pd.DataFrame:
    """
    Same as load_all_data() but reads local files — useful for testing
    without AWS credentials.

    walmart_files: list of .csv paths
    kroger_files:  list of .gz or .csv paths
    """
    parts = []

    # Walmart
    wdfs = []
    for path in walmart_files:
        df = pd.read_csv(path)
        df = df[df["sale_price"].notna() & (df["sale_price"] > 0)].copy()
        if df.empty:
            continue

        # Map product + apply price cap
        df["_product_tmp"] = df["category"].str.lower().str.strip().map(TERM_TO_PRODUCT)
        for product, cap in PRICE_CAPS.items():
            mask = (df["_product_tmp"] == product) & (df["sale_price"] > cap)
            df = df[~mask]
        df["product"] = df["_product_tmp"].fillna("unknown")
        df = df.drop(columns=["_product_tmp"])

        if df.empty:
            continue

        df["retailer"] = "walmart"
        df["date"]     = pd.to_datetime(df["run_date"], errors="coerce")
        df["price"]    = df["sale_price"].astype(float)
        df["location"] = df.get("location_name", pd.Series("unknown", index=df.index))
        keep = ["date", "retailer", "product", "name", "brand", "price", "discount_pct", "location"]
        wdfs.append(df[[c for c in keep if c in df.columns]].rename(columns={"name": "product_name"}))
    if wdfs:
        parts.append(pd.concat(wdfs, ignore_index=True))

    # Kroger
    kdfs = []
    for path in kroger_files:
        if path.endswith(".gz"):
            with gzip.open(path, "rt") as f:
                df = pd.read_csv(f)
        else:
            df = pd.read_csv(path)
        df = df[df["effective_price"].notna() & (df["effective_price"] > 0)].copy()
        if df.empty:
            continue
        df["product"]  = df["term"].str.lower().str.strip().map(TERM_TO_PRODUCT).fillna("unknown")
        df["retailer"] = "kroger"
        df["date"] = (
            pd.to_datetime(df["snapshot_utc"], utc=True, errors="coerce")
            .dt.normalize().dt.tz_localize(None)
        )
        df["price"]    = df["effective_price"].astype(float)
        df["location"] = df.get("city", pd.Series("unknown", index=df.index))
        keep = ["date", "retailer", "product", "description", "brand", "price", "size", "location"]
        kdfs.append(df[[c for c in keep if c in df.columns]].rename(columns={"description": "product_name"}))
    if kdfs:
        parts.append(pd.concat(kdfs, ignore_index=True))

    if not parts:
        raise ValueError("No data loaded.")

    combined = pd.concat(parts, ignore_index=True)
    daily    = get_daily_avg(combined)
    print(f"\nDaily records: {len(daily)}")
    print(daily.groupby(["retailer", "product"]).agg(
        days  = ("date", "nunique"),
        from_ = ("date", "min"),
        to_   = ("date", "max"),
    ))
    return daily


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))

    walmart_files = [
        os.path.join(base, "eggs_2026-03-01.csv"),
        os.path.join(base, "olive_oil_2026-03-01.csv"),
        os.path.join(base, "paper_towel_2026-03-01.csv"),
    ]
    kroger_files = [
        os.path.join(base, "kroger_wa_eggs_20260217T112042Z.csv.gz"),
        os.path.join(base, "kroger_wa_olive_oil_20260217T112042Z.csv.gz"),
        os.path.join(base, "kroger_wa_paper_towels_20260217T112042Z.csv.gz"),
    ]

    df = load_all_data_local(walmart_files, kroger_files)
    print(df.head(12).to_string())