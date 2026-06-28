import os
from datetime import datetime
from pyspark.sql.functions import col


def process_bronze_feature_table(source_name, csv_path, snapshot_date_str, bronze_dir, spark):
    """
    Ingest one raw feature CSV and save to bronze layer.
    source_name: 'clickstream' | 'financials' | 'attributes'

    Design note: clickstream is a genuine monthly time-series fact table
    (one row per customer per month) and is filtered to snapshot_date.
    financials and attributes are static customer dimension tables
    (one row per customer, captured once) — they are ingested as a full
    snapshot on every run (Type-1 SCD refresh), so that every monthly
    partition has complete coverage of all 12,500 customer profiles to
    join against that month's clickstream activity.
    """
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    df = spark.read.csv(csv_path, header=True, inferSchema=True)

    if source_name == "clickstream":
        df = df.filter(col("snapshot_date") == snapshot_date)
    # financials / attributes: full customer dimension snapshot, no date filter

    row_count = df.count()
    print(f"[Bronze] {source_name} | {snapshot_date_str} | rows: {row_count}")

    out_dir = os.path.join(bronze_dir, source_name)
    os.makedirs(out_dir, exist_ok=True)
    filename = f"bronze_{source_name}_{snapshot_date_str.replace('-', '_')}.csv"
    filepath = os.path.join(out_dir, filename)
    df.toPandas().to_csv(filepath, index=False)
    print(f"[Bronze] Saved to: {filepath}")

    return df


def process_bronze_lms_table(snapshot_date_str, bronze_lms_dir, spark):
    """
    Bronze ingestion for lms_loan_daily.
    """
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    csv_file_path = "data/lms_loan_daily.csv"
    df = (
        spark.read
        .csv(csv_file_path, header=True, inferSchema=True)
        .filter(col("snapshot_date") == snapshot_date)
    )
    print(f"[Bronze-LMS] {snapshot_date_str} | rows: {df.count()}")

    os.makedirs(bronze_lms_dir, exist_ok=True)
    filename = f"bronze_loan_daily_{snapshot_date_str.replace('-', '_')}.csv"
    filepath = os.path.join(bronze_lms_dir, filename)
    df.toPandas().to_csv(filepath, index=False)
    print(f"[Bronze-LMS] Saved to: {filepath}")
    return df
