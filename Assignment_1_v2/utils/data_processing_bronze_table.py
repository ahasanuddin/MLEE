import os
from datetime import datetime
from pyspark.sql.functions import col


def process_bronze_table(snapshot_date_str, bronze_dir, spark):
    """
    Ingest raw feature CSVs for a given snapshot_date and save to bronze layer.
    Handles 3 sources: clickstream, financials, attributes.
    Returns a dict of DataFrames for each source.
    """
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    results = {}

    sources = {
        "clickstream":  "data/feature_clickstream.csv",
        "financials":   "data/features_financials.csv",
        "attributes":   "data/features_attributes.csv",
    }

    for source_name, csv_path in sources.items():
        # Load raw CSV and filter to snapshot_date
        df = (
            spark.read
            .csv(csv_path, header=True, inferSchema=True)
            .filter(col("snapshot_date") == snapshot_date)
        )

        row_count = df.count()
        print(f"[Bronze] {source_name} | {snapshot_date_str} | rows: {row_count}")

        if row_count == 0:
            print(f"[Bronze] No data for {source_name} on {snapshot_date_str}, skipping.")
            continue

        # Save as CSV partition (raw / no transformation)
        out_dir = os.path.join(bronze_dir, source_name)
        os.makedirs(out_dir, exist_ok=True)
        filename = f"bronze_{source_name}_{snapshot_date_str.replace('-', '_')}.csv"
        filepath = os.path.join(out_dir, filename)
        df.toPandas().to_csv(filepath, index=False)
        print(f"[Bronze] Saved to: {filepath}")

        results[source_name] = df

    return results
