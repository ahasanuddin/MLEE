import os
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, DateType


# ---------------------------------------------------------------------------
# Gold Feature Store
# ---------------------------------------------------------------------------

def process_gold_feature_store(snapshot_date_str, silver_dir, gold_feature_dir, spark):
    """
    Join silver clickstream (driver) + financials + attributes on Customer_ID
    to produce the gold feature store partition for one snapshot date.
    """
    suffix = snapshot_date_str.replace('-', '_')

    click_path = os.path.join(silver_dir, "clickstream", f"silver_clickstream_{suffix}.parquet")
    df = spark.read.parquet(click_path)
    print(f"[Gold-Feature] clickstream rows: {df.count()}")

    fin_path = os.path.join(silver_dir, "financials", f"silver_financials_{suffix}.parquet")
    if os.path.exists(fin_path):
        df_fin = spark.read.parquet(fin_path).drop("snapshot_date")
        df = df.join(df_fin, on="Customer_ID", how="left")
        print("[Gold-Feature] joined financials")

    att_path = os.path.join(silver_dir, "attributes", f"silver_attributes_{suffix}.parquet")
    if os.path.exists(att_path):
        df_att = spark.read.parquet(att_path).drop("snapshot_date")
        df = df.join(df_att, on="Customer_ID", how="left")
        print("[Gold-Feature] joined attributes")

    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

    os.makedirs(gold_feature_dir, exist_ok=True)
    filepath = os.path.join(gold_feature_dir, f"gold_feature_store_{suffix}.parquet")
    df.write.mode("overwrite").parquet(filepath)
    print(f"[Gold-Feature] Saved to: {filepath} | rows: {df.count()}")
    return df


# ---------------------------------------------------------------------------
# Gold Label Store
# ---------------------------------------------------------------------------

def process_gold_label_store(snapshot_date_str, silver_lms_dir, gold_label_dir, spark, dpd=30, mob=6):
    suffix = snapshot_date_str.replace('-', '_')
    filepath = os.path.join(silver_lms_dir, f"silver_loan_daily_{suffix}.parquet")
    df = spark.read.parquet(filepath)
    print(f"[Gold-Label] loaded rows: {df.count()}")

    df = df.filter(col("mob") == mob)
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(f"{dpd}dpd_{mob}mob").cast(StringType()))
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    os.makedirs(gold_label_dir, exist_ok=True)
    filepath = os.path.join(gold_label_dir, f"gold_label_store_{suffix}.parquet")
    df.write.mode("overwrite").parquet(filepath)
    print(f"[Gold-Label] Saved to: {filepath} | rows: {df.count()}")
    return df
