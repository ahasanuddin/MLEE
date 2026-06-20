import os
from datetime import datetime
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


# ---------------------------------------------------------------------------
# Gold Feature Store
# ---------------------------------------------------------------------------

def process_feature_store_gold_table(snapshot_date_str, silver_dir, gold_feature_dir, spark):
    """
    Join silver clickstream + financials + attributes on Customer_ID + snapshot_date
    to produce the gold feature store partition.

    Clickstream is the driver (time-series, one row per customer per month).
    Financials and attributes are joined on Customer_ID only (one row per customer).
    """
    # --- Load silver clickstream (driver: has Customer_ID + snapshot_date) ---
    click_path = os.path.join(
        silver_dir, "clickstream",
        f"silver_clickstream_{snapshot_date_str.replace('-', '_')}.parquet"
    )
    if not os.path.exists(click_path):
        print(f"[Gold-Feature] No clickstream data for {snapshot_date_str}, skipping.")
        return None

    df_click = spark.read.parquet(click_path)
    print(f"[Gold-Feature] Clickstream loaded | rows: {df_click.count()}")

    # --- Load silver financials (static per customer) ---
    fin_path = os.path.join(
        silver_dir, "financials",
        f"silver_financials_{snapshot_date_str.replace('-', '_')}.parquet"
    )
    if os.path.exists(fin_path):
        df_fin = spark.read.parquet(fin_path).drop("snapshot_date")
        df_click = df_click.join(df_fin, on="Customer_ID", how="left")
        print(f"[Gold-Feature] Joined financials")
    else:
        print(f"[Gold-Feature] No financials for {snapshot_date_str}, skipping financials join.")

    # --- Load silver attributes (static per customer, PII already removed) ---
    att_path = os.path.join(
        silver_dir, "attributes",
        f"silver_attributes_{snapshot_date_str.replace('-', '_')}.parquet"
    )
    if os.path.exists(att_path):
        df_att = spark.read.parquet(att_path).drop("snapshot_date")
        df_click = df_click.join(df_att, on="Customer_ID", how="left")
        print(f"[Gold-Feature] Joined attributes")
    else:
        print(f"[Gold-Feature] No attributes for {snapshot_date_str}, skipping attributes join.")

    df_feature_store = df_click

    # Ensure snapshot_date is a date type
    df_feature_store = df_feature_store.withColumn(
        "snapshot_date", col("snapshot_date").cast(DateType())
    )

    print(f"[Gold-Feature] Feature store rows: {df_feature_store.count()}")

    # Save gold feature store partition
    os.makedirs(gold_feature_dir, exist_ok=True)
    filename = f"gold_feature_store_{snapshot_date_str.replace('-', '_')}.parquet"
    filepath = os.path.join(gold_feature_dir, filename)
    df_feature_store.write.mode("overwrite").parquet(filepath)
    print(f"[Gold-Feature] Saved to: {filepath}")

    return df_feature_store


# ---------------------------------------------------------------------------
# Gold Label Store  (reused from Lab 2)
# ---------------------------------------------------------------------------

def process_bronze_lms_table(snapshot_date_str, bronze_lms_dir, spark):
    """
    Bronze ingestion for lms_loan_daily (unchanged from Lab 2).
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


def process_silver_lms_table(snapshot_date_str, bronze_lms_dir, silver_lms_dir, spark):
    """
    Silver cleaning for lms_loan_daily (unchanged from Lab 2).
    Adds mob and dpd columns.
    """
    filename = f"bronze_loan_daily_{snapshot_date_str.replace('-', '_')}.csv"
    filepath = os.path.join(bronze_lms_dir, filename)
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print(f"[Silver-LMS] Loaded from: {filepath} | rows: {df.count()}")

    column_type_map = {
        "loan_id":          StringType(),
        "Customer_ID":      StringType(),
        "loan_start_date":  DateType(),
        "tenure":           IntegerType(),
        "installment_num":  IntegerType(),
        "loan_amt":         FloatType(),
        "due_amt":          FloatType(),
        "paid_amt":         FloatType(),
        "overdue_amt":      FloatType(),
        "balance":          FloatType(),
        "snapshot_date":    DateType(),
    }
    for c, dtype in column_type_map.items():
        df = df.withColumn(c, col(c).cast(dtype))

    df = df.withColumn("mob", col("installment_num").cast(IntegerType()))
    df = df.withColumn(
        "installments_missed",
        F.ceil(col("overdue_amt") / col("due_amt")).cast(IntegerType())
    ).fillna(0)
    df = df.withColumn(
        "first_missed_date",
        F.when(col("installments_missed") > 0,
               F.add_months(col("snapshot_date"), -1 * col("installments_missed"))
               ).cast(DateType())
    )
    df = df.withColumn(
        "dpd",
        F.when(col("overdue_amt") > 0.0,
               F.datediff(col("snapshot_date"), col("first_missed_date"))
               ).otherwise(0).cast(IntegerType())
    )

    os.makedirs(silver_lms_dir, exist_ok=True)
    filename = f"silver_loan_daily_{snapshot_date_str.replace('-', '_')}.parquet"
    filepath = os.path.join(silver_lms_dir, filename)
    df.write.mode("overwrite").parquet(filepath)
    print(f"[Silver-LMS] Saved to: {filepath}")
    return df


def process_labels_gold_table(snapshot_date_str, silver_lms_dir, gold_label_dir, spark,
                               dpd=30, mob=6):
    """
    Gold label store (unchanged from Lab 2).
    Filters at mob=6 and labels loans with dpd >= 30 as default (label=1).
    """
    filename = f"silver_loan_daily_{snapshot_date_str.replace('-', '_')}.parquet"
    filepath = os.path.join(silver_lms_dir, filename)
    df = spark.read.parquet(filepath)
    print(f"[Gold-Label] Loaded from: {filepath} | rows: {df.count()}")

    df = df.filter(col("mob") == mob)
    df = df.withColumn(
        "label",
        F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType())
    )
    df = df.withColumn(
        "label_def",
        F.lit(f"{dpd}dpd_{mob}mob").cast(StringType())
    )
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    os.makedirs(gold_label_dir, exist_ok=True)
    filename = f"gold_label_store_{snapshot_date_str.replace('-', '_')}.parquet"
    filepath = os.path.join(gold_label_dir, filename)
    df.write.mode("overwrite").parquet(filepath)
    print(f"[Gold-Label] Saved to: {filepath}")
    return df
