"""
Assignment 1 - Data Processing Pipelines
CS611 - Machine Learning Engineering

Runs the full Medallion Architecture pipeline:
  Bronze -> Silver -> Gold (Feature Store + Label Store)

Usage:
    python main.py
"""

import os
import glob
from datetime import datetime
import pyspark

import utils.data_processing_bronze_table as bronze
import utils.data_processing_silver_table as silver
import utils.data_processing_gold_table as gold


# ---------------------------------------------------------------------------
# Spark
# ---------------------------------------------------------------------------
spark = pyspark.sql.SparkSession.builder \
    .appName("CS611_Assignment1") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
START_DATE = "2023-01-01"
END_DATE   = "2024-12-01"

LABEL_DPD = 30   # days past due threshold for default label
LABEL_MOB = 6    # month-on-book at which label is observed


def generate_first_of_month_dates(start_date_str, end_date_str):
    start = datetime.strptime(start_date_str, "%Y-%m-%d")
    end   = datetime.strptime(end_date_str,   "%Y-%m-%d")
    dates = []
    current = datetime(start.year, start.month, 1)
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)
    return dates


dates = generate_first_of_month_dates(START_DATE, END_DATE)
print(f"Processing {len(dates)} monthly snapshots: {dates[0]} to {dates[-1]}")


# ---------------------------------------------------------------------------
# ── FEATURE STORE PIPELINE ──────────────────────────────────────────────────
# ---------------------------------------------------------------------------

# Bronze feature tables
bronze_feature_dir = "datamart/bronze/features/"
os.makedirs(bronze_feature_dir, exist_ok=True)
print("\n" + "="*60)
print("BRONZE - Feature Sources")
print("="*60)
for date_str in dates:
    bronze.process_bronze_table(date_str, bronze_feature_dir, spark)

# Silver feature tables
silver_feature_dir = "datamart/silver/features/"
os.makedirs(silver_feature_dir, exist_ok=True)
print("\n" + "="*60)
print("SILVER - Feature Sources")
print("="*60)
for date_str in dates:
    silver.process_silver_table(date_str, bronze_feature_dir, silver_feature_dir, spark)

# Gold feature store
gold_feature_dir = "datamart/gold/feature_store/"
os.makedirs(gold_feature_dir, exist_ok=True)
print("\n" + "="*60)
print("GOLD - Feature Store")
print("="*60)
for date_str in dates:
    gold.process_feature_store_gold_table(date_str, silver_feature_dir, gold_feature_dir, spark)


# ---------------------------------------------------------------------------
# ── LABEL STORE PIPELINE  (reused from Lab 2) ───────────────────────────────
# ---------------------------------------------------------------------------

# Bronze LMS
bronze_lms_dir = "datamart/bronze/lms/"
os.makedirs(bronze_lms_dir, exist_ok=True)
print("\n" + "="*60)
print("BRONZE - LMS Loan Daily")
print("="*60)
for date_str in dates:
    gold.process_bronze_lms_table(date_str, bronze_lms_dir, spark)

# Silver LMS
silver_lms_dir = "datamart/silver/loan_daily/"
os.makedirs(silver_lms_dir, exist_ok=True)
print("\n" + "="*60)
print("SILVER - LMS Loan Daily")
print("="*60)
for date_str in dates:
    gold.process_silver_lms_table(date_str, bronze_lms_dir, silver_lms_dir, spark)

# Gold label store
gold_label_dir = "datamart/gold/label_store/"
os.makedirs(gold_label_dir, exist_ok=True)
print("\n" + "="*60)
print("GOLD - Label Store")
print("="*60)
for date_str in dates:
    gold.process_labels_gold_table(date_str, silver_lms_dir, gold_label_dir, spark,
                                   dpd=LABEL_DPD, mob=LABEL_MOB)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("PIPELINE COMPLETE - Summary")
print("="*60)

# Feature store summary
feat_files = glob.glob(os.path.join(gold_feature_dir, "*.parquet"))
if feat_files:
    df_feat = spark.read.option("header", "true").parquet(*feat_files)
    print(f"Gold Feature Store  | rows: {df_feat.count()} | columns: {len(df_feat.columns)}")
    df_feat.show(5)

# Label store summary
label_files = glob.glob(os.path.join(gold_label_dir, "*.parquet"))
if label_files:
    df_label = spark.read.option("header", "true").parquet(*label_files)
    print(f"Gold Label Store    | rows: {df_label.count()} | columns: {len(df_label.columns)}")
    df_label.show(5)
    print("Label distribution:")
    df_label.groupBy("label").count().show()
