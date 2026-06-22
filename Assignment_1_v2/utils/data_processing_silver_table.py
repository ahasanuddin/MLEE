import os
import re
from datetime import datetime
import pyspark.sql.functions as F
from pyspark.sql.functions import col, regexp_replace, trim, when
from pyspark.sql.types import (
    StringType, IntegerType, FloatType, DateType
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_numeric_str(df, col_name):
    """
    Strip non-numeric trailing characters (e.g. '52312.68_' -> '52312.68').
    Replaces anything after the last digit or decimal point with nothing.
    """
    return df.withColumn(
        col_name,
        regexp_replace(col(col_name), r"[^0-9\.]", "")
    )


def _clean_category(df, col_name, invalid_values=("_", "nan", "NA", "N/A", "")):
    """
    Replace known dirty category placeholders with null.
    """
    condition = col(col_name).isin(list(invalid_values))
    return df.withColumn(col_name, when(condition, None).otherwise(col(col_name)))


# ---------------------------------------------------------------------------
# Per-source silver processing
# ---------------------------------------------------------------------------

def _process_silver_clickstream(df):
    """
    Clickstream: fe_1..fe_20 are numeric click-behaviour features.
    Enforce integer types and keep Customer_ID + snapshot_date.
    """
    feature_cols = [f"fe_{i}" for i in range(1, 21)]
    for c in feature_cols:
        df = df.withColumn(c, col(c).cast(IntegerType()))

    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

    return df


def _process_silver_financials(df):
    """
    Financials: mixed numeric / categorical columns with dirty string suffixes.
    - Strip '_' noise from numeric strings (e.g. '52312.68_')
    - Replace '_' category placeholder with null
    - Parse Credit_History_Age into total months
    - Drop Type_of_Loan (high cardinality free-text, not useful without NLP)
    """
    # --- Numeric columns that arrived as strings due to dirty suffixes ---
    numeric_str_cols = [
        "Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment",
        "Changed_Credit_Limit", "Outstanding_Debt",
        "Amount_invested_monthly", "Monthly_Balance",
    ]
    for c in numeric_str_cols:
        df = _clean_numeric_str(df, c)
        df = df.withColumn(c, col(c).cast(FloatType()))

    # --- Categorical clean-up ---
    cat_cols = ["Credit_Mix", "Payment_Behaviour", "Payment_of_Min_Amount"]
    for c in cat_cols:
        df = _clean_category(df, c)

    # --- Parse Credit_History_Age: 'X Years and Y Months' -> total months ---
    df = df.withColumn(
        "credit_history_months",
        (
            F.coalesce(
                F.regexp_extract(col("Credit_History_Age"), r"(\d+)\s+Year", 1).cast(IntegerType()),
                F.lit(0)
            ) * 12
            +
            F.coalesce(
                F.regexp_extract(col("Credit_History_Age"), r"(\d+)\s+Month", 1).cast(IntegerType()),
                F.lit(0)
            )
        ).cast(IntegerType())
    )

    # --- Drop columns not useful for ML ---
    df = df.drop("Type_of_Loan", "Credit_History_Age")

    # --- Enforce remaining types ---
    type_map = {
        "Customer_ID":              StringType(),
        "Monthly_Inhand_Salary":    FloatType(),
        "Num_Bank_Accounts":        IntegerType(),
        "Num_Credit_Card":          IntegerType(),
        "Interest_Rate":            IntegerType(),
        "Delay_from_due_date":      IntegerType(),
        "Num_Credit_Inquiries":     FloatType(),
        "Credit_Utilization_Ratio": FloatType(),
        "Total_EMI_per_month":      FloatType(),
        "snapshot_date":            DateType(),
    }
    for c, dtype in type_map.items():
        df = df.withColumn(c, col(c).cast(dtype))

    return df


def _process_silver_attributes(df):
    """
    Attributes: Customer demographics.
    - Drop PII columns (Name, SSN)
    - Enforce types
    """
    # Drop PII
    df = df.drop("Name", "SSN")

    # Clean Age: sometimes stored with noise
    df = _clean_numeric_str(df, "Age")

    type_map = {
        "Customer_ID":  StringType(),
        "Age":          IntegerType(),
        "Occupation":   StringType(),
        "snapshot_date": DateType(),
    }
    for c, dtype in type_map.items():
        df = df.withColumn(c, col(c).cast(dtype))

    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def process_silver_table(snapshot_date_str, bronze_dir, silver_dir, spark):
    """
    Read bronze partitions for each feature source, clean and save as parquet.
    Returns a dict of cleaned DataFrames.
    """
    results = {}

    processors = {
        "clickstream": _process_silver_clickstream,
        "financials":  _process_silver_financials,
        "attributes":  _process_silver_attributes,
    }

    for source_name, processor_fn in processors.items():
        bronze_path = os.path.join(
            bronze_dir, source_name,
            f"bronze_{source_name}_{snapshot_date_str.replace('-', '_')}.csv"
        )

        if not os.path.exists(bronze_path):
            print(f"[Silver] No bronze file for {source_name} on {snapshot_date_str}, skipping.")
            continue

        df = spark.read.csv(bronze_path, header=True, inferSchema=True)
        print(f"[Silver] {source_name} | {snapshot_date_str} | loaded rows: {df.count()}")

        df = processor_fn(df)

        # Save as parquet partition
        out_dir = os.path.join(silver_dir, source_name)
        os.makedirs(out_dir, exist_ok=True)
        filename = f"silver_{source_name}_{snapshot_date_str.replace('-', '_')}.parquet"
        filepath = os.path.join(out_dir, filename)
        df.write.mode("overwrite").parquet(filepath)
        print(f"[Silver] Saved to: {filepath}")

        results[source_name] = df

    return results
