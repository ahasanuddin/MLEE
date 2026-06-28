import os
import pyspark.sql.functions as F
from pyspark.sql.functions import col, regexp_replace, when
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_numeric_str(df, col_name):
    """Strip non-numeric trailing characters (e.g. '52312.68_' -> '52312.68').
    Empty results (e.g. from a value that was just '_') become null so the
    subsequent cast to Float/Integer doesn't fail under ANSI mode."""
    df = df.withColumn(col_name, regexp_replace(col(col_name), r"[^0-9\.]", ""))
    df = df.withColumn(col_name, when(col(col_name) == "", None).otherwise(col(col_name)))
    return df


def _clean_category(df, col_name, invalid_values=("_", "nan", "NA", "N/A", "")):
    """Replace known dirty category placeholders with null."""
    condition = col(col_name).isin(list(invalid_values))
    return df.withColumn(col_name, when(condition, None).otherwise(col(col_name)))


# ---------------------------------------------------------------------------
# Clickstream
# ---------------------------------------------------------------------------

def process_silver_clickstream(snapshot_date_str, bronze_dir, silver_dir, spark):
    bronze_path = os.path.join(bronze_dir, "clickstream", f"bronze_clickstream_{snapshot_date_str.replace('-', '_')}.csv")
    df = spark.read.csv(bronze_path, header=True, inferSchema=True)
    print(f"[Silver] clickstream | {snapshot_date_str} | loaded rows: {df.count()}")

    feature_cols = [f"fe_{i}" for i in range(1, 21)]
    for c in feature_cols:
        df = df.withColumn(c, col(c).cast(IntegerType()))

    df = df.withColumn("Customer_ID", col("Customer_ID").cast(StringType()))
    df = df.withColumn("snapshot_date", col("snapshot_date").cast(DateType()))

    out_dir = os.path.join(silver_dir, "clickstream")
    os.makedirs(out_dir, exist_ok=True)
    filepath = os.path.join(out_dir, f"silver_clickstream_{snapshot_date_str.replace('-', '_')}.parquet")
    df.write.mode("overwrite").parquet(filepath)
    print(f"[Silver] Saved to: {filepath}")
    return df


# ---------------------------------------------------------------------------
# Financials
# ---------------------------------------------------------------------------

def process_silver_financials(snapshot_date_str, bronze_dir, silver_dir, spark):
    bronze_path = os.path.join(bronze_dir, "financials", f"bronze_financials_{snapshot_date_str.replace('-', '_')}.csv")
    df = spark.read.csv(bronze_path, header=True, inferSchema=True)
    print(f"[Silver] financials | {snapshot_date_str} | loaded rows: {df.count()}")

    # Numeric columns arrived as strings due to dirty suffixes
    numeric_str_cols = [
        "Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment",
        "Changed_Credit_Limit", "Outstanding_Debt",
        "Amount_invested_monthly", "Monthly_Balance",
    ]
    for c in numeric_str_cols:
        df = _clean_numeric_str(df, c)
        df = df.withColumn(c, col(c).cast(FloatType()))

    # Categorical clean-up
    for c in ["Credit_Mix", "Payment_Behaviour", "Payment_of_Min_Amount"]:
        df = _clean_category(df, c)

    # Parse Credit_History_Age -> total months
    df = df.withColumn(
        "credit_history_months",
        (
            F.coalesce(F.regexp_extract(col("Credit_History_Age"), r"(\d+)\s+Year", 1).cast(IntegerType()), F.lit(0)) * 12
            + F.coalesce(F.regexp_extract(col("Credit_History_Age"), r"(\d+)\s+Month", 1).cast(IntegerType()), F.lit(0))
        ).cast(IntegerType())
    )

    df = df.drop("Type_of_Loan", "Credit_History_Age")

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

    out_dir = os.path.join(silver_dir, "financials")
    os.makedirs(out_dir, exist_ok=True)
    filepath = os.path.join(out_dir, f"silver_financials_{snapshot_date_str.replace('-', '_')}.parquet")
    df.write.mode("overwrite").parquet(filepath)
    print(f"[Silver] Saved to: {filepath}")
    return df


# ---------------------------------------------------------------------------
# Attributes
# ---------------------------------------------------------------------------

def process_silver_attributes(snapshot_date_str, bronze_dir, silver_dir, spark):
    bronze_path = os.path.join(bronze_dir, "attributes", f"bronze_attributes_{snapshot_date_str.replace('-', '_')}.csv")
    df = spark.read.csv(bronze_path, header=True, inferSchema=True)
    print(f"[Silver] attributes | {snapshot_date_str} | loaded rows: {df.count()}")

    # Drop PII
    df = df.drop("Name", "SSN")

    # Clean Age: strip non-numeric chars, then null-out implausible values
    df = _clean_numeric_str(df, "Age")
    df = df.withColumn("Age", col("Age").cast(IntegerType()))
    df = df.withColumn("Age", when((col("Age") < 0) | (col("Age") > 110), None).otherwise(col("Age")))

    type_map = {
        "Customer_ID":   StringType(),
        "Occupation":    StringType(),
        "snapshot_date": DateType(),
    }
    for c, dtype in type_map.items():
        df = df.withColumn(c, col(c).cast(dtype))

    out_dir = os.path.join(silver_dir, "attributes")
    os.makedirs(out_dir, exist_ok=True)
    filepath = os.path.join(out_dir, f"silver_attributes_{snapshot_date_str.replace('-', '_')}.parquet")
    df.write.mode("overwrite").parquet(filepath)
    print(f"[Silver] Saved to: {filepath}")
    return df


# ---------------------------------------------------------------------------
# LMS Loan Daily
# ---------------------------------------------------------------------------

def process_silver_lms(snapshot_date_str, bronze_lms_dir, silver_lms_dir, spark):
    filepath = os.path.join(bronze_lms_dir, f"bronze_loan_daily_{snapshot_date_str.replace('-', '_')}.csv")
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
        F.when(col("due_amt") > 0.0, F.ceil(col("overdue_amt") / col("due_amt")))
         .otherwise(F.lit(0))
         .cast(IntegerType())
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
    filepath = os.path.join(silver_lms_dir, f"silver_loan_daily_{snapshot_date_str.replace('-', '_')}.parquet")
    df.write.mode("overwrite").parquet(filepath)
    print(f"[Silver-LMS] Saved to: {filepath}")
    return df
