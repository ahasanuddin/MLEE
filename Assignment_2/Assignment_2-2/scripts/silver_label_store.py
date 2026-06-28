import argparse
import pyspark

import utils.data_processing_silver_table as silver

# to call: python silver_label_store.py --snapshotdate "2023-01-01"


def main(snapshotdate):
    print('\n\n---starting job: silver_label_store---\n\n')

    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    bronze_lms_dir = "datamart/bronze/lms/"
    silver_lms_dir = "datamart/silver/loan_daily/"

    silver.process_silver_lms(snapshotdate, bronze_lms_dir, silver_lms_dir, spark)

    spark.stop()
    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)
