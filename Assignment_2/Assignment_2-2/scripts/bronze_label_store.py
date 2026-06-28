import argparse
import os
import pyspark

import utils.data_processing_bronze_table as bronze

# to call: python bronze_label_store.py --snapshotdate "2023-01-01"


def main(snapshotdate):
    print('\n\n---starting job: bronze_label_store---\n\n')

    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    bronze_lms_dir = "datamart/bronze/lms/"
    os.makedirs(bronze_lms_dir, exist_ok=True)

    bronze.process_bronze_lms_table(snapshotdate, bronze_lms_dir, spark)

    spark.stop()
    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)
