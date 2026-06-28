import argparse
import pyspark

import utils.data_processing_gold_table as gold

# to call: python gold_feature_store.py --snapshotdate "2023-01-01"


def main(snapshotdate):
    print('\n\n---starting job: gold_feature_store---\n\n')

    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    silver_dir = "datamart/silver/features/"
    gold_feature_dir = "datamart/gold/feature_store/"

    gold.process_gold_feature_store(snapshotdate, silver_dir, gold_feature_dir, spark)

    spark.stop()
    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)
