import argparse
import pyspark

import utils.data_processing_gold_table as gold

# to call: python gold_label_store.py --snapshotdate "2023-01-01" --dpd 30 --mob 6

LABEL_DPD = 30
LABEL_MOB = 6


def main(snapshotdate, dpd=LABEL_DPD, mob=LABEL_MOB):
    print('\n\n---starting job: gold_label_store---\n\n')

    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    silver_lms_dir = "datamart/silver/loan_daily/"
    gold_label_dir = "datamart/gold/label_store/"

    gold.process_gold_label_store(snapshotdate, silver_lms_dir, gold_label_dir, spark, dpd=dpd, mob=mob)

    spark.stop()
    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--dpd", type=int, default=LABEL_DPD)
    parser.add_argument("--mob", type=int, default=LABEL_MOB)
    args = parser.parse_args()
    main(args.snapshotdate, args.dpd, args.mob)
