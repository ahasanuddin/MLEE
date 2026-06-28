import argparse
import pyspark

import utils.data_processing_silver_table as silver

# to call: python silver_feature_store.py --snapshotdate "2023-01-01" --source clickstream

PROCESSORS = {
    "clickstream": silver.process_silver_clickstream,
    "financials":  silver.process_silver_financials,
    "attributes":  silver.process_silver_attributes,
}


def main(snapshotdate, source):
    print('\n\n---starting job: silver_feature_store [' + source + ']---\n\n')

    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    bronze_dir = "datamart/bronze/features/"
    silver_dir = "datamart/silver/features/"

    PROCESSORS[source](snapshotdate, bronze_dir, silver_dir, spark)

    spark.stop()
    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--source", type=str, required=True, choices=list(PROCESSORS.keys()))
    args = parser.parse_args()
    main(args.snapshotdate, args.source)
