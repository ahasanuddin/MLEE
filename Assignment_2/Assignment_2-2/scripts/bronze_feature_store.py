import argparse
import os
import pyspark

import utils.data_processing_bronze_table as bronze

# to call: python bronze_feature_store.py --snapshotdate "2023-01-01" --source clickstream

SOURCES = {
    "clickstream": "data/feature_clickstream.csv",
    "financials":  "data/features_financials.csv",
    "attributes":  "data/features_attributes.csv",
}


def main(snapshotdate, source):
    print('\n\n---starting job: bronze_feature_store [' + source + ']---\n\n')

    spark = pyspark.sql.SparkSession.builder \
        .appName("dev") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    bronze_dir = "datamart/bronze/features/"
    os.makedirs(bronze_dir, exist_ok=True)

    bronze.process_bronze_feature_table(source, SOURCES[source], snapshotdate, bronze_dir, spark)

    spark.stop()
    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--source", type=str, required=True, choices=list(SOURCES.keys()))
    args = parser.parse_args()
    main(args.snapshotdate, args.source)
