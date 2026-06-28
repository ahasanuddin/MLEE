import argparse
import os
import glob
import pickle
import pprint
from datetime import datetime

import pyspark
from pyspark.sql.functions import col

# to call: python model_inference.py --snapshotdate "2024-09-01" --modelname credit_model_2024_09_01.pkl
#
# Design: inference for a given month only runs if a trained model
# artefact already exists in the model bank AND the snapshot date is on
# or after the model's training date. This mirrors production: you
# cannot serve predictions with a model that doesn't exist yet.


def main(snapshotdate, modelname):
    print('\n\n---starting job: model_inference---\n\n')

    model_bank_directory = "model_bank/"
    model_artefact_filepath = os.path.join(model_bank_directory, modelname)

    if not os.path.exists(model_artefact_filepath):
        print(f"[model_inference] Model artefact {model_artefact_filepath} not found. "
              f"Skipping inference for {snapshotdate} (model not yet trained).")
        print('\n\n---completed job (skipped)---\n\n')
        return

    with open(model_artefact_filepath, 'rb') as f:
        model_artefact = pickle.load(f)

    train_date = model_artefact['data_dates']['model_train_date']
    snapshot_dt = datetime.strptime(snapshotdate, "%Y-%m-%d")
    if snapshot_dt < train_date:
        print(f"[model_inference] snapshot {snapshotdate} is before model training date "
              f"{train_date.date()}. Skipping inference (model not yet in production).")
        print('\n\n---completed job (skipped)---\n\n')
        return

    spark = pyspark.sql.SparkSession.builder.appName("dev").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    config = {
        "snapshot_date_str": snapshotdate,
        "snapshot_date": snapshot_dt,
        "model_name": modelname,
        "model_artefact_filepath": model_artefact_filepath,
    }
    pprint.pprint(config)

    print("Model loaded successfully! " + model_artefact_filepath)

    # --- load feature store for this snapshot date ---
    feature_folder = "datamart/gold/feature_store/"
    suffix = snapshotdate.replace('-', '_')
    feature_path = os.path.join(feature_folder, f"gold_feature_store_{suffix}.parquet")

    if not os.path.exists(feature_path):
        print(f"[model_inference] Feature store partition not found: {feature_path}. Skipping.")
        spark.stop()
        print('\n\n---completed job (skipped)---\n\n')
        return

    features_sdf = spark.read.parquet(feature_path)
    features_pdf = features_sdf.toPandas()
    print("features rows:", features_pdf.shape[0])

    # --- preprocess + predict ---
    feature_cols = model_artefact['feature_cols']
    X_inference = features_pdf[feature_cols]

    preprocessor = model_artefact['preprocessing_pipeline']
    X_inference_processed = preprocessor.transform(X_inference)

    model = model_artefact['model']
    y_inference = model.predict_proba(X_inference_processed)[:, 1]

    y_inference_pdf = features_pdf[["Customer_ID", "snapshot_date"]].copy()
    y_inference_pdf["model_name"] = modelname
    y_inference_pdf["model_predictions"] = y_inference

    print("predictions:", y_inference_pdf.shape[0], "| mean score:", round(float(y_inference.mean()), 4))

    # --- save to gold model_predictions table ---
    gold_directory = f"datamart/gold/model_predictions/{modelname[:-4]}/"
    os.makedirs(gold_directory, exist_ok=True)
    filename = f"{modelname[:-4]}_predictions_{suffix}.parquet"
    filepath = os.path.join(gold_directory, filename)
    spark.createDataFrame(y_inference_pdf).write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    spark.stop()
    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--modelname", type=str, required=True, help="model_name (e.g. credit_model_2024_09_01.pkl)")
    args = parser.parse_args()
    main(args.snapshotdate, args.modelname)
