import argparse
import os
import pickle
import pprint
from datetime import datetime

import numpy as np
import pyspark
from pyspark.sql.functions import col

# to call: python model_monitor.py --snapshotdate "2024-09-01" --modelname credit_model_2024_09_01.pkl
#
# Design: monitoring for a given month only runs if predictions exist for
# that month (i.e. inference has already run). It computes:
#   1. Population Stability Index (PSI) of the prediction-score
#      distribution vs. the training-time baseline distribution
#      (population/score stability — does the model still see a similar
#      population to what it was trained on?)
#   2. If actual labels are available for this snapshot date
#      (gold_label_store), computes AUC / Gini and compares the actual
#      default rate to the model's average predicted probability
#      (performance stability — is the model still discriminating well?)


def compute_psi(actual_scores, baseline_bins, baseline_pct):
    """PSI = sum((actual_pct - baseline_pct) * ln(actual_pct / baseline_pct))"""
    actual_counts, _ = np.histogram(actual_scores, bins=baseline_bins)
    actual_pct = actual_counts / actual_counts.sum()

    baseline_pct = np.array(baseline_pct, dtype=float)
    # Floor both distributions away from zero, then renormalize.
    eps = 1e-4
    actual_pct = np.clip(actual_pct, eps, None)
    baseline_pct = np.clip(baseline_pct, eps, None)
    actual_pct = actual_pct / actual_pct.sum()
    baseline_pct = baseline_pct / baseline_pct.sum()
    psi = np.sum((actual_pct - baseline_pct) * np.log(actual_pct / baseline_pct))
    return float(psi)


def main(snapshotdate, modelname):
    print('\n\n---starting job: model_monitor---\n\n')

    model_bank_directory = "model_bank/"
    model_artefact_filepath = os.path.join(model_bank_directory, modelname)

    if not os.path.exists(model_artefact_filepath):
        print(f"[model_monitor] Model artefact {model_artefact_filepath} not found. Skipping.")
        print('\n\n---completed job (skipped)---\n\n')
        return

    with open(model_artefact_filepath, 'rb') as f:
        model_artefact = pickle.load(f)

    suffix = snapshotdate.replace('-', '_')
    pred_path = f"datamart/gold/model_predictions/{modelname[:-4]}/{modelname[:-4]}_predictions_{suffix}.parquet"

    if not os.path.exists(pred_path):
        print(f"[model_monitor] Predictions not found for {snapshotdate}: {pred_path}. Skipping.")
        print('\n\n---completed job (skipped)---\n\n')
        return

    spark = pyspark.sql.SparkSession.builder.appName("dev").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    config = {"snapshot_date_str": snapshotdate, "model_name": modelname}
    pprint.pprint(config)

    # --- load predictions ---
    pred_sdf = spark.read.parquet(pred_path)
    pred_pdf = pred_sdf.toPandas()
    scores = pred_pdf["model_predictions"].values
    print("predictions rows:", len(scores))

    # --- PSI vs training baseline ---
    baseline = model_artefact['monitoring_baseline']
    psi = compute_psi(scores, baseline['psi_bins'], baseline['psi_bin_pct'])

    monitor_row = {
        "snapshot_date": datetime.strptime(snapshotdate, "%Y-%m-%d").date(),
        "model_name": modelname,
        "n_predictions": int(len(scores)),
        "score_mean": float(scores.mean()),
        "score_std": float(scores.std()),
        "psi_score": psi,
        "training_score_mean": baseline['score_mean'],
        "training_default_rate": baseline['default_rate'],
        "actual_default_rate": None,
        "auc": None,
        "gini": None,
        "n_labels": 0,
    }

    # --- if actual labels are available, compute AUC / Gini ---
    label_path = f"datamart/gold/label_store/gold_label_store_{suffix}.parquet"
    if os.path.exists(label_path):
        label_sdf = spark.read.parquet(label_path)
        joined = pred_sdf.join(label_sdf, on=["Customer_ID", "snapshot_date"], how="inner")
        joined_pdf = joined.toPandas()

        if len(joined_pdf) > 0 and joined_pdf["label"].nunique() > 1:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(joined_pdf["label"], joined_pdf["model_predictions"])
            monitor_row["auc"] = float(auc)
            monitor_row["gini"] = round(2 * auc - 1, 3)
            monitor_row["actual_default_rate"] = float(joined_pdf["label"].mean())
            monitor_row["n_labels"] = int(len(joined_pdf))
            print(f"AUC: {auc:.4f}  Gini: {monitor_row['gini']}  "
                  f"actual_default_rate: {monitor_row['actual_default_rate']:.3f}  "
                  f"n_labels: {monitor_row['n_labels']}")
        else:
            print("Labels found but insufficient class variation; skipping AUC.")
    else:
        print(f"No label store partition for {snapshotdate} yet (labels lag by MOB). "
              f"Logging stability metrics only.")

    print(f"PSI (score distribution vs training): {psi:.4f}")
    if psi < 0.1:
        flag = "STABLE"
    elif psi < 0.25:
        flag = "WATCH"
    else:
        flag = "ALERT - SIGNIFICANT DRIFT"
    monitor_row["psi_flag"] = flag
    print(f"PSI flag: {flag}")

    # --- save monitoring row to gold table ---
    out_dir = f"datamart/gold/model_monitoring/{modelname[:-4]}/"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{modelname[:-4]}_monitoring_{suffix}.parquet")

    monitor_row = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in monitor_row.items()}
    monitor_sdf = spark.createDataFrame([monitor_row])
    monitor_sdf.write.mode("overwrite").parquet(out_path)
    print('saved to:', out_path)

    spark.stop()
    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--modelname", type=str, required=True, help="model_name (e.g. credit_model_2024_09_01.pkl)")
    args = parser.parse_args()
    main(args.snapshotdate, args.modelname)
