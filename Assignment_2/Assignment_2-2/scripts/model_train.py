import argparse
import os
import glob
import pickle
import pprint
from datetime import datetime, timedelta

import numpy as np
import pyspark
from pyspark.sql.functions import col
from dateutil.relativedelta import relativedelta

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, roc_auc_score
import xgboost as xgb

# to call: python model_train.py --snapshotdate "2024-09-01"
#
# Design: training is only performed on TRAIN_DATE. For every other
# snapshot date this script is a deliberate no-op — in production you
# would not retrain every month, only on a governed schedule. This keeps
# the DAG graph identical across all monthly runs while only doing real
# work once enough history has accumulated.

TRAIN_DATE = "2024-09-01"
TRAIN_TEST_PERIOD_MONTHS = 12
OOT_PERIOD_MONTHS = 2
TRAIN_TEST_RATIO = 0.8

FEATURE_COLS = (
    [f"fe_{i}" for i in range(1, 21)] +
    [
        "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts", "Num_Credit_Card",
        "Interest_Rate", "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment",
        "Changed_Credit_Limit", "Num_Credit_Inquiries", "Outstanding_Debt",
        "Credit_Utilization_Ratio", "Total_EMI_per_month", "Amount_invested_monthly",
        "Monthly_Balance", "credit_history_months", "Age",
    ]
)


def main(snapshotdate):
    print('\n\n---starting job: model_train---\n\n')

    if snapshotdate != TRAIN_DATE:
        print(f"[model_train] snapshot {snapshotdate} != TRAIN_DATE {TRAIN_DATE}. "
              f"Skipping training (model governance: train only on scheduled date).")
        print('\n\n---completed job (skipped)---\n\n')
        return

    spark = pyspark.sql.SparkSession.builder.appName("dev").master("local[*]").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # --- set up config ---
    config = {}
    config["model_train_date_str"] = snapshotdate
    config["train_test_period_months"] = TRAIN_TEST_PERIOD_MONTHS
    config["oot_period_months"] = OOT_PERIOD_MONTHS
    config["model_train_date"] = datetime.strptime(snapshotdate, "%Y-%m-%d")
    config["oot_end_date"] = config["model_train_date"] - timedelta(days=1)
    config["oot_start_date"] = config["model_train_date"] - relativedelta(months=OOT_PERIOD_MONTHS)
    config["train_test_end_date"] = config["oot_start_date"] - timedelta(days=1)
    config["train_test_start_date"] = config["oot_start_date"] - relativedelta(months=TRAIN_TEST_PERIOD_MONTHS)
    config["train_test_ratio"] = TRAIN_TEST_RATIO
    pprint.pprint(config)

    # --- load label store ---
    label_folder = "datamart/gold/label_store/"
    label_files = [label_folder + os.path.basename(f) for f in glob.glob(os.path.join(label_folder, '*'))]
    label_sdf = spark.read.option("header", "true").parquet(*label_files)
    labels_sdf = label_sdf.filter(
        (col("snapshot_date") >= config["train_test_start_date"]) &
        (col("snapshot_date") <= config["oot_end_date"])
    )
    print("labels rows:", labels_sdf.count())

    # --- load feature store ---
    feature_folder = "datamart/gold/feature_store/"
    feature_files = [feature_folder + os.path.basename(f) for f in glob.glob(os.path.join(feature_folder, '*'))]
    feature_sdf = spark.read.option("header", "true").parquet(*feature_files)
    features_sdf = feature_sdf.filter(
        (col("snapshot_date") >= config["train_test_start_date"]) &
        (col("snapshot_date") <= config["oot_end_date"])
    )
    print("features rows:", features_sdf.count())

    # --- join label + feature on Customer_ID + snapshot_date ---
    data_pdf = labels_sdf.join(features_sdf, on=["Customer_ID", "snapshot_date"], how="left").toPandas()
    print("joined rows:", data_pdf.shape[0])

    # --- split train / test / oot ---
    oot_pdf = data_pdf[(data_pdf['snapshot_date'] >= config["oot_start_date"].date()) &
                        (data_pdf['snapshot_date'] <= config["oot_end_date"].date())]
    train_test_pdf = data_pdf[(data_pdf['snapshot_date'] >= config["train_test_start_date"].date()) &
                               (data_pdf['snapshot_date'] <= config["train_test_end_date"].date())]

    X_oot = oot_pdf[FEATURE_COLS]
    y_oot = oot_pdf["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        train_test_pdf[FEATURE_COLS], train_test_pdf["label"],
        test_size=1 - TRAIN_TEST_RATIO,
        random_state=88,
        shuffle=True,
        stratify=train_test_pdf["label"]
    )
    print('X_train', X_train.shape[0], 'X_test', X_test.shape[0], 'X_oot', X_oot.shape[0])
    print('y_train mean', round(y_train.mean(), 3), 'y_test mean', round(y_test.mean(), 3), 'y_oot mean', round(y_oot.mean(), 3))

    # --- preprocessing: impute (median) then scale ---
    # Median imputation chosen because Silver cleaning produced nulls for
    # Age outliers and Credit_Mix-style placeholders — median is robust
    # to the remaining skew/outliers in financial columns.
    preprocessor = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    preprocessor.fit(X_train)

    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    X_oot_processed = preprocessor.transform(X_oot)

    # --- train XGBoost with randomized hyperparameter search ---
    xgb_clf = xgb.XGBClassifier(eval_metric='logloss', random_state=88)

    param_dist = {
        'n_estimators': [25, 50, 75],
        'max_depth': [2, 3, 4],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1],
        'min_child_weight': [1, 3, 5],
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [1, 1.5, 2],
    }

    auc_scorer = make_scorer(roc_auc_score, needs_proba=True) #response_method="predict_proba")

    random_search = RandomizedSearchCV(
        estimator=xgb_clf,
        param_distributions=param_dist,
        scoring=auc_scorer,
        n_iter=20,
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1,
    )
    random_search.fit(X_train_processed, y_train)

    print("Best parameters found: ", random_search.best_params_)
    print("Best CV AUC score: ", random_search.best_score_)

    best_model = random_search.best_estimator_

    train_auc = roc_auc_score(y_train, best_model.predict_proba(X_train_processed)[:, 1])
    test_auc = roc_auc_score(y_test, best_model.predict_proba(X_test_processed)[:, 1])
    oot_auc = roc_auc_score(y_oot, best_model.predict_proba(X_oot_processed)[:, 1])

    print("Train AUC:", train_auc, " Gini:", round(2*train_auc-1, 3))
    print("Test  AUC:", test_auc, " Gini:", round(2*test_auc-1, 3))
    print("OOT   AUC:", oot_auc, " Gini:", round(2*oot_auc-1, 3))

    # --- compute baseline prediction-score distribution for monitoring (PSI baseline) ---
    train_scores = best_model.predict_proba(X_train_processed)[:, 1]
    psi_bins = np.quantile(train_scores, np.linspace(0, 1, 11))   # 10 bins by training decile
    psi_bins[0], psi_bins[-1] = 0.0, 1.0
    train_bin_counts, _ = np.histogram(train_scores, bins=psi_bins)
    train_bin_pct = train_bin_counts / train_bin_counts.sum()

    # --- assemble model artefact ---
    model_artefact = {}
    model_artefact['model'] = best_model
    model_artefact['model_version'] = "credit_model_" + snapshotdate.replace('-', '_')
    model_artefact['preprocessing_pipeline'] = preprocessor
    model_artefact['feature_cols'] = FEATURE_COLS
    model_artefact['data_dates'] = config
    model_artefact['data_stats'] = {
        'X_train': X_train.shape[0], 'X_test': X_test.shape[0], 'X_oot': X_oot.shape[0],
        'y_train': round(float(y_train.mean()), 3),
        'y_test': round(float(y_test.mean()), 3),
        'y_oot': round(float(y_oot.mean()), 3),
    }
    model_artefact['results'] = {
        'auc_train': train_auc, 'auc_test': test_auc, 'auc_oot': oot_auc,
        'gini_train': round(2*train_auc-1, 3), 'gini_test': round(2*test_auc-1, 3), 'gini_oot': round(2*oot_auc-1, 3),
    }
    model_artefact['hp_params'] = random_search.best_params_
    model_artefact['monitoring_baseline'] = {
        'psi_bins': psi_bins.tolist(),
        'psi_bin_pct': train_bin_pct.tolist(),
        'score_mean': float(train_scores.mean()),
        'score_std': float(train_scores.std()),
        'default_rate': float(y_train.mean()),
    }

    pprint.pprint(model_artefact['results'])

    # --- save artefact to model bank ---
    model_bank_directory = "model_bank/"
    os.makedirs(model_bank_directory, exist_ok=True)
    file_path = os.path.join(model_bank_directory, model_artefact['model_version'] + '.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(model_artefact, f)
    print(f"Model saved to {file_path}")

    spark.stop()
    print('\n\n---completed job---\n\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run job")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    args = parser.parse_args()
    main(args.snapshotdate)
