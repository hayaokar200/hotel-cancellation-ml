from hotel_cancel import config
from hotel_cancel import data
from hotel_cancel import evaluate
from hotel_cancel import features
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import ClassifierMixin
import joblib
import json
from lightgbm import LGBMClassifier
from datetime import datetime

def build_model(preprocessor: ColumnTransformer, classifier: ClassifierMixin) -> Pipeline:
    """Combine the preprocessor and any classifier into one Pipeline.  (notebook cell 38)"""
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])
    return model


def main():
    # Step 1: Load the raw data
    df = data.load_raw(config.DATA_PATH)

    # Step 2: Clean the data
    df = data.drop_unused_columns(df)
    df = data.add_dates(df)

    # Step 3: Split the data into train, validation, and test sets
    train_df, val_df, test_df = data.temporal_split(df)

    # Step 4: Split the data into features and target
    X_train, y_train = data.split_xy(train_df)
    X_val, y_val = data.split_xy(val_df)
    X_test, y_test = data.split_xy(test_df)

    # Step 5: Build the preprocessor
    preprocessor = features.build_preprocessor(X_train)

    # Step 6: Build the model
    model_selection = {
        "rf":  (RandomForestClassifier, config.RF_PARAMS),
        "lgb": (LGBMClassifier, config.LGBM_PARAMS),
    }
    model_class, params = model_selection[config.MODEL_NAME]
    model = build_model(preprocessor, model_class(**params)  )

    # Step 7: Fit on train
    model.fit(X_train, y_train)

    # Step 8: Get validation probabilities and choose the best threshold
    y_prob_val = model.predict_proba(X_val)[:, 1]
    threshold = evaluate.get_best_threshold(y_val, y_prob_val)
    val_metrics = evaluate.get_eval_metrics(y_val, y_prob_val, threshold)

    report_data = {
        "model": config.MODEL_NAME,
        "threshold": threshold,
        "val_f1": val_metrics["f1"]
    }

    if(config.EVALUATE_TEST):
        # Step 9: Get test probabilities and evaluate ONCE
        y_prob_test = model.predict_proba(X_test)[:, 1]
        test_metrics = evaluate.get_eval_metrics(y_test, y_prob_test, threshold)
        report_data['test_metrics'] = test_metrics

    # Step 10: Save the model
    models_dir = config.MODEL_PATH.parent
    models_dir.mkdir(parents=True, exist_ok=True)

    model_artifact = {
        "model": model,
        "threshold": threshold,
        "features": X_train.columns.tolist(),
    }
    joblib.dump(model_artifact, config.MODEL_PATH)

    # Step 11: Save the metrics report
    reports_dir = config.METRICS_PATH.parent
    reports_dir.mkdir(parents=True, exist_ok=True)

    with open(config.METRICS_PATH, "w") as f:
        json.dump(
            report_data,
            f,
            indent=2,
        )

    with open(config.EXPERIMENTS_PATH, "a", encoding="utf-8") as f:
        f.write(f"\n## Experiment - {datetime.now():%Y-%m-%d %H:%M}\n")
        f.write(f"- **Model:** {config.MODEL_NAME}\n")
        f.write(f"- **Parameters:** `{params}`\n")
        f.write(f"- **Threshold:** `{threshold}`\n")
        for metric, value in val_metrics.items():
            f.write(f"- **{metric}:** {value:.4f}\n")

        f.write("\n---\n")


if __name__ == "__main__":
    main()
