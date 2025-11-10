# src/train.py
import os
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, classification_report, precision_recall_curve
from sklearn.pipeline import Pipeline
from preprocess import load_and_prepare

# Optional MLflow tracking
USE_MLFLOW = True
try:
    import mlflow
    import mlflow.sklearn
except Exception:
    USE_MLFLOW = False
    mlflow = None


def train(save_path="models"):
    """
    Train a Gradient Boosting model on the marketing campaign dataset.
    Logs metrics and optimized threshold to MLflow if available,
    and saves the model locally.
    """
    os.makedirs(save_path, exist_ok=True)

    # Load preprocessed data
    preprocessor, X_train, X_test, y_train, y_test = load_and_prepare()
    print(f"Data loaded successfully: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")

    # Define model pipeline
    clf = GradientBoostingClassifier(random_state=42)
    pipe = Pipeline([
        ("prep", preprocessor),
        ("clf", clf)
    ])

    # --- TRAINING WITH MLFLOW ---
    if USE_MLFLOW:
        mlflow.set_experiment("AdSmartAI_MarketingCampaign")

        with mlflow.start_run():
            print("Training model...")
            pipe.fit(X_train, y_train)

            proba = pipe.predict_proba(X_test)[:, 1]
            preds = (proba >= 0.5).astype(int)

            # Base metrics
            auc = roc_auc_score(y_test, proba)
            report = classification_report(y_test, preds, output_dict=True)

            print(f"\nAUC: {auc:.3f}")
            print("Classification Report:")
            print(classification_report(y_test, preds))

            # --- Threshold Optimization ---
            precision, recall, thresholds = precision_recall_curve(y_test, proba)
            f1_scores = 2 * (precision * recall) / (precision + recall)
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx]
            best_f1 = f1_scores[best_idx]

            print(f"\nOptimized threshold: {best_threshold:.3f}")
            print(f"Best F1 score: {best_f1:.3f}")
            print(f"Precision at best threshold: {precision[best_idx]:.3f}")
            print(f"Recall at best threshold: {recall[best_idx]:.3f}")

            # Log metrics and params
            mlflow.log_metric("AUC", float(auc))
            mlflow.log_metric("Precision", float(report["1"]["precision"]))
            mlflow.log_metric("Recall", float(report["1"]["recall"]))
            mlflow.log_metric("F1_Score", float(report["1"]["f1-score"]))
            mlflow.log_metric("best_f1", float(best_f1))
            mlflow.log_param("optimized_threshold", float(best_threshold))

            # Log model to MLflow
            mlflow.sklearn.log_model(pipe, "model")
            print("\nModel training complete and logged to MLflow.")

            # --- Save model locally for prediction ---
            import joblib
            local_model_path = os.path.join(save_path, "marketing_response_model.joblib")
            joblib.dump(pipe, local_model_path)
            print(f"Model also saved locally at: {local_model_path}")

            return auc, report, best_threshold

    # --- TRAINING WITHOUT MLFLOW ---
    else:
        print("Training model (MLflow not available)...")
        pipe.fit(X_train, y_train)

        proba = pipe.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)

        auc = roc_auc_score(y_test, proba)
        print(f"\nAUC: {auc:.3f}")
        print("Classification Report:")
        print(classification_report(y_test, preds))

        # Threshold optimization
        precision, recall, thresholds = precision_recall_curve(y_test, proba)
        f1_scores = 2 * (precision * recall) / (precision + recall)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]

        print(f"\nOptimized threshold: {best_threshold:.3f}")
        print(f"Best F1 score: {best_f1:.3f}")
        print(f"Precision at best threshold: {precision[best_idx]:.3f}")
        print(f"Recall at best threshold: {recall[best_idx]:.3f}")

        # Save model locally
        try:
            import joblib
            model_path = os.path.join(save_path, "marketing_response_model.joblib")
            joblib.dump(pipe, model_path)
            print(f"Model saved locally at: {model_path}")
        except Exception as e:
            print(f"Could not save model: {e}")

        return auc, None, best_threshold


if __name__ == "__main__":
    train()
