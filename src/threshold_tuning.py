# src/threshold_tuning.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_auc_score, classification_report
from preprocess import load_and_prepare
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline

def optimize_threshold():
    preprocessor, X_train, X_test, y_train, y_test = load_and_prepare()

    # Train the same model as before
    model = Pipeline([
        ('prep', preprocessor),
        ('clf', GradientBoostingClassifier(random_state=42))
    ])
    model.fit(X_train, y_train)

    # Get predicted probabilities
    proba = model.predict_proba(X_test)[:, 1]

    # Compute precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, proba)
    f1_scores = 2 * (precision * recall) / (precision + recall)

    # Find the threshold that maximizes F1 score
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    print(f"Best threshold: {best_threshold:.3f}")
    print(f"Best F1 score: {best_f1:.3f}")
    print(f"Precision at best threshold: {precision[best_idx]:.3f}")
    print(f"Recall at best threshold: {recall[best_idx]:.3f}")

    # Plot Precision-Recall Curve
    plt.figure(figsize=(8,6))
    plt.plot(thresholds, precision[:-1], label='Precision')
    plt.plot(thresholds, recall[:-1], label='Recall')
    plt.plot(thresholds, f1_scores[:-1], label='F1 Score', linestyle='--')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision, Recall, and F1 Score vs Threshold')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Evaluate model using optimized threshold
    preds = (proba >= best_threshold).astype(int)
    print("\nClassification Report using optimized threshold:")
    print(classification_report(y_test, preds))

if __name__ == "__main__":
    optimize_threshold()
