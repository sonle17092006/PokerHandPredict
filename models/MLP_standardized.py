"""
MLP — Poker Hand Prediction (Standardized)
===========================================
Pipeline chuẩn hóa cho MLP (Multi-Layer Perceptron):
  - Train + Evaluate trên RAW data
  - Train + Evaluate trên FE data
  - Metrics: Accuracy, Classification Report, Confusion Matrix
"""

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from utils import load_raw_data, load_fe_data, evaluate_model


def build_pipeline():
    """Tạo pipeline MLP chuẩn."""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(max_iter=2000, random_state=42))
    ])
    return pipeline


def run_mlp(X_train, y_train, X_test, y_test, dataset_type="RAW"):
    """Train + Evaluate MLP trên 1 loại data."""
    pipeline = build_pipeline()

    # GridSearchCV
    param_grid = {
        'mlp__hidden_layer_sizes': [(512, 256)],
        'mlp__alpha': [0.0001, 0.001],
        'mlp__learning_rate_init': [0.001, 0.01]
    }

    print(f"\n[{dataset_type}] Đang chạy GridSearchCV (5 folds)...")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)

    print(f"[{dataset_type}] Best parameters: {grid_search.best_params_}")
    print(f"[{dataset_type}] Best CV accuracy: {grid_search.best_score_*100:.2f}%")

    # Predict với best estimator
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Evaluate
    evaluate_model("MLP", y_test, y_pred, dataset_type)


def main():
    print("=" * 60)
    print("  MLP — Poker Hand Prediction")
    print("=" * 60)

    # --- RAW Data ---
    print("\n>>> Loading RAW data...")
    X_train, y_train, X_test, y_test = load_raw_data()
    run_mlp(X_train, y_train, X_test, y_test, dataset_type="RAW")

    # --- FE Data ---
    print("\n>>> Loading FE data...")
    X_train, y_train, X_test, y_test = load_fe_data()
    run_mlp(X_train, y_train, X_test, y_test, dataset_type="FE")


if __name__ == "__main__":
    main()
