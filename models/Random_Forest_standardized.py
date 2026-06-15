"""
Random Forest — Poker Hand Prediction (Standardized)
=====================================================
Pipeline chuẩn hóa cho Random Forest Classifier:
  - Mode 1: Default Random Forest
  - Mode 2: Tuned Random Forest
  - Train + Evaluate trên RAW data
  - Train + Evaluate trên FE data
  - Metrics: Accuracy, Classification Report, Confusion Matrix
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from utils import load_raw_data, load_fe_data, evaluate_model


def run_random_forest(X_train, y_train, X_test, y_test, dataset_type="RAW"):
    """Train + Evaluate Random Forest (default + tuned) trên 1 loại data."""

    # ---- Mode 1: Default Random Forest ----
    print(f"\n[{dataset_type}] Đang huấn luyện Default Random Forest...")
    rf_default = RandomForestClassifier(random_state=42, n_jobs=-1)
    rf_default.fit(X_train, y_train)
    y_pred_default = rf_default.predict(X_test)
    print(f"[{dataset_type}] Hoàn thành!")
    evaluate_model("Random Forest (Default)", y_test, y_pred_default, dataset_type)

    # Feature importance
    import pandas as pd
    importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance (%)': rf_default.feature_importances_ * 100
    }).sort_values(by='Importance (%)', ascending=False)
    print(f"\n[{dataset_type}] Feature Importance (Default):")
    print(importance.to_string(index=False))

    # ---- Mode 2: Tuned Random Forest (GridSearchCV) ----
    from sklearn.model_selection import GridSearchCV
    print(f"\n[{dataset_type}] Đang chạy GridSearchCV cho Random Forest (3 folds)...")
    
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    param_grid = {
        'n_estimators': [100, 300],
        'max_depth': [10, 20, None],
        'min_samples_leaf': [1, 2]
    }
    
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"[{dataset_type}] Best parameters: {grid_search.best_params_}")
    print(f"[{dataset_type}] Best CV accuracy: {grid_search.best_score_*100:.2f}%")

    y_pred_tuned = grid_search.best_estimator_.predict(X_test)
    print(f"[{dataset_type}] Hoàn thành!")
    evaluate_model("Random Forest (Tuned)", y_test, y_pred_tuned, dataset_type)


def main():
    print("=" * 60)
    print("  Random Forest — Poker Hand Prediction")
    print("=" * 60)

    # --- RAW Data ---
    print("\n>>> Loading RAW data...")
    X_train, y_train, X_test, y_test = load_raw_data()
    run_random_forest(X_train, y_train, X_test, y_test, dataset_type="RAW")

    # --- FE Data ---
    print("\n>>> Loading FE data...")
    X_train, y_train, X_test, y_test = load_fe_data()
    run_random_forest(X_train, y_train, X_test, y_test, dataset_type="FE")


if __name__ == "__main__":
    main()
