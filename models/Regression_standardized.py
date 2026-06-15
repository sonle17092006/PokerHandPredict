"""
Linear Regression — Poker Hand Prediction (Standardized)
=========================================================
Pipeline chuẩn hóa cho Linear Regression:
  - Train + Evaluate trên RAW data
  - Train + Evaluate trên FE data
  - Metrics: MSE, Accuracy, Classification Report, Confusion Matrix

Note: Linear Regression output là giá trị liên tục, cần round + clip
      về khoảng [0, 9] để dùng làm classification.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from utils import load_raw_data, load_fe_data, evaluate_model


def run_regression(X_train, y_train, X_test, y_test, dataset_type="RAW"):
    """Train + Evaluate Ridge Regression trên 1 loại data."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GridSearchCV
    
    print(f"\n[{dataset_type}] Đang chạy GridSearchCV cho Ridge Regression (3 folds)...")
    
    model = Ridge(random_state=42)
    param_grid = {
        'alpha': [0.1, 1.0, 10.0],
        'fit_intercept': [True, False]
    }
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    print(f"[{dataset_type}] Best parameters: {grid_search.best_params_}")
    print(f"[{dataset_type}] Best CV MSE: {-grid_search.best_score_:.4f}")

    # Predict — regression output → round + clip to [0, 9]
    best_model = grid_search.best_estimator_
    y_pred_raw = best_model.predict(X_test)
    y_pred_rounded = np.clip(np.round(y_pred_raw), 0, 9).astype(int)

    # MSE (dùng giá trị raw, không round)
    mse = mean_squared_error(y_test, y_pred_raw)
    print(f"\n[{dataset_type}] Test Mean Squared Error (MSE): {mse:.4f}")

    # Evaluate chuẩn (dùng giá trị đã round)
    evaluate_model("Ridge Regression", y_test, y_pred_rounded, dataset_type)


def main():
    print("=" * 60)
    print("  Linear Regression — Poker Hand Prediction")
    print("=" * 60)

    # --- RAW Data ---
    print("\n>>> Loading RAW data...")
    X_train, y_train, X_test, y_test = load_raw_data()
    run_regression(X_train, y_train, X_test, y_test, dataset_type="RAW")

    # --- FE Data ---
    print("\n>>> Loading FE data...")
    X_train, y_train, X_test, y_test = load_fe_data()
    run_regression(X_train, y_train, X_test, y_test, dataset_type="FE")


if __name__ == "__main__":
    main()
