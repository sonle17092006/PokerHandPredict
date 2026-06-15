"""
AdaBoost — Poker Hand Prediction (Standardized)
================================================
Pipeline chuẩn hóa cho AdaBoost:
  - Train + Evaluate trên RAW data
  - Train + Evaluate trên FE data
  - Metrics: Accuracy, Classification Report, Confusion Matrix
"""

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score

from utils import load_raw_data, load_fe_data, evaluate_model


def build_pipeline():
    """Tạo pipeline AdaBoost chuẩn."""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=50,
            learning_rate=1.0,
            random_state=42
        ))
    ])
    return pipeline


def run_adaboost(X_train, y_train, X_test, y_test, dataset_type="RAW"):
    """Train + Evaluate AdaBoost trên 1 loại data."""
    pipeline = build_pipeline()

    # GridSearchCV
    param_grid = {
        'clf__n_estimators': [50, 100],
        'clf__learning_rate': [0.1, 1.0],
        'clf__estimator__max_depth': [1, 2]
    }

    print(f"\n[{dataset_type}] Đang chạy GridSearchCV (3 folds)...")
    from sklearn.model_selection import GridSearchCV
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=3,
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
    evaluate_model("AdaBoost", y_test, y_pred, dataset_type)


def main():
    print("=" * 60)
    print("  AdaBoost — Poker Hand Prediction")
    print("=" * 60)

    # --- RAW Data ---
    print("\n>>> Loading RAW data...")
    X_train, y_train, X_test, y_test = load_raw_data()
    run_adaboost(X_train, y_train, X_test, y_test, dataset_type="RAW")

    # --- FE Data ---
    print("\n>>> Loading FE data...")
    X_train, y_train, X_test, y_test = load_fe_data()
    run_adaboost(X_train, y_train, X_test, y_test, dataset_type="FE")


if __name__ == "__main__":
    main()
