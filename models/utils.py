"""
Poker Hand Prediction — Shared Utilities
=========================================
Module chứa các hàm dùng chung cho tất cả model scripts:
  - load_raw_data()   : Load dữ liệu thô (train.csv / test.csv)
  - load_fe_data()    : Load dữ liệu đã Feature Engineering (train_fe.csv / test_fe.csv)
  - evaluate_model()  : Đánh giá model với bộ metrics chuẩn
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Đường dẫn tương đối từ thư mục models/ tới data/
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

TARGET_COL = 'Class'


def load_raw_data():
    """
    Load dữ liệu thô (raw) từ ../data/train.csv và ../data/test.csv.

    Returns
    -------
    X_train, y_train, X_test, y_test : pandas objects
    """
    train_path = os.path.join(_DATA_DIR, 'train.csv')
    test_path = os.path.join(_DATA_DIR, 'test.csv')

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    X_train = df_train.drop(TARGET_COL, axis=1)
    y_train = df_train[TARGET_COL]

    X_test = df_test.drop(TARGET_COL, axis=1)
    y_test = df_test[TARGET_COL]

    print(f"[RAW] Train shape: {df_train.shape}, Test shape: {df_test.shape}")
    return X_train, y_train, X_test, y_test


def load_fe_data():
    """
    Load dữ liệu đã Feature Engineering từ ../data/train_fe.csv và ../data/test_fe.csv.

    Returns
    -------
    X_train, y_train, X_test, y_test : pandas objects
    """
    train_path = os.path.join(_DATA_DIR, 'train_fe.csv')
    test_path = os.path.join(_DATA_DIR, 'test_fe.csv')

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    X_train = df_train.drop(TARGET_COL, axis=1)
    y_train = df_train[TARGET_COL]

    X_test = df_test.drop(TARGET_COL, axis=1)
    y_test = df_test[TARGET_COL]

    print(f"[FE]  Train shape: {df_train.shape}, Test shape: {df_test.shape}")
    return X_train, y_train, X_test, y_test


def evaluate_model(model_name, y_true, y_pred, dataset_type="RAW"):
    """
    In kết quả benchmark chuẩn cho 1 lần đánh giá:
      - Accuracy
      - Classification Report (precision, recall, f1-score)
      - Confusion Matrix (heatmap)

    Parameters
    ----------
    model_name : str
        Tên model (dùng cho tiêu đề).
    y_true : array-like
        Nhãn thật.
    y_pred : array-like
        Nhãn dự đoán.
    dataset_type : str
        Loại dữ liệu ("RAW" hoặc "FE").
    """
    title = f"{model_name} — {dataset_type} Data"
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

    acc = accuracy_score(y_true, y_pred)
    incorrect_count = (y_true != y_pred).sum()
    total_count = len(y_true)
    print(f"\nAccuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"Incorrect Predictions: {incorrect_count} / {total_count}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    # Confusion Matrix — heatmap
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(f'Confusion Matrix: {title}')
    plt.tight_layout()
    
    # Save the screenshot
    screenshots_dir = os.path.join(_DATA_DIR, '..', 'screenshots of benchmark')
    os.makedirs(screenshots_dir, exist_ok=True)
    filename = f"{model_name.replace(' ', '_')}_{dataset_type}_confusion_matrix.png"
    filepath = os.path.join(screenshots_dir, filename)
    plt.savefig(filepath)
    print(f"\n[INFO] Saved confusion matrix to {filepath}")
    
    # Close plot to avoid displaying and consuming memory
    plt.close(fig)

    return acc
