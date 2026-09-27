"""
======================================================================
DU AN: IDS LUONG TU - MODULE 3: TEST & EVALUATION
FILE: code/v2/test.py
MO TA:
  - Nap mo hinh QSVC da huan luyen tu file .dill bang QSVC.from_dill().
  - Nap ma tran precomputed Kernel Test (N_test x N_train).
  - Du doan, danh gia hieu nang va xuat bieu do (CM, ROC).
======================================================================
"""

import os
import json
import logging
import argparse
import dill # Them vao dau file
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_auc_score, 
                             roc_curve, ConfusionMatrixDisplay)
from qiskit_machine_learning.algorithms import QSVC
from sklearn.svm import SVC
from qiskit_machine_learning.algorithms import QSVC

# Cau hinh Logging mac dinh (Tieng Viet khong dau)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Danh gia mo hinh QSVC voi Precomputed Kernel")
    parser.add_argument('--dataset_test', type=str, required=True, help="Ten file test CSV de lay nhan y_test")
    parser.add_argument('--version_name', type=str, required=True, help="Ten thu muc phien ban (giong luc chay train va precompute)")
    args = parser.parse_args()

    # Thiet lap cac duong dan
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    test_csv_path = os.path.join(base_dir, "data", "test", args.dataset_test)
    checkpoint_dir = os.path.join(base_dir, "checkpoints", args.version_name)
    kernel_test_path = os.path.join(checkpoint_dir, "kernel_test.npy")
    
    # Thu muc run (da duoc tao tu buoc train)
    run_dir = os.path.join(base_dir, "runs", "checkpoint-test", args.version_name)
    model_path = os.path.join(run_dir, "model.dill")
    
    # Ghi them log vao file test.log trong thu muc run
    log_file = os.path.join(run_dir, "test.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)

    logging.info("=== BAT DAU MODULE 3: TEST & EVALUATION ===")

    # 1. Kiem tra va nap mo hinh
    if not os.path.exists(model_path):
        logging.error(f"Khong tim thay mo hinh tai {model_path}. Vui long chay train.py truoc!")
        return
        
    logging.info(f"Dang khoi phuc mo hinh tu file: {model_path}")
    
    # Thay vi qsvc = QSVC.from_dill(model_path), ta dung:
    with open(model_path, 'rb') as f:
        qsvc = dill.load(f)

    # 2. Nap ma tran Kernel Test va Nhan thuc te
    if not os.path.exists(kernel_test_path):
        logging.error(f"Khong tim thay {kernel_test_path}. Vui long chay precompute.py!")
        return
        
    logging.info("Dang nap ma tran Kernel Test (N_test x N_train)...")
    kernel_test = np.load(kernel_test_path)
    
    logging.info(f"Dang nap nhan y_test tu file: {test_csv_path}")
    try:
        df_test = pd.read_csv(test_csv_path)
        y_test = df_test.iloc[:, -1].values
    except FileNotFoundError:
        logging.error(f"Khong tim thay file du lieu test: {test_csv_path}")
        return

    # Kiem tra kich thuoc ma tran voi so luong nhan Test
    if kernel_test.shape[0] != len(y_test):
        logging.error(f"Loi kich thuoc! So hang cua ma tran test ({kernel_test.shape[0]}) khong khop voi so nhan y_test ({len(y_test)}).")
        return

    # 3. Du doan (Predict) bang cach goi thang class cha Scikit-learn
    logging.info("Dang thuc thi predict() tren tap Test (Bypass Qiskit wrapper)...")
    y_pred = SVC.predict(qsvc, kernel_test)

    # 4. Tinh toan Metrics
    logging.info("Dang tinh toan cac chi so danh gia (Metrics)...")
    metrics = {
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1_Score": round(f1_score(y_test, y_pred, zero_division=0), 4)
    }
    
    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logging.info(f"Da luu chi so danh gia vao: {metrics_path}")

    # 5. Ve va luu Confusion Matrix
    logging.info("Dang ve va luu bieu do Confusion Matrix...")
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap=plt.cm.Blues)
    plt.title("Quantum SVC - Confusion Matrix")
    plt.savefig(os.path.join(run_dir, "confusion_matrix.png"))
    plt.close()

    # 6. Ve va luu ROC Curve
    logging.info("Dang ve va luu bieu do ROC Curve...")
    try:
        # Goi decision_function cua class cha Scikit-learn
        y_scores = SVC.decision_function(qsvc, kernel_test)
        auc_score = roc_auc_score(y_test, y_scores)
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Quantum SVC - Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(run_dir, "roc_curve.png"))
        plt.close()
    except AttributeError:
        logging.warning("Mo hinh QSVC hien tai khong ho tro decision_function de ve ROC.")

    logging.info("=== HOAN TAT GIAI DOAN TEST ===")

if __name__ == "__main__":
    main()