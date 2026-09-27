"""
======================================================================
DU AN: IDS LUONG TU - MODULE 2: TRAIN QSVC
FILE: code/v2/train.py
MO TA:
  - Nap ma tran Kernel da duoc tinh toan san (precomputed) tu Module 1.
  - Huan luyen mo hinh QSVC (Sieu toc do vi khong can tinh lai luong tu).
  - Luu mo hinh duoi dang .dill bang ham to_dill().
======================================================================
"""

import os
import json
import logging
import argparse
import dill # Them vao dau file
import numpy as np
import pandas as pd
from qiskit_machine_learning.algorithms import QSVC

# Cau hinh Logging mac dinh (Tieng Viet khong dau de tuong thich Slurm)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Huan luyen QSVC voi Precomputed Kernel")
    parser.add_argument('--dataset_train', type=str, required=True, help="Ten file train CSV de lay nhan y_train")
    parser.add_argument('--version_name', type=str, required=True, help="Ten thu muc checkpoint da tao o Module 1")
    parser.add_argument('--c_param', type=float, default=1.0, help="Tham so C cua SVC")
    args = parser.parse_args()

    # Thiet lap cac duong dan tuong doi
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    train_csv_path = os.path.join(base_dir, "data", "train", args.dataset_train)
    checkpoint_dir = os.path.join(base_dir, "checkpoints", args.version_name)
    kernel_train_path = os.path.join(checkpoint_dir, "kernel_train.npy")
    
    # Thu muc luu ket qua chay (runs) duoc nhom theo version_name de de quan ly
    run_dir = os.path.join(base_dir, "runs", "checkpoint-test", args.version_name)
    os.makedirs(run_dir, exist_ok=True)
    
    # Ghi log file vao thu muc runs
    log_file = os.path.join(run_dir, "train.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)

    logging.info("=== BAT DAU MODULE 2: TRAIN QSVC ===")
    
    # 1. Kiem tra va nap file ma tran Kernel
    if not os.path.exists(kernel_train_path):
        logging.error(f"Khong tim thay {kernel_train_path}. Vui long chay precompute.py truoc!")
        return
        
    logging.info(f"Dang nap ma tran Kernel tu: {kernel_train_path}")
    kernel_train = np.load(kernel_train_path)
    
    # 2. Nap nhan y_train tu file CSV
    logging.info(f"Dang nap nhan y_train tu file: {train_csv_path}")
    try:
        df_train = pd.read_csv(train_csv_path)
        y_train = df_train.iloc[:, -1].values  # Cot cuoi cung mac dinh la nhan (label)
    except FileNotFoundError:
        logging.error(f"Khong tim thay file du lieu: {train_csv_path}")
        return
    
    # Kiem tra an toan: Kich thuoc ma tran co khop voi so luong nhan khong?
    if kernel_train.shape[0] != len(y_train) or kernel_train.shape[1] != len(y_train):
        logging.error(f"Loi kich thuoc! Ma tran {kernel_train.shape} khong khop voi so nhan ({len(y_train)}).")
        return

    # 3. Huan luyen mo hinh QSVC
    logging.info(f"Khoi tao QSVC (C={args.c_param}) va bypass loi Qiskit 0.8.2")
    
    # Buoc 1: Khoi tao QSVC mac dinh de tranh loi .evaluate cua Qiskit
    qsvc = QSVC(C=args.c_param, random_state = 42) 
    
    # Buoc 2: Ep truc tiep kernel cua class Scikit-learn cha thanh precomputed
    qsvc.kernel = "precomputed"
    
    logging.info("Dang thuc thi qsvc.fit()... Qua trinh nay se dien ra rat nhanh!")
    qsvc.fit(kernel_train, y_train)
    
    # 4. Luu cau hinh vao file JSON
    config = {
        "version_name": args.version_name,
        "dataset_train": args.dataset_train,
        "c_param_svc": args.c_param,
        "kernel_shape": kernel_train.shape
    }
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
        
    # 5. Luu mo hinh thanh file .dill (Su dung thu vien dill truc tiep)
    model_path = os.path.join(run_dir, "model.dill")
    logging.info(f"Dang luu mo hinh QSVC vao: {model_path}")
    
    # Thay vi qsvc.to_dill(model_path), ta dung:
    with open(model_path, 'wb') as f:
        dill.dump(qsvc, f)

    logging.info("=== HOAN TAT GIAI DOAN TRAIN ===")

if __name__ == "__main__":
    main()