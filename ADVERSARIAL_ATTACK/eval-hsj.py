# -*- coding: utf-8 -*-
"""
SCRIPT: GỘP VÀ ĐÁNH GIÁ CHUNG CUỘC KẾT QUẢ ADVERSARIAL (SLURM)
"""
import os
import sys
import argparse
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import pandas as pd
import dill
import time
import logging
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score
from sklearn.model_selection import train_test_split

import qiskit
if not hasattr(qiskit, 'circuit'):
    import qiskit.circuit
sys.modules['qiskit.circuit.quantumregister'] = sys.modules.get('qiskit.circuit')

from art.estimators.classification import ClassifierMixin
from art.estimators.estimator import BaseEstimator
from qiskit.circuit.library import ZZFeatureMap
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.kernels import FidelityQuantumKernel

# Lấy số Core trên Slurm để chấm điểm siêu tốc
NUM_CORES = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))

class QSVC_ART_Classifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model, q_kernel, X_train, nb_features, clip_values=(0, 1)):
        super().__init__(model=model, clip_values=clip_values)
        self._model = model
        self.q_kernel = q_kernel
        self._nb_classes = 2
        self._input_shape = (nb_features,)
        self.n_jobs = NUM_CORES

        model_inner = self._model.steps[-1][1] if hasattr(self._model, 'steps') else self._model
        self.sv_indices = model_inner.support_
        self.X_sv = X_train[self.sv_indices]
        self.dual_coef = model_inner.dual_coef_
        self.intercept = model_inner.intercept_

    @property
    def nb_classes(self): return self._nb_classes
    @property
    def input_shape(self): return self._input_shape

    def fit(self, x, y, **kwargs): pass

    def predict(self, x, **kwargs):
        n_chunks = min(len(x), self.n_jobs)
        chunks = np.array_split(x, n_chunks) if n_chunks > 0 else []
        def _evaluate_chunk(xc):
            return self.q_kernel.evaluate(x_vec=xc, y_vec=self.X_sv) if len(xc) > 0 else np.empty((0, len(self.X_sv)))

        if n_chunks > 1:
            results = Parallel(n_jobs=n_chunks, backend="loky")(delayed(_evaluate_chunk)(c) for c in chunks)
            kernel_matrix = np.vstack(results)
        else:
            kernel_matrix = _evaluate_chunk(x)
        
        decision = np.dot(kernel_matrix, self.dual_coef.T) + self.intercept
        prob_1 = 1 / (1 + np.exp(-decision.ravel()))
        return np.vstack([1 - prob_1, prob_1]).T.astype(np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_train', type=str, required=True)
    parser.add_argument('--dataset_test', type=str, required=True)
    parser.add_argument('--version_name', type=str, required=True)
    parser.add_argument('--test_size', type=int, default=1000)
    parser.add_argument('--npy_files', nargs='+', required=True, help='Danh sach cac file .npy can gop')
    parser.add_argument('--output_csv', type=str, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', stream=sys.stdout)
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    model_path = os.path.join(base_dir, "runs", "checkpoint-test", args.version_name, "model.dill")
    
    logging.info(f"[*] Dang tai mo hinh: {model_path}")
    with open(model_path, "rb") as f: qsvc_model = dill.load(f)

    # 1. Tải và xử lý dữ liệu (Cố định seed=42 để lấy đúng thứ tự Test)
    df_train = pd.read_csv(os.path.join(base_dir, "data", "train", args.dataset_train))
    X_train = df_train.iloc[:, :-1].values.astype(np.float32)
    
    df_test = pd.read_csv(os.path.join(base_dir, "data", "test", args.dataset_test))
    X_test_all = df_test.iloc[:, :-1].values.astype(np.float32)
    y_test_all = df_test.iloc[:, -1].values

    _, X_test_sub, _, y_test_sub = train_test_split(
        X_test_all, y_test_all, test_size=args.test_size, stratify=y_test_all, random_state=42
    )

    # 2. Gộp các file .npy theo đúng thứ tự truyền vào
    logging.info("[*] Dang tien hanh gop cac file Checkpoint...")
    X_adv_list = []
    for file in args.npy_files:
        if not os.path.exists(file):
            logging.error(f"[!] Khong tim thay file: {file}")
            sys.exit(1)
        part_data = np.load(file)
        X_adv_list.append(part_data)
        logging.info(f"    -> Load: {file} ({len(part_data)} mau)")
        
    X_adv_total = np.vstack(X_adv_list)
    total_adv_samples = len(X_adv_total)
    logging.info(f"[*] Tong so mau nhieu da gop: {total_adv_samples}")

    # (Linh hoạt: Nếu bạn chưa chạy đủ 1000 mẫu, code sẽ tự cắt test set cho vừa với số mẫu đã chạy)
    X_test_sub = X_test_sub[:total_adv_samples]
    y_test_sub = y_test_sub[:total_adv_samples]

    # Khởi tạo Classifier đa luồng
    q_kernel = FidelityQuantumKernel(
        fidelity=ComputeUncompute(sampler=StatevectorSampler(seed=42)),
        feature_map=ZZFeatureMap(X_train.shape[1], reps=2)
    )
    classifier = QSVC_ART_Classifier(qsvc_model, q_kernel, X_train, X_train.shape[1])

    # 3. CHẤM ĐIỂM
    logging.info(f"[*] Dang danh gia tren {total_adv_samples} mau SẠCH (Baseline)...")
    preds_clean = classifier.predict(X_test_sub)
    y_pred_clean = np.argmax(preds_clean, axis=1)
    correct_indices = np.where(y_pred_clean == y_test_sub)[0]
    
    clean_acc = accuracy_score(y_test_sub, y_pred_clean)
    clean_recall = recall_score(y_test_sub, y_pred_clean)

    logging.info(f"[*] Dang danh gia tren {total_adv_samples} mau NHIỄU (Adversarial)...")
    preds_adv = classifier.predict(X_adv_total)
    y_pred_adv = np.argmax(preds_adv, axis=1)
    
    robust_acc = accuracy_score(y_test_sub, y_pred_adv)
    robust_recall = recall_score(y_test_sub, y_pred_adv)
    robust_auc = roc_auc_score(y_test_sub, preds_adv[:, 1])
    
    # Tính các chỉ số Drop và ASR
    recall_drop = clean_recall - robust_recall
    asr = np.sum(y_pred_adv[correct_indices] != y_test_sub[correct_indices]) / len(correct_indices) if len(correct_indices) > 0 else 0

    # 4. LƯU KẾT QUẢ VÀO CSV
    res_dict = {
        "Attack": "HopSkipJump_Final",
        "Total_Samples": total_adv_samples,
        "Clean_Acc": clean_acc,
        "Clean_Recall": clean_recall,
        "Robust_Acc": robust_acc,
        "Robust_Recall": robust_recall,
        "Recall_Drop": recall_drop,
        "ASR": asr,
        "Robust_AUC": robust_auc
    }
    
    df_res = pd.DataFrame([res_dict])
    df_res.to_csv(args.output_csv, index=False)
    
    logging.info("=========================================")
    logging.info(f"KẾT QUẢ CHUNG CUỘC ({total_adv_samples} MẪU):")
    logging.info(f" Clean Acc    : {clean_acc:.4f}")
    logging.info(f" Robust Acc   : {robust_acc:.4f}")
    logging.info(f" Clean Recall : {clean_recall:.4f}")
    logging.info(f" Robust Recall: {robust_recall:.4f}")
    logging.info(f" Recall Drop  : {recall_drop:.4f}")
    logging.info(f" Final ASR    : {asr:.4f}")
    logging.info(f" Robust AUC   : {robust_auc:.4f}")
    logging.info("=========================================")
    logging.info(f"[*] Báo cáo đã được lưu tại: {args.output_csv}")

if __name__ == "__main__": main()