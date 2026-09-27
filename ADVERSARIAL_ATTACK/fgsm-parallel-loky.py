# -*- coding: utf-8 -*-
"""
======================================================================
DU AN: IDS LUONG TU - MODULE 4: ADVERSARIAL ROBUSTNESS (ART)
FILE: code/checkpoint-test/run-art-qsvc-fgsm-parallel.py
MO TA:
  - Nap mo hinh QSVC (.dill).
  - TOI UU HOA: Chi trich xuat va tinh toan tren Support Vectors.
  - SONG SONG HOA (PARALLEL): Dung joblib de chia tai qua cac CPU Cores.
======================================================================
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import gc
import dill
import logging
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score
from sklearn.model_selection import train_test_split

# Va loi module Qiskit cho moi truong HPC
import qiskit
if not hasattr(qiskit, 'circuit'):
    import qiskit.circuit
sys.modules['qiskit.circuit.quantumregister'] = sys.modules.get('qiskit.circuit')

from art.attacks.evasion import FastGradientMethod
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.estimators.classification import ClassifierMixin

from qiskit.circuit.library import ZZFeatureMap
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.kernels import FidelityQuantumKernel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Tu dong doc so luong Core CPU tu Slurm de Joblib biet cach chia viec
NUM_CORES = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))

# ==========================================================
# CUSTOM ART CLASSIFIER (PARALLELIZED & SV FILTERED)
# ==========================================================
class QSVC_ART_Classifier(BaseEstimator, ClassifierMixin, LossGradientsMixin):
    def __init__(self, model, q_kernel, X_train, nb_features, clip_values=(0, 1)): 
        super().__init__(model=model, clip_values=clip_values)
        self._model = model
        self.q_kernel = q_kernel  
        self._nb_classes = 2
        self._input_shape = (nb_features,)
        self.n_jobs = NUM_CORES

        # TRICH XUAT SUPPORT VECTORS VA TRONG SO
        model_inner = self._model.steps[-1][1] if hasattr(self._model, 'steps') else self._model
        self.sv_indices = model_inner.support_
        self.X_sv = X_train[self.sv_indices]
        self.dual_coef = model_inner.dual_coef_ 
        self.intercept = model_inner.intercept_
        
        logging.info(f"[*] Da loc du lieu: -> {len(self.X_sv)} Support Vectors.")
        logging.info(f"[*] He thong se chay song song tren {self.n_jobs} CPU Cores.")

    @property
    def nb_classes(self): return self._nb_classes

    @property
    def input_shape(self): return self._input_shape

    def fit(self, x, y, **kwargs):
        pass

    def predict(self, x, **kwargs):
        """Tinh toan luong tu song song bang cach chia nho mẻ du lieu (chunks)"""
        # Chia mang x thanh cac phan nho (chunks) vua voi so luong CPU
        n_chunks = min(len(x), self.n_jobs)
        chunks = np.array_split(x, n_chunks) if n_chunks > 0 else []

        def _evaluate_chunk(x_chunk):
            if len(x_chunk) == 0:
                return np.empty((0, self.X_sv.shape[0]))
            return self.q_kernel.evaluate(x_vec=x_chunk, y_vec=self.X_sv)

        # Chay song song (threading an toan cho Qiskit C++)
        if n_chunks > 1:
            kernel_matrices = Parallel(n_jobs=n_chunks, backend="loky")(
                delayed(_evaluate_chunk)(chunk) for chunk in chunks
            )
            kernel_matrix_adv = np.vstack(kernel_matrices)
        else:
            kernel_matrix_adv = self.q_kernel.evaluate(x_vec=x, y_vec=self.X_sv)
        
        decision = np.dot(kernel_matrix_adv, self.dual_coef.T) + self.intercept
        decision = decision.ravel()
        
        prob_1 = 1 / (1 + np.exp(-decision))
        prob_0 = 1 - prob_1
        return np.vstack([prob_0, prob_1]).T.astype(np.float32)

    def loss_gradient(self, x, y, **kwargs):
        """Tinh gradient dao ham sai phan SONG SONG cho cac dac trung"""
        eps = 1e-4
        y_predict = np.argmax(y, axis=1) if len(y.shape) > 1 else y.astype(int)
        n_features = x.shape[1]
        
        def _compute_feature_grad(i):
            x_p, x_m = x.copy(), x.copy()
            x_p[:, i] += eps
            x_m[:, i] -= eps
            p_p, p_m = self.predict(x_p), self.predict(x_m)
            l_p = -np.log(p_p[np.arange(len(x)), y_predict] + 1e-9)
            l_m = -np.log(p_m[np.arange(len(x)), y_predict] + 1e-9)
            return (l_p - l_m) / (2 * eps)
        
        # Ném tung đặc trưng vao tung Core de tinh toan cung luc
        grads_list = Parallel(n_jobs=min(self.n_jobs, n_features), backend="loky")(
            delayed(_compute_feature_grad)(i) for i in range(n_features)
        )
        
        grads = np.zeros_like(x)
        for i, grad_val in enumerate(grads_list):
            grads[:, i] = grad_val
            
        return grads.astype(np.float32)

# ==========================================================
# MAIN EXECUTION
# ==========================================================
def main():
    parser = argparse.ArgumentParser(description="Adversarial Robustness cho QSVC - FGSM Parallel")
    parser.add_argument('--dataset_train', type=str, required=True)
    parser.add_argument('--dataset_test', type=str, required=True)
    parser.add_argument('--version_name', type=str, required=True)
    parser.add_argument('--test_size', type=int, default=1000)
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    train_csv_path = os.path.join(base_dir, "data", "train", args.dataset_train)
    test_csv_path = os.path.join(base_dir, "data", "test", args.dataset_test)
    run_dir = os.path.join(base_dir, "runs", "checkpoint-test", args.version_name)
    model_path = os.path.join(run_dir, "model.dill")
    output_file = os.path.join(run_dir, "qsvc_fgsm_parallel_results.csv")

    os.makedirs(run_dir, exist_ok=True)

    log_file = os.path.join(run_dir, "art_eval_fgsm_parallel.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)

    logging.info("=== BAT DAU CHAY THU NGHIEM FGSM PARALLEL ===")
    
    with open(model_path, "rb") as f:
        qsvc_model = dill.load(f)

    df_train = pd.read_csv(train_csv_path)
    X_train_full = df_train.iloc[:, :-1].values.astype(np.float32)
    
    df_test = pd.read_csv(test_csv_path)
    y_test_full = df_test.iloc[:, -1].values
    X_test_full = df_test.iloc[:, :-1].values.astype(np.float32)
    nb_features = X_train_full.shape[1]

    feature_map = ZZFeatureMap(feature_dimension=nb_features, reps=2, entanglement='linear')
    sampler = StatevectorSampler()
    fidelity = ComputeUncompute(sampler=sampler)
    q_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)

    _, X_sub, _, y_sub = train_test_split(
        X_test_full, y_test_full, test_size=args.test_size, stratify=y_test_full, random_state=42
    )

    classifier_art = QSVC_ART_Classifier(
        model=qsvc_model, 
        q_kernel=q_kernel, 
        X_train=X_train_full, 
        nb_features=nb_features,
        clip_values=(0, 1) 
    )
    
    # 1. Baseline Evaluation
    logging.info("Dang danh gia baseline (Du lieu sach)...")
    preds_clean = classifier_art.predict(X_sub)
    y_pred_clean = np.argmax(preds_clean, axis=1)
    correct_indices = np.where(y_pred_clean == y_sub)[0]
    
    clean_acc = accuracy_score(y_sub, y_pred_clean)
    clean_recall = recall_score(y_sub, y_pred_clean)
    clean_auc = roc_auc_score(y_sub, preds_clean[:, 1])
    logging.info(f"[OK] Baseline - Acc: {clean_acc:.4f}, Recall: {clean_recall:.4f}, AUC: {clean_auc:.4f}")

    results_list = []

    # 2. White-box Attack (FGSM)
    logging.info("BAT DAU CHAY WHITE-BOX ATTACK (CHI FGSM)...")
    for eps in [0.01, 0.05, 0.1]:
        attack = FastGradientMethod(estimator=classifier_art, eps=eps, batch_size=32)
        
        logging.info(f" -> Dang chay FGSM (eps={eps})...")
        X_adv = attack.generate(x=X_sub)
        preds_adv = classifier_art.predict(X_adv)
        y_pred_adv = np.argmax(preds_adv, axis=1)
        
        asr = np.sum(y_pred_adv[correct_indices] != y_sub[correct_indices]) / len(correct_indices) if len(correct_indices) > 0 else 0
        
        robust_recall = recall_score(y_sub, y_pred_adv)
        recall_drop = clean_recall - robust_recall
        
        results_list.append({
            "Attack": "FGSM", "Epsilon": eps, 
            "Robust_Acc": accuracy_score(y_sub, y_pred_adv),
            "Robust_Recall": robust_recall,
            "Recall_Drop": recall_drop,
            "ASR": asr,
            "Robust_AUC": roc_auc_score(y_sub, preds_adv[:, 1])
        })
        logging.info(f"    [XONG] FGSM (eps={eps}) - ASR: {asr:.4f} | Recall Drop: {recall_drop:.4f}")
        del X_adv; gc.collect()

    df_res = pd.DataFrame(results_list)
    df_res.to_csv(output_file, index=False)
    logging.info(f"[OK] Hoan tat. Ket qua duoc luu tai: {output_file}")

if __name__ == "__main__":
    main()