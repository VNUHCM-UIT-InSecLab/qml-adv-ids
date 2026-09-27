"""
======================================================================
DU AN: IDS LUONG TU - MODULE 4: ADVERSARIAL ROBUSTNESS (ART)
FILE: code/checkpoint-test/run-art-qsvc.py
MO TA:
  - Nap mo hinh QSVC (.dill).
  - TOI UU HOA: Chi trich xuat va tinh toan tren Support Vectors.
  - TAN CONG: Chi chay duy nhat thuat toan ZOO (Black-box).
  - Tich hop do luong Recall Drop.
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
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score
from sklearn.model_selection import train_test_split

# Va loi module Qiskit cho moi truong HPC
import qiskit
if not hasattr(qiskit, 'circuit'):
    import qiskit.circuit
sys.modules['qiskit.circuit.quantumregister'] = sys.modules.get('qiskit.circuit')

from art.attacks.evasion import ZooAttack
from art.estimators.estimator import BaseEstimator
from art.estimators.classification import ClassifierMixin

# Import Qiskit de tinh Kernel on-the-fly
from qiskit.circuit.library import ZZFeatureMap
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.kernels import FidelityQuantumKernel

# Cau hinh Logging mac dinh (Tieng Viet khong dau de tuong thich Slurm)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==========================================================
# CUSTOM ART CLASSIFIER (SUPPORT VECTOR FILTERED - BLACKBOX)
# ==========================================================
class QSVC_ART_Classifier(BaseEstimator, ClassifierMixin):
    # Da loai bo LossGradientsMixin vi ZOO khong can tinh dao ham truc tiep
    def __init__(self, model, q_kernel, X_train, nb_features, clip_values=(0, 1)): 
        super().__init__(model=model, clip_values=clip_values)
        self._model = model
        self.q_kernel = q_kernel  
        self._nb_classes = 2
        self._input_shape = (nb_features,)

        # TRICH XUAT SUPPORT VECTORS VA TRONG SO
        model_inner = self._model.steps[-1][1] if hasattr(self._model, 'steps') else self._model
        
        self.sv_indices = model_inner.support_
        self.X_sv = X_train[self.sv_indices]  # Chi giu lai cac Support Vectors
        
        # dual_coef_ chua gia tri (alpha_i * y_i) cua cac Support Vectors
        self.dual_coef = model_inner.dual_coef_ 
        self.intercept = model_inner.intercept_
        
        logging.info(f"[*] Da loc du lieu: Tu {len(X_train)} mau train -> {len(self.X_sv)} Support Vectors.")

    @property
    def nb_classes(self): return self._nb_classes

    @property
    def input_shape(self): return self._input_shape

    def fit(self, x, y, **kwargs):
        pass

    def predict(self, x, **kwargs):
        """Tinh toan ma tran luong tu CHI tren tap Support Vectors (Qiskit Engine)"""
        # Ma tran kernel_matrix_adv bay gio co kich thuoc (N_test x N_sv)
        kernel_matrix_adv = self.q_kernel.evaluate(x_vec=x, y_vec=self.X_sv)
        
        # Tu tinh ham quyet dinh (Decision Function) bang Dai so do sklearn khong ho tro ma tran rut gon
        decision = np.dot(kernel_matrix_adv, self.dual_coef.T) + self.intercept
        decision = decision.ravel()
        
        # Chuyen doi sang xac suat cho thu vien ART
        prob_1 = 1 / (1 + np.exp(-decision))
        prob_0 = 1 - prob_1
        return np.vstack([prob_0, prob_1]).T.astype(np.float32)

# ==========================================================
# MAIN EXECUTION
# ==========================================================
def main():
    parser = argparse.ArgumentParser(description="Adversarial Robustness cho QSVC - Chi ZOO Attack")
    parser.add_argument('--dataset_train', type=str, required=True, help="Ten file train CSV")
    parser.add_argument('--dataset_test', type=str, required=True, help="Ten file test CSV")
    parser.add_argument('--version_name', type=str, required=True, help="Ten thu muc phien ban (chua file model.dill)")
    parser.add_argument('--test_size', type=int, default=100, help="So luong mau de chay tan cong (Khuyen nghi: 100 cho ZOO Qiskit)")
    args = parser.parse_args()

    # Thiet lap duong dan
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    train_csv_path = os.path.join(base_dir, "data", "train", args.dataset_train)
    test_csv_path = os.path.join(base_dir, "data", "test", args.dataset_test)
    run_dir = os.path.join(base_dir, "runs", "checkpoint-test", args.version_name)
    model_path = os.path.join(run_dir, "model.dill")
    output_file = os.path.join(run_dir, "qsvc_zoo_only_results.csv")

    os.makedirs(run_dir, exist_ok=True)

    log_file = os.path.join(run_dir, "art_eval_zoo.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)

    logging.info("=== BAT DAU CHAY THU NGHIEM ZOO ATTACK TREN QISKIT ENGINE ===")
    
    logging.info(f"Dang nap mo hinh tu {model_path}")
    try:
        with open(model_path, "rb") as f:
            qsvc_model = dill.load(f)
    except FileNotFoundError:
        logging.error(f"Khong tim thay mo hinh tai: {model_path}")
        return

    logging.info("Dang nap du lieu train va test...")
    df_train = pd.read_csv(train_csv_path)
    X_train_full = df_train.iloc[:, :-1].values.astype(np.float32)
    
    df_test = pd.read_csv(test_csv_path)
    y_test_full = df_test.iloc[:, -1].values
    X_test_full = df_test.iloc[:, :-1].values.astype(np.float32)
    nb_features = X_train_full.shape[1]

    logging.info("Dang khoi tao moi truong Luong tu (Quantum Kernel)...")
    feature_map = ZZFeatureMap(feature_dimension=nb_features, reps=2, entanglement='linear')
    sampler = StatevectorSampler()
    fidelity = ComputeUncompute(sampler=sampler)
    q_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)

    # Voi ZOO chay tren Qiskit, nen giam test_size neu thoi gian chay qua lau tren HPC
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

    # 2. Black-box Attack (Chi tinh ZOO)
    logging.info("BAT DAU CHAY BLACK-BOX ATTACK (CHI ZOO)...")
    
    # Cau hinh ZOO: max_iter=30 la muc vuot qua du de danh gia ranh gioi ma khong bi timeout
    attack = ZooAttack(
        classifier=classifier_art,
        max_iter=30,             
        learning_rate=0.1,
        nb_parallel=1, 
        batch_size=1,
        targeted=False,
        use_importance=False
    )
    
    logging.info(" -> Dang chay ZOO Attack...")
    try:
        X_adv = attack.generate(x=X_sub)
        preds_adv = classifier_art.predict(X_adv)
        y_pred_adv = np.argmax(preds_adv, axis=1)
        
        asr = np.sum(y_pred_adv[correct_indices] != y_sub[correct_indices]) / len(correct_indices) if len(correct_indices) > 0 else 0
        
        # Tinh toan do suy giam Recall (Recall Drop)
        robust_recall = recall_score(y_sub, y_pred_adv)
        recall_drop = clean_recall - robust_recall
        
        results_list.append({
            "Attack": "ZOO", "Epsilon": "N/A", 
            "Robust_Acc": accuracy_score(y_sub, y_pred_adv),
            "Robust_Recall": robust_recall,
            "Recall_Drop": recall_drop,
            "ASR": asr,
            "Robust_AUC": roc_auc_score(y_sub, preds_adv[:, 1])
        })
        logging.info(f"    [XONG] ZOO - ASR: {asr:.4f} | Recall Drop: {recall_drop:.4f}")
    except Exception as e:
        logging.error(f" [!] Loi trong ZOO Attack: {e}")
    
    del X_adv; gc.collect()

    # 3. Save results
    df_res = pd.DataFrame(results_list)
    df_res.to_csv(output_file, index=False)
    logging.info(f"[OK] Hoan tat. Ket qua duoc luu tai: {output_file}")
    logging.info("=== HOAN TAT MODULE 4 ===")

if __name__ == "__main__":
    main()