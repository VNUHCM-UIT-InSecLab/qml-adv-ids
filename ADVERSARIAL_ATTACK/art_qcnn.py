# -*- coding: utf-8 -*-
# ==========================================================
# 1. INSTALL + IMPORT
# ==========================================================

import dill
import numpy as np
import pandas as pd
import gc
import os
from sklearn.metrics import recall_score, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

import art
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, HopSkipJump, ZooAttack
from art.estimators.estimator import BaseEstimator, LossGradientsMixin
from art.estimators.classification import ClassifierMixin

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZZFeatureMap

# ==========================================================
# 2. CONFIG & OUTPUT PATH
# ==========================================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
art.config.progress_bar = False

# ÄÆ¯á»œNG DáºªN OUTPUT KIá»†T YÃŠU Cáº¦U
OUTPUT_DIR = "/datastore/inseclab/quach_kiet/qcnn_ids/experiments/v1/art-result"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
# ==========================================================
# 3. ÐINH NGHIA MACH QCNN 
# ==========================================================
def create_qcnn_circuit(n_qubits=4):
    feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=2, entanglement='linear')
    theta = ParameterVector("θ", 60)
    ansatz = QuantumCircuit(n_qubits)

    # Convolution 1
    for i in range(3):
        base = i * 8
        ansatz.cx(0, 1); ansatz.ry(theta[base], 0); ansatz.ry(theta[base+1], 1)
        ansatz.cx(2, 3); ansatz.ry(theta[base+2], 2); ansatz.ry(theta[base+3], 3)
        ansatz.cx(1, 2); ansatz.ry(theta[base+4], 1); ansatz.ry(theta[base+5], 2)
        ansatz.cx(3, 0); ansatz.ry(theta[base+6], 3); ansatz.ry(theta[base+7], 0)
        ansatz.barrier()

    # Pooling 1
    ansatz.cx(1, 0); ansatz.rz(theta[24], 0); ansatz.ry(theta[25], 0)
    ansatz.cx(3, 2); ansatz.rz(theta[26], 2); ansatz.ry(theta[27], 2)
    ansatz.barrier()

    # Convolution 2
    for i in range(6):
        base = 28 + i * 4
        ansatz.cx(0, 2); ansatz.ry(theta[base], 0); ansatz.ry(theta[base+1], 2)
        ansatz.cx(2, 0); ansatz.ry(theta[base+2], 0); ansatz.ry(theta[base+3], 2)
    ansatz.barrier()

    # Pooling 2
    ansatz.cx(2, 0); ansatz.rz(theta[52], 0); ansatz.ry(theta[53], 0)
    ansatz.rz(theta[54], 0); ansatz.ry(theta[55], 0)
    ansatz.ry(theta[56], 0); ansatz.rz(theta[57], 0)
    ansatz.ry(theta[58], 0); ansatz.rz(theta[59], 0)

    return feature_map.compose(ansatz)

# ==========================================================
# 4. LOAD MODEL & FULL DATA
# ==========================================================
print(">>> Loading QCNN model & Full Test Data...")
QCNN_MODEL_PATH = "/datastore/inseclab/quach_kiet/qcnn_ids/experiments/v1/model/qcnn_hur_model_1k_s42.pkl"
DATA_PATH = "/datastore/inseclab/quach_kiet/qcnn_ids/data/test/1000_test_aic_ (1).csv"

df_full = pd.read_csv(DATA_PATH)
y_test_full = df_full['Attack_label'].values
X_test_full = df_full.drop(columns=['Attack_label']).values.astype(np.float32)

with open(QCNN_MODEL_PATH, "rb") as f:
    model_raw = dill.load(f)

# ==========================================================
# 5. CUSTOM ART CLASSIFIER (GIá»® NGUYÃŠN Cáº¤U HÃŒNH Cá»¦A KIá»†T)
# ==========================================================
class QiskitARTClassifier(BaseEstimator, ClassifierMixin, LossGradientsMixin):
    def __init__(self, model, clip_values=(0, np.pi)):
        super().__init__(model=model, clip_values=clip_values)
        self._model = model
        self._nb_classes = 2
        self._input_shape = (X_test_full.shape[1],) 

    @property
    def nb_classes(self): return self._nb_classes
    @property
    def input_shape(self): return self._input_shape
    def fit(self, x, y, **kwargs): pass
    def predict(self, x, **kwargs):
        return self._model.predict_proba(x).astype(np.float32)

    def loss_gradient(self, x, y, **kwargs):
        eps = 1e-4
        grads = np.zeros_like(x)
        y_true = np.argmax(y, axis=1) if len(y.shape) > 1 else y.astype(int)
        for i in range(x.shape[1]):
            x_p, x_m = x.copy(), x.copy()
            x_p[:, i] += eps; x_m[:, i] -= eps
            p_p, p_m = self.predict(x_p), self.predict(x_m)
            l_p = -np.log(p_p[np.arange(len(x)), y_true] + 1e-9)
            l_m = -np.log(p_m[np.arange(len(x)), y_true] + 1e-9)
            grads[:, i] = (l_p - l_m) / (2 * eps)
        return grads.astype(np.float32)

classifier = QiskitARTClassifier(model=model_raw)

# ==========================================================
# 6. Táº O Táº¬P SUBSET 1000 MáºªU
# ==========================================================
print("\n>>> Preparing Subset 1000...")
_, X_sub, _, y_sub = train_test_split(
    X_test_full, y_test_full, 
    test_size=1000, 
    stratify=y_test_full, 
    random_state=42
)

preds_clean_sub = classifier.predict(X_sub)
y_pred_clean_sub = np.argmax(preds_clean_sub, axis=1)
prob_clean_sub = preds_clean_sub[:, 1]

clean_acc_sub = accuracy_score(y_sub, y_pred_clean_sub)
clean_recall_sub = recall_score(y_sub, y_pred_clean_sub)
clean_auc_sub = roc_auc_score(y_sub, prob_clean_sub)

correct_indices_sub = np.where(y_pred_clean_sub == y_sub)[0]
results_list = []
EPSILONS = [0.01, 0.05, 0.1]

# ==========================================================
# 7. WHITE-BOX ATTACK (GIá»® NGUYÃŠN THAM Sá» PGD Cá»¦A KIá»†T)
# ==========================================================
for eps in EPSILONS:
    white_attacks = {
        "FGSM": FastGradientMethod(estimator=classifier, eps=eps, batch_size=32),
        "PGD": ProjectedGradientDescent(estimator=classifier, eps=eps, eps_step=eps/20, max_iter=50, num_random_init=2, batch_size=32)
    }

    for name, attack in white_attacks.items():
        X_adv = attack.generate(x=X_sub)
        preds_adv = classifier.predict(X_adv)
        y_pred_adv = np.argmax(preds_adv, axis=1)
        prob_adv = preds_adv[:, 1]

        robust_acc = accuracy_score(y_sub, y_pred_adv)
        robust_recall = recall_score(y_sub, y_pred_adv)
        robust_auc = roc_auc_score(y_sub, prob_adv)
        asr = np.sum(y_pred_adv[correct_indices_sub] != y_sub[correct_indices_sub]) / len(correct_indices_sub)

        results_list.append({
            "Attack": name, "Epsilon": eps,
            "Robust_Acc": robust_acc, "Robust_Recall": robust_recall,
            "Recall_Drop": clean_recall_sub - robust_recall, 
            "ASR": asr, "Robust_AUC": robust_auc
        })
        del X_adv; gc.collect()

# ==========================================================
# 8. BLACK-BOX ATTACK
# ==========================================================
black_attacks = {
    "ZOO": ZooAttack(classifier=classifier, max_iter=30, learning_rate=0.1, nb_parallel=1, use_resize=False, use_importance=False, batch_size=1),
    "HopSkipJump": HopSkipJump(classifier=classifier, targeted=False, max_iter=10, max_eval=150)
}

for name, attack in black_attacks.items():
    try:
        X_adv_sub = attack.generate(x=X_sub)
        preds_adv_sub = classifier.predict(X_adv_sub)
        y_pred_adv_sub = np.argmax(preds_adv_sub, axis=1)
        
        robust_acc_sub = accuracy_score(y_sub, y_pred_adv_sub)
        robust_recall_sub = recall_score(y_sub, y_pred_adv_sub)
        robust_auc_sub = roc_auc_score(y_sub, preds_adv_sub[:, 1])
        asr_sub = np.sum(y_pred_adv_sub[correct_indices_sub] != y_sub[correct_indices_sub]) / len(correct_indices_sub)

        results_list.append({
            "Attack": name, "Epsilon": "N/A",
            "Robust_Acc": robust_acc_sub, "Robust_Recall": robust_recall_sub,
            "Recall_Drop": clean_recall_sub - robust_recall_sub, 
            "ASR": asr_sub, "Robust_AUC": robust_auc_sub
        })
    except Exception as e:
        print(f" [ERROR] {name} failed: {e}")
    gc.collect()

# ==========================================================
# 9. RESULT DISPLAY & SAVE (LÆ¯U VÃ€O ÄÆ¯á»œNG DáºªN Má»šI)
# ==========================================================
df_res = pd.DataFrame(results_list)
df_res = df_res[["Attack", "Epsilon", "Robust_Acc", "Robust_Recall", "Recall_Drop", "ASR", "Robust_AUC"]]

print("\n" + "="*90)
print(df_res.to_string(index=False))

# LÆ¯U FILE
final_path = os.path.join(OUTPUT_DIR, "adversarial_results_qcnn_1k_seed42_hur.csv")
df_res.to_csv(final_path, index=False)
print(f"\n>>> File saved to: {final_path}")