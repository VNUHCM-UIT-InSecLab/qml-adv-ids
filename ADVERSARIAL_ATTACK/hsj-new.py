# -*- coding: utf-8 -*-
import os
import sys
import argparse
import warnings

# BỘ LỌC CẢNH BÁO
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import dill
import time
import logging
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split

import qiskit
if not hasattr(qiskit, 'circuit'):
    import qiskit.circuit
sys.modules['qiskit.circuit.quantumregister'] = sys.modules.get('qiskit.circuit')

import pandas as pd # Chi dung de doc file csv ban dau
from art.attacks.evasion import HopSkipJump
from art.estimators.classification import ClassifierMixin
from art.estimators.estimator import BaseEstimator
from qiskit.circuit.library import ZZFeatureMap
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.kernels import FidelityQuantumKernel

# [SLURM] Lấy số Core trực tiếp từ cấu hình của Slurm
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

    def fit(self, x, y, **kwargs):
        pass

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
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--end_idx', type=int, default=-1)
    parser.add_argument('--output_npy', type=str, required=True) # File dich cuoi cung
    parser.add_argument('--log_file', type=str, required=True)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(args.log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    model_path = os.path.join(base_dir, "runs", "checkpoint-test", args.version_name, "model.dill")
    
    logging.info(f"[*] He thong Slurm dang cap phat: {NUM_CORES} CPU Cores.")
    
    with open(model_path, "rb") as f: qsvc_model = dill.load(f)

    df_train = pd.read_csv(os.path.join(base_dir, "data", "train", args.dataset_train))
    X_train = df_train.iloc[:, :-1].values.astype(np.float32)
    df_test = pd.read_csv(os.path.join(base_dir, "data", "test", args.dataset_test))
    X_test_all = df_test.iloc[:, :-1].values.astype(np.float32)
    y_test_all = df_test.iloc[:, -1].values

    _, X_test_sub, _, y_test_sub = train_test_split(
        X_test_all, y_test_all, test_size=args.test_size, stratify=y_test_all, random_state=42
    )

    end = args.end_idx if args.end_idx != -1 else len(X_test_sub)
    X_part = X_test_sub[args.start_idx:end]
    
    logging.info(f"[*] NHIEM VU (PHASE 1): Tao nhieu tu index {args.start_idx} den {end} (Tong: {len(X_part)} mau)")

    q_kernel = FidelityQuantumKernel(
        fidelity=ComputeUncompute(sampler=StatevectorSampler(seed=42)),
        feature_map=ZZFeatureMap(X_train.shape[1], reps=2)
    )
    classifier = QSVC_ART_Classifier(qsvc_model, q_kernel, X_train, X_train.shape[1])

    attack = HopSkipJump(classifier=classifier, targeted=False, max_iter=10, max_eval=150)
    
    # KHOI PHUC CHECKPOINT HOAC TAO MOI
    if os.path.exists(args.output_npy):
        X_adv_list = [np.load(args.output_npy)]
        curr_start = len(X_adv_list[0])
        logging.info(f"[*] Phat hien file .npy cu, khoi phuc tu vi tri: {curr_start}")
    else:
        X_adv_list, curr_start = [], 0

    start_time = time.time()
    batch_size = 5
    
    # VONG LAP TAO NHIEU
    for i in range(curr_start, len(X_part), batch_size):
        batch = X_part[i:i+batch_size]
        X_adv_list.append(attack.generate(x=batch))
        
        # Ghi de ngay vao file output
        np.save(args.output_npy, np.vstack(X_adv_list))
        
        done = i + len(batch)
        elapsed = time.time() - start_time
        eta = (elapsed / (done - curr_start)) * (len(X_part) - done) if (done - curr_start) > 0 else 0
        logging.info(f"Progress: {done}/{len(X_part)} | ETA: {int(eta//3600)}h {int((eta%3600)//60)}m")

    logging.info(f"=== HOAN TAT PHASE 1 ===")
    logging.info(f"[*] Toan bo du lieu nhieu da duoc an toan luu tai: {args.output_npy}")

if __name__ == "__main__": main()