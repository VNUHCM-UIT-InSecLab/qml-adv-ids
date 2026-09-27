"""
======================================================================
DU AN: IDS LUONG TU - MODULE 1: PRECOMPUTE (LUONG TU HOA)
FILE: code/v2/precompute.py
MO TA: 
  - Tinh toan ma tran Kernel luong tu cho ca tap Train va Test.
  - Quan ly phien ban (Version Control) thu muc checkpoint.
  - Su dung co che Checkpoint de chong sap (bo qua block da tinh).
  - Tich hop da luong (Joblib) tan dung 100% CPU tren HPC.
======================================================================
"""

import os
import argparse
import logging
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Thu vien Qiskit
from qiskit.circuit.library import ZZFeatureMap
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit.primitives import StatevectorSampler 
from qiskit_machine_learning.kernels import FidelityQuantumKernel

# Cau hinh Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_quantum_kernel(feature_dim=4, reps=2, entanglement='linear', shots=1024):
    """Khoi tao Quantum Kernel voi StatevectorSampler."""
    feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=reps, entanglement=entanglement)
    sampler = StatevectorSampler(default_shots=shots)
    fidelity = ComputeUncompute(sampler=sampler)
    q_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=feature_map)
    return q_kernel

def compute_block(i, j, batch_x, batch_y, checkpoint_dir, prefix, q_kernel):
    """Tinh toan va luu mot block (sub-matrix) cua Kernel."""
    file_path = os.path.join(checkpoint_dir, f"{prefix}_block_{i}_{j}.npy")
    
    # Co che Fault Tolerance: Bo qua neu block da ton tai
    if os.path.exists(file_path):
        return file_path 
    
    logging.info(f"Dang tinh {prefix} block ({i}, {j})...")
    
    # Tinh ma tran tuong dong giua 2 batch
    block_matrix = q_kernel.evaluate(batch_x, batch_y)
    
    # Luu block xuong dia
    np.save(file_path, block_matrix)
    return file_path

def compute_and_assemble_train(X_train, q_kernel, version_dir, n_jobs, batch_size=200):
    """Tinh toan ma tran Train x Train (Khai thac tinh doi xung)."""
    n_samples = len(X_train)
    n_batches = int(np.ceil(n_samples / batch_size))
    batches = [X_train[i*batch_size : min((i+1)*batch_size, n_samples)] for i in range(n_batches)]
    
    logging.info(f"Bat dau tinh Train Kernel ({n_samples}x{n_samples}) - Chia lam {n_batches} batches.")
    
    # Giai doan 1: Chay song song tinh cac block (chi tinh tam giac tren)
    Parallel(n_jobs=n_jobs)(
        delayed(compute_block)(i, j, batches[i], batches[j], version_dir, "train", q_kernel)
        for i in range(n_batches) for j in range(i, n_batches)
    )
    
    # Giai doan 2: Lap rap cac block thanh ma tran hoan chinh
    logging.info("Dang lap rap cac blocks thanh Train Kernel hoan chinh...")
    kernel_matrix = np.zeros((n_samples, n_samples))
    
    for i in range(n_batches):
        for j in range(i, n_batches):
            file_path = os.path.join(version_dir, f"train_block_{i}_{j}.npy")
            block = np.load(file_path)
            
            row_start = i * batch_size
            row_end = row_start + block.shape[0]
            col_start = j * batch_size
            col_end = col_start + block.shape[1]
            
            # Dien vao tam giac tren
            kernel_matrix[row_start:row_end, col_start:col_end] = block
            
            # Sao chep doi xung xuong tam giac duoi
            if i != j:
                kernel_matrix[col_start:col_end, row_start:row_end] = block.T
                
    # Luu ma tran hoan chinh
    final_path = os.path.join(version_dir, "kernel_train.npy")
    np.save(final_path, kernel_matrix)
    logging.info(f"Da luu Train Kernel hoan chinh tai: {final_path}")
    return kernel_matrix

def compute_and_assemble_test(X_test, X_train, q_kernel, version_dir, n_jobs, batch_size=200):
    """Tinh toan ma tran Test x Train (Toan bo luoi chu nhat)."""
    n_test = len(X_test)
    n_train = len(X_train)
    
    n_batches_test = int(np.ceil(n_test / batch_size))
    n_batches_train = int(np.ceil(n_train / batch_size))
    
    batches_test = [X_test[i*batch_size : min((i+1)*batch_size, n_test)] for i in range(n_batches_test)]
    batches_train = [X_train[j*batch_size : min((j+1)*batch_size, n_train)] for j in range(n_batches_train)]
    
    logging.info(f"Bat dau tinh Test Kernel ({n_test}x{n_train}).")
    
    # Giai doan 1: Chay song song tinh toan bo luoi
    Parallel(n_jobs=n_jobs)(
        delayed(compute_block)(i, j, batches_test[i], batches_train[j], version_dir, "test", q_kernel)
        for i in range(n_batches_test) for j in range(n_batches_train)
    )
    
    # Giai doan 2: Lap rap cac block thanh ma tran hoan chinh
    logging.info("Dang lap rap cac blocks thanh Test Kernel hoan chinh...")
    kernel_matrix = np.zeros((n_test, n_train))
    
    for i in range(n_batches_test):
        for j in range(n_batches_train):
            file_path = os.path.join(version_dir, f"test_block_{i}_{j}.npy")
            block = np.load(file_path)
            
            row_start = i * batch_size
            row_end = row_start + block.shape[0]
            col_start = j * batch_size
            col_end = col_start + block.shape[1]
            
            kernel_matrix[row_start:row_end, col_start:col_end] = block
            
    # Luu ma tran hoan chinh
    final_path = os.path.join(version_dir, "kernel_test.npy")
    np.save(final_path, kernel_matrix)
    logging.info(f"Da luu Test Kernel hoan chinh tai: {final_path}")
    return kernel_matrix

def main():
    parser = argparse.ArgumentParser(description="Luong tu hoa du lieu & Tinh ma tran Kernel")
    parser.add_argument('--dataset_train', type=str, required=True, help="Ten file train CSV")
    parser.add_argument('--dataset_test', type=str, required=True, help="Ten file test CSV")
    parser.add_argument('--version_name', type=str, required=True, help="Ten thu muc phien ban checkpoint (vd: run_500_samples)")
    parser.add_argument('--batch_size', type=int, default=200, help="So samples moi block")
    parser.add_argument('--n_jobs', type=int, default=-1, help="So luong CPU (-1 la dung toi da)")
    args = parser.parse_args()

    # Thiet lap duong dan tuong doi
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    train_path = os.path.join(base_dir, "data", "train", args.dataset_train)
    test_path = os.path.join(base_dir, "data", "test", args.dataset_test)
    
    # Thu muc chua cac block checkpoint cua phien ban nay
    version_dir = os.path.join(base_dir, "checkpoints", args.version_name)
    os.makedirs(version_dir, exist_ok=True)

    # Nap du lieu
    logging.info(f"Dang nap du lieu Train tu: {train_path}")
    X_train = pd.read_csv(train_path).iloc[:, :-1].values
    
    logging.info(f"Dang nap du lieu Test tu: {test_path}")
    X_test = pd.read_csv(test_path).iloc[:, :-1].values

    # Khoi tao Kernel
    q_kernel = create_quantum_kernel()

    # Tinh toan
    compute_and_assemble_train(X_train, q_kernel, version_dir, args.n_jobs, args.batch_size)
    compute_and_assemble_test(X_test, X_train, q_kernel, version_dir, args.n_jobs, args.batch_size)
    
    logging.info("=== HOAN TAT MODULE LUONG TU HOA ===")

if __name__ == "__main__":
    main()