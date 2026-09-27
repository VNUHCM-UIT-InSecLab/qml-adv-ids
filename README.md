# Adversarial Robustness of Quantum Learning-based Intrusion Detection Systems in IoT Networks

Official implementation and experimental code for the paper:
> **"Adversarial Robustness of Quantum Learning-based Intrusion Detection Systems in IoT Networks"**  
> *IEEE ICPADS 2026*  
> Information Security Laboratory (InSecLab), VNU-HCM University of Information Technology.

---

## 📌 Overview

This repository provides the complete experimental framework comparing classical machine learning models (**CNN**, **SVC**) against Quantum Machine Learning (QML) models (**QCNN**, **QSVC**) for Intrusion Detection Systems (IDS) in IoT networks.

We comprehensively evaluate both:
1. **Clean Performance**: Baseline detection accuracy, AUC, Precision, Recall, and F1-score without attacks.
2. **Adversarial Robustness**: Vulnerability and robustness against state-of-the-art white-box and black-box evasion attacks (FGSM, ZOO, HopSkipJump, PGD) utilizing the Adversarial Robustness Toolbox (ART).

---

## 🗂️ Project Structure

```text
├── QCNN/                      # Quantum Convolutional Neural Network
│   ├── preprocess_data_qcnn.py   # Dimensionality reduction (PCA to 4 qubits) & MinMax scaling for train set
│   ├── preprocess_test_qcnn.py   # Test set preprocessing pipeline
│   ├── train_qcnn.py             # QCNN architecture design (ZZFeatureMap + Conv/Pooling) and training
│   └── test_QCNN.py              # Clean evaluation metrics (Accuracy, AUC, F1, Precision, Recall)
│
├── QSVC/                      # Quantum Support Vector Classifier
│   ├── precompute.py             # Module 1: Precomputes Quantum Kernel matrices with fault-tolerance & multiprocessing
│   ├── train.py                  # Module 2: Fast QSVC training using precomputed kernels (.dill)
│   └── test.py                   # Module 3: Evaluation on precomputed test kernels (ROC, Confusion Matrix)
│
├── CNN/                       # Classical 1D-CNN Baseline
│   ├── train_cnn.py              # 1D-CNN model architecture, log feature transformation, and training
│   └── test_CNN.py               # Evaluation on test dataset (Classification Report, ROC-AUC)
│
├── SVC/                       # Classical Linear SVC Baseline
│   ├── train_svc.py              # LinearSVC model training with balanced class weights
│   └── test_SVC.py               # Baseline testing and evaluation
│
├── ADVERSARIAL_ATTACK/        # Adversarial Attacks & Robustness Evaluation (ART)
│   ├── art_qcnn.py               # Evasion attacks on QCNN (FGSM, PGD, HopSkipJump, ZOO)
│   ├── art_for_qcnn.py           # ART wrapper for quantum neural network classifiers
│   ├── fgsm-parallel-loky.py     # Fast Gradient Sign Method on QSVC (Support Vector filtering & multi-core)
│   ├── run-art-qsvc-zoo-sv.py    # Zeroth-Order Optimization (ZOO) black-box attack on QSVC
│   ├── hsj-new.py                # HopSkipJump decision-based black-box attack on QSVC
│   └── eval-hsj.py               # Aggregation & evaluation script for HopSkipJump on Slurm HPC
│
├── LICENSE                    # Apache-2.0 License
├── .gitignore
└── README.md
```

---

## ⚙️ Requirements & Installation

Recommended Python version: `>= 3.9`

```bash
pip install numpy pandas scikit-learn matplotlib dill joblib
pip install qiskit qiskit-machine-learning qiskit-algorithms
pip install adversarial-robustness-toolbox tensorflow
```

---

## 🚀 Workflow

### 1. Data Preprocessing
Preprocess and extract principal features for quantum circuits (e.g., 4-qubit encoding):
```bash
python QCNN/preprocess_data_qcnn.py
python QCNN/preprocess_test_qcnn.py
```

### 2. Classical Models (CNN & SVC)
- Train & evaluate 1D-CNN:
  ```bash
  python CNN/train_cnn.py
  python CNN/test_CNN.py
  ```
- Train & evaluate Linear SVC:
  ```bash
  python SVC/train_svc.py
  python SVC/test_SVC.py
  ```

### 3. Quantum Models (QCNN & QSVC)
- **QCNN**:
  ```bash
  python QCNN/train_qcnn.py
  python QCNN/test_QCNN.py
  ```
- **QSVC** (3-stage pipeline):
  ```bash
  # Step 1: Precompute quantum kernel
  python QSVC/precompute.py --dataset_train <train.csv> --version_name v1

  # Step 2: Train QSVC
  python QSVC/train.py --dataset_train <train.csv> --version_name v1

  # Step 3: Evaluate QSVC
  python QSVC/test.py --dataset_test <test.csv> --version_name v1
  ```

### 4. Adversarial Robustness Assessment
Evaluate model resilience under various adversarial threat models:
```bash
# Evasion attacks on QCNN
python ADVERSARIAL_ATTACK/art_qcnn.py

# FGSM / ZOO / HopSkipJump attacks on QSVC
python ADVERSARIAL_ATTACK/fgsm-parallel-loky.py
python ADVERSARIAL_ATTACK/run-art-qsvc-zoo-sv.py
python ADVERSARIAL_ATTACK/hsj-new.py
```

---

## 📄 License
This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
