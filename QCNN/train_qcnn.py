!pip install qiskit qiskit-machine-learning scikit-learn
!pip install qiskit-algorithms
!pip install pylatexenc
!pip install imbalanced-learn

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorSampler

from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier

from qiskit_algorithms.optimizers import COBYLA



# --- 1. Load Data ---
df = pd.read_csv("/kaggle/input/datasets/tuankietquach/qcnn-2k-lan2-5/2000_train_aic_remake4quantum_seed_0 (1).csv")
X = df.drop(columns=["Attack_label"]).values
y = df["Attack_label"].values

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)


from qiskit.circuit.library import ZZFeatureMap
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector # Import đúng chỗ cho Qiskit 1.x

n_qubits = 4
feature_map = ZZFeatureMap(feature_dimension=n_qubits, reps=2, entanglement='linear')
x = feature_map.parameters # Đây là 4 tham số đầu vào

# 2. Khởi tạo Ansatz (40 tham số trọng số)
theta = ParameterVector("θ", 60) 
ansatz = QuantumCircuit(n_qubits)

for i in range(3):
    base = i * 8
    # Cặp (0,1) và (2,3)
    ansatz.cx(0, 1); ansatz.ry(theta[base], 0); ansatz.ry(theta[base+1], 1)
    ansatz.cx(2, 3); ansatz.ry(theta[base+2], 2); ansatz.ry(theta[base+3], 3)
    # Cặp lệch (1,2) và (3,0) để tạo vướng víu chéo
    ansatz.cx(1, 2); ansatz.ry(theta[base+4], 1); ansatz.ry(theta[base+5], 2)
    ansatz.cx(3, 0); ansatz.ry(theta[base+6], 3); ansatz.ry(theta[base+7], 0)
    ansatz.barrier()

# --- POOLING 1: Nén về Qubit 0 và 2 (θ 24-27) ---
ansatz.cx(1, 0); ansatz.rz(theta[24], 0); ansatz.ry(theta[25], 0)
ansatz.cx(3, 2); ansatz.rz(theta[26], 2); ansatz.ry(theta[27], 2)
ansatz.barrier()

# --- STAGE 2: 6 Lớp Conv sâu trên Qubit 0 và 2 (θ 28-51) ---
for i in range(6):
    base = 28 + i * 4
    ansatz.cx(0, 2); ansatz.ry(theta[base], 0); ansatz.ry(theta[base+1], 2)
    ansatz.cx(2, 0); ansatz.ry(theta[base+2], 0); ansatz.ry(theta[base+3], 2)
ansatz.barrier()

# --- POOLING 2: Nén về Qubit 0 (θ 52-55) ---
ansatz.cx(2, 0); ansatz.rz(theta[52], 0); ansatz.ry(theta[53], 0)
ansatz.rz(theta[54], 0); ansatz.ry(theta[55], 0) # Thêm cổng để tăng độ sâu nén

# --- FINAL STAGE: Phân loại cuối (θ 56-59) ---
ansatz.ry(theta[56], 0); ansatz.rz(theta[57], 0)
ansatz.ry(theta[58], 0); ansatz.rz(theta[59], 0)

# Kết hợp với ZZFeatureMap
qc = feature_map.compose(ansatz)
print(f"Tổng tham số mạch: {len(qc.parameters)}") # Sẽ là 64 (4 x + 60 θ)# 3. Kết hợp mạch

# Vẽ mạch để kiểm tra cấu trúc "ZZ + Conv + Pool"
qc.draw("mpl")

import numpy as np
import matplotlib.pyplot as plt
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
from qiskit_algorithms.optimizers import SPSA
from IPython.display import clear_output
import matplotlib.pyplot as plt

from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.primitives import StatevectorSampler

# --- BƯỚC QUAN TRỌNG: ĐỊNH NGHĨA QNN ---

# 1. Khởi tạo Sampler (Bộ thực thi mạch lượng tử)
sampler = StatevectorSampler()

# 2. Định nghĩa kiến trúc QNN
qnn = SamplerQNN(
    circuit=qc,              # Mạch lượng tử đã kết hợp ZZFeatureMap và Ansatz
    input_params=x,          # Các tham số x từ feature_map.parameters
    weight_params=theta,     # 12 tham số theta để học
    sampler=sampler,
    # interpret giúp ánh xạ kết quả đo từ mạch về nhãn 0 hoặc 1
    interpret=lambda x: (x & 1), 
    output_shape=2           # Số lượng lớp (2 lớp: Attack và Normal)
)

# --- SAU ĐÓ MỚI ĐẾN PHẦN CLASSIFIER CỦA BẠN ---

# Mảng lưu giá trị hàm mục tiêu (Loss)
objective_func_vals = []


def callback_graph(nfev, weights, obj_func_eval, stepsize, accepted):
    """
    Sửa lỗi TypeError bằng cách nhận đủ 5 tham số từ SPSA:
    1. nfev: Số lần đánh giá hàm
    2. weights: Trọng số hiện tại (theta)
    3. obj_func_eval: Giá trị Loss (quan trọng nhất để vẽ hình)
    4. stepsize: Kích thước bước nhảy của thuật toán
    5. accepted: Bước nhảy có được chấp nhận hay không
    """
    clear_output(wait=True)
    objective_func_vals.append(obj_func_eval)
    
    plt.figure(figsize=(10, 5))
    plt.title(f"Huấn luyện QCNN với SPSA - Bước thứ {len(objective_func_vals)}")
    plt.xlabel("Iteration")
    plt.ylabel("Loss (Hàm mục tiêu)")
    
    # Vẽ đường Loss
    plt.plot(range(len(objective_func_vals)), objective_func_vals, 
             color='tab:red', label='Current Loss', marker='o', markersize=2)
    
    # In giá trị Loss hiện tại lên biểu đồ để dễ theo dõi
    plt.text(len(objective_func_vals)-1, obj_func_eval, f"{obj_func_eval:.4f}")
    
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

# Khởi tạo Optimizer và Initial Point
# Khởi tạo Optimizer SPSA đúng chuẩn Qiskit
optimizer = SPSA(
    maxiter=200,           # Tăng thời gian để các tập dữ liệu "khó" kịp hội tụ
    blocking=True,         # Tuyệt đối giữ True để đường Loss không bị nhảy ngược
    resamplings=5,         # Tăng lên 5 để lọc nhiễu cực tốt cho tập 500 mẫu
    learning_rate=None,    # ĐỂ TỰ ĐỘNG: SPSA sẽ tự tính toán bước nhảy phù hợp cho từng Seed
    perturbation=None,     # ĐỂ TỰ ĐỘNG: SPSA sẽ tự rung lắc dựa trên độ nhiễu của dữ liệu
    last_avg=30            # Lấy trung bình 30 bước cuối để triệt tiêu biến động ngẫu nhiên
)
# --- CẤU HÌNH COLD START ---
seed_value = 42
np.random.seed(seed_value)

# Xác định số lượng tham số cần khởi tạo
num_weights = qnn.num_weights 

# Khởi tạo cực nhỏ (Cold Start) để giảm thiểu sự khác biệt giữa các Seed
initial_point = np.random.uniform(-0.01, 0.01, num_weights)

print(f"Khởi tạo Cold Start cho {num_weights} tham số thành công.")
print(f"Ví dụ 3 tham số đầu tiên: {initial_point[:3]}")


# Khởi tạo Classifier với qnn đã định nghĩa ở trên
classifier = NeuralNetworkClassifier(
    neural_network=qnn,
    optimizer=optimizer,
    initial_point=initial_point,
    callback=callback_graph
)

# Reset mảng và bắt đầu huấn luyện
objective_func_vals = []
print("Bắt đầu huấn luyện QCNN với ZZFeatureMap...")
classifier.fit(X_train, y_train)
print("Huấn luyện hoàn tất!")



import dill
with open("qcnn_model.pkl", "wb") as f:
    dill.dump(classifier, f)

from sklearn.metrics import roc_auc_score

valid_probs = classifier.predict_proba(X_valid)[:,1]

valid_auc = roc_auc_score(y_valid, valid_probs)

print("VALID AUC:", valid_auc)

test_probs = classifier.predict_proba(X_test)[:,1]

test_auc = roc_auc_score(y_test, test_probs)

print("TEST AUC:", test_auc)

print("mean:", test_probs.mean())
print("std:", test_probs.std())
print("min:", test_probs.min())
print("max:", test_probs.max())


