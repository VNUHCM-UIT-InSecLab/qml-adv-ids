import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# ==========================================
# 1. CẤU HÌNH & LOAD DỮ LIỆU
# ==========================================
DATA_PATH = "/kaggle/input/edge-2k5-x2-train-tong-test/2k5_test_edge.csv" # <--- Sửa nếu cần
EXPORT_SAMPLE_SIZE = 516281
FILE_RAW_NAME = "test-2k5-raw.csv"
FILE_QUANTUM_NAME = "500_test_aic_.csv"
MODEL_PREPROCESSOR_NAME = "test-2k5-preprocessors.joblib" # File lưu PCA/Scaler mới để dùng lại sau này

print(f"--- BẮT ĐẦU QUY TRÌNH (Size: {EXPORT_SAMPLE_SIZE}) ---")

# Load dữ liệu gốc
if 'df' not in globals() or 'X' not in globals():
    print("📥 Đang đọc dữ liệu gốc...")
    try:
        df = pd.read_csv(DATA_PATH, skipinitialspace=True)
    except:
        df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    X_full = df.drop(columns=['Attack_label']) 
    y_full = df['Attack_label']
else:
    print("✅ Sử dụng dữ liệu có sẵn trong RAM.")
    X_full = X
    y_full = y

# ==========================================
# 2. LẤY MẪU (SAMPLE)
# ==========================================
print(f"\n🔄 Đang lấy mẫu ngẫu nhiên {EXPORT_SAMPLE_SIZE} dòng...")
if len(X_full) > EXPORT_SAMPLE_SIZE:
    X_sample, _, y_sample, _ = train_test_split(
        X_full, y_full, train_size=EXPORT_SAMPLE_SIZE, stratify=y_full, random_state=42
    )
else:
    X_sample, y_sample = X_full, y_full

# ==========================================
# 3. ÁP DỤNG CẤU HÌNH PCA & SCALER (THEO YÊU CẦU CỦA BẠN)
# ==========================================
# print("\n⚙️ Đang áp dụng cấu hình PCA (n=4) và Scaler (0-1)...")

# # --- ĐOẠN CODE BẠN YÊU CẦU ĐƯỢC TÍCH HỢP TẠI ĐÂY ---
# # 1. Giảm chiều xuống 4 (để khớp với 4 Qubit theo bài báo)
# pca = PCA(n_components=4)
# # Vì đây là tạo dataset train mới, ta dùng fit_transform lên X_sample
# X_sample_pca = pca.fit_transform(X_sample)

# # 2. Chuẩn hóa về [0, 1] (Bắt buộc cho Rotation Encoding)
# scaler = MinMaxScaler(feature_range=(0, 1))
# X_sample_scaled = scaler.fit_transform(X_sample_pca)

# print(f"   -> Kích thước sau xử lý: {X_sample_scaled.shape}")
# # ---------------------------------------------------

log_features = ['FPR', 'FPLR', 'FTRLR','FLRR','FDiR']

for df_ in [X_sample]:
    df_[log_features] = np.log1p(df_[log_features])



# Load bundle
bundle = joblib.load("/kaggle/input/datasets/tuankietquach/qt-cnn-500/test-500-preprocessors (2).joblib")

pca = bundle["pca"]
scaler = bundle["scaler"]

# Chỉ transform
print("\n⚙️ Transform dữ liệu SAMPLE bằng PCA & Scaler đã train...")

X_sample_pca = pca.transform(X_sample)
X_sample_scaled = scaler.transform(X_sample_pca)

print(f"   -> Sample shape sau xử lý: {X_sample_scaled.shape}")

# ==========================================
# 4. XUẤT FILE 1: RAW (CHO CLASSICAL ML)
# ==========================================
df_export_raw = X_sample.copy()
df_export_raw['Attack_label'] = y_sample
print(f"\n💾 Đang lưu FILE 1 (Raw): {FILE_RAW_NAME}...")
df_export_raw.to_csv(FILE_RAW_NAME, index=False)

# ==========================================
# 5. XUẤT FILE 2: PROCESSED (CHO QUANTUM)
# ==========================================
# Tạo tên cột mới: PC1, PC2, PC3, PC4
cols_quantum = [f"Feature_{i+1}" for i in range(X_sample_scaled.shape[1])]
df_export_quantum = pd.DataFrame(X_sample_scaled, columns=cols_quantum)
df_export_quantum['Attack_label'] = y_sample.values

print(f"💾 Đang lưu FILE 2 (Processed): {FILE_QUANTUM_NAME}...")
df_export_quantum.to_csv(FILE_QUANTUM_NAME, index=False)

# ==========================================
# 6. LƯU LẠI PCA & SCALER (QUAN TRỌNG)
# ==========================================
# Lưu lại 2 cái này để sau này bạn load file model lên test thì dùng lại được hệ quy chiếu này
print(f"💾 Đang lưu PCA & Scaler vào '{MODEL_PREPROCESSOR_NAME}'...")
# joblib.dump({'pca': pca, 'scaler': scaler}, MODEL_PREPROCESSOR_NAME)

print("\n=== ✅ HOÀN TẤT ===")
print("Bạn đã có 3 file mới:")
print(f"1. Dataset thô:   {FILE_RAW_NAME}")
print(f"2. Dataset sạch:  {FILE_QUANTUM_NAME} (Dùng train Quantum ngay)")
print(f"3. Preprocessors: {MODEL_PREPROCESSOR_NAME} (Dùng để test dữ liệu mới sau này)")