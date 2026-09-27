import pandas as pd
import numpy as np
df = pd.read_csv("test_path")

features = [  'FDiR', 'FPR', 'FDuR',
    'FPRR', 'FITR', 'FPLR', 'FLRR', 'FTRLR' ]

import joblib

scaler = joblib.load("scaler_of_model_cnn_path")

from sklearn.preprocessing import StandardScaler,RobustScaler
import joblib
X = df[features]
y = df["Attack_label"]

log_features = ['FPR', 'FPLR', 'FTRLR','FLRR','FDiR']

for df_ in [X]:
    df_[log_features] = np.log1p(df_[log_features])

X_scaled = scaler.transform(X)

import tensorflow as tf
import numpy as np

# Load model Keras
model = tf.keras.models.load_model(
    "model_cnn_path"
)

y_prob = model.predict(X_scaled, batch_size=1024).ravel()

from sklearn.metrics import confusion_matrix, classification_report

# Giả sử bạn đã có mô hình MLP được huấn luyện là 'model_mlp'
# và tập dữ liệu kiểm tra là 'X_test' (dạng numpy array)


# --- BƯỚC 2: PHÂN NGƯỠNG HÓA (Đưa về dạng 0 hoặc 1) ---
# Đặt ngưỡng 0.5: Nếu xác suất >= 0.5 thì là 1 (tấn công), ngược lại là 0 (bình thường)
y_pred = (y_prob > 0.5).astype(int) 
# --- BƯỚC 3: ĐÁNH GIÁ (Sử dụng y_test và y_pred đã được phân ngưỡng) ---

print('\nconfusion_matrix:')
print(confusion_matrix(y, y_pred))

print('\nclassification_report:')
print(classification_report(y, y_pred))

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Tính ROC
fpr, tpr, thresholds = roc_curve(y, y_prob)
roc_auc = auc(fpr, tpr)
# Vẽ
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='darkorange',
         lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Random')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)

plt.show()



