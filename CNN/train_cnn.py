import os, random
import numpy as np
import tensorflow as tf
import pandas as pd

df = pd.read_csv('dataset_path')
from sklearn.model_selection import train_test_split

X = df.drop(columns=['Attack_label'])
y = df['Attack_label']




x_train, x_temp, y_train, y_temp = train_test_split(
    X, y, 
    test_size=0.3, 
    random_state=42,
    shuffle = True,
    stratify=y  # Quan trọng: giữ phân bố lớp
)
x_test, x_valid, y_test, y_valid = train_test_split(
    x_temp, y_temp,
    test_size=0.5,
    random_state=42,
    shuffle = True,
    stratify=y_temp
)


log_features = ['FPR', 'FPLR', 'FTRLR','FLRR','FDiR']

for df_ in [x_train, x_test, x_valid]:
    df_[log_features] = np.log1p(df_[log_features])


from sklearn.preprocessing import StandardScaler
import joblib

scaler = StandardScaler()
X_train = scaler.fit_transform(x_train)
X_valid = scaler.transform(x_valid)
X_test  = scaler.transform(x_test)
joblib.dump(scaler, "scaler_aic_check_1k_seed0.pkl")
print(X_train.shape)


X_train = np.expand_dims(X_train, axis=-1)
X_valid = np.expand_dims(X_valid, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)



from tensorflow import keras
from tensorflow.keras import layers

model_cnn = keras.Sequential([
    layers.Conv1D(filters = 16,kernel_size = 3,activation = 'relu',padding = 'same',input_shape = input_shape),
    layers.MaxPooling1D(),
    layers.Conv1D(filters = 32,kernel_size = 3,activation = 'relu',padding = 'same'),
    layers.MaxPooling1D(),
    layers.Flatten(),
    layers.Dense(units=32, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(units=1,activation = 'sigmoid')
])


from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

opt = Adam(learning_rate = 3e-4) 

model_cnn.compile(
    optimizer = opt,
    loss = 'binary_crossentropy',
    metrics = ['binary_accuracy']
)

call_backs = [
    ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,            
    patience=2,            
    min_lr=0.000001,        
    verbose=1              
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=4,
        restore_best_weights=True,
        verbose=1
    )]


history_cnn = model_cnn.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=64,
    epochs=50, 
    callbacks=call_backs
)
import matplotlib.pyplot as plt

history_df = pd.DataFrame(history_cnn.history)

plt.figure(figsize = (10,5))

# ve loss
plt.subplot(1,2,1)
plt.plot(history_df['loss'], label = 'Train Loss')
plt.plot(history_df['val_loss'],label = 'Val loss')
plt.title('Loss')
plt.legend()
# ve accuracy
plt.subplot(1,2,2)
plt.plot(history_df['binary_accuracy'], label = 'Train Acc')
plt.plot(history_df['val_binary_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.legend()

plt.show()

y_prob = model_cnn.predict(X_test) 

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Giả sử bạn đã có mô hình MLP được huấn luyện là 'model_mlp'
# và tập dữ liệu kiểm tra là 'X_test' (dạng numpy array)

# --- BƯỚC 1: DỰ ĐOÁN XÁC SUẤT (Sẽ ra giá trị liên tục [0, 1]) ---


# --- BƯỚC 2: PHÂN NGƯỠNG HÓA (Đưa về dạng 0 hoặc 1) ---
# Đặt ngưỡng 0.5: Nếu xác suất >= 0.5 thì là 1 (tấn công), ngược lại là 0 (bình thường)
y_pred = (y_prob > 0.45).astype(int) 

# --- BƯỚC 3: ĐÁNH GIÁ (Sử dụng y_test và y_pred đã được phân ngưỡng) ---

print('\nconfusion_matrix:')
print(confusion_matrix(y_test, y_pred))

print('\nclassification_report:')
print(classification_report(y_test, y_pred))


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Tính ROC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
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

