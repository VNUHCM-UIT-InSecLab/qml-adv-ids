import pandas as pd
import numpy as np
df = pd.read_csv("test_path")

features = [  'FDiR', 'FPR', 'FDuR',
    'FPRR', 'FITR', 'FPLR', 'FLRR', 'FTRLR' ]

import joblib

scaler = joblib.load("scaler_of_svc_path")

from sklearn.preprocessing import StandardScaler,RobustScaler
import joblib
X = df[features]
y = df["Attack_label"]


X_scaled = scaler.transform(X)

import tensorflow as tf
import numpy as np
import joblib

svm_model = joblib.load(
    "/kaggle/input/datasets/tuankietquach/svm-2k-cuoicung/svm_aci_iot_2k_seed42.pkl"
)

y_score = svm_model.decision_function(X_scaled)

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
threshold = 0   # mặc định của SVM
y_pred = (y_score > threshold).astype(int)


print('\nconfusion_matrix:')
print(confusion_matrix(y, y_pred))

print('\nclassification_report:')
print(classification_report(y, y_pred))

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Tính ROC
fpr, tpr, thresholds = roc_curve(y, y_score)
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
