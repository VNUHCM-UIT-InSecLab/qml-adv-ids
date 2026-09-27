import os, random
import numpy as np
import tensorflow as tf
import pandas as pd

df = pd.read_csv('data_set_path')

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

from sklearn.preprocessing import StandardScaler
import joblib

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_valid = scaler.transform(x_valid)
x_test  = scaler.transform(x_test)
joblib.dump(scaler, "scaler_aic_1k-seed42.pkl")


from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score



svm_model = LinearSVC(
    C=0.1,  
    loss="squared_hinge",
    class_weight="balanced", 
    max_iter=10000,
    dual=False,
    random_state=42
)

svm_model.fit(x_train, y_train)

y_val_score = svm_model.decision_function(x_valid)
y_train_score = svm_model.decision_function(x_train)


y_score = svm_model.decision_function(x_test)


import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
threshold = 0   # mặc định của SVM
y_pred = (y_score > threshold).astype(int)


print('\nconfusion_matrix:')
print(confusion_matrix(y_test, y_pred))

print('\nclassification_report:')
print(classification_report(y_test, y_pred))


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Tính ROC
fpr, tpr, thresholds = roc_curve(y_test, y_score)
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

import joblib

joblib.dump(svm_model, "model_svc.pkl")
