# ------------------------------
# HEART DISEASE ML PROJECT
# Python + Visual Studio Compatible Version
# ------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ===============================
# 1) VERİYİ OKUMA
# ===============================

df = pd.read_csv("heart.csv")   # Dosya aynı klasörde olmalı

print("\n--- İlk 5 Satır ---")
print(df.head())

print("\n--- Veri Bilgisi ---")
print(df.info())

print("\n--- Veri İstatistikleri ---")
print(df.describe())

print("\n--- Target Değer Dağılımı ---")
print(df["target"].value_counts())


# ===============================
# 2) KEŞİFSEL VERİ ANALİZİ (EDA)
# ===============================

plt.figure(figsize=(4,4))
sns.countplot(x="target", data=df)
plt.title("Kalp Hastalığı Dağılımı (0: Yok, 1: Var)")
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=False, cmap="coolwarm")
plt.title("Korelasyon Isı Haritası")
plt.show()


# ===============================
# 3) ÖZELLİK / HEDEF AYRIMI
# ===============================

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nEğitim seti boyutu:", X_train.shape)
print("Test seti boyutu:", X_test.shape)


# ===============================
# 4) ÖLÇEKLEME (Scaler)
# ===============================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ===============================
# 5) LOJİSTİK REGRESYON MODELİ
# ===============================

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)

y_pred_log = log_model.predict(X_test_scaled)

print("\n==============================")
print(" LOJİSTİK REGRESYON")
print("==============================")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("\nClassification Report:\n", classification_report(y_test, y_pred_log))

cm_log = confusion_matrix(y_test, y_pred_log)
print("\nConfusion Matrix:\n", cm_log)

plt.figure(figsize=(4,4))
sns.heatmap(cm_log, annot=True, fmt="d", cmap="Blues")
plt.title("Lojistik Regresyon - Confusion Matrix")
plt.show()


# ===============================
# 6) RANDOM FOREST MODELİ
# ===============================

rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("\n==============================")
print(" RANDOM FOREST")
print("==============================")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))

cm_rf = confusion_matrix(y_test, y_pred_rf)
print("\nConfusion Matrix:\n", cm_rf)

plt.figure(figsize=(4,4))
sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Greens")
plt.title("Random Forest - Confusion Matrix")
plt.show()


# ===============================
# 7) RANDOM FOREST FEATURE IMPORTANCE
# ===============================

importances = rf_model.feature_importances_
feature_names = X.columns

feat_imp = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\n--- Özellik Önemleri (Feature Importance) ---")
print(feat_imp)

plt.figure(figsize=(8,5))
sns.barplot(x="Importance", y="Feature", data=feat_imp)
plt.title("Özellik Önemleri (Random Forest)")
plt.show()