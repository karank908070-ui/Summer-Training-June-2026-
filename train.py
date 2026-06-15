"""
Automotive Market Intelligence System - Model Training
Trains:
  1. Linear Regression -> predicts log(selling_price)
  2. Logistic Regression -> classifies car as High Value / Low Value
        (High Value = selling_price >= median selling_price)
Saves models, scaler, encoders to disk.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix

DATA_PATH = "cardekho_dataset.csv"   # <-- put your CSV in same folder
OUT_DIR = "."

# ---------------- Load & Clean ----------------
df = pd.read_csv(DATA_PATH)
df = df.drop(columns=["Unnamed: 0", "car_name", "model"])

# Encode categorical columns
cat_cols = ["brand", "seller_type", "fuel_type", "transmission_type"]
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Feature columns
feature_cols = ["brand", "vehicle_age", "km_driven", "seller_type",
                 "fuel_type", "transmission_type", "mileage",
                 "engine", "max_power", "seats"]

X = df[feature_cols]

# ---------------- 1. Linear Regression: Price Prediction (log target) ----------------
y_price_log = np.log(df["selling_price"])

X_train, X_test, y_train, y_test = train_test_split(X, y_price_log, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lin_model = LinearRegression()
lin_model.fit(X_train_scaled, y_train)

y_pred = lin_model.predict(X_test_scaled)
print("Linear Regression -> R2 (log price):", r2_score(y_test, y_pred),
      "RMSE (log price):", np.sqrt(mean_squared_error(y_test, y_pred)))

# ---------------- 2. Logistic Regression: High/Low Value Classification ----------------
median_price = df["selling_price"].median()
y_class = (df["selling_price"] >= median_price).astype(int)  # 1 = High Value, 0 = Low Value

X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y_class, test_size=0.2, random_state=42)

scaler2 = StandardScaler()
X_train2_scaled = scaler2.fit_transform(X_train2)
X_test2_scaled = scaler2.transform(X_test2)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train2_scaled, y_train2)

y_pred2 = log_model.predict(X_test2_scaled)
print("Logistic Regression -> Accuracy:", accuracy_score(y_test2, y_pred2))
print("Confusion Matrix:\n", confusion_matrix(y_test2, y_pred2))

# ---------------- Save Everything ----------------
joblib.dump(lin_model, f"{OUT_DIR}/linear_model.pkl")
joblib.dump(scaler, f"{OUT_DIR}/scaler_linear.pkl")

joblib.dump(log_model, f"{OUT_DIR}/logistic_model.pkl")
joblib.dump(scaler2, f"{OUT_DIR}/scaler_logistic.pkl")

joblib.dump(encoders, f"{OUT_DIR}/encoders.pkl")
joblib.dump(median_price, f"{OUT_DIR}/median_price.pkl")
joblib.dump(feature_cols, f"{OUT_DIR}/feature_cols.pkl")

print("\nMedian price used for classification:", median_price)
print("All models & objects saved successfully.")