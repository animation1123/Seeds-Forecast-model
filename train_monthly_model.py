import pandas as pd
import numpy as np
import pickle
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

np.random.seed(42)

df = pd.read_csv("synthetic_demo_data.csv")
df["Month"] = pd.to_datetime(df["Month"])

product_cols = [c for c in df.columns if c != "Month"]

# reshape wide -> long: one row per (month, product)
long_rows = []
for _, row in df.iterrows():
    for p in product_cols:
        long_rows.append({
            "month_dt": row["Month"],
            "product": p,
            "quantity": row[p],
        })
long_df = pd.DataFrame(long_rows)

long_df["month_num"] = long_df["month_dt"].dt.month
long_df["year"] = long_df["month_dt"].dt.year
long_df["month_sin"] = np.sin(2 * np.pi * long_df["month_num"] / 12)
long_df["month_cos"] = np.cos(2 * np.pi * long_df["month_num"] / 12)

le = LabelEncoder()
long_df["product_enc"] = le.fit_transform(long_df["product"])

features = ["product_enc", "month_num", "month_sin", "month_cos", "year"]
X = long_df[features]
y = long_df["quantity"].astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.08, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred = np.maximum(y_pred, 0)
test_mae = mean_absolute_error(y_test, y_pred)
print(f"Test MAE: {test_mae:.3f}  (mean qty: {y.mean():.2f})")

# retrain on ALL data for the production model that powers the live demo
model_final = XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.08, random_state=42)
model_final.fit(X, y)

with open("xgb_model.pkl", "wb") as f:
    pickle.dump(model_final, f, protocol=4)
with open("product_encoder.pkl", "wb") as f:
    pickle.dump(le, f, protocol=4)
with open("product_list.pkl", "wb") as f:
    pickle.dump(sorted(product_cols), f, protocol=4)

long_df.to_csv("monthly_long_data.csv", index=False)
print("Saved xgb_model.pkl, product_encoder.pkl, product_list.pkl, monthly_long_data.csv")
print("Products:", sorted(product_cols))
