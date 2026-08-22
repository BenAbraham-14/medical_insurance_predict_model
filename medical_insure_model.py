# train.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Load Data
df = pd.read_csv("C:/Users/chris/OneDrive/Desktop/medical insurance model/insurance.csv")
# 2. Encode Categoricals (drop_first to prevent dummy variable trap)
df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

X = df_encoded.drop('charges', axis=1)
y = df_encoded['charges']

# Save column order for consistent inference in app.py
feature_columns = list(X.columns)
joblib.dump(feature_columns, 'columns.pkl')

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Standard Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, 'scaler.pkl')

# 5. Build ANN Regressor (Capacity tuned for R² ~0.70 - 0.80)
model = models.Sequential([
    layers.Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.Dropout(0.35),
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.25),
    layers.Dense(1)  # Linear activation for continuous cost
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss='mae',
    metrics=['mae']
)

# 6. Train Model
model.fit(
    X_train_scaled, 
    y_train, 
    epochs=85, 
    batch_size=32, 
    verbose=0
)

# 7. Evaluate Performance
y_pred = model.predict(X_test_scaled).flatten()
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Model Training Complete!")
print(f"Test R² Score: {r2:.3f}")
print(f"Mean Absolute Error (MAE): ${mae:.2f}")

# 8. Save Model
model.save('insurance_model.keras')
print("Saved: insurance_model.keras, scaler.pkl, columns.pkl")