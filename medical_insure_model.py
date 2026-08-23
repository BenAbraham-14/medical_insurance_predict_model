import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# 1. Load the Indian Dataset
df = pd.read_csv('Medicalpremium.csv')

# 2. Features and Target (PremiumPrice in INR)
X = df.drop('PremiumPrice', axis=1)
y = df['PremiumPrice']

# Save feature column names
joblib.dump(list(X.columns), 'columns.pkl')

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Standard Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'scaler.pkl')

# 5. Train Neural Network (MLP)
model = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    learning_rate_init=0.01,
    max_iter=1500,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# 6. Evaluate
y_pred = model.predict(X_test_scaled)
print(f"R² Score: {r2_score(y_test, y_pred):.3f}")
print(f"MAE: ₹{mean_absolute_error(y_test, y_pred):.2f}")

# 7. Save Model
joblib.dump(model, 'insurance_model.pkl')
print("Successfully generated: insurance_model.pkl, scaler.pkl, columns.pkl")