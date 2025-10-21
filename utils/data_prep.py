import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# ----------------------------
# Create folders if not exist
# ----------------------------
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ----------------------------
# Generate synthetic dataset
# ----------------------------
np.random.seed(42)
N = 200  # number of applicants

data = {
    'income': np.random.randint(2000, 10000, N),
    'family_size': np.random.randint(1, 8, N),
    'employment_years': np.random.randint(0, 20, N),
    'assets': np.random.randint(0, 50000, N),
    'age': np.random.randint(18, 65, N),
}

df = pd.DataFrame(data)

# Eligibility rule (synthetic)
df['eligible'] = ((df['income'] < 5000) | (df['assets'] < 10000)) & (df['family_size'] > 3)
df['eligible'] = df['eligible'].astype(int)

# Save dataset
df.to_csv('data/applicants.csv', index=False)

# ----------------------------
# Train ML model
# ----------------------------
X = df[['income', 'family_size', 'employment_years', 'assets', 'age']]
y = df['eligible']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save trained model
joblib.dump(model, 'models/eligibility_model.joblib')

print("✅ Synthetic data generated and ML model trained successfully!")
