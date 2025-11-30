import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import json

df = pd.read_csv('encoded_dataset.csv')

# Prepare data
X = df.drop('G3', axis=1)
y = df['G3'].values

# Save feature order
feature_order = X.columns.tolist()

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train sklearn Linear Regression model
model = LinearRegression()
model.fit(X_scaled, y)

# Build export dictionary
model_export = {
    'feature_order': feature_order,             # List of feature names in order
    'scaler_mean': scaler.mean_.tolist(),       # List of means used for scaling
    'scaler_scale': scaler.scale_.tolist(),     # List of scales used for scaling
    'coefficients': model.coef_.tolist(),       # Weights
    'intercept': model.intercept_               # Bias
}

# Save to JSON into /docs/model.json for GitHub Pages
with open('../docs/model.json', 'w') as f:
    json.dump(model_export, f, indent=2)

print('Model exported to ../docs/model.json')
print('Number of features:', len(feature_order))

i = 6  # example row
x_row = df.drop(columns=["G3"]).iloc[i:i+1]   # DataFrame with 1 row
y_true = df["G3"].iloc[i]
print(x_row)
print(y_true)
X_scaled = scaler.transform(x_row.values)
y_pred = model.predict(X_scaled)[0]
print(y_true, y_pred)