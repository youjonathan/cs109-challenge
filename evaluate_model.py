import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import add_dummy_feature
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv('encoded_dataset.csv')

# Prepare data
X = df.drop('G3', axis=1).values
y = df['G3'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Add bias term
X_train = add_dummy_feature(X_train)
X_test = add_dummy_feature(X_test)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Evaluate model using RMSE
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, predictions))}") # RMSE: 3.700920835179676 -> good to go!