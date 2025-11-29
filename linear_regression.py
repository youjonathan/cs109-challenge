import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import add_dummy_feature

df = pd.read_csv('encoded_dataset.csv')

# Prepare data
X = df.drop('G3', axis=1).values
y = df['G3'].values

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Add bias term
X = add_dummy_feature(X)

# Initialize parameters
alpha = 0.01
epochs = 10000
theta = np.random.randn(X.shape[1])

def gradient_descent(X, y, theta, alpha, epochs):
    n = len(y)
    for _ in range(epochs):
        predictions = X.dot(theta)
        errors = predictions - y
        gradients = (2/n) * X.T.dot(errors)
        theta = theta - alpha * gradients
    return theta

# Train model
theta = gradient_descent(X, y, theta, alpha, epochs)
predictions = X.dot(theta)

# Evaluate model using RMSE
def rmse_evaluation(pred_y, y):
    return np.sqrt(np.mean((pred_y - y)**2))

test = pd.read_csv('encoded_dataset.csv')
y = test['G3'].values
pred_y = predictions
print(f"RMSE: {rmse_evaluation(pred_y, y)}")

# Compare to sklearn's LinearRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

gold_X = scaler.fit_transform(df.drop('G3', axis=1).values)
model = LinearRegression()
model.fit(gold_X, y)
predictions_sklearn = model.predict(gold_X)
print(f"RMSE (sklearn): {rmse_evaluation(predictions_sklearn, y)}")
print(f"RMSE (sklearn.metrics): {np.sqrt(mean_squared_error(y, predictions_sklearn))}")