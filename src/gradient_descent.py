import pandas as pd
import numpy as np
from sklearn.preprocessing import add_dummy_feature

df = pd.read_csv('encoded_dataset.csv')

X = df['Dalc'].values.reshape(-1, 1)
y = df['G3'].values

X = add_dummy_feature(X)
theta = np.random.randn(X.shape[1])
L = 0.01
epochs = 1000

def gradient_descent(X, y, theta, L, epochs):
    m = len(y)
    for _ in range(epochs):
        predictions = X.dot(theta)
        theta0_partial = (2/m) * np.sum((predictions - y))
        theta1_partial = (2/m) * np.sum((predictions - y).dot(X))

        theta[0] = theta[0] - L * theta0_partial
        theta[1] = theta[1] - L * theta1_partial
    return theta

theta = gradient_descent(X, y, theta, L, epochs)
print(theta)

predictions = X.dot(theta)

# Plotting the results
import matplotlib.pyplot as plt
plt.scatter(df['Dalc'], df['G3'])
plt.xlabel('Dalc')
plt.ylabel('G3')
plt.plot(df['Dalc'], predictions, color='red')
plt.show()