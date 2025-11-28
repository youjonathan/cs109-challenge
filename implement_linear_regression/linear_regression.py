import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('train.csv')

# Basic Linear Regression Implementation
def loss_function(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].x
        y = points.iloc[i].y
        total_error += (y - (m * x + b)) ** 2
    total_error /= float(len(points))
    return total_error

def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0

    n = len(points)

    for i in range(n):
        x = points.iloc[i].x
        y = points.iloc[i].y

        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))

    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    return m, b

def linear_regression(data, L, epochs):
    m = 0
    b = 0

    for i in range(epochs):
        m, b = gradient_descent(m, b, data, L)
    return m, b

L = 0.0001
epochs = 1000
m, b = linear_regression(data, L, epochs)
print(f"Final parameters: m = {m}, b = {b}")

def rmse_evaluation(pred_y, y):
    n = len(y)
    total_error = 0
    for i in range(n):
        total_error += (y.iloc[i] - pred_y.iloc[i]) ** 2
    rmse = (total_error / n) ** 0.5
    return rmse

test = pd.read_csv('test.csv')
y = test.y
pred_y = m * test.x + b
print(f"RMSE: {rmse_evaluation(pred_y, y)}")