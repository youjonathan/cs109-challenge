import pandas as pd
import numpy as np
from sklearn.preprocessing import add_dummy_feature

df = pd.read_csv('encoded_dataset.csv')

X = df['Dalc'].values.reshape(-1, 1)
y = df['G3'].values

X = add_dummy_feature(X)

def normal_equation(X, y):
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

theta = normal_equation(X, y)
print(theta)

predictions = X.dot(theta)

# Plotting the results
import matplotlib.pyplot as plt
plt.scatter(df['Dalc'], df['G3'])
plt.xlabel('Dalc')
plt.ylabel('G3')
plt.plot(df['Dalc'], predictions, color='red')
plt.show()