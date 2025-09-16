import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1500], [1600], [1700], [1800], [1900]])
y = np.array([300000, 320000, 340000, 360000, 380000])
model = LinearRegression().fit(X, y)
print(model.coef_, model.intercept_)

