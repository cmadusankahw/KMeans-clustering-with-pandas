import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4, 5])
y = np.array([4, 2, 1, 3, 7])
plt.scatter(x, y)

x=np.reshape(x,(x.shape[0],1))

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

poly=PolynomialFeatures(degree=2,include_bias=False)
x_new=poly.fit_transform(x)

print(x_new)

model=LinearRegression()
model.fit(x_new,y)

result=model.predict(x_new)

plt.plot(x,result,'r--',marker='x')

plt.show()
