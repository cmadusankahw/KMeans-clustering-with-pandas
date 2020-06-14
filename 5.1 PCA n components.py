from sklearn import datasets

digits=datasets.load_digits()

data=digits.data
target=digits.target

from sklearn.decomposition import PCA

pca=PCA()
pca.fit(data)

from matplotlib import pyplot as plt
import numpy as np

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('number of components')
plt.ylabel('cumalative variance')

plt.show()
