from sklearn import datasets

iris=datasets.load_iris()

data=iris.data
target=iris.target

print(data[0])

#######################################
from sklearn.decomposition import PCA

pca=PCA(n_components=2) #n_components- number of features required
pca.fit(data)
data=pca.transform(data)

print(data[0])
#######################################

from matplotlib import pyplot as plt

for i in range(0,150):

    if target[i]==0:
        plt.scatter(data[i][0],data[i][1],c='r')
    elif target[i]==1:
        plt.scatter(data[i][0],data[i][1],c='g')
    elif target[i]==2:
        plt.scatter(data[i][0],data[i][1],c='b')

plt.show()
     
from sklearn.model_selection import train_test_split

train_data,test_data,train_target,test_target=train_test_split(data,target,test_size=0.1)

from sklearn import tree

clsfr=tree.DecisionTreeClassifier()

clsfr.fit(train_data,train_target)
results=clsfr.predict(test_data)

from sklearn import metrics

accuracy=metrics.accuracy_score(test_target,results)
print('Accuaracy:',accuracy)

print(data.shape)
