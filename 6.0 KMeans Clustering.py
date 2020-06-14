from sklearn import datasets

iris=datasets.load_iris()

data=iris.data
target=iris.target

from sklearn.model_selection import train_test_split

train_data,test_data,train_target,test_target=train_test_split(data,target,test_size=0.1)

from sklearn.cluster import KMeans

clsfr=KMeans(n_clusters=3)
clsfr.fit(train_data)
train_results=clsfr.labels_

test_result=clsfr.predict(test_data)

print(test_result)
