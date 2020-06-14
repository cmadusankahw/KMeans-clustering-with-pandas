import pandas as pd

df=pd.read_csv('diabetes.csv')
#dataframe referred in pandas

cols=df.columns

#test_array=df[['Pregnancies','Glucose']]
#print(test_array)

data=df[cols[0:8]]
target=df[cols[8]]

from sklearn.preprocessing import Imputer

imp=Imputer(strategy='mean') #or median, most frequent etc.

data=imp.fit_transform(data)

print(data[0:10,3])

from sklearn.model_selection import train_test_split

train_data,test_data,train_target,test_target=train_test_split(data,target,test_size=0.2)

from sklearn.tree import DecisionTreeClassifier

clsfr=DecisionTreeClassifier()
clsfr.fit(train_data,train_target)
results=clsfr.predict(test_data)

from sklearn import metrics

accuracy=metrics.accuracy_score(test_target,results)
print('Accuaracy:',accuracy)
