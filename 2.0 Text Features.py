sample = ['problem of evil',
          'evil queen',
          'horizon problem']

from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer()
X = vec.fit_transform(sample)
#print(X)

import pandas as pd

df=pd.DataFrame(X.toarray(),columns=vec.get_feature_names())

print(df)
