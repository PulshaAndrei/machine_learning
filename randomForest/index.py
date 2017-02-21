import numpy as np
import pandas
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

data = pandas.read_csv('../data/abalone.csv')
data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))
X = data[['Sex','Length','Diameter','Height','WholeWeight','ShuckedWeight','VisceraWeight','ShellWeight']]
y = data[['Rings']]

cross_validation = KFold(n_splits=5, shuffle=True, random_state=1)
clf = RandomForestRegressor(n_estimators=1, random_state=1)

train = cross_validation.split(X, y)
# clf.fit(train, test)
# predictions = clf.predict(train)

print(list(train))
# print(r2_score([2, 3.1], [2, 3]))