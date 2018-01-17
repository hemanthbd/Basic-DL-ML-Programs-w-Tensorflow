import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

data = pd.read_csv('bank_note_data.csv')

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(data.drop('Class',axis=1))
scaled_features = scaler.fit_transform(data.drop('Class',axis=1))

df_feat = pd.DataFrame(scaled_features,columns=data.columns[:-1])

X = df_feat
y = data['Class']
X = X.as_matrix()
y = y.as_matrix()

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

import tensorflow.contrib.learn.python.learn as learn

classifier = learn.DNNClassifier(hidden_units=[10, 20, 10], n_classes=2)
classifier.fit(X_train, y_train, steps=200, batch_size=20)

predictions = classifier.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


