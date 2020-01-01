# UFC Winnter Prediction ANN

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('UFCdata.csv')
X = dataset.iloc[:, 2:38].values
y = dataset.iloc[:, 38].values

# Imputing Numeric Data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 0:13])
X[:, 0:13] = imputer.transform(X[:, 0:13])
imputer = imputer.fit(X[:, 14:30])
X[:, 14:30] = imputer.transform(X[:, 14:30])
imputer = imputer.fit(X[:, 31:36])
X[:, 31:36] = imputer.transform(X[:, 31:36])

# Imputing Categorical Data
from sklearn_pandas import CategoricalImputer
data = np.array(X[:, 13], dtype=object)
imputer = CategoricalImputer()
X[:, 13] = imputer.fit_transform(data)
data = np.array(X[:, 30], dtype=object)
imputer = CategoricalImputer()
X[:, 30] = imputer.fit_transform(data)

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_13 = LabelEncoder()
X[:, 13] = labelencoder_X_13.fit_transform(X[:, 13])
labelencoder_X_30 = LabelEncoder()
X[:, 30] = labelencoder_X_30.fit_transform(X[:, 30])
onehotencoder = OneHotEncoder(categorical_features = [[13], [30]])
X = onehotencoder.fit_transform(X).toarray()
Z = pd.DataFrame(X)
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 30, init = 'uniform', activation = 'relu', input_dim = 43))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 30, init = 'uniform', activation = 'relu'))

# Adding the third hidden layer
classifier.add(Dense(output_dim = 30, init = 'uniform', activation = 'relu'))

# Adding the fourth hidden layer
classifier.add(Dense(output_dim = 30, init = 'uniform', activation = 'relu'))

# Adding the fith hidden layer
classifier.add(Dense(output_dim = 30, init = 'uniform', activation = 'relu'))

# Adding the sixth hidden layer
classifier.add(Dense(output_dim = 30, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adamax', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 25)

# Predicting the Test set results 
y_pred = classifier.predict(X_test) 
y_pred = (y_pred > 0.50) # Converting to T/F based on 50% threshold

# Making the Confusion Matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)