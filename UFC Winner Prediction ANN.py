# UFC Winnter Prediction ANN

import numpy as np
import pandas as pd

# Importing the UFC_dataset
UFC_dataset = pd.read_csv('UFCdata.csv')
X = UFC_dataset.iloc[:, 2:42].values
y = UFC_dataset.iloc[:, 42].values # 0 = Red Corner Won, 1 = Blue Corner Won
# Z = pd.DataFrame(X)

# Imputing Numeric Data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, 0:15])
X[:, 0:15] = imputer.transform(X[:, 0:15])
imputer = imputer.fit(X[:, 16:35])
X[:, 16:35] = imputer.transform(X[:, 16:35])
imputer = imputer.fit(X[:, 36:40])
X[:, 36:40] = imputer.transform(X[:, 36:40])

# Imputing Categorical Data
from sklearn_pandas import CategoricalImputer
data = np.array(X[:, 15], dtype=object)
imputer = CategoricalImputer()
X[:, 15] = imputer.fit_transform(data)
data = np.array(X[:, 35], dtype=object)
imputer = CategoricalImputer()
X[:, 35] = imputer.fit_transform(data)

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_15 = LabelEncoder()
X[:, 15] = labelencoder_X_15.fit_transform(X[:, 15])
labelencoder_X_35 = LabelEncoder()
X[:, 35] = labelencoder_X_35.fit_transform(X[:, 35])
onehotencoder = OneHotEncoder(categorical_features = [[15], [35]])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
X = np.delete(X, [1, 4, 6], 1) # deleting columns of zeros

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
classifier.add(Dense(output_dim = 30, init = 'uniform', activation = 'relu', input_dim = np.size(X_train, 1)))

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
y_pred_percentage = classifier.predict(X_test) # chance that blue corner will win, expressed as a percentage
y_pred_bool = (y_pred_percentage > 0.50)
y_pred = np.where(y_pred_percentage > 0.50, 'Blue', 'Red')

# Making the Confusion Matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_bool)
