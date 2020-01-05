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

# =============================================================================
# # Making a User-Defined Prediction (Have user enter all the data, then put it into an array, preprocess it, and predict)
# =============================================================================
answer = input('The Neural Network has been trained with the data provided. Would you like to have it make a fight-winner-predicition based on user-defined data? Enter "yes" or "no": ').lower()
loop = True
while loop:
    if answer == 'yes' or answer == 'no':
        loop = False
    else:
        answer = input('Please enter either "yes" or "no": ').lower()
        
if answer == 'yes':
    user_data = []
    
# =============================================================================
#     Blue Fighter
# =============================================================================
    B_current_lose_streak = input("Please enter the blue fighter's current losing streak: ")
    loop = True
    while loop:
        if B_current_lose_streak.isdigit():
            loop = False
            user_data.append(int(B_current_lose_streak))
        else:
            B_current_lose_streak = input('Please enter a valid number: ')
    
    B_current_win_streak = input("Please enter the blue fighter's current win streak: ")
    loop = True
    while loop:
        if B_current_win_streak.isdigit():
            loop = False
            user_data.append(int(B_current_win_streak))
        else:
            B_current_win_streak = input('Please enter a valid number: ')
            
    B_draw = input("Please enter the blue fighter's current draw streak: ")
    loop = True
    while loop:
        if B_draw.isdigit():
            loop = False
            user_data.append(int(B_draw))
        else:
            B_draw = input('Please enter a valid number: ')
            
    B_longest_win_streak = input("Please enter the blue fighter's longest win streak: ")
    loop = True
    while loop:
        if B_longest_win_streak.isdigit():
            loop = False
            user_data.append(int(B_longest_win_streak))
        else:
            B_longest_win_streak = input('Please enter a valid number: ')
            
    B_losses = input("Please enter the blue fighter's total number of losses: ")
    loop = True
    while loop:
        if B_losses.isdigit():
            loop = False
            user_data.append(int(B_losses))
        else:
            B_losses = input('Please enter a valid number: ')
            
    B_total_rounds_fought = input("Please enter the blue fighter's number of total rounds fought: ")
    loop = True
    while loop:
        if B_total_rounds_fought.isdigit():
            loop = False
            user_data.append(int(B_total_rounds_fought))
        else:
            B_total_rounds_fought = input('Please enter a valid number: ')
            
    B_total_time_fought = input("Please enter the blue fighter's total time fought (in seconds): ")
    loop = True
    while loop:
        if B_total_time_fought.isdigit():
            loop = False
            user_data.append(int(B_total_time_fought))
        else:
            B_total_time_fought = input('Please enter a valid number: ')
            
    B_total_title_bouts = input("Please enter the blue fighter's total title bouts fought in: ")
    loop = True
    while loop:
        if B_total_title_bouts.isdigit():
            loop = False
            user_data.append(int(B_total_title_bouts))
        else:
            B_total_title_bouts = input('Please enter a valid number: ')
            
    B_win_by_Decision_Majority = input("Please enter the blue fighter's wins by decision (majority): ")
    loop = True
    while loop:
        if B_win_by_Decision_Majority.isdigit():
            loop = False
            user_data.append(int(B_win_by_Decision_Majority))
        else:
            B_win_by_Decision_Majority = input('Please enter a valid number: ')
            
    B_win_by_Decision_Split = input("Please enter the blue fighter's wins by decision (split): ")
    loop = True
    while loop:
        if B_win_by_Decision_Split.isdigit():
            loop = False
            user_data.append(int(B_win_by_Decision_Split))
        else:
            B_win_by_Decision_Split = input('Please enter a valid number: ')
            
    B_win_by_Decision_Unanimous = input("Please enter the blue fighter's wins by decision (unamimous): ")
    loop = True
    while loop:
        if B_win_by_Decision_Unanimous.isdigit():
            loop = False
            user_data.append(int(B_win_by_Decision_Unanimous))
        else:
            B_win_by_Decision_Unanimous = input('Please enter a valid number: ')
            
    B_win_by_KO_or_TKO = input("Please enter the blue fighter's wins by KO/TKO: ")
    loop = True
    while loop:
        if B_win_by_KO_or_TKO.isdigit():
            loop = False
            user_data.append(int(B_win_by_KO_or_TKO))
        else:
            B_win_by_KO_or_TKO = input('Please enter a valid number: ')
            
    B_win_by_Submission = input("Please enter the blue fighter's wins by submission: ")
    loop = True
    while loop:
        if B_win_by_Submission.isdigit():
            loop = False
            user_data.append(int(B_win_by_Submission))
        else:
            B_win_by_Submission = input('Please enter a valid number: ')
            
    B_win_by_TKO_Doctor_Stoppage = input("Please enter the blue fighter's wins by doctor stoppage: ")
    loop = True
    while loop:
        if B_win_by_TKO_Doctor_Stoppage.isdigit():
            loop = False
            user_data.append(int(B_win_by_TKO_Doctor_Stoppage))
        else:
            B_win_by_TKO_Doctor_Stoppage = input('Please enter a valid number: ')
            
    B_wins = input("Please enter the blue fighter's total number of wins: ")
    loop = True
    while loop:
        if B_wins.isdigit():
            loop = False
            user_data.append(int(B_wins))
        else:
            B_wins = input('Please enter a valid number: ')
            
    B_Stance = input("Please enter the blue fighter's stance (Orthodox, Southpaw, or Switch: ").lower()
    loop = True
    while loop:
        if B_Stance == 'orthodox' or B_Stance == 'southpaw' or B_Stance == 'switch':
            loop = False
            user_data.append(B_Stance)
        else:
            B_Stance = input('Please enter a valid stance: ')
            
    B_Height_cms = input("Please enter the blue fighter's height (in cms): ")
    loop = True
    while loop:
        if B_Height_cms.isdigit():
            loop = False
            user_data.append(int(B_Height_cms))
        else:
            B_Height_cms = input('Please enter a valid number: ')
            
    B_Reach_cms = input("Please enter the blue fighter's reach (in cms): ")
    loop = True
    while loop:
        if B_Reach_cms.isdigit():
            loop = False
            user_data.append(int(B_Reach_cms))
        else:
            B_Reach_cms = input('Please enter a valid number: ')
            
    B_Weight_lbs = input("Please enter the blue fighter's weight (in lbs): ")
    loop = True
    while loop:
        if B_Weight_lbs.isdigit():
            loop = False
            user_data.append(int(B_Weight_lbs))
        else:
            B_Weight_lbs = input('Please enter a valid number: ')
            
    B_age = input("Please enter the blue fighter's age: ")
    loop = True
    while loop:
        if B_age.isdigit():
            loop = False
            user_data.append(int(B_age))
        else:
            B_age = input('Please enter a valid number: ')
                        
# =============================================================================
#     Red Fighter
# =============================================================================
    R_current_lose_streak = input("Please enter the red fighter's current losing streak: ")
    loop = True
    while loop:
        if R_current_lose_streak.isdigit():
            loop = False
            user_data.append(int(R_current_lose_streak))
        else:
            R_current_lose_streak = input('Please enter a valid number: ')
    
    R_current_win_streak = input("Please enter the red fighter's current win streak: ")
    loop = True
    while loop:
        if R_current_win_streak.isdigit():
            loop = False
            user_data.append(int(R_current_win_streak))
        else:
            R_current_win_streak = input('Please enter a valid number: ')
            
    R_draw = input("Please enter the red fighter's current draw streak: ")
    loop = True
    while loop:
        if R_draw.isdigit():
            loop = False
            user_data.append(int(R_draw))
        else:
            R_draw = input('Please enter a valid number: ')
            
    R_longest_win_streak = input("Please enter the red fighter's longest win streak: ")
    loop = True
    while loop:
        if R_longest_win_streak.isdigit():
            loop = False
            user_data.append(int(R_longest_win_streak))
        else:
            R_longest_win_streak = input('Please enter a valid number: ')
            
    R_losses = input("Please enter the red fighter's total number of losses: ")
    loop = True
    while loop:
        if R_losses.isdigit():
            loop = False
            user_data.append(int(R_losses))
        else:
            R_losses = input('Please enter a valid number: ')
            
    R_total_rounds_fought = input("Please enter the red fighter's number of total rounds fought: ")
    loop = True
    while loop:
        if R_total_rounds_fought.isdigit():
            loop = False
            user_data.append(int(R_total_rounds_fought))
        else:
            R_total_rounds_fought = input('Please enter a valid number: ')
            
    R_total_time_fought = input("Please enter the red fighter's total time fought (in seconds): ")
    loop = True
    while loop:
        if R_total_time_fought.isdigit():
            loop = False
            user_data.append(int(R_total_time_fought))
        else:
            R_total_time_fought = input('Please enter a valid number: ')
            
    R_total_title_bouts = input("Please enter the red fighter's total title bouts fought in: ")
    loop = True
    while loop:
        if R_total_title_bouts.isdigit():
            loop = False
            user_data.append(int(R_total_title_bouts))
        else:
            R_total_title_bouts = input('Please enter a valid number: ')
            
    R_win_by_Decision_Majority = input("Please enter the red fighter's wins by decision (majority): ")
    loop = True
    while loop:
        if R_win_by_Decision_Majority.isdigit():
            loop = False
            user_data.append(int(R_win_by_Decision_Majority))
        else:
            R_win_by_Decision_Majority = input('Please enter a valid number: ')
            
    R_win_by_Decision_Split = input("Please enter the red fighter's wins by decision (split): ")
    loop = True
    while loop:
        if R_win_by_Decision_Split.isdigit():
            loop = False
            user_data.append(int(R_win_by_Decision_Split))
        else:
            R_win_by_Decision_Split = input('Please enter a valid number: ')
            
    R_win_by_Decision_Unanimous = input("Please enter the red fighter's wins by decision (unamimous): ")
    loop = True
    while loop:
        if R_win_by_Decision_Unanimous.isdigit():
            loop = False
            user_data.append(int(R_win_by_Decision_Unanimous))
        else:
            R_win_by_Decision_Unanimous = input('Please enter a valid number: ')
            
    R_win_by_KO_or_TKO = input("Please enter the red fighter's wins by KO/TKO: ")
    loop = True
    while loop:
        if R_win_by_KO_or_TKO.isdigit():
            loop = False
            user_data.append(int(R_win_by_KO_or_TKO))
        else:
            R_win_by_KO_or_TKO = input('Please enter a valid number: ')
            
    R_win_by_Submission = input("Please enter the red fighter's wins by submission: ")
    loop = True
    while loop:
        if R_win_by_Submission.isdigit():
            loop = False
            user_data.append(int(R_win_by_Submission))
        else:
            R_win_by_Submission = input('Please enter a valid number: ')
            
    R_win_by_TKO_Doctor_Stoppage = input("Please enter the red fighter's wins by doctor stoppage: ")
    loop = True
    while loop:
        if R_win_by_TKO_Doctor_Stoppage.isdigit():
            loop = False
            user_data.append(int(R_win_by_TKO_Doctor_Stoppage))
        else:
            R_win_by_TKO_Doctor_Stoppage = input('Please enter a valid number: ')
            
    R_wins = input("Please enter the red fighter's total number of wins: ")
    loop = True
    while loop:
        if R_wins.isdigit():
            loop = False
            user_data.append(int(R_wins))
        else:
            R_wins = input('Please enter a valid number: ')
            
    R_Stance = input("Please enter the red fighter's stance (Orthodox, Southpaw, or Switch: ").lower()
    loop = True
    while loop:
        if R_Stance == 'orthodox' or R_Stance == 'southpaw' or R_Stance == 'switch':
            loop = False
            user_data.append(B_Stance)
        else:
            R_Stance = input('Please enter a valid stance: ')
            
    R_Height_cms = input("Please enter the red fighter's height (in cms): ")
    loop = True
    while loop:
        if R_Height_cms.isdigit():
            loop = False
            user_data.append(int(R_Height_cms))
        else:
            R_Height_cms = input('Please enter a valid number: ')
            
    R_Reach_cms = input("Please enter the red fighter's reach (in cms): ")
    loop = True
    while loop:
        if R_Reach_cms.isdigit():
            loop = False
            user_data.append(int(R_Reach_cms))
        else:
            R_Reach_cms = input('Please enter a valid number: ')
            
    R_Weight_lbs = input("Please enter the red fighter's weight (in lbs): ")
    loop = True
    while loop:
        if R_Weight_lbs.isdigit():
            loop = False
            user_data.append(int(R_Weight_lbs))
        else:
            R_Weight_lbs = input('Please enter a valid number: ')
    
    R_age = input("Please enter the red fighter's age: ")        
    loop = True
    while loop:
        if R_age.isdigit():
            loop = False
            user_data.append(int(R_age))
        else:
            R_age = input('Please enter a valid number: ')

    # Create array of user-defined fighter stats
    temp = user_data
    user_data = pd.DataFrame(np.array(user_data).reshape(1,40))
    user_data = user_data.iloc[:, :].values
    user_data = np.delete(user_data, [15, 35], 1)
    
    # Add six columns of zeros to the beginning of the array to manually encode each fighter's stance
    user_data = np.insert(user_data, 0, 0, axis=1)
    user_data = np.insert(user_data, 0, 0, axis=1)
    user_data = np.insert(user_data, 0, 0, axis=1)
    user_data = np.insert(user_data, 0, 0, axis=1)
    user_data = np.insert(user_data, 0, 0, axis=1)
    user_data = np.insert(user_data, 0, 0, axis=1)

    # Encoding each fighter's stance
    if temp[15] == 'orthodox':
        user_data[0][0] = 1
    elif temp[15] == 'southpaw':
        user_data[0][1] = 0
    elif temp[15] == 'switch':
        user_data[0][2] = 1
        
    if temp[35] == 'orthodox':
        user_data[0][3] = 1
    elif temp[35] == 'southpaw':
        user_data[0][4] = 0
    elif temp[35] == 'switch':
        user_data[0][5] = 1
    
    # Return percentage that blue fighter wins and winner prediction based on 50% threshold
    user_data_pred_percentage = classifier.predict(user_data) # chance that blue corner will win, expressed as a percentage
    user_data_pred = np.where(user_data_pred_percentage > 0.50, 'Blue', 'Red')
    blue_percent = (user_data_pred_percentage[0][0]) * 100
    red_percent = 100 - blue_percent
    print(f'The Artificial Neural Network predicts that the fighter in the {user_data_pred[0][0]} corner will win, while the fighter in the Blue corner has a {blue_percent:0.2f} % chance of winning and the fighter in the Red corner has a {red_percent:0.2f} % chance of winning.')
            
else:
    pass