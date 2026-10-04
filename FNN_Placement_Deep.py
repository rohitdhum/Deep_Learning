###################################################
# Deep Learning Pipeline
###################################################
# 1. Read the Data from CSV
# 2. Data Analysis (EDA)
# 3. Preprocessing
# 4. Train Test Split
# 5. Feature Scaling
# 6. FNN Model traing
# 7. Model Evaluation 
# 8. Graphical Representation
# 9. Model Preserve
# 10. Model loading and Preserve
# 11. Test unseen data
###################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

###################################################
# 1. Read the Data from CSV
###################################################

print("1. Read the Data from CSV")

data = pd.read_csv("placement_data.csv")

print("Complete Dataset :")
print(data)

###################################################
# 2. Data Analysis (EDA)
###################################################

print("2. Data Analysis (EDA)")

print("First 5 rows :")
print(data.head())

print("Column names :")
print(data.columns)

print("Shape of Dataset :")
print(data.shape)

print("Statistical Sammary :")
print(data.describe())

###################################################
# 3. Preprocessing
###################################################

print("3. Preprocessing")

X = data[['Aptitude', 'Coding', 'Communication', 'Academics', 'Internship']]
Y = data['Placed']

print("Input Feature :")
print(X.head())

print("Target :")
print(Y.head())

###################################################
# 4. Train Test Split
###################################################

print("4. Train Test Split")

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=42)

print("Training Input Shape :", X_train.shape)
print("Testing Input Shape :", X_test.shape)
print("Training Output Shape :", Y_train.shape)
print("Testing Output Shape :", Y_test.shape)

###################################################
# 5. Feature Scaling
###################################################

print("5. Feature Scaling")

scalar = StandardScaler()

X_train_scaled = scalar.fit_transform(X_train) 
X_test_scaled =  scalar.fit_transform(X_test)

print("Scaled Training data :")
print(X_train_scaled[:5])

###################################################
# 6. FNN Model traing
###################################################

print("6. FNN Model traing")

model = MLPClassifier(
    hidden_layer_sizes=(8,4),
    activation='relu',
    solver="adam",
    max_iter=1000,
    random_state=42
)

print(model)

print("Train the model")

model.fit(X_train_scaled, Y_train)

print("Model training completed")

###################################################
# 7. Model Evaluation 
###################################################

print("7. Model Evaluation")

Y_pred = model.predict(X_test_scaled)

Accuracy = accuracy_score(Y_test, Y_pred)

print("Accuracy is :", Accuracy)

cm = confusion_matrix(Y_test, Y_pred)

print("Consusion Matrix :",cm)

print("Predict the probabilty :")

Y_prob = model.predict_proba(X_train_scaled)

print(Y_prob[:5])

###################################################
# 9. Model Preserve 
###################################################

print("9. Model Preserve")

joblib.dump(model, "placement_fnn_model.pkl")
joblib.dump(scalar,"placement_scalar.pkl")

print("Model and Scalar gest dump successfully")

###################################################
# 10. Model loading and Preserve
###################################################

print("10. Model loading and Preserve")

loaded_model = joblib.load("placement_fnn_model.pkl")
loaded_scalar = joblib.load("placement_scalar.pkl")

print("Model is loaded successfully")

###################################################
# 11. Test unseen data
# Apptitute :      70
# Coding :         75
# Communication :  80 
# Acadamics :      85
# Internship :     1
###################################################

new_student = pd.DataFrame([[70,75,80,85,1]], columns=['Aptitude', 'Coding', 'Communication', 'Academics', 'Internship'])

new_student_scalad = loaded_scalar.transform(new_student)

new_prediction = loaded_model.predict(new_student_scalad)

new_probability = loaded_model.predict_proba(new_student_scalad)

print("New Students Data :")
print(new_student)

print("Prediction Probability :", new_probability)

if new_prediction[0] == 1:
    print("Prediction : Placed")
else:
    print("Prediction : Not Placed")