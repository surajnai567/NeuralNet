import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix

location = "C:\\Users\\ani\\Desktop\\dataset\\diabetes.csv"
df = pd.read_csv(location)

#data processing

df["BMI"] = df["BMI"].replace(0, np.nan)
df["Insulin"] = df["Insulin"].replace(0, np.nan)
df['Glucose'] = df['Glucose'].replace(0, np.nan)
df['BloodPressure'] = df['BloodPressure'].replace(0, np.nan)
df['SkinThickness'] = df['SkinThickness'].replace(0, np.nan)

#filling nan with mean

df['Glucose'] = df['Glucose'].fillna(df['Glucose'].mean())
df['BloodPressure'] = df['BloodPressure'].fillna(df['BloodPressure'].mean())
df['SkinThickness'] = df['SkinThickness'].fillna(df['SkinThickness'].mean())
df['Insulin'] = df['Insulin'].fillna(df['Insulin'].mean())
df['BMI'] = df['BMI'].fillna(df['BMI'].mean())

# data standarlization
preprocessed = preprocessing.scale(df)

# scale doest return DF
pre_df = pd.DataFrame(preprocessed, columns=df.columns)

# adding target variable
pre_df["Outcome"] = df["Outcome"]

# splitting the data into traing testingset
x = pre_df.loc[: , df.columns != 'Outcome']
y = pre_df.loc[: , 'Outcome']

# 80:20
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8)
X_train, X_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2)

# creating model
model = Sequential()

# first layer
model.add(Dense(32, activation="relu", input_dim=8))

# second layer
model.add(Dense(32, activation="relu"))

# final layer
model.add(Dense(1, activation="sigmoid"))

# defining training process
# by compiling model

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model for 200 epochs
model.fit(X_train, y_train, epochs=200)

# test model
scores = model.evaluate(X_train, y_train)
print("Training Accuracy: %.2f%%\n" % (scores[1]*100))

scores = model.evaluate(x_test, y_test)
print("Testing Accuracy: %.2f%%\n" % (scores[1]*100))

# drawing confusion matrix
y_test_pred = model.predict_classes(x_test)

confu = pd.DataFrame(confusion_matrix(y_test, y_test_pred), columns=["negative", "positive"], index=["NODB", "DB"])
print(confu)
