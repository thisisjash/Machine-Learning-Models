import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import pickle
import matplotlib.pyplot as plt
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")
# Data in the CSV file is seperated by semi colons, so we are using it

# print(data) -----> prints the whole CSV file to terminal
# print(data.head()) ------> prints first 5 rows of CSV file

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
# This trims down the data to only the columns mentioned

# We are seperating our data such that, to predict a attribute(column)
# We are going to Predict G3 attribute using the other 5 attributes
# The attribute to be predicted is called "label"
# The attribute that will determine our label are known as "features"

predict = "G3"

X = np.array(data.drop([predict], 1))
# The features, here are all of the "data", except predict(G3)
y = np.array(data[predict])
# The label, here is just the predict(G3) column of "data"

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# We are splitting the testing data and training data as 90%-10%

linear = linear_model.LinearRegression()
# This essentially creates the machine learning model(resembles a line), we want

linear.fit(x_train, y_train)
# We are passing the training data to make the regression line

acc = linear.score(x_test, y_test)
# We are predicting how accurate it is using score function

print(acc)
# This prints around, 86% accuracy... The results vary for succesive implementations

# For this dataset, out machine learning model just took a few seconds
# to train the dataset. But, IRL when we deal with huge datasets
# It takes hours and hours to train the data. We just can't train the datasets
# Everytime. Hence, we save our ML model using Pickles

# We are saving our ML Model in "trainedStudentModel.pickle" file
with open("trainedStudentModel.pickle", "wb") as f:
    pickle.dump(linear, f)

# Our models vary in accuracy. This is because when we split the
# data into training and testing data it is divided differently each
# time. Since our model trains very quickly it may be worth training
# multiple models and saving the best one.

# TRAIN MODEL MULTIPLE TIMES FOR BEST SCORE, 100 in this case
# best = 0
# for _ in range(100):
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
#
#     linear = linear_model.LinearRegression()
#
#     linear.fit(x_train, y_train)
#     acc = linear.score(x_test, y_test)
#     print("Accuracy: " + str(acc))
#
#     # If the current model has a better score than one we've already trained then save it
#     if acc > best:
#         best = acc
#         with open("studentgrades.pickle", "wb") as f:
#             pickle.dump(linear, f)

# We are using our saved ML model as "linear2" again here
pickle_in = open("trainedStudentModel.pickle", "rb")
linear2 = pickle.load(pickle_in)

# To print the attributes of the created line
print('Coefficient: \n', linear.coef_)
# This prints the slope attributes of the line
# Since there are 5 attributes we got 5 results
print('Intercept: \n', linear.intercept_)
# This prints the intercept of the line

# To compare the predictions to the actual results
predictions = linear.predict(x_test)
# Gets a list of all predictions for the input features

# len(any array) gives length of an array
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

# Drawing and plotting model
plot = "G1"
# Change this to G1, G2, studytime or absences to see other graphs
plt.scatter(data[plot], data["G3"])
# Creates the Graph
plt.legend(loc=4)
plt.xlabel(plot)
# Assigns the X Axis Label
plt.ylabel("Final Grade")
# Assigns the Y Axis Label
plt.show()
# Prints the Graph
