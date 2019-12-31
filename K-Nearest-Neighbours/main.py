from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import sklearn
import pandas as pd

# Creating a dataframe out of the given data using pandas
# It is more like cleaning and organising data for further use
path = "C:\Creed Stone\Python Projects\Machine Learning Models\Machine-Learning-Models\K-Nearest-Neighbours\car.data"
data = pd.read_csv(path)

# The data we have is not all numeric
# To Train KNeighborsClassifier we need to convert string data
# to some kind of a number

# For that we are going to use a Method from sklearn module

# creating a label encoder object and then use that to encode
# each column of our data into integers.
le = preprocessing.LabelEncoder()

# The method fit_transform() takes a list (each of our columns)
# and will return to us an array containing our new values.
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cl = le.fit_transform(list(data["class"]))

# Now dividing our data again into Labels and Features
# We shall take all except "class" as Labels

# zip method binds single arrays into one array
X = list(zip(buying, maint, door, persons, lug_boot, safety))  # features
y = list(cl)  # labels

# Splitting the data randomly for training and testing, 90%-10%
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# Creating the KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=9)

# Feeding the Algorithm with the Data
model.fit(x_train, y_train)

# Checking the accuracy score
acc = model.score(x_test, y_test)
print(acc)

# Testing our MODEL

predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])
    n = model.kneighbors([x_test[x]], 9, True)
    print("N: ", n)

# This will display the predicted class, our data and the actual class
# We create a names list so that we can convert our integer predictions into
# their string representation
# Now we will we see the neighbors of each point in our testing data
