import sklearn
from sklearn import svm
from sklearn import datasets
from sklearn import metrics

# sklearn has few already existing datasets, in its modules
# Digits Datasets for image classification
# Boston Realestates Dataset
# Iris Datasets about flowers
# Breast Cancer Dataset
# We are gonna use Breast Cancer Dataset now

# Loading Datasets to Dataframes
cancer = datasets.load_breast_cancer()

# Printing features and targets of Breast Cancer Dataset
print(cancer.feature_names)
print(cancer.target_names)

# Seperating features and labels
x = cancer.data     # features
y = cancer.target   # labels

# Seperating Data Points Randomly into Testing Set and Training Set
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

# Create a classifier
# clf = svm.SVC()

# Using Kernel, along with parameter
clf = svm.SVC(C=2, kernel="linear")
# Fit the data to make the classifier
clf.fit(x_train, y_train)

# Predict values for our test data
y_pred = clf.predict(x_test)
# Test them against correct values
acc = metrics.accuracy_score(y_test, y_pred)

# Print the Accuracy
print(acc)
