from sklearn.preprocessing import scale
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn import metrics

# Loading the dataset into the dataframe
digits = load_digits()

# We use scale method to scale our data down
data = scale(digits.data)
y = digits.target

# We define the amount of clusters by creating a variable k
k = 10
# We define how many samples and featurres we have by getting
# the dataset shape
samples, features = data.shape


def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))


# Training the model
clf = KMeans(n_clusters=k, init="random", n_init=10)
bench_k_means(clf, "1", data)
