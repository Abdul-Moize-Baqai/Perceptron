import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Perceptron(object):

    def __init__(self, eta=0.1, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter

    def weighted_sum(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def predict_proba(self, X):
        return self.sigmoid(self.weighted_sum(X))

    def predict(self, X):
        return np.where(self.predict_proba(X) >= 0.5, 1, 0)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0

            for xi, target in zip(X, y):
                y_hat = self.predict_proba(xi)

                update = self.eta * (target - y_hat)

                self.w_[1:] += update * xi
                self.w_[0] += update

                errors += int(self.predict(xi) != target)

            self.errors_.append(errors)

        return self

def read_float(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("  Please enter a valid number.")

def main():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

    try:
        df = pd.read_csv(url, header=None)
        print("Dataset loaded from UCI repository.")
    except Exception as e:
        print("Could not download the dataset (", e, ")")
        print("Using scikit-learn's built-in Iris dataset instead.")
        from sklearn.datasets import load_iris
        iris = load_iris()
        df = pd.DataFrame(iris.data)
        names = np.array(["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
        df[4] = names[iris.target]

    df = shuffle(df)
    print(df.head())

    X = df.iloc[:, 0:4].values
    y = df.iloc[:, 4].values
    print(X[0:5])
    print(y[0:5])

    train_data, test_data, train_labels, test_labels = train_test_split(
        X, y, test_size=0.25
    )

    train_labels = np.where(train_labels == "Iris-setosa", 1, 0)
    test_labels = np.where(test_labels == "Iris-setosa", 1, 0)

    print("Train data:", train_data[0:2])
    print("Train labels:", train_labels[0:2])
    print("Test data:", test_data[0:2])
    print("Test labels:", test_labels[0:2])

    perceptron = Perceptron(eta=0.1, n_iter=50)
    perceptron.fit(train_data, train_labels)

    print("\nBias (w0):", perceptron.w_[0])
    print("Feature weights (w1..w4):", perceptron.w_[1:])
    print("Errors per epoch:", perceptron.errors_)

    test_preds = perceptron.predict(test_data)
    print("\nTest predictions:", test_preds)

    accuracy = accuracy_score(test_labels, test_preds)
    print("Accuracy: {:.2f}%".format(accuracy * 100))

    print("\n--- Manual prediction ---")
    sepal_length = read_float("Sepal length (cm): ")
    sepal_width = read_float("Sepal width  (cm): ")
    petal_length = read_float("Petal length (cm): ")
    petal_width = read_float("Petal width  (cm): ")

    sample = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prob = perceptron.predict_proba(sample)[0]
    pred = perceptron.predict(sample)[0]

    print("\nSigmoid output (probability of Iris-setosa): {:.4f}".format(prob))
    if pred == 1:
        print("Prediction: 1 -> Iris-setosa")
    else:
        print("Prediction: 0 -> NOT Iris-setosa (Iris-versicolor or Iris-virginica)")

if __name__ == "__main__":
    main()