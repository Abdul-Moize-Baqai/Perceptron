# Perceptron

A from-scratch implementation of a Perceptron classifier, built for a Deep Learning course assignment. It uses a **sigmoid activation function** instead of the classic step function, and predicts whether an Iris flower is **Iris-setosa (1)** or **not (0)**.

## Overview

This project implements a `Perceptron` class in NumPy — no machine learning libraries used for the model itself — trained on the classic [Iris dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data).

**Key features:**
- Weighted sum computed manually (`w · x + bias`)
- Sigmoid activation instead of a step function
- Binary labels: `1` for Iris-setosa, `0` for everything else
- Trained with online (per-sample) gradient updates
- Manual input mode: enter flower measurements and get a live prediction

## How it works

1. **Load data** — reads the Iris dataset from the UCI repository (falls back to scikit-learn's built-in copy if offline).
2. **Preprocess** — shuffles the data, splits it 75/25 into train/test sets, and encodes labels as `1` (setosa) / `0` (not setosa).
3. **Train** — the `Perceptron` class updates its weights and bias over multiple epochs using the rule:

   ```
   update = eta * (y - y_hat)
   w = w + update * x
   bias = bias + update
   ```

   where `y_hat` is the sigmoid output of the weighted sum.
4. **Evaluate** — predictions are made on the held-out test set and accuracy is reported.
5. **Manual prediction** — the script prompts for sepal length, sepal width, petal length, and petal width, then prints the predicted class.

## Usage

```bash
python "Assignment DL/Perceptron.py"
```

You'll need:
```bash
pip install numpy pandas scikit-learn
```

Example output:
```
Accuracy: 100.00%

--- Manual prediction ---
Sepal length (cm): 5.1
Sepal width  (cm): 3.5
Petal length (cm): 1.4
Petal width  (cm): 0.2

Sigmoid output (probability of Iris-setosa): 0.9993
Prediction: 1 -> Iris-setosa
```

## Dataset

The [Iris dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data) contains 150 samples across three species (*Iris-setosa*, *Iris-versicolor*, *Iris-virginica*), each described by four features: sepal length, sepal width, petal length, and petal width. Since Iris-setosa is linearly separable from the other two species, the Perceptron typically achieves 100% test accuracy on this task.

## Author

Abdul Moize Baqai
