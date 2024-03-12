# Import required modules
import numpy as np
from sklearn.linear_model import LogisticRegression


# Define the model
def log_reg_model(X, y):
    """
    Defines a Logistic Regression Models and uses it to predict target
    Args:
        X (ndarray (m, n)) : Data, m examples with n features
        y (ndarray (m,))   : target values
    Returns:
        lr_accuracy (scalar) : Accuracy of the LR model in predicting y

    """
    lr_model = LogisticRegression()
    lr_model.fit(X, y)
    y_pred = lr_model.predict(X)
    lr_accuracy = lr_model.score(X, y)

    return lr_accuracy


if __name__ == "__main__":
    X = np.array([[0.5, 1.5], [1, 1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
    y = np.array([0, 0, 0, 1, 1, 1])

    accuracy = log_reg_model(X, y)

    print(f"Accuracy on training set: {accuracy} ")
