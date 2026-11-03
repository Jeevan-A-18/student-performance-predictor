# model.py
# ------------------------------------------------------------------
# This file handles everything related to Machine Learning:
#   - Loading the dataset
#   - Splitting data into training and testing sets
#   - Training the Linear Regression model
#   - Making predictions
#   - Evaluating model performance
# ------------------------------------------------------------------

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


def load_dataset(filepath="dataset.csv"):
    """
    Load the student dataset from a CSV file.
    Returns a Pandas DataFrame with columns: study_hours, marks
    """
    df = pd.read_csv(filepath)
    return df


def prepare_data(df):
    """
    Separate the dataset into Features (X) and Labels (y).

    - Feature (X): study_hours  → the input we give to the model
    - Label   (y): marks        → the output we want the model to predict

    We reshape X into a 2-D array because scikit-learn expects that shape.
    """
    X = df[["study_hours"]]   # Feature — double brackets keep it as a DataFrame (2-D)
    y = df["marks"]           # Label   — single bracket gives a Series (1-D)
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into Training set and Testing set.

    - Training set (80 %): used to teach the model
    - Testing  set (20 %): used to evaluate how well the model learned

    random_state=42 makes the split reproducible (same result every run).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Train a Linear Regression model.

    Linear Regression finds the best-fit straight line through the data
    by learning two numbers:
        - slope (m)     : how much marks increase per extra study hour
        - intercept (b) : the predicted marks when study hours = 0

    The line equation is:  marks = m × study_hours + b
    """
    model = LinearRegression()
    model.fit(X_train, y_train)   # ← this is where "learning" happens
    return model


def evaluate_model(model, X_test, y_test):
    """
    Measure how accurate the trained model is on unseen test data.

    Returns:
        mae   – Mean Absolute Error   : average error in marks (lower is better)
        r2    – R² Score              : how well the model explains the data
                                        (1.0 = perfect, 0.0 = no better than guessing)
        y_pred – predicted marks for the test set
    """
    y_pred = model.predict(X_test)
    mae    = mean_absolute_error(y_test, y_pred)
    r2     = r2_score(y_test, y_pred)
    return mae, r2, y_pred


def predict_marks(model, study_hours):
    """
    Predict examination marks for a given number of study hours.

    study_hours : float  (e.g. 6.5)
    Returns     : float  (predicted marks, clipped to 0–100 range)
    """
    # scikit-learn expects a 2-D input with the same column name used during training
    input_df  = pd.DataFrame({"study_hours": [study_hours]})
    predicted = model.predict(input_df)[0]

    # Clamp the result so it never goes below 0 or above 100
    predicted = max(0, min(100, predicted))
    return round(predicted, 2)


def get_regression_line_points(model, df):
    """
    Generate (x, y) points that describe the fitted regression line,
    so it can be drawn on a scatter plot.
    """
    import numpy as np
    x_line = numpy_linspace(df["study_hours"].min(), df["study_hours"].max())
    y_line = model.predict(x_line.reshape(-1, 1))
    return x_line, y_line


# Small helper to avoid importing numpy at the top level unnecessarily
def numpy_linspace(start, stop, num=100):
    import numpy as np
    return np.linspace(start, stop, num)
