import sys
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from utils.visualize_utils import visualize_lag_plot, visualize_plot
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error


def reverse_price_feature(y, x_0):
    """
    Given a diff feature and the initial point x_0, return the price feature
    Args:
        y:      The diff feature
        x_0:    The initial price point
    Returns:
        price_y: The price at x_1
    """

    price_y = y.copy()

    # adding the first price point
    price_y = price_y + x_0

    return price_y


def transform_diff(X, y):
    """
    Given the X price features and y groundtruth price, this function returns the difference between current price and the one-step previous price
    Args:
        X:      The input price features
        y:      The output price desired
    Returns:
        X_diff, y_diff: The differential price features for training
    """

    X_diff = X.copy()
    y_diff = y.copy()

    # calculate the X difference between today and the previous day
    X_diff.iloc[:, 2:] = X.iloc[:, 2:] - X.iloc[:, 2:].copy().shift(1)

    # calculate the y difference between today and the previous day
    y_diff = y - y.copy().shift(1)

    X_diff = X_diff.dropna(how="any")
    y_diff = y_diff.dropna(how="any")

    return X_diff, y_diff


def plot_and_print_result(
    model,
    X_train,
    X_valid,
    y_train,
    price_y_train,
    y_valid,
    price_y_valid,
    model_name="",
):

    # get output from the models and reverse them into price features
    y_train_pred = model.predict(X_train.iloc[:, 2:])
    price_y_train_pred = reverse_price_feature(y_train_pred, X_train.iloc[:, 1])
    y_valid_pred = model.predict(X_valid.iloc[:, 2:])
    price_y_valid_pred = reverse_price_feature(y_valid_pred, X_valid.iloc[:, 1])

    # print the evaluation metrics
    print(
        model_name + " train error",
        mean_squared_error(price_y_train, price_y_train_pred, squared=False),
    )
    print(
        model_name + " valid error",
        mean_squared_error(price_y_valid, price_y_valid_pred, squared=False),
    )

    # plot the output diff prediction
    visualize_plot(
        X_train["date"].iloc[:100],
        y_train.iloc[:100],
        y_train_pred[:100],
        model_name + " Single Step Diff Prediction on Train Set",
    )
    visualize_plot(
        X_valid["date"].iloc[:100],
        y_valid.iloc[:100],
        y_valid_pred[:100],
        model_name + " Single Step Diff Prediction on Validation Set",
    )

    # plot the price prediction
    visualize_plot(
        X_train["date"].iloc[100:200],
        price_y_train.iloc[100:200],
        price_y_train_pred[100:200],
        model_name + " Single Step Price Prediction on Train Set",
    )

    visualize_plot(
        X_valid["date"].iloc[100:200],
        price_y_valid.iloc[100:200],
        price_y_valid_pred[100:200],
        model_name + " Single Step Price Prediction on Validation Set",
    )


def prepare_data(data, input_lag=1):
    X = data[
        ["date", "BTCUSDT_close"]
    ]  # preseve the date and one BTCUSDT_close column in the first cols for easier indexing, since the number of columns is arbitrary so ignore the name and indexing is better
    y = []

    # iteratively add #input_lag columns from 0,1,...,input_lag date before the current date
    for i in range(input_lag):
        lag_data = data.iloc[:, 1:].shift(i)
        X = pd.concat([X, lag_data], axis=1)

    X = X.iloc[input_lag - 1 :]

    # use the X to align when shifting y, y is one step in the future of X.iloc[:, 1]
    X["y"] = X.iloc[:, 1].shift(-1)
    X = X[X["y"].notnull()]
    y = X["y"]
    X = X.drop(columns=["y"])

    X, y = transform_diff(X, y)

    return X, y


def main(file_name, input_lag):
    data = pd.read_csv(
        file_name,
        parse_dates=["date"],
    )

    X, y = prepare_data(data, input_lag)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, shuffle=False
    )  # since we have limited data, shuffle = False so the model can generalize on better

    price_y_train = reverse_price_feature(y_train, X_train.iloc[:, 1])
    price_y_valid = reverse_price_feature(y_valid, X_valid.iloc[:, 1])

    # KneighborsRegressor
    KNN_model = make_pipeline(
        StandardScaler(), KNeighborsRegressor(n_neighbors=20)
    ).fit(X_train.iloc[:, 2:], y_train)

    plot_and_print_result(
        KNN_model,
        X_train,
        X_valid,
        y_train,
        price_y_train,
        y_valid,
        price_y_valid,
        "KNN",
    )

    # RandomForrest
    rf_model = make_pipeline(StandardScaler(), RandomForestRegressor()).fit(
        X_train.iloc[:, 2:], y_train
    )

    plot_and_print_result(
        rf_model,
        X_train,
        X_valid,
        y_train,
        price_y_train,
        y_valid,
        price_y_valid,
        "Random Forrest",
    )

    # Neural network
    mlp_model = make_pipeline(
        StandardScaler(), MLPRegressor((100, 100), max_iter=5000)
    ).fit(X_train.iloc[:, 2:], y_train)

    plot_and_print_result(
        mlp_model,
        X_train,
        X_valid,
        y_train,
        price_y_train,
        y_valid,
        price_y_valid,
        "Neural Network",
    )

    return 0


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
