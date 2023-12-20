import sys
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from ml_diff_features import transform_diff
from utils.visualize_utils import visualize_lag_plot, visualize_plot
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error


def reverse_price_feature(y, x_0):
    """
    Given a diff feature and the initial point x_0, return the price feature
    Args:
        y:      The diff features
        x_0:    The initial price point
    Returns:
        price_y: The price at [x_1, x_2, x_3, ...]
    """

    price_y = y.copy()

    # accumulate the diff, output [d_0, d_0 + d_1, d_0 + d_1 + d_2, ...]
    price_y = price_y.cumsum(axis=1)

    # adding the first price point, output [x_0 + d_0, x_0 + d_0 + d_1, x_0 + d_0 + d_1 + d_2, ...]
    price_y = price_y + np.expand_dims(x_0.values, axis=1)

    return price_y


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

    # plot some data points showing the prediction
    visualize_lag_plot(
        X_train,
        price_y_train,
        price_y_train_pred,
        model_name + " Multiple Step Price Prediction on Train Set",
    )

    visualize_lag_plot(
        X_valid,
        price_y_valid,
        price_y_valid_pred,
        model_name + " Multiple Step Price Prediction on Validation Set",
    )


def prepare_data(data, input_lag=1, output_lag=1):
    # preseve the date and one BTCUSDT_close column in the first cols for easier indexing, since the number of columns is arbitrary so ignore the name and indexing is better
    X = data[["date", "BTCUSDT_close"]]
    y = []

    # iteratively add #input_lag columns from x_0,x_-1,...,x_-input_lag date before the current date
    for i in range(input_lag):
        lag_data = data.iloc[:, 1:].shift(i)
        X = pd.concat([X, lag_data], axis=1)

    # filter out the rows that do not have all the features, since we shifted at most input_lag date, there will be input_lag rows on the top to be removed
    X = X.iloc[input_lag - 1 : -output_lag]

    # iteratively add #output_lag columns from x_1,x_2,..., the next output lag date after the current date
    for i in range(output_lag):
        y.append(data["BTCUSDT_close"].shift(-i - 1))

    y = pd.concat(y, axis=1)

    # filter out the rows that do not have all the features, since we shifted at most output_lag date, there will be output_lag rows on the bottom to be removed
    y = y.iloc[input_lag - 1 : -output_lag]

    # transform into diff
    X, y = transform_diff(X, y)

    return X, y


def main(file_name, input_lag, output_lag):
    data = pd.read_csv(
        file_name,
        parse_dates=["date"],
    )

    assert output_lag > 1

    X, y = prepare_data(data, input_lag, output_lag)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, shuffle=False
    )  # since we have limited data, shuffle = False so we can evaluate

    # reverse difference groundtruth back to price
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
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
