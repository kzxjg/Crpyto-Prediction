import sys
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from utils.visualize_utils import visualize_lag_plot, visualize_plot
from sklearn.metrics import mean_squared_error


def prepare_data(data, input_lag=1, output_lag=1):
    X = data[
        ["date", "BTCUSDT_close"]
    ]  # preseve the date and one BTCUSDT_close column in the first cols for easier indexing, since the number of columns is arbitrary so ignore the name and indexing is better
    y = []

    # iteratively add #input_lag columns from 0,1,...,input_lag date before the current date
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

    return X, y


def plot_and_print_result(
    model,
    X_train,
    X_valid,
    y_train,
    y_valid,
    model_name="",
):
    # get prediction output for train and valid set
    y_train_pred = model.predict(X_train.iloc[:, 2:])
    y_valid_pred = model.predict(X_valid.iloc[:, 2:])

    # print evaluation metrircs
    print(
        model_name + " train error",
        mean_squared_error(y_train, y_train_pred, squared=False),
    )
    print(
        model_name + " valid error",
        mean_squared_error(y_valid, y_valid_pred, squared=False),
    )

    # plot the prediction and the groundtruth
    visualize_lag_plot(
        X_train,
        y_train,
        y_train_pred,
        model_name + " Multiple Step Price Prediction on Train Set",
    )
    visualize_lag_plot(
        X_valid,
        y_valid,
        y_valid_pred,
        model_name + " Multiple Step Price Prediction on Validation Set",
    )


def main(file_name, input_lag, output_lag):
    data = pd.read_csv(
        file_name,
        parse_dates=["date"],
    )

    assert output_lag > 1

    X, y = prepare_data(data, input_lag, output_lag)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, shuffle=False)

    # KneighborsRegressor
    KNN_model = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors= 10)).fit(
        X_train.iloc[:, 2:], y_train
    )

    plot_and_print_result(KNN_model, X_train, X_valid, y_train, y_valid, "KNN")

    # RandomForrest
    rf_model = make_pipeline(StandardScaler(), RandomForestRegressor()).fit(
        X_train.iloc[:, 2:], y_train
    )

    plot_and_print_result(
        rf_model, X_train, X_valid, y_train, y_valid, "Random Forrest"
    )

    # Neural network
    mlp_model = make_pipeline(
        StandardScaler(), MLPRegressor((100, 100), max_iter=5000)
    ).fit(X_train.iloc[:, 2:], y_train)

    plot_and_print_result(
        mlp_model, X_train, X_valid, y_train, y_valid, "Neural Network"
    )

    return 0


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
