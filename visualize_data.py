from pykalman import KalmanFilter
import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
import sys
import seaborn

seaborn.set()


def loess_smooth(data, symbol):

    # smooth using lowess
    smoothed = lowess(data[symbol + "_close"], data["date"], frac=0.05)

    # plot price and smooth price
    plt.clf()
    plt.plot(data["date"], data[symbol + "_close"], "b-")
    plt.plot(data["date"], smoothed[:, 1], "r-")
    plt.xlabel("Time")
    plt.ylabel("Price in USD")
    plt.legend(["Price", "Smoothed Price"])
    plt.show()


def main(filename, symbol):
    data = pd.read_csv(filename, parse_dates=["date"])
    loess_smooth(data, symbol)
    return 0


if __name__ == "__main__":
    filename = sys.argv[1]
    symbol = sys.argv[2]
    main(filename, symbol)
