import pandas as pd
import sys
import os
from tqdm import tqdm


def read_price_data(folder_path):
    data_frames = []
    colnames = [
        "unix",
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "volume_usdt",
        "tradecount",
    ]

    # scanning the folder and read each csv file
    for filename in tqdm(os.listdir(folder_path)):
        if os.path.splitext(filename)[1] == ".csv":
            filepath = os.path.join(folder_path, filename)
            data = pd.read_csv(
                filepath, header=0, names=colnames, dtype={"symbol": object}
            )

            data["date"] = pd.to_datetime(data["date"], format="mixed")
            data_frames.append(data)

    # concat each csv into a giant table
    full_data = pd.concat(data_frames)

    # unix is not needed so cann be dropped
    full_data = full_data.drop(columns=["unix"])

    # The symbol of crypto can be different even though they refer to the same crypto, e.g.: ADAUSDT and ADA/USDT are the same, remove the / to unify them
    full_data["symbol"] = full_data["symbol"].str.replace("/", "")

    # define features to keep
    feature_names = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "volume_usdt",
        "tradecount",
    ]

    # construct a dataframe with only date col, and append horizontally each crypto data later
    df = pd.DataFrame(full_data["date"].unique(), columns=["date"]).set_index("date")

    # construct a dataframe with each coin data as a column. We need to prefix the coin symbol to the feature of each coin so have to loop
    for symbol in full_data["symbol"].unique():
        coin_data = full_data[full_data["symbol"] == symbol].set_index("date")
        for feature in feature_names:
            new_column_name = str(symbol) + "_" + feature
            df[new_column_name] = coin_data[feature]

    df = df.dropna(how="any")

    return df


def main(input_dir, output_file):

    data = read_price_data(input_dir)
    data = data.sort_values("date", ascending=True)
    data.to_csv(output_file, sep=",", compression="gzip")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
