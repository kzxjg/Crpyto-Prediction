# The raw csv dataset usually have the same schema in all of them however, there is a specific file has additionally redundant columns should be removed
import pandas as pd
import sys
import os
from tqdm import tqdm


def main(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for filename in tqdm(os.listdir(input_dir)):
        file_path = os.path.join(input_dir, filename)  # Replace with your file path
        output_file_path = os.path.join(output_dir, filename)

        with open(file_path, "r") as file:
            lines = file.readlines()

        # Remove the first line, it is "https://www.CryptoDataDownload.com"
        lines = lines[1:]

        # Write the modified content back to the original file
        with open(output_file_path, "w") as file:
            file.writelines(lines)

        if (
            filename == "Binance_BTCUSDT_2021_minute.csv"
        ):  # for this specific file, there will be additional columns like date_close,close_unix which inconsistent with others
            data = pd.read_csv(output_file_path)

            data = data[
                [
                    "unix",
                    "date",
                    "symbol",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "volume_from",
                    "tradecount",
                ]
            ]  # only keep the neccessary columns

            data.to_csv(output_file_path, index=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
