# Forecasting Bitcoin Price using Machine Learning

## Introduction

Cryptocurrency is well-known as a volatile and high-risk, high-return asset. The ability to predict the market's movement for the next hour or day is incredibly valuable, aiding investors in mitigating losses and potentially achieving substantial profits. With the use of fundamental machine learning algorithms, this project reflects our attempt to forecast the price of Bitcoin or the larger cryptocurrency market. Our goal is to explore the existing states and challenges associated with time series forecasting problems in this domain.

---

## Datasets

The dataset we used contained daily cryptocurrency prices from the Binance exchange, found on [https://www.CryptoDataDownload.com](https://www.CryptoDataDownload.com). CryptoDataDownload provides datasets for a plethora of different coins; however, we only selected a few which had pronounced market cap for our analysis and prediction tasks, such as **BTC, ETH, ADA, BNB, and DOGE**. We chose these additional coins because their prices often correlate with Bitcoin, and money flow to other coins may cause Bitcoin's price to alter.

The dataset contains:
* **Unix**: Timestamp indicating the record's time
* **Symbol**: Cryptocurrency symbol (e.g., BTCUSDT, ETHUSDT, etc.)
* **Open Price**: Beginning-of-day price
* **Close Price**: End-of-day price
* **Low Price**: Minimum price observed during the day
* **Highest Price**: Maximum price observed during the day
* **Volume**: Trading volume within the day measured in cryptocurrency units
* **Volume of USDT**: Trading volume within the day measured in US Dollars
* **Tradecount**: Total count of buy and sell orders executed

---

## Data Preparation

### Data Cleaning

First, we needed to unify all the CSVs into a readable format. Every CSV file was marked with a sentence "https://www.CryptoDataDownload.com" as the header, so we removed it and saved the files. The CSV file "Binance_BTCUSDT_2021_minute.csv" contained several columns like "marketorder_volume," "marketorder_volume_from," "date_close," and "close_unix." However, due to inconsistencies, these lacked relevance to our analysis and were removed rather than attempting to fill missing values or average them, which could lead to inaccurate results. After this process, we had a similar number of files to our raw dataset, but they were much cleaner and readily readable by the `read_csv` function.

### Data Preprocessing

To speed up the procedure without manually going through each individual crypto dataset, we merged separate cryptocurrency dataframes into a single one. Each column was prefixed with its appropriate symbol (e.g., `ADAUSDT_open`, `ETHUSDT_tradecount`). We then removed the `Unix` timestamp and `Symbol` columns as they were unnecessary. Finally, we saved the CSV files using `gz` compression to optimize file sizes.

### Feature Selection

We used **open price, close price, low price, high price, volume, volume in USD, and trade count** as our main features.

While we initially experimented with daily, hourly, and minute datasets, all three showed identical patterns in trends. Due to resource and framework limitations, we decided to run our experiments using only the **daily dataset**. This choice also benefits from the daily dataset capturing a wider view of market trends, being less prone to trading noise, and generally being more useful for investors.

Our ultimate goal is to predict the **Bitcoin closing price for the next day** using this daily dataset. To incorporate historical context, we implemented **sequence input features**, meaning the model can take data from multiple days to predict the next day. We achieved this by creating duplications of features in the dataframe and shifting each column according to its relative date to the current one.

---

## Machine Learning Experiments

### Evaluation Metric

Since this is a regression problem, **Root Mean Square Error (RMSE)** will be the metric to evaluate our models. RMSE calculates the Euclidean distance between the ground truth price and the predicted price. Lower RMSE values indicate better model performance for predicting continuous values.

### Ordinary Features

We started by taking all selected features and feeding them as input to our models. We then deployed and experimented with models ranging from simple to advanced, such as **KNeighborsRegressor (KNN), Random Forests (RF), and Neural Network (MLP)** to evaluate which performed best.

To ensure scale uniformity across all variables, we used the `MinMaxScaler` to normalize the features. This prevents features with higher values (like pricing and volume in USD) from dominating and causing disproportionate effects on the models.

In time series data, the `shuffle` option in `train_test_split` can reduce generalization because shuffling before splitting allows train and validation sets to contain similar data points, preventing a true evaluation of performance on unseen data. To test this, we split our dataset into three portions: training, validation, and testing. We first constructed the testing set with `shuffle=False`, then split the remaining data into validation and training sets with `shuffle=True`. We observed significant differences: train, validation, and test errors were 1533.97, 1727.48, and 2884.81 (USD) respectively. This showed that the validation set had interlacing data points with the training set, making them appear similar, and despite good validation performance, the model performed poorly on new data. Therefore, keeping `shuffle` off is crucial for properly evaluating time series models, assuming we don't know the future.

Following testing with a single input sequence, the **MLP model** yielded the best results with an **RMSE validation error of about 708.8003 (USD)**. While this error might seem acceptable for unpredictable assets like BTC, a closer look at the graph revealed that the predictions consistently lagged behind the actual values by approximately 1 unit of time.

Expanding input sequence length for the neural network model:
* With a 2-day input sequence, the validation error was **755.58458**.
* Extending to a 3-day input sequence yielded a validation error of **806.87314**.
* Further increasing to a 5-day input sequence resulted in a validation error of **929.24621**.

This was counter-intuitive, as we expected more prior information to lead to better predictions by capturing trends more effectively. However, it appears that more inputs tended to introduce noise, confusing the model and hindering its learning. This might be because the models we experimented with were not complex enough, and we were limited by resources from using a very large hidden layer in the MLP.

### Differential Features

Instead of using the original features, we replaced them by calculating the **differences between the current day's features and the previous day's features**. This was achieved by shifting the entire dataframe by one step and subtracting it from the original.

For normalization with these differential features, we decided to use `StandardScaler`, as it tends to yield better results in such cases.

After experimenting with a single input sequence, our best valid score with the smallest generalization gap was achieved by the **Random Forest model**, with a train error of **871.4660 (USD)** and a validation error of **783.05 (USD)**. This makes sense, as difference values might be easier for decision-tree-based models to process, potentially allowing them to establish if-else rules where large price changes tend to continue in the same direction for several future steps.

Differential features seemed to improve all models, resulting in validation errors that were lower by around **20-30%**. However, we noticed some models (specifically Neural Network) attempted to circumvent the prediction problem by repeating the previous step's price, resulting in an output difference close to 0. Although the validation error of MLP in this case was **584.0263**, which was an improvement from using ordinary features, the predictions were unusable.

Expanding input sequence length for the random forest model:
* With a 2-day input sequence, the validation error was **789.75150**.
* Extending to a 3-day input sequence yielded a validation error of **787.97543**.
* Further increasing to a 5-day input sequence resulted in a validation error of **796.58988**.

### Multi-step Time Series Forecasting Trials

In addition to using multiple inputs, we tried allowing the models to output predictions for multiple "next days" to see how far we could push their capabilities. Multi-step time series forecasting can be formulated as multiple-output regression models. The results we obtained were not accurate, and as the number of outputs increased, the RMSE increased significantly. Regardless of how we expanded the model trying to overfit the data, it proved incapable of accurate multi-step predictions. For example, one result involved using 10 days of input to predict 5 future days using Random Forest with ordinary features.

---

## Conclusion

Despite experimenting with various models and trying different feature engineering techniques, the best result we obtained that seemed to provide useful prediction was with an **RMSE error around 708.8003** using a neural network and ordinary features input. Unfortunately, this level of error and the observed lag in predictions make it unsuitable for practical application. Further exploration and model refinement may be necessary to enhance performance and decrease prediction error, aiming for more reliable and accurate outcomes in practical real-world scenarios.

---

## Challenges

* **Data Limitation**: The data we obtained covers only one cycle of the market. This made it challenging to decide which portion to use for training and which for evaluation, ensuring the model learned patterns without simply memorizing prices.
* **Framework Limitation**: While Scikit-learn served as a reliable tool, its lack of GPU support hindered training and prediction acceleration. We were restricted to using a small number of input sequences because the amount of computation grows massively with an increased quantity of features. A framework such as PyTorch or TensorFlow might produce better results.
* **Machine Learning Model Limitation**: As we increased our input sequence number, the results worsened, indicating that the models could not effectively incorporate all the features. This signifies the need for more complex and sequence-specialized models such as RNN, LSTM, or Transformer networks.

---
