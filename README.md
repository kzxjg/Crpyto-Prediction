CMPT 353 Project Report: Forecasting Bitcoin Price using Machine Learning
Introduction
Cryptocurrency is well-known as a volatile and high-risk, high-return asset. The ability to predict the market's movement for the next hour or day is incredibly valuable, aiding investors in mitigating losses and potentially achieving substantial profits. With the use of fundamental machine learning algorithms, this project reflects our attempt to forecast the price of Bitcoin or the larger cryptocurrency market. Our goal is to explore the existing states and challenges associated with time series forecasting problems in this domain.

Datasets
The dataset we used contained daily cryptocurrency prices from the Binance exchange, found on https://www.CryptoDataDownload.com. CryptoDataDownload provides datasets for a plethora of different coins; however, we only selected a few which had pronounced market cap for our analysis and prediction tasks, such as BTC, ETH, ADA, BNB, and DOGE. The reason for using the additional coins is because their prices correlate with Bitcoin. Money flow to other coins may cause the price of Bitcoin to alter, which therefore causes either increases or or decreases in price.

The dataset contains:

Unix: Timestamp indicating the record's time
Symbol: Cryptocurrency symbol (e.g., BTCUSDT, ETHUSDT, etc.)
Open Price: Beginning-of-day price
Close Price: End-of-day price
Low Price: Minimum price observed during the day
Highest Price: Maximum price observed during the day
Volume: Trading volume within the day measured in cryptocurrency units
Volume of USDT: Trading volume within the day measured in US Dollars
Tradecount: Total count of buy and sell orders executed
Data Cleaning
First, we needed to unify all the CSVs to a readable format. Every CSV file was marked with a sentence "https://www.CryptoDataDownload.com" as the header, so we removed it from the CSV and saved it to make it readable. The CSV file "Binance_BTCUSDT_2021_minute.csv" contained several columns, including "marketorder_volume," "marketorder_volume_from," "date_close," and "close_unix." However, due to inconsistencies, they lacked relevance to our analysis, which resulted in their removal rather than tampering with our data by filling the missing values or averaging, which can lead to inaccurate results. After this process, we produced a similar number of files to our raw dataset, however, much cleaner and readable by the read_csv function.

Data Preprocessing
To speed up the procedure without spending time manually going through each individual crypto dataset, we merged separate cryptocurrency dataframes into a single one. The name of the appropriate symbol precedes each column. Furthermore, we removed the Unix timestamp and symbol column since they are unnecessary. In conclusion, we created a dataframe with columns containing all the features of these cryptocurrencies with their symbol as a prefix: Date, ADAUSDT_open, ADAUSDT_close, ..., ETHUSDT_tradecount. After that, we saved the CSV files as gz compression to optimize the file sizes.

After preprocessing the data, below is a visualization of some cryptocurrency daily close prices:

BTC ETH
(Self-correction: The provided text doesn't include the actual visualization, so I will omit describing it beyond what's stated.)

We also applied a LOESS filter to the visualization in order to show the true signal shape and facilitate a better comprehension of the data trends and patterns.

Feature Selection
We used open price, close price, low price, high price, volume, volume in USD, and trade count as our main features.

Originally, we experimented with the daily, hourly, and minute datasets; however, the three showed identical patterns in trends, and due to resource and framework limitations, we decided to run our experiments using only the daily dataset. Furthermore, the daily dataset captures a wider view of market trends, is less prone to trading noise, and is more useful for investors. Therefore, we only work with the daily dataset.

By combining all of those features, our ultimate goal is to predict Bitcoin closing price the next day in the future using the daily dataset. In addition, we also implemented sequence input features, meaning, the model can take data from multiple days to predict the next day. This is done by creating many duplications of the features in the dataframe and shifting each column accordingly to the number of their relative dates to current.

Machine Learning Experiments
Evaluation Metric
Since this is a regression problem, Root Mean Square Error (RMSE) will be the metric to evaluate our models. RMSE takes the Euclidean distance between the ground truth price and predicted price. The lower the RMSE values are, the better our model's performance will be for predicting continuous values.

Ordinary Features
We started by taking all selected features and feeding them as input to our model. Then we deployed and experimented with models ranging from simple to advanced, such as KNeighborsRegressor (KNN), Random Forests (RF), and Neural Network (MLP) to evaluate which model is the best.

To ensure scale uniformity across all variables, we used the MinMaxScaler to normalize the features. As a result, some features—like pricing and volume in USD—that have higher values are kept from dominating and causing disproportionate effects for the models.

The shuffle option in train_test_split is a great way to avoid biased datasets for training and validating. However, in our time series data cases, this option tends to reduce the generalization because when it shuffles the dataset before splitting, the train and valid set will contain similar data points, making the model see all kinds of samples, and we cannot know how it performs on examples it has never seen. To test this hypothesis, we split the dataset into three portions instead of two: training set, validation set, and testing set. First, splitting the dataset to construct the testing with shuffle to False, then we have another split on the remaining dataset to get the validation and training sets, shuffle option to True. Below is a simple setup with a KNN model using single day input. We observed significant differences between valid set and test set errors. The train, validation, and test errors are 1533.97, 1727.48, and 2884.81 (USD) respectively. Below are some plots demonstrating how these figures represent:
(Self-correction: The provided text doesn't include the actual plots, so I will omit describing them beyond what's stated.)

The validation set has interlacing data points with the training set since we turned shuffling on; therefore, the appearance of the two plots looks similar. Although the model seems to perform well on the validation set, it shows poor performance on new data. Therefore, keeping the shuffle option off is equivalent to the assumption that we don’t know anything about the future, so the model cannot see similar data points in the train set, and we can evaluate it better.

Following testing with one input sequence, the MLP model yields the best results with a RMSE validation error of about 708.8003 (USD). It seems the error is not that bad for unpredictable assets like BTC. However, a closer look at the graph reveals that the predictions lag behind the actual values by around 1 unit of time.

Expanding input sequence length for the neural network model:

With a 2-day input sequence, the validation error is 755.58458.
Extending to a 3-day input sequence yields a validation error of 806.87314.
Further increasing to a 5-day input sequence results in a validation error of 929.24621.
It is counter-intuitive since we thought more prior information would lead to better prediction because it can capture the trend better. However, it was not the case here, and more inputs tended to cause more noise for the model to be confused and incapable of learning. This is maybe because the models we are experimenting with are not complex enough, and we cannot experiment with a very large hidden layer of MLP due to resource limitation.

Differential Features
Instead of using the original features, we replaced them by using the differences between the current day features with one previous step features. This is done by shifting the entire dataframe by 1 and subtracting the original one from it.

For normalization, we decided to use StandardScaler in this case as it will give better results with these differential features.

After experimenting with a single input sequence, our best valid score with the smallest generalization gap is the Random Forest model with a train error of 871.4660 (USD) and a validation error of 783.05 (USD). This makes sense because maybe difference values can be easier to threshold by if-else statements, where if price has a large change, it will tend to continue to change in that direction for several future steps.

Differential features seem to improve all the models with validation errors that were lower by around 20-30%. However, we noticed some models (specifically Neural Network) try to workaround the prediction problem by repeating the previous step price though output difference is close to 0. Although the validation error of MLP in this case is 584.0263, which is an improvement from using ordinary features, it is unusable since the predictions are useless.

Expanding input sequence length for the random forest model:

With a 2-day input sequence, the validation error is 789.75150.
Extending to a 3-day input sequence yields a validation error of 787.97543.
Further increasing to a 5-day input sequence results in a validation error of 796.58988.
Multi-step Time Series Forecasting Trials
In addition to multiple inputs, we tried allowing the models to output multiple “next days” to see how far we can push our model. Multi-step time series forecasting can be formulated as multiple output regression models. The results we obtained are not accurate, and as the number of outputs increases, the RMSE increases significantly. Regardless of how we expanded the model trying to overfit the data, the model could not. Here is one output result we got by using 10 days and predicting 5 days using Random Forest using ordinary features.

Conclusion
Despite experimenting with various models and trying different feature engineering techniques, the best result we got which seems to provide useful prediction is with an RMSE error around 708.8003 by using a neural network and ordinary features input. Unfortunately, this level of error and the observed lag in predictions make it unsuitable for practical application. Further exploration and model refinement may be necessary to enhance the performance and decrease the prediction error, aiming for more reliable and accurate outcomes in practical real-world scenarios.

Challenges
Data limitation: The data we got covers only one cycle of the market, so it is hard to decide which part to train on and which part is used to evaluate so that the model can learn the patterns but not memorize the prices.
Framework limitation: While Sklearn serves as a reliable tool, its lack of GPU support hinders training and prediction acceleration. We can only use a small number of input sequences since the amount of computation grows massively as we increase the quantity of features. A framework such as PyTorch or TensorFlow may produce better results.
Machine learning model limitation: As we increase our input sequence number, the results are worse, meaning the models cannot incorporate all the features. This signifies the need for more complex and sequence-specialized models such as RNN, LSTM, or Transformer.
