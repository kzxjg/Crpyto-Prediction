Forecasting Bitcoin Price using Machine Learning
Introduction
This project investigates the application of fundamental machine learning algorithms to forecast Bitcoin prices. Our goal is to explore the existing states and challenges of time series forecasting within the volatile cryptocurrency market. The ability to predict market movements can be incredibly valuable for investors, helping to mitigate losses and potentially achieve substantial profits.

Datasets
We used daily cryptocurrency price data from the Binance exchange, sourced from CryptoDataDownload.com. For our analysis and prediction tasks, we selected cryptocurrencies with pronounced market capitalization like BTC, ETH, ADA, BNB, and DOGE. We chose these additional coins because their prices often correlate with Bitcoin, and money flow to or from them can influence Bitcoin's value.

The dataset includes the following features:

Unix: Timestamp of the record
Symbol: Cryptocurrency symbol (e.g., BTCUSDT, ETHUSDT)
Open Price: Beginning-of-day price
Close Price: End-of-day price
Low Price: Minimum price observed during the day
Highest Price: Maximum price observed during the day
Volume: Trading volume in cryptocurrency units
Volume of USDT: Trading volume in US Dollars
Tradecount: Total count of buy and sell orders executed
Data Preparation
Data Cleaning
Our initial data cleaning steps involved:

Removing the extraneous header line, "https://www.CryptoDataDownload.com," from all CSV files.
Identifying and removing inconsistent or irrelevant columns such as marketorder_volume, marketorder_volume_from, date_close, and close_unix from Binance_BTCUSDT_2021_minute.csv to avoid data contamination and ensure relevance to our analysis.
Data Preprocessing
To streamline our workflow, we merged separate cryptocurrency dataframes into a single, unified one. Each column was prefixed with its respective symbol (e.g., ADAUSDT_open, ETHUSDT_tradecount). We then removed the Unix timestamp and Symbol columns as they were not needed for our models. The processed dataframes were saved as .gz compressed CSV files to optimize file sizes.

Feature Selection
We used open price, close price, low price, high price, volume, volume in USD, and trade count as our primary features.

While we initially experimented with daily, hourly, and minute datasets, we ultimately decided to focus on the daily dataset. This choice was made due to resource and framework limitations, and because the daily dataset offers a broader view of market trends, is less susceptible to trading noise, and is generally more useful for investors.

Our main goal is to predict the Bitcoin closing price for the next day. To capture temporal dependencies, we implemented sequence input features. This means our models can take data from multiple preceding days to predict the next day's price. We achieved this by duplicating features within the dataframe and shifting each column to represent data from relative past dates.

Machine Learning Experiments
Evaluation Metric
As this is a regression problem, we chose Root Mean Square Error (RMSE) as our evaluation metric. RMSE calculates the Euclidean distance between the actual (ground truth) price and the predicted price. A lower RMSE indicates better model performance for continuous value prediction.

Ordinary Features
We began our experiments by feeding all selected features as input to our models. We explored models ranging from simple to advanced, including KNeighborsRegressor (KNN), Random Forests (RF), and Neural Network (MLP).

Normalization: We applied MinMaxScaler to normalize features, ensuring all variables were on a uniform scale. This prevents features with higher values (like price and volume) from disproportionately affecting the models.
Data Splitting for Time Series: Crucially, we avoided shuffling the dataset before splitting for time series data. Instead, we split the data into training, validation, and testing sets sequentially (shuffle=False). This approach ensures that the training set does not contain data points similar to the validation or test sets, allowing for a more realistic evaluation of how the model performs on truly unseen future data. Our experiments showed significant differences between validation and test errors when shuffling was used, confirming that an unshuffled split is vital for proper time series evaluation.
Single-Day Input Results: The MLP model yielded the best results with a validation RMSE of approximately 708.80 USD. While this error might seem reasonable for an unpredictable asset like Bitcoin, a closer look revealed that predictions consistently lagged behind actual values by about one unit of time.
Impact of Input Sequence Length for MLP (Ordinary Features):
2-day input sequence: 755.58 RMSE
3-day input sequence: 806.87 RMSE
5-day input sequence: 929.25 RMSE Counterintuitively, increasing the input sequence length led to worse performance. We believe this might be due to the models not being complex enough to effectively process more input features, potentially introducing noise rather than useful patterns.
Differential Features
Instead of using raw feature values, we transformed them into differential features, representing the difference between the current day's features and the previous day's features. This was done by shifting the entire dataframe by one step and subtracting it from the original.

Normalization: For differential features, we used StandardScaler, which yielded better results.
Single-Day Input Results: With differential features, the Random Forest model achieved the best validation score with the smallest generalization gap: a train RMSE of 871.47 USD and a validation RMSE of 783.05 USD. This suggests that difference values might be easier for tree-based models to learn, as they can represent thresholds for changes in direction.
Observations: Differential features generally improved model performance, leading to 20-30% lower validation errors across models. However, we observed that some models, particularly the Neural Network, tended to predict differences close to zero, effectively repeating the previous day's price. While this lowered the validation RMSE (e.g., 584.03 USD for MLP), the predictions were largely unusable.
Impact of Input Sequence Length for RF (Differential Features):
2-day input sequence: 789.75 RMSE
3-day input sequence: 787.98 RMSE
5-day input sequence: 796.59 RMSE
Multi-step Time Series Forecasting Trials
We also attempted multi-step time series forecasting, allowing models to output predictions for multiple "next days." This was formulated as a multiple-output regression problem. The results were not accurate, and the RMSE increased significantly as the number of output steps grew. The models struggled to generalize for longer-term predictions, suggesting difficulties in overfitting even with expanded model complexity.

Conclusion
Despite experimenting with various models and feature engineering techniques, the best practical result we obtained was with an RMSE of approximately 708.80 USD using a neural network and ordinary features. Unfortunately, this level of error, combined with the observed lag in predictions, renders it unsuitable for practical, real-world applications in mitigating losses or achieving substantial profits in cryptocurrency trading. Further exploration and significant model refinement, likely involving more sophisticated architectures, would be necessary to achieve more reliable and accurate outcomes.

Challenges
Data Limitation: Our dataset covered only one market cycle. This made it challenging to identify the optimal training and evaluation splits that would allow the model to learn general market patterns without simply memorizing past prices.
Framework Limitation: Using Scikit-learn, which lacks GPU support, significantly hindered training and prediction acceleration. This constrained our ability to experiment with larger input sequences, as computational requirements grew massively with increased features. Frameworks like PyTorch or TensorFlow, with GPU capabilities, would likely yield better results.
Machine Learning Model Limitation: As we increased the input sequence length, our models' performance deteriorated. This suggests that the models we experimented with (KNN, RF, MLP) were not complex enough or specialized for sequence data. More advanced, sequence-aware models like Recurrent Neural Networks (RNN), Long Short-Term Memory networks (LSTM), or Transformer models are likely needed to effectively incorporate more temporal features.
