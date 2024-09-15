# ml_capstone
This is my capstone project for machine learning. I wanted to predict earnings call beats and misses via historical earning data and general stock data. I created an ensemble model that would generate a latent vector of the historical earnings data and then use that along with the stock data to output the degree of earnings beats and misses between -1 and 1 (-1 being a large miss and 1 being a large beat) I used [this dataset](https://www.kaggle.com/datasets/tsaustin/us-historical-stock-prices-with-earnings-data) as a base and supplemented it with data from the [yfinance](https://pypi.org/project/yfinance/) API. I preprocessed it in preprocess.py. The code for the model is in model.ipynb.

[In-Depth Discussion](Capstone%20Paper.pdf)
