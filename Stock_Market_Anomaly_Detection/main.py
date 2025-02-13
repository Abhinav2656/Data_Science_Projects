# Using the dataset

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
from scipy.stats import zscore


stock_data = pd.read_csv("stock_market.csv")

stock_data['Date'] = pd.to_datetime(stock_data['Date'])
stock_data.set_index('Date', inplace=True)

stock_data.dropna(inplace=True)


# visualization of stock prices and volume
plt.figure(figsize=(14, 7))
for ticker in stock_data['Ticker'].unique():
    subset = stock_data[stock_data['Ticker'] == ticker]
    plt.plot(subset.index, subset['Adj Close'], label=ticker)
plt.title('Adjusted close prices Over Time')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price')
plt.legend()
plt.show()

plt.figure(figsize=(14, 7))
for ticker in stock_data['Ticker'].unique():
    subset = stock_data[stock_data['Ticker'] == ticker]
    plt.plot(subset.index, subset['Volume'], label=ticker)
plt.title('Trading Volume Over Time')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.legend()
plt.show()


# Anomaly Detection Using Z-Score
def detect_anomalies(df, column):
    df_copy = df.copy()
    df_copy['Z-score'] = zscore(df_copy[column].dropna())
    anomalies = df_copy[abs(df_copy['Z-score']) > 2]
    return anomalies

anomalies_adj_close = pd.DataFrame()
anomalies_volume = pd.DataFrame()

for ticker in stock_data['Ticker'].unique():
    data_ticker = stock_data[stock_data['Ticker'] == ticker]
    adj_close_anomalies = detect_anomalies(data_ticker, 'Adj Close')
    volume_anomalies = detect_anomalies(data_ticker, 'Volume')
    anomalies_adj_close = pd.concat([anomalies_adj_close, adj_close_anomalies])
    anomalies_volume = pd.concat([anomalies_volume, volume_anomalies])

print("Anomalies in Adjusted Close Prices:")
print(anomalies_adj_close[['Ticker', 'Adj Close', 'Z-score']])

print('\nAnomalies in Trading Volume')
print(anomalies_volume[['Ticker', 'Volume', 'Z-score']])


# visualization of anomalies
def plot_anomalies(ticker, anomalies_adj_close, anomalies_volume):
    data_ticker = stock_data[stock_data['Ticker'] == ticker]
    adj_close_anomalies = anomalies_adj_close[anomalies_adj_close['Ticker'] == ticker]
    volume_anomalies = anomalies_volume[anomalies_volume['Ticker'] == ticker]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    ax1.plot(data_ticker.index, data_ticker['Adj Close'], label='Adj Close', color='blue')
    ax1.scatter(adj_close_anomalies.index, adj_close_anomalies['Adj Close'], color='red', label='Anomalies')
    ax1.set_title(f'{ticker} Adjusted Close Price and Anomalies')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Adjusted Close Prices')
    ax1.legend()

    ax2.plot(data_ticker.index, data_ticker['Volume'], label='volume', color='green')
    ax2.scatter(volume_anomalies.index, volume_anomalies['Volume'], color='orange', label='Anomalies')
    ax2.set_title(f'{ticker} Trading volume and Anomalies')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Volume')
    ax2.legend()

    plt.tight_layout()
    plt.show()


for ticker in stock_data['Ticker'].unique():
    plot_anomalies(ticker, anomalies_adj_close, anomalies_volume)


# Correlation Analysis of Anomalies
anomaly_flags = stock_data.copy()
anomaly_flags['Adj Close Anomaly'] = 0
anomaly_flags['Volume Anomaly'] = 0

anomaly_flags.loc[anomalies_adj_close.index, 'ADj Close Anomaly'] = 1
anomaly_flags.loc[anomalies_volume.index, 'Volume Anomaly'] = 1

adj_close_anomalies_pivot = anomaly_flags.pivot_table(index='Date', columns='Ticker', values='Adj Close Anomaly', fill_value=0)
volume_anomalies_pivot = anomaly_flags.pivot_table(index='Date', columns='Ticker', values='Volume Anomaly', fill_value=0)

adj_close_corr = adj_close_anomalies_pivot.corr()
volume_corr = volume_anomalies_pivot.corr()

plt.figure(figsize=(10, 6))
sns.heatmap(adj_close_corr, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f")
plt.title("Correlation fo Adjusted close Price Anomalies")
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(volume_corr, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f")
plt.title("Correlation of Volume Anomalies")
plt.show()


# Risk Analysis based on Anomalies
adj_close_risk = anomalies_adj_close.groupby('Ticker')['Z-score'].apply(lambda x: abs(x).mean() )
volume_risk = anomalies_volume.groupby('Ticker')['Z-score'].apply(lambda x: abs(x).mean())

total_risk = adj_close_risk.add(volume_risk, fill_value=0)
risk_rating = (total_risk - total_risk.min()) / (total_risk.max() - total_risk.min(0))

print("Risk Ratings for Each Stock:")
print(risk_rating)

plt.figure(figsize=(10, 5))
sns.barplot(x=risk_rating.index, y=risk_rating.values, palette='viridis', hue='None')
plt.title('Relative Risk Ratings for Each Stock')
plt.xlabel('Stock Market')
plt.ylabel('Normalized Risk Score')
plt.ylim(0, 1)
plt.show()











# Using yfinance module
# import pandas as pd
# import yfinance as yf
# from datetime import date, timedelta
#
# end_date = date.today().strftime("%Y-%m-%d")
# start_date = (date.today() - timedelta(days=365)).strftime("%Y-%m-%d")
#
# tickers = ['AAPL', 'MSFT', 'NFLX', 'GOOG', 'TSLA']
#
# data = yf.download(tickers, start=start_date, end=end_date, progress=False)
# data = data.reset_index()
#
# data.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in data.columns]
#
# data_melted = data.melt(id_vars=['Date'], var_name='Attribute_Ticker', value_name='value')
#
# data_melted[['Attribute', 'Ticker']] = data_melted['Attribute_Ticker'].str.split('_', expand=True)
# data_melted.drop(columns=['Attribute_Ticker'], inplace=True)
#
# data_pivoted = data_melted.pivot_table(index=['Date', 'Ticker'], columns='Attribute', values='value', aggfunc='first')
#
# stock_data = data_pivoted.reset_index()
#
# stock_data['Date'] = pd.to_datetime(stock_data['Date'])
# stock_data.set_index('Date', inplace=True)
# print(stock_data.head())