# Import necessary libraries
import numpy as np
import pandas as pd
import requests
import os
from scipy import stats
from statistics import mean

# Input data files are available in the read-only "../input/" directory
# For example, listing all files under the input directory
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Getting data from CSV files
value_strategy = pd.read_csv("/kaggle/input/algorithmic-trading-dataset/value_strategy_1.csv.csv")
recommended_trades = pd.read_csv("/kaggle/input/algorithmic-trading-dataset/recommended_trades_1.csv.csv")
momentum_strategy = pd.read_csv("/kaggle/input/algorithmic-trading-dataset/momentum_strategy_1.csv.csv")
trades = pd.read_csv("/kaggle/input/algorithmic-trading-dataset/sp_500_stocks.csv")

# Displaying the head of the DataFrames
print("Momentum Strategy Data:")
print(momentum_strategy.head())

print("\nValue Strategy Data:")
print(value_strategy.head())

print("\nRecommended Trades Data:")
print(recommended_trades.head())

print("\nTrades Data:")
print(trades.head())

# Collecting the data
# Importing IEX Cloud API token (Make sure to keep your API token secure)
IEX_CLOUD_API_TOKEN = "YOUR_IEX_CLOUD_API_TOKEN"  # Replace with your actual API token

symbol = 'AAPL'
# Getting the data of a single stock and storing it
api_url = f'https://sandbox.iexapis.com/stable/stock/{symbol}/quote?token={IEX_CLOUD_API_TOKEN}'
data = requests.get(api_url).json()
print("\nSample Data for AAPL:")
print(data)

# Splitting stocks into groups of 100 and making batch API calls
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# Removing specific tickers if necessary
trades.drop(index=trades[trades['Ticker'] == 'VIAC'].index, inplace=True)

symbol_groups = list(chunks(trades['Ticker'], 100))
symbol_strings = []
for i in range(len(symbol_groups)):
    symbol_strings.append(','.join(symbol_groups[i]))

# Adding required columns to trades DataFrame
my_columns = ['Ticker', 'Price', 'One-Year Price Return', 'Number of Shares to Buy']
trades_data = pd.DataFrame(columns=my_columns)

for symbol_string in symbol_strings:
    batch_api_call_url = f'https://sandbox.iexapis.com/stable/stock/market/batch?types=stats,quote&symbols={symbol_string}&token={IEX_CLOUD_API_TOKEN}'
    data = requests.get(batch_api_call_url).json()
    for symbol in symbol_string.split(','):
        try:
            trades_data = trades_data.append(
                pd.Series([
                    symbol,
                    data[symbol]['quote']['latestPrice'],
                    data[symbol]['stats']['year1ChangePercent'],
                    'N/A'
                ], index=my_columns),
                ignore_index=True
            )
        except KeyError:
            print(f"Data for {symbol} not found. Skipping.")

print("\nTrades Data with Additional Columns:")
print(trades_data.head())

# Momentum Strategy
hqm_columns = [
    'Ticker',
    'Price',
    'Market Capitalization',
    'Number of Shares to Buy',
    'One-Year Price Return',
    'One-Year Return Percentile',
    'Six-Month Price Return',
    'Six-Month Return Percentile',
    'Three-Month Price Return',
    'Three-Month Return Percentile',
    'One-Month Price Return',
    'One-Month Return Percentile',
    'HQM Score'
]

hqm_dataframe = pd.DataFrame(columns=hqm_columns)

# Batch API Calls for Momentum Strategy
for symbol_string in symbol_strings:
    batch_api_call_url = f'https://sandbox.iexapis.com/stable/stock/market/batch?types=stats,quote&symbols={symbol_string}&token={IEX_CLOUD_API_TOKEN}'
    data = requests.get(batch_api_call_url).json()
    for symbol in symbol_string.split(','):
        try:
            hqm_dataframe = hqm_dataframe.append(
                pd.Series([
                    symbol,
                    data[symbol]['quote']['latestPrice'],
                    data[symbol]['quote']['marketCap'],
                    'N/A',
                    data[symbol]['stats']['year1ChangePercent'],
                    'N/A',
                    data[symbol]['stats']['month6ChangePercent'],
                    'N/A',
                    data[symbol]['stats']['month3ChangePercent'],
                    'N/A',
                    data[symbol]['stats']['month1ChangePercent'],
                    'N/A',
                    'N/A'
                ], index=hqm_columns),
                ignore_index=True
            )
        except KeyError:
            print(f"Data for {symbol} not found. Skipping.")

# Sorting the DataFrame based on One-Year Price Return and selecting top 50 stocks
hqm_dataframe.sort_values('One-Year Price Return', ascending=False, inplace=True)
hqm_dataframe = hqm_dataframe[:50]
hqm_dataframe.reset_index(drop=True, inplace=True)

# Dropping rows with missing values
hqm_dataframe.dropna(inplace=True)

# Calculating Momentum Percentiles
time_periods = [
    'One-Year',
    'Six-Month',
    'Three-Month',
    'One-Month'
]

for row in hqm_dataframe.index:
    for time_period in time_periods:
        hqm_dataframe.loc[row, f'{time_period} Return Percentile'] = stats.percentileofscore(
            hqm_dataframe[f'{time_period} Price Return'],
            hqm_dataframe.loc[row, f'{time_period} Price Return']
        ) / 100

# Calculating HQM Score
for row in hqm_dataframe.index:
    momentum_percentiles = []
    for time_period in time_periods:
        momentum_percentiles.append(hqm_dataframe.loc[row, f'{time_period} Return Percentile'])
    hqm_dataframe.loc[row, 'HQM Score'] = mean(momentum_percentiles) * 100

print("\nHigh Quality Momentum DataFrame:")
print(hqm_dataframe.head())

# Real Value Investing Strategy
rv_columns = [
    "Ticker",
    "Price",
    "Number of Shares to Buy",
    "Price_to_Earnings_Ratio",
    "PE Percentile",
    "Price_to_Book_Ratio",
    "PB Percentile",
    "Price_to_Sales_Ratio",
    "PS Percentile",
    "EV/EBITDA",
    "EV/EBITDA Percentile",
    "EV/GP",
    "EV/GP Percentile",
    "RV Score"
]

rv_dataframe = pd.DataFrame(columns=rv_columns)

# Batch API Calls for Value Strategy
for symbol_string in symbol_strings:
    batch_api_call_url = f'https://sandbox.iexapis.com/stable/stock/market/batch?symbols={symbol_string}&types=quote,advanced-stats&token={IEX_CLOUD_API_TOKEN}'
    data = requests.get(batch_api_call_url).json()
    for symbol in symbol_string.split(','):
        try:
            enterprise_value = data[symbol]["advanced-stats"]["enterpriseValue"]
            ebitda = data[symbol]["advanced-stats"]["EBITDA"]
            gross_profit = data[symbol]["advanced-stats"]["grossProfit"]

            try:
                ev_to_ebitda = enterprise_value / ebitda
            except (TypeError, ZeroDivisionError):
                ev_to_ebitda = np.NaN
            try:
                ev_to_gross_profit = enterprise_value / gross_profit
            except (TypeError, ZeroDivisionError):
                ev_to_gross_profit = np.NaN

            rv_dataframe = rv_dataframe.append(
                pd.Series([
                    symbol,
                    data[symbol]['quote']['latestPrice'],
                    'N/A',
                    data[symbol]['quote']['peRatio'],
                    'N/A',
                    data[symbol]["advanced-stats"]["priceToBook"],
                    'N/A',
                    data[symbol]["advanced-stats"]["priceToSales"],
                    'N/A',
                    ev_to_ebitda,
                    'N/A',
                    ev_to_gross_profit,
                    'N/A',
                    'N/A'
                ], index=rv_columns),
                ignore_index=True
            )
        except KeyError:
            print(f"Data for {symbol} not found. Skipping.")

# Dropping rows with missing values
rv_dataframe.dropna(inplace=True)

# Calculating Value Percentiles
metrics = {
    "Price_to_Earnings_Ratio": "PE Percentile",
    "Price_to_Book_Ratio": "PB Percentile",
    "Price_to_Sales_Ratio": "PS Percentile",
    "EV/EBITDA": "EV/EBITDA Percentile",
    "EV/GP": "EV/GP Percentile"
}

for metric in metrics.keys():
    for row in rv_dataframe.index:
        rv_dataframe.loc[row, metrics[metric]] = stats.percentileofscore(
            rv_dataframe[metric],
            rv_dataframe.loc[row, metric]
        )

# Calculating RV Score
for row in rv_dataframe.index:
    value_percentiles = []
    for metric in metrics.keys():
        value_percentiles.append(rv_dataframe.loc[row, metrics[metric]])
    rv_dataframe.loc[row, "RV Score"] = mean(value_percentiles)

# Sorting the DataFrame and getting top 50 stocks with highest Value Score
rv_dataframe.sort_values("RV Score", ascending=True, inplace=True)
rv_dataframe = rv_dataframe[:50]
rv_dataframe.reset_index(drop=True, inplace=True)

print("\nValue Investing DataFrame:")
print(rv_dataframe.head())

# Backtesting the Strategy
# Comparing with existing strategies
print("\nBacktesting the Value Strategy:")
print(rv_dataframe['Ticker'].isin(value_strategy['Ticker']).value_counts())

print("\nBacktesting the Momentum Strategy:")
print(hqm_dataframe['Ticker'].isin(momentum_strategy['Ticker']).value_counts())

print("\nScript execution completed.")
