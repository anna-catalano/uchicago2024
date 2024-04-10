import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from itertools import combinations


data = pd.read_csv('Case 2 Data 2024.csv', index_col = 0)

TRAIN, TEST = train_test_split(data, test_size = 0.3, shuffle = False)

class Allocator():
    def __init__(self, train_data):
        self.running_price_paths = train_data.copy()
        self.train_data = train_data.copy()
        
    def calculate_correlation_coefficient(self, portfolio_stocks):
        if len(self.running_price_paths) > 252:
            last_year_data = self.running_price_paths.iloc[-252:]
        else:
            last_year_data = self.running_price_paths

        daily_returns = last_year_data[portfolio_stocks].pct_change().dropna()
        # Calculate the correlation coefficient of the returns
        correlation_matrix = daily_returns.corr()
        return correlation_matrix.iloc[0, 1]  # Return the correlation between the two stocks
    
    def calculate_sharpe_ratio(self, stock_name):
        if len(self.running_price_paths) > 252:
            last_year_data = self.running_price_paths.iloc[-252:]
        else:
            last_year_data = self.running_price_paths
        
        daily_returns = last_year_data[stock_name].pct_change().dropna()
        mean_return = daily_returns.mean()
        std_dev = daily_returns.std()

        if std_dev != 0:
            sharpe_ratio = mean_return / std_dev
        else:
            sharpe_ratio = 0

        return sharpe_ratio

        
    def allocate_portfolio(self, asset_prices):
        # Assumptions made for simplification; adjust as needed.
        new_row = pd.DataFrame([asset_prices], columns=self.train_data.columns)
        self.running_price_paths = pd.concat([self.running_price_paths, new_row], ignore_index=True)

        column_names_map = {name: idx for idx, name in enumerate(self.running_price_paths.columns)}

        # Calculate Sharpe ratios for individual stocks
        sharpe_ratios = {stock: self.calculate_sharpe_ratio(stock) for stock in self.running_price_paths.columns}

        # Calculate and sort correlation coefficients
        correlations = [(self.calculate_correlation_coefficient([first, second]), first, second) 
                        for first, second in combinations(self.running_price_paths.columns, 2)]
        correlations.sort(key=lambda x: abs(x[0]), reverse=True)

        # Select the top 3 most correlated pairs
        top_correlations = correlations[:3]

        # Identify unique stocks from top pairs
        unique_stocks = set([stock for _, stock1, stock2 in top_correlations for stock in [stock1, stock2]])

        # Determine weights
        weights = np.zeros(len(column_names_map))
        if unique_stocks:
            # Determine which stock has the highest Sharpe ratio among the unique stocks
            highest_sharpe_stock = max(unique_stocks, key=lambda x: sharpe_ratios[x])
            highest_sharpe_ratio = sharpe_ratios[highest_sharpe_stock]

            # Allocate weights
            for stock in unique_stocks:
                if stock == highest_sharpe_stock:
                    if len(unique_stocks) == 4:
                        weights[column_names_map[stock]] = 0.4
                    elif len(unique_stocks) == 5:
                        weights[column_names_map[stock]] = 0.3
                    elif len(unique_stocks) == 6:
                        weights[column_names_map[stock]] = 0.2
                else:
                    if len(unique_stocks) == 4:
                        weights[column_names_map[stock]] = 0.2
                    elif len(unique_stocks) == 5:
                        weights[column_names_map[stock]] = 0.175
                    elif len(unique_stocks) == 6:
                        weights[column_names_map[stock]] = 0.16

        return weights




def grading(train_data, test_data): 
    '''
    Grading Script
    '''
    weights = np.full(shape=(len(test_data.index),6), fill_value=0.0)
    alloc = Allocator(train_data)
    for i in range(0,len(test_data)):
        weights[i,:] = alloc.allocate_portfolio(test_data.iloc[i,:])
        if np.sum(weights < -1) or np.sum(weights > 1):
            raise Exception("Weights Outside of Bounds")
    
    capital = [1]
    for i in range(len(test_data) - 1):
        shares = capital[-1] * weights[i] / np.array(test_data.iloc[i,:])
        balance = capital[-1] - np.dot(shares, np.array(test_data.iloc[i,:]))
        net_change = np.dot(shares, np.array(test_data.iloc[i+1,:]))
        capital.append(balance + net_change)
    capital = np.array(capital)
    returns = (capital[1:] - capital[:-1]) / capital[:-1]
    
    if np.std(returns) != 0:
        sharpe = np.mean(returns) / np.std(returns)
    else:
        sharpe = 0
        
    return sharpe, capital, weights

sharpe, capital, weights = grading(TRAIN, TEST)
#Sharpe gets printed to command line
print(sharpe)

plt.figure(figsize=(10, 6), dpi=80)
plt.title("Capital")
plt.plot(np.arange(len(TEST)), capital)
plt.show()

plt.figure(figsize=(10, 6), dpi=80)
plt.title("Weights")
plt.plot(np.arange(len(TEST)), weights)
plt.legend(TEST.columns)
plt.show()
