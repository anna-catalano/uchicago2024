import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from itertools import combinations


data = pd.read_csv('Case 2 Data 2024.csv', index_col = 0)

'''
We recommend that you change your train and test split
'''

TRAIN, TEST = train_test_split(data, test_size = 0.3, shuffle = False)

class Allocator():
    def __init__(self, train_data):
        '''
        Anything data you want to store between days must be stored in a class field
        '''
        
        self.running_price_paths = train_data.copy()
        
        self.train_data = train_data.copy()
        
        # Do any preprocessing here -- do not touch running_price_paths, it will store the price path up to that data
   
    def calculate_portfolio_sharpe_ratio(self, portfolio_stocks):
        if len(self.running_price_paths) > 252:
            last_year_data = self.running_price_paths.iloc[-252:]
        else:
            last_year_data = self.running_price_paths

        daily_returns = last_year_data[portfolio_stocks].pct_change().dropna()
        average_daily_return = daily_returns.mean(axis=1)
        mean_return = average_daily_return.mean()
        std_dev = average_daily_return.std()

        sharpe_ratio = mean_return / std_dev if std_dev != 0 else 0
        return abs(sharpe_ratio)     
        
    def allocate_portfolio(self, asset_prices):
        #print("asset", asset_prices, type(asset_prices))
        #if not isinstance(asset_prices, np.ndarray) or len(asset_prices) != 6:
        #    raise ValueError("asset_prices must be a numpy array of length 6.")

        #new_row = pd.DataFrame([asset_prices], columns=self.train_data.columns)
        #self.running_price_paths = pd.concat([self.running_price_paths, new_row], ignore_index=True)
        self.running_price_paths = pd.concat([self.running_price_paths, asset_prices], ignore_index=True)
        self.running_price_paths.drop(0, axis=1, inplace=True)
        
        column_names_map = {name: idx for idx, name in enumerate(self.running_price_paths.columns)}
        
        max_sharpes = [-1, -1, -1]  # Dummy populate with -1
        max_pairs = ['', '', '']
        for first, second in combinations(self.running_price_paths.columns, 2):
            print("fs", first, second)
            sr = self.calculate_portfolio_sharpe_ratio([first, second])
            min_sharpe = min(max_sharpes)
            if sr > min_sharpe:
                min_index = max_sharpes.index(min_sharpe)
                max_sharpes[min_index] = sr
                max_pairs[min_index] = first + second
        
        unique_stocks = set(''.join(max_pairs))
        weights = np.zeros(6)
        
        if len(unique_stocks) == 4:
            max_index = max_sharpes.index(max(max_sharpes))
            for stock in unique_stocks:
                if stock in max_pairs[max_index]:
                    weights[column_names_map[stock]] = 0.4
                else:
                    weights[column_names_map[stock]] = 0.2
        elif len(unique_stocks) == 5:
            max_index = max_sharpes.index(max(max_sharpes))
            for stock in unique_stocks:
                if stock in max_pairs[max_index]:
                    weights[column_names_map[stock]] = 0.3
                else:
                    weights[column_names_map[stock]] = 0.175
        elif len(unique_stocks) == 6:
            for stock in unique_stocks:
                weights[column_names_map[stock]] = 0.16
            max_index = max_sharpes.index(max(max_sharpes))
            weights[column_names_map[max_pairs[max_index][0]]] = 0.2
            weights[column_names_map[max_pairs[max_index][1]]] = 0.2
        
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
