#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 14:27:15 2021

@author: stefanosbaros
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from scipy import stats
import seaborn as sb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.linalg import norm



# Data cleaning: data quality checks, outliers, missing data, imputation, time-series forecasting and linear regression

# loading data
path = '/Users/stefanosbaros/Desktop/Millenium/Sample_dataset.xlsx'
features=['Date', 'Signal', 'Open', 'High', 'Low', 'Close', 'Adj Close']
price_data = pd.read_excel(path,  names=features)
col=['Signal', 'Open', 'High', 'Low', 'Close', 'Adj Close']


# error_type_1: when price in 'low' column not the lowest, impute partly using LOCF and partly using linear regression
error_type_1 =price_data.loc[(price_data['Low']> price_data['High']) | (price_data['Low']> price_data['Open']) | (price_data['Low']> price_data['Close'])]

# For isolated observations use last observation carried forward to impute (LOCF)
for index in error_type_1.index:
    if ((index+1) not in error_type_1.index) and ((index-1) not in error_type_1.index):  
        price_data.loc[index,col] = price_data.loc[index-1,col]
        error_type_1 = error_type_1.drop(index)
        
        

# error_type_2:  when price in 'high' column not the highest, impute using LOCF
error_type_2=price_data.loc[(price_data['High']<price_data['Low']) | (price_data['High']< price_data['Open']) | (price_data['High']< price_data['Close'])] 


# Last observation carried forward (LOCF) for all columns
for index in error_type_2.index:
    price_data.loc[index,col] = price_data.loc[index-1,col]


# error_type_3: if adjusted closing prices are higher than closing prices, impute using LOCF
error_type_3=price_data.loc[price_data['Close']<price_data['Adj Close']]

# Last observation carried forward (LOCF)
for index in error_type_3.index:
    price_data.loc[index,'Adj Close'] = price_data.loc[index-1,'Adj Close']


#error_type_4: if any price has consistent value but with negative sign, impute absolute value of price
columns=['Signal', 'Open', 'High', 'Low', 'Close', 'Adj Close']
error_type_4=price_data.loc[(price_data['Signal']<0) | (price_data['Open']<0) | (price_data['High']<0) | (price_data['Low']<0) | (price_data['Close']<0) | (price_data['Adj Close']<0) ]

# Impute using absolute value of price
for index in error_type_4.index:
    price_data.loc[index,col] = abs(price_data.loc[index, col])



# check for duplicated rows
boolean=price_data.duplicated()
print('Number of duplicated rows:', sum(boolean))



# Detection of outliers

#plotting prices to visually check for outliers
price_data.plot(y='Signal')
price_data.plot(y='Open')
price_data.plot(y='Close')
price_data.plot(y='Low')
price_data.plot(y='High')
price_data.plot(y='Adj Close')



#Outlier detection on 'Signal' using median absolute deviation around median criterion
mad_score=price_data['Signal'].mad()
price_data['scores_deviation']=abs((price_data['Signal']-price_data['Signal'].median()))/mad_score
scores=price_data['scores_deviation']
plt.figure()
plt.plot(scores)
outliers=price_data.loc[price_data['scores_deviation'] >3 ]


#Excluding missing values from outliers as these will be corrected later using time-series forecasting
outliers=outliers[outliers['Signal']>0]
price_data = price_data.drop(outliers.index)
price_data.reset_index(drop=True, inplace=True)


#Plotting 'Signal' again to verify that outliers have been removed
plt.figure()
price_data.plot(y='Signal')


#Dealing with missing values, correcting zeros using Holts-Winters triple exponential smoothing time-series forecasting method

#Checking for missing data with 0 values (assumption: zero values correspond to missing data)
missing_data_rows=price_data.loc[(price_data==0).any(axis=1),:]
n_miss=missing_data_rows.shape[0]

#Forecast missing values using Holts winters triple exponential smoothing method
train_data=price_data.iloc[:-n_miss,:]      # exclude data with missing values from training
x_train = train_data['Signal'].index        #training data
y_train = train_data['Signal']
x_miss=price_data.tail(n_miss).index          #missing data

#Fitting the model
model = ExponentialSmoothing(y_train, trend="add", seasonal="add", seasonal_periods=260).fit()

#Correcting zero values
y_pred = model.forecast(n_miss)
for index in x_miss:
    price_data.loc[index,'Signal'] = y_pred[index]


#Append first forecasted element to end of time-series so that the curve is continuous
y_train = y_train.append(y_pred[:1], ignore_index=True)
x_train = x_train.append(x_miss[:1])


#Plotting actual and forecasted values 
plt.figure()
plt.plot(x_train[-50:], y_train[-50:], label='Actual series (Signal)')
plt.plot(x_miss, y_pred, label='Holt-Winters forecast (Signal)')
plt.legend(loc='best')



#Assessing effectiveness of 'Signal' in predicting prices

#Computing correlation matrix to assess if a signal is a good predictor of prices
corr_matrix = price_data[columns].corr()
sb.heatmap(corr_matrix, 
            xticklabels=corr_matrix.columns,
            yticklabels=corr_matrix.columns,
            cmap='RdBu_r',
            annot=True,
            linewidth=0.5, fmt=".3g")

# Building a regression model using signal as a feature and Adjusted closing price as output
X=price_data['Signal'].values.reshape(-1,1)
X = np.delete(X, error_type_1.index , axis=0)
Y=price_data['Adj Close'].values.reshape(-1,1)
Y = np.delete(Y, error_type_1.index , axis=0)
X_train_lr,X_test_lr,Y_train_lr,Y_test_lr=train_test_split(X,Y,test_size=0.25,random_state=0)
lr=LinearRegression()
lr.fit(X_train_lr,Y_train_lr)
Y_pred_lr=lr.predict(X_test_lr)

# Construct all regression models for predicting prices in rows with Error type 1

Y1=price_data['Low'].values.reshape(-1,1)
Y1 = np.delete(Y1, error_type_1.index , axis=0)
X_train_lr1,X_test_lr1,Y_train_lr1,Y_test_lr1=train_test_split(X,Y1,test_size=0.25,random_state=0)
lr1=LinearRegression()
lr1.fit(X_train_lr1,Y_train_lr1)

Y2=price_data['High'].values.reshape(-1,1)
Y2 = np.delete(Y2, error_type_1.index , axis=0)
X_train_lr2,X_test_lr2,Y_train_lr2,Y_test_lr2=train_test_split(X,Y2,test_size=0.25,random_state=0)
lr2=LinearRegression()
lr2.fit(X_train_lr2,Y_train_lr2)

Y3=price_data['Open'].values.reshape(-1,1)
Y3 = np.delete(Y3, error_type_1.index , axis=0)
X_train_lr3,X_test_lr3,Y_train_lr3,Y_test_lr3=train_test_split(X,Y3,test_size=0.25,random_state=0)
lr3=LinearRegression()
lr3.fit(X_train_lr3,Y_train_lr3)

Y4=price_data['Close'].values.reshape(-1,1)
Y4 = np.delete(Y4, error_type_1.index , axis=0)
X_train_lr4,X_test_lr4,Y_train_lr4,Y_test_lr4=train_test_split(X,Y4,test_size=0.25,random_state=0)
lr4=LinearRegression()
lr4.fit(X_train_lr4,Y_train_lr4)


# Correcting all prices in consecutive rows with Error type 1
X_error_type_1=error_type_1['Signal'].values.reshape(-1,1)
Y_pred_error1=lr.predict(X_error_type_1)
Y_pred_error1_1=lr1.predict(X_error_type_1)
Y_pred_error1_2=lr2.predict(X_error_type_1)
Y_pred_error1_3=lr3.predict(X_error_type_1)
Y_pred_error1_4=lr4.predict(X_error_type_1)


price_data.loc[error_type_1.index,'Adj Close'] = Y_pred_error1
price_data.loc[error_type_1.index,'Low'] = Y_pred_error1_1
price_data.loc[error_type_1.index,'High'] = Y_pred_error1_2
price_data.loc[error_type_1.index,'Open'] = Y_pred_error1_3
price_data.loc[error_type_1.index,'Close'] = Y_pred_error1_4




# Plotting actual vs predicted close prices
plt.figure()
plt.plot(Y_test_lr[-100:], label='Actual adjusted closing price')
plt.plot(Y_pred_lr[-100:], label='Predicted adjusted closing price from signal')
plt.legend(loc='best')

# Plotting Signal-Adjusted close price characteristic
plt.figure()
plt.scatter(X, Y, label='Signal-Adjusted close price characteristic')
plt.legend(loc='best')
  

# Assessing how good model with one 'Signal' as feature is using goodness-of-fit and accuracy metrics

# Goodness-of-fit assessment criterion - Adjusted R2 baseline model
r2 = r2_score(Y_test_lr, Y_pred_lr)
r2_adj = 1 - (1-r2)*(len(Y_train_lr)-1)/(len(Y_train_lr)-X_train_lr.shape[1]-1)
print("Adjusted R^2 for the model: ", r2_adj)

#P rediction accuracy asssessment criteria

# MSE baseline model
MSE = mean_squared_error(Y_test_lr,Y_pred_lr)
Percentage_MSE=MSE/norm(Y_test_lr,2)
print("Percentage % MSE for the regression model with Signal as single feature: ", round(Percentage_MSE*100,3))

# MAE baseline model
MAE = mean_absolute_error(Y_test_lr,Y_pred_lr)
Percentage_MAE = mean_absolute_error(Y_test_lr,Y_pred_lr)/norm(Y_test_lr,1)
print("Percentage % MAE for the regression model with Signal as single feature: ", round(Percentage_MAE*100,3))



# Exporting cleaned data
price_data.to_csv('Clean_dataset.csv')

