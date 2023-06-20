# Project description
### There is a module which can forecast daily data for the next day. 
### Data description
As an example was taken a dataset with banks balance.
Also were used exogenous data which contained: inflation rate(Bank of Russia), MSCI rate(Yahoo Finance), RTSI rate(Bank of Russia), USD/RUB rate(Yahoo Finance),
BTC/RUB rate(Yahoo Finance), Key rates(Bank of Russia), Tax days.
Also there is feature(balance) with a daily lag (30 lags) and moving averages for 7,14,30,60 and 90 days.
Additional features were also used: day of the week, week number, month number, year, day off.
### Description of methods
As a key methods were used: Regression Lasso, Linear Regression and Random Forest. 
### Description of code and functions
df - daily data.

df.data_process() - creating features for model.

df.fit_model('date') - training the model, an argument ``date`` is the end date of training. 

df.predict_one_step('date', is_fact=True) - prediction for the next day, argument ``date`` is a date which we want to predict, argument ``is_fact=True/False`` used for calculation of metrics. If there is no data for last day metrics do not be calculated.
