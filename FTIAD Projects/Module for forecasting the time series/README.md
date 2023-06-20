# Project description
### There is a module which can forecast daily data for the next day. 
### Data used in module
As an example was taken a dataset with banks balance.
Also were used exogenous data which contained: Inflation rate(Bank of Russia), MSCI rate(Yahoo Finance), RTSI rate(Bank of Russia), USD/RUB rate(Yahoo Finance),
BTC/RUB rate(Yahoo Finance), Key rates(Bank of Russia), Tax days.
Also there is feature(balance) with a daily lag (30 lags) and moving averages for 7,14,30,60 and 90 days.
Additional features were also used: day of the week, week number, month number, year, day off.


**Warning:** Key rates, RTSI rate and Inflation rate were added as a csv files, before using this module you should update data in these csv files.

### Description of methods
This module uses these methods: Regression Lasso, Linear Regression and Random Forest for feature selection

### Description of code and functions
**For using functions from module we need to install module using these command ``pip install module``**

df - daily data.

df.data_process() - creating features for model.

df.fit_model('date') - training the model, an argument ``date`` is the end date of training. 

df.predict_one_step('date', is_fact=True) - prediction for the next day, argument ``date`` is a date which we want to predict, argument ``is_fact=True/False`` used for calculation of metrics. If there is no data (`False`) for last day metrics do not be calculated.

### Description of files
[dataset.xlsx](https://github.com/Spacelightpony/University/blob/Main/FTIAD%20Projects/Module%20for%20forecasting%20the%20time%20series/dataset.xlsx) - data with bank balance

[example_code.ipynb](https://github.com/Spacelightpony/University/blob/Main/FTIAD%20Projects/Module%20for%20forecasting%20the%20time%20series/example_code.ipynb) - example of using module

[module.py](https://github.com/Spacelightpony/University/blob/Main/FTIAD%20Projects/Module%20for%20forecasting%20the%20time%20series/module.py) - module which you need to install before using it

[additional datasets](https://github.com/Spacelightpony/University/tree/Main/FTIAD%20Projects/Module%20for%20forecasting%20the%20time%20series/additional%20datasets) - exogenous data which you should update before using module

[All files in zip.zip](https://github.com/Spacelightpony/University/blob/Main/FTIAD%20Projects/Module%20for%20forecasting%20the%20time%20series/All%20files%20in%20zip.zip) - for easy downloading all data for module(module is also included) use this file
