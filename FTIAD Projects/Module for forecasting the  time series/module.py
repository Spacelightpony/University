import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import bayesian_changepoint_detection.online_changepoint_detection as oncd
import seaborn as sns
import numpy as np
import scipy.stats as scs
import yfinance as yf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tools.tools import add_constant
from functools import partial
from sklearn.model_selection import TimeSeriesSplit
from catboost import CatBoostRegressor
from datetime import datetime

class DataModule:
    def get_additional_data(
        self, start_date: pd.Timestamp, final_date: pd.Timestamp
    ) -> list[pd.DataFrame]:
        # start_date = start_date.strftime("%d.%m.%Y")
        # final_date = final_date.strftime("%d.%m.%Y")

        # MSCI
        msci = yf.Ticker("ERUS").history("10y")
        msci_data = pd.DataFrame(msci["Close"]).rename(columns={"Close": "MSCI"})
        msci_data.index = pd.to_datetime(
            msci_data.index, dayfirst=True, format="%Y-%d-%m", utc=True
        )

        # USDRUB
        usdrub = yf.Ticker("RUB=X").history("10y")
        usdrub_data = pd.DataFrame(usdrub["Close"]).rename(columns={"Close": "USDRUB"})
        usdrub_data.index = pd.to_datetime(
            usdrub_data.index, dayfirst=True, format="%Y-%d-%m", utc=True
        )

        # BTCRUB
        btcrub = yf.Ticker("BTC-RUB").history("10y")
        btcrub_data = pd.DataFrame(btcrub["Close"]).rename(columns={"Close": "BTCRUB"})
        btcrub.index = pd.to_datetime(
            btcrub.index, dayfirst=True, format="%Y-%d-%m", utc=True
        )

        # RTSI
        rtsi_data = pd.read_csv("additional_datasets/RTSI.csv", index_col=[0])

        # Key Rates
        kr_data = pd.read_csv("additional_datasets/key_rates.csv", index_col=[0])

        # Inflation Rates
        infl_data = pd.read_csv("additional_datasets/infl.csv", index_col=[0])

        all_data = [msci_data, usdrub_data, btcrub_data, rtsi_data, kr_data, infl_data]

        return all_data

    def create_features(self, df, start_date, final_date, plot_autocorr=False):
        df = df.copy()

        df["Weekday"] = df.index.dayofweek + 1
        df["Day"] = df.index.day
        df["Weekend"] = (df["Weekday"] > 5).astype(int)
        df["Month"] = df.index.month
        df["Year"] = df.index.year

        all_data = list(self.get_additional_data(start_date, final_date))
        infl_data = all_data[-1]

        date_index = pd.DataFrame(df.index)
        date_index["Year"] = pd.DatetimeIndex(date_index["Date"]).year
        date_index["month"] = pd.DatetimeIndex(date_index["Date"]).month
        inflation = date_index.merge(
            infl_data, how="inner", left_on=["Year", "month"], right_on=["Год", "month"]
        )
        inflation = inflation[["Date", "value"]]
        inflation = inflation.rename(
            columns={"Date": "Date", "value": "Inflation"}
        ).set_index("Date")
        all_data[-1] = inflation

        df.index = pd.Index(df.index.date).rename("Date")

        for data in all_data:
            data.index = pd.Index(pd.to_datetime(data.index).date).rename("Date")
            df = df.merge(data, on="Date", how="left")

        df.fillna(method="ffill", inplace=True)

        df["tax_days"] = 0
        df.loc[
            (df["Month"] == 3) & (df["Day"].isin([1, 25, 28, 30, 31])), "tax_days"
        ] = 1
        df.loc[(df["Month"] == 4) & (df["Day"].isin([20, 25, 28, 30])), "tax_days"] = 1
        df.loc[(df["Month"] == 6) & (df["Day"].isin([25])), "tax_days"] = 1
        df.loc[(df["Month"] == 7) & (df["Day"].isin([20, 25, 28, 31])), "tax_days"] = 1
        df.loc[(df["Month"] == 9) & (df["Day"].isin([25])), "tax_days"] = 1
        df.loc[(df["Month"] == 10) & (df["Day"].isin([20, 25, 28, 31])), "tax_days"] = 1
        df.loc[(df["Month"] == 12) & (df["Day"].isin([25])), "tax_days"] = 1
        df.loc[(df["Month"] == 1) & (df["Day"].isin([20])), "tax_days"] = 1

        ts_cols = [
            "Income",
            "Outcome",
            "Balance",
            "MSCI",
            "USDRUB",
            "BTCRUB",
            "RTSI Price",
            "CB_Rate",
            "Inflation",
        ]

        if plot_autocorr:
            for col in ts_cols:
                print("Признак:", col)
                self.tsplot(df[col], 90)

        for feature in ["Balance"]:
            for lag in range(1, 31):
                df[f"{feature}_lag_{lag}"] = df[feature].shift(lag)
        for feature in [
            "Income",
            "Outcome",
            "MSCI",
            "USDRUB",
            "BTCRUB",
            "RTSI Price",
            "CB_Rate",
            "Inflation",
        ]:
            df[f"{feature}_lag_1"] = df[feature].shift(1)

        # скользящее среднее
        for feature in ts_cols:
            for roll in [7, 14, 30, 60, 90]:
                df[f"{feature}_rolling_mean_{roll}"] = (
                    df[feature].shift().rolling(roll).mean()
                )

        df = df.dropna()

        # удалим признаки, значения которых на текущий день не можем получить (заменены на значение предыдущего дня)
        df = df.drop(
            [
                "Income",
                "Outcome",
                "MSCI",
                "USDRUB",
                "BTCRUB",
                "RTSI Price",
                "CB_Rate",
                "Inflation",
            ],
            axis=1,
        )
        return df

    def tsplot(self, y: pd.Series, lags=None, figsize=(15, 8), style="bmh"):
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        with plt.style.context(style):
            _ = plt.figure(figsize=figsize)

            layout = (3, 2)
            ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
            acf_ax = plt.subplot2grid(layout, (1, 0))
            pacf_ax = plt.subplot2grid(layout, (1, 1))
            qq_ax = plt.subplot2grid(layout, (2, 0))
            pp_ax = plt.subplot2grid(layout, (2, 1))

            y.plot(ax=ts_ax)
            ts_ax.set_title("Time Series Analysis Plots")

            sm.graphics.tsa.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.05)
            sm.graphics.tsa.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.05)
            sm.qqplot(y, line="s", ax=qq_ax)
            qq_ax.set_title("QQ Plot")
            scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

            plt.tight_layout()
        return
    
    def Selection_OlsMethod(self,X, y, lasso=False):
        if lasso:
            model = sm.OLS(y, add_constant(X, has_constant="add")).fit_regularized(
                alpha=0.01
            )
            params = model.params.drop("const")
            return params[params != 0].index
        else:
            model = sm.OLS(y, add_constant(X, has_constant="add")).fit()
            return list(model.pvalues.drop("const").sort_values(ascending=True).index)


    def Selection_ForestMethod(self, X, y, all_features):
        max_depth = int(np.ceil(0.3 * len(all_features)))

        model = RandomForestRegressor(max_depth=max_depth, n_estimators=50)
        model.fit(X, y)

        importances = [
            [model.feature_names_in_[i], model.feature_importances_[i]]
            for i in range(X.shape[1])
        ]
        return list(np.array(sorted(importances, key=lambda x: x[1], reverse=True))[:, 0])


    def getStability(self, Z):
        Z = np.concatenate([Z, np.zeros([Z.shape[0], 1])], axis=1)
        M, d = Z.shape

        hatPF = np.mean(Z, axis=0)
        kbar = np.sum(hatPF)
        denom = (kbar / d) * (1 - kbar / d)
        return 1 - (M / (M - 1)) * np.mean(np.multiply(hatPF, 1 - hatPF)) / denom


    def selected_to_dummy(self, selected, features):
        dm_features = []
        for f in features:
            dm_features.append((f in selected) * 1)

        return dm_features


    def GetCrossVAl(self, df, test_length=90, first_train_length=180):
        train_list = []
        test_list = []

        L = len(df)
        for i in range((L - first_train_length) // test_length):
            train_list.append(df.iloc[0 : first_train_length + test_length * i])
            test_list.append(
                df.iloc[
                    first_train_length
                    + test_length * i : first_train_length
                    + test_length * (i + 1)
                ]
            )

        return train_list, test_list


    def find_best_selector(self, df, all_features, target="Balance", max_k=None):
        methods = ["lasso", "ols", "forest"]
        k_list = [15, 25, 35, 45]

        stab_dict = {
            k: {
                "lasso": [],
                "ols": [],
                "corr": [],
                "forest": [],
            }
            for k in k_list
        }

        train_list, _ = self.GetCrossVAl(df)

        for i in range(len(train_list)):
            X = train_list[i][all_features]
            y = train_list[i][target]

            lasso = self.Selection_OlsMethod(X, y, lasso=True)
            ols = self.Selection_OlsMethod(X, y, lasso=False)
            forest = self.Selection_ForestMethod(X, y, all_features)

            for k in k_list:
                stab_dict[k]["lasso"].append(self.selected_to_dummy(lasso, all_features))
                stab_dict[k]["ols"].append(self.selected_to_dummy(ols[:k], all_features))
                stab_dict[k]["forest"].append(self.selected_to_dummy(forest[:k], all_features))

        mean_stab_dict = {}
        for k in k_list:
            mean_stab_dict[k] = {}
            for m in methods:
                mean_stab_dict[k][m] = self.getStability(np.array(stab_dict[k][m]))
        result_dict = {}
        for method in methods:
            result_dict[method] = []
        for k in mean_stab_dict:
            for method in methods:
                result_dict[method].append(mean_stab_dict[k][method])
        for method in result_dict:
            result_dict[method] = np.mean(result_dict[method])

        return (
            sorted(
                list(zip(result_dict.keys(), result_dict.values())),
                key=lambda x: x[1],
                reverse=True,
            )[0][0],
            result_dict,
        )


    def FeatureSelection(self, df, all_features, target="Balance"):
        method, result_dict = self.find_best_selector(df, all_features)
        features = None

        X, y = df[all_features], df[target]

        if method == "ols":
            features = self.Selection_OlsMethod(X, y, lasso=False)

        if method == "lasso":
            features = self.Selection_OlsMethod(X, y, lasso=True)

        if method == "forest":
            features = self.Selection_ForestMethod(X, y, all_features)

        return features, method, result_dict

    def find_change_point(
        self,
        data,
        column="Balance",
        threshold_diff=1,
        p_constant_hazard=250,
        alpha=0.1,
        beta=0.01,
        kappa=1,
        mu=0,
    ):
        data_with_changepoints = data.copy()

        R, maxes = oncd.online_changepoint_detection(
            data[column],
            partial(oncd.constant_hazard, p_constant_hazard),
            oncd.StudentT(alpha=alpha, beta=beta, kappa=kappa, mu=mu),
        )
        run_length_diff = pd.DataFrame(maxes[1:-1] - maxes[:-2])

        change_points = data_with_changepoints.index[
            run_length_diff[run_length_diff[0] < threshold_diff].index
        ]

        rolling_mean = data_with_changepoints[column].rolling(window=30).mean().shift(1)
        rolling_std = data_with_changepoints[column].rolling(window=30).std().shift(1)
        upper_bond = rolling_mean + 1.96 * rolling_std
        lower_bond = rolling_mean - 1.96 * rolling_std

        anomalies = data_with_changepoints[
            (data_with_changepoints[column] < lower_bond)
            | (data_with_changepoints[column] > upper_bond)
        ].index

        data_with_changepoints["anomalies"] = 0
        data_with_changepoints.loc[anomalies, "anomalies"] = 1

        data_with_changepoints["changepoints"] = 0
        data_with_changepoints.loc[change_points, "changepoints"] = 1

        return R, maxes, data_with_changepoints, change_points, anomalies

    def plot_change_points(self, data_with_changepoints, column="Balance"):
        change_points = data_with_changepoints[column][
            data_with_changepoints.changepoints == 1
        ]
        anomalies = data_with_changepoints[column][
            data_with_changepoints.anomalies == 1
        ]

        fig, ax = plt.subplots(figsize=[15, 8])
        ax.plot(data_with_changepoints.index, data_with_changepoints[column].values)
        ax.plot(
            change_points.index,
            change_points.values,
            "ro",
            markersize=5,
            label="Changepoints",
        )

        rolling_mean = data_with_changepoints[column].rolling(window=30).mean().shift(1)
        rolling_std = data_with_changepoints[column].rolling(window=30).std().shift(1)
        upper_bond = rolling_mean + 1.96 * rolling_std
        lower_bond = rolling_mean - 1.96 * rolling_std
        ax.plot(rolling_mean, "g", label="Rolling mean trend")
        ax.plot(upper_bond, "r--", label="Upper Bond / Lower Bond")
        ax.plot(lower_bond, "r--")
        ax.plot(
            anomalies.index, anomalies.values, "y*", markersize=7, label="Anomalies"
        )
        ax.legend(loc="upper left")

class CustomMetric(object): 
    def get_final_error(self, error, weight): 
        return error 

    def is_max_optimal(self): 
        return False 

    def evaluate(self, approxes, target, weight): 

        assert len(approxes) == 1 
        assert len(target) == len(approxes[0]) 

        approx = approxes[0] 

        res = np.abs(target - approx) 
        res = np.where(res>0.42, res*1.5, res) 

        error_sum = np.mean(res) 
        weight_sum = 1 
        return error_sum,weight_sum 

class ModelThings:
    def fit_model(self, df, features, target = 'Balance', metric='RMSE'):
        model = CatBoostRegressor(loss_function='RMSE', 
                                    eval_metric=CustomMetric(), 
                                    logging_level='Silent',
                                    od_wait=50)

        params = {
            'iterations' : np.arange(50, 150, 50),
            'depth': np.arange(10,50,10),
            'min_data_in_leaf' : np.arange(2,8,2),
            'l2_leaf_reg' : np.arange(0, 11, 2)/10}

        tscv = TimeSeriesSplit(n_splits=5)
        
        gs = model.grid_search(params, df[features], df[target], cv=tscv,  plot=False, verbose=False)

        return model


class Pipeline:
    def __init__(self, init_dataset: pd.DataFrame, target: str) -> None:
        self.init_dataset = init_dataset
        self.start_date = self.init_dataset.index[0]
        self.final_date = self.init_dataset.index[-1]
        self.target = target
        self.dataModule = DataModule()

        self.df = None
        self.all_features = None

        self.selected_features = None
        self.feature_selection_method = None
        self.result_dict = None

        self.model = None
        self.train_set = None
        self.eval_set = None


    def data_process(self):
        self.df = self.dataModule.create_features(self.init_dataset, self.start_date, self.final_date, plot_autocorr=False)
        _ , _ , _ , self.change_points, self.anomalies = self.dataModule.find_change_point(self.df)

        self.all_features = list(self.df.columns)
        if self.target  in self.all_features:
            self.all_features.remove(self.target)   

    def fit_model(self, first_test_date):
        first_test_date = datetime.strptime(first_test_date, '%Y-%m-%d').date()

        self.train_set = self.df[self.df.index < first_test_date]
        self.train_set = self.train_set[~self.train_set.index.isin(self.anomalies)]

        self.eval_set = self.df[self.df.index >= first_test_date]
        self.eval_set = self.eval_set[~self.eval_set.index.isin(self.anomalies)]

        self.first_test_date = first_test_date
        self.last_fit_date = self.train_set.index.max()

        self.selected_features, self.feature_selection_method, self.result_dict = self.dataModule.FeatureSelection(self.train_set, self.all_features)
        self.model = ModelThings().fit_model(self.train_set, self.selected_features)

    def calculate_metrics(self):
        fact = self.eval_set[self.target]
        predict = self.model.predict(self.eval_set[self.selected_features])

        ae = np.abs(fact - predict)
        mae = np.mean(ae)
        bad_points_rate = ae[ae>0.42].shape[0] / ae.shape[0]
        bad_points_mean = np.mean(ae[ae>0.42]) - 0.42

        return {'MAE': mae, 'BPR': bad_points_rate, 'BPM': bad_points_mean}
    
    def predict_one_step(self, date, fit_limit=7, is_fact=True):
        if not self.model:
            raise Exception('Not fitted yet')

        if np.datetime64(date)-np.timedelta64(1,'D') in self.anomalies:
            raise Exception('Last train point is anomaly.')

        if (datetime.strptime(date, "%Y-%m-%d") - datetime.strptime(str(self.last_fit_date)[:10], "%Y-%m-%d")).days >= fit_limit:
            self.fit_model(date)

        date = datetime.strptime(date, '%Y-%m-%d').date()
        predict = self.model.predict(self.df.loc[date][self.selected_features])
        
        ae = 'No fact data'
        is_anomaly = 'No fact data'
        is_change = 'No fact data'
        
        if is_fact:
            ae = np.abs(self.df.loc[date][self.target] - predict)
            is_anomaly = date in self.anomalies
            is_change = date in self.change_points
        
        
        result_dict = {'Predict value': predict,
                      'Abs.Error' : ae,
                      'Is changepoint' : is_change,
                      'Is anomaly' : is_anomaly,
                      'Last train date': str(self.train_set.index.max())[:10]}


        return result_dict
    