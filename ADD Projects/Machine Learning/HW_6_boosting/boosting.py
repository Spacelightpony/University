from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot as plt

def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
    ):
        self.base_model_class = DecisionTreeRegressor
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample

        self.early_stopping_rounds: int = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)

        self.plot: bool = plot

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)

    def fit_new_base_model(self, x, y, predictions):
        # Определение размера выборки 
        sample_size = int(x.shape[0] * self.subsample) 

        # Генерация случайных индексов для выборки 
        random_indices = np.random.randint(0, x.shape[0], sample_size) 

        # Получение выборок данных и соответствующих предсказаний 
        x_sampled = x[random_indices, :] 
        y_sampled = y[random_indices] 
        predictions_sampled = predictions[random_indices] 

        # Вычисление градиента функции потерь 
        loss_gradient = -self.loss_derivative(y_sampled, predictions_sampled) 

        # Создание и обучение новой модели на выборочных данных 
        new_model = self.base_model_class(**self.base_model_params) 
        new_model.fit(x_sampled, loss_gradient) 

        # Получение обновленных предсказаний с учетом новой модели 
        updated_predictions = predictions + new_model.predict(x) 

        # Нахождение оптимального значения гамма и обновление коэффициента обучения 
        optimal_gamma = self.find_optimal_gamma(y, predictions, updated_predictions) 
        new_gamma = optimal_gamma * self.learning_rate
        
        self.gammas.append(new_gamma)
        self.models.append(new_model)

    def fit(self, x_train, y_train, x_valid, y_valid):
        """
        :param x_train: features array (train set)
        :param y_train: targets array (train set)
        :param x_valid: features array (validation set)
        :param y_valid: targets array (validation set)
        """
         # Инициализация предсказаний для тренировочного и валидационного наборов 
        train_preds = np.zeros(y_train.shape[0]) 
        valid_preds = np.zeros(y_valid.shape[0]) 

        # Запись начальной оценки потерь в историю 
        self.history['train'].append(self.loss_fn(y_train, train_preds)) 
        self.history['val'].append(self.loss_fn(y_valid, valid_preds)) 

        bad_rounds = 0  # Счетчик плохих итераций 

        for _ in range(self.n_estimators): 
            # Обучение новой базовой модели 
            self.fit_new_base_model(x_train, y_train, train_preds) 
            train_preds += self.gammas[-1] * self.models[-1].predict(x_train) 

            # Обновление истории потерь для тренировочного набора 
            train_loss = self.loss_fn(y_train, train_preds) 
            self.history['train'].append(train_loss) 

            if self.early_stopping_rounds is not None: 
                # Обновление предсказаний и потерь для валидационного набора 
                valid_preds += self.gammas[-1] * self.models[-1].predict(x_valid) 
                val_loss = self.loss_fn(y_valid, valid_preds) 
                self.history['val'].append(val_loss) 

                # Проверка условий ранней остановки 
                if val_loss >= self.history['val'][-2]: 
                    bad_rounds += 1 
                else: 
                    bad_rounds = 0 

                if bad_rounds >= self.early_stopping_rounds: 
                    break 
 
        # Построение графика истории потерь, если необходимо 
        if self.plot: 
            plt.plot(self.history['train'], label='Train Loss') 
            plt.xlabel('Number of Models') 
            plt.ylabel('Loss') 

            if 'val' in self.history: 
                plt.plot(self.history['val'], label='Validation Loss') 

            plt.legend() 
            plt.show()

    def predict_proba(self, x):
        res = 0
        for gamma, model in zip(self.gammas, self.models):
            res +=  model.predict(x) * gamma
        res = self.sigmoid(res)

        return np.vstack([1 - res, res]).T

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]

        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)

    @property
    def feature_importances_(self):
        # Проверка, обучена ли модель 
        if not self.models: 
            raise Exception('Модель не обучена') 

        # Инициализация суммы важностей признаков 
        total_importances = np.zeros(self.models[0].feature_importances_.shape[0]) 

        # Суммирование важностей признаков всех моделей 
        for model in self.models: 
            total_importances += model.feature_importances_ 

        # Вычисление средних важностей признаков 
        average_importances = total_importances / len(self.models) 

        # Применение экспоненциальной функции для нормализации 
        normalized_importances = np.exp(average_importances - np.max(average_importances)) 

        # Возвращение нормализованных важностей признаков 
        return normalized_importances / normalized_importances.sum()
