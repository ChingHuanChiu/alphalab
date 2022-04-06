
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.model_selection import cross_val_score
from time import time
from functools import wraps
from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor


class ManagerBase:
    def __init__(self, x, y, cv, hyper_param, return_hyper_result, return_record, is_time_series, scoring):
        """
        :param x:  x
        :param y:  y
        :param cv: int, cross-validation generator or an iterable, default=10
                * integer, to specify the number of folds in a (Stratified)KFold,
                * CV splitter,
                * An iterable yielding (train, test) splits as arrays of indices.
        :param hyper_param: the hyper_param field to tune
                            default is "None", which means there is no action to find the best hyper parameters

        :param return_hyper_result: whether return the result after doing cv or tuning hyper parameters

        :param is_time_series: if True, the  cv represents the split in TimeSeriesSplit, or KFold

        """
        self.hyper_param_field = hyper_param
        self.x = x
        self.y = y

        self.cv = TimeSeriesSplit(n_splits=cv) if is_time_series else cv
        self.hyper_param = hyper_param
        self.return_result = return_hyper_result
        self.return_record = return_record
        self.scoring = scoring

    def tune_hyper(self, model, *args, **kwargs):
        """
        :return: return the model after tuning the hyper parameter
        """

        g_search = GridSearchCV(estimator=model, cv=self.cv, param_grid=self.hyper_param_field,
                                scoring=self.scoring)
        print("Start to tune the hyper parameters.....")
        start = time()
        g_search.fit(self.x, self.y, *args, **kwargs)
        end = time()
        print(f'Your model is just finished tuning hyper parameters and it cost {round(end - start, 2)} seconds')
        return g_search

    def cv_score(self, model):
        print('Start to do cross validation....')
        start = time()
        sc = cross_val_score(model, self.x, self.y, cv=self.cv, scoring=self.scoring)
        end = time()
        print(f'Your model is just finished doing CV and it cost {round(end - start, 2)} seconds')

        return sc, sc.mean(), sc.std()

    def record_tune_hyper(self, grid_result):
        """
        record the result with different hyper parameters every step
        :return: dict
        """
        record = {}
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        n = 1
        for mean, stdev, param in zip(means, stds, params):
            record[f'{n}_mean'] = mean
            record[f'{n}_stdev'] = stdev
            record[f'{n}_param'] = param
            n += 1
        return record


class MLManager(ManagerBase):

    def __init__(self, x, *, y=None, cv=None, hyper_param=None, return_hyper_result=False,
                 return_record=False,
                 is_time_series=False,
                 scoring='accuracy'):

        super(MLManager, self).__init__(x, y, cv,
                                        hyper_param=hyper_param,
                                        return_hyper_result=return_hyper_result,
                                        return_record=return_record,
                                        is_time_series=is_time_series,
                                        scoring=scoring)

    def __call__(self, func):
        result = {}

        @wraps(func)
        def wrapper(*args, **kwargs):
            if self.hyper_param is not None:
                all_new_model = self.tune_hyper(func(*args, **kwargs))
                if self.return_record is True:
                    result['record'] = self.record_tune_hyper(all_new_model)
                result['model'] = all_new_model.best_estimator_
                if self.return_result is True:
                    result['best_score'] = all_new_model.best_score_
                    result['best_param'] = all_new_model.best_params_
            if self.cv is not None:
                model = func(*args, **kwargs) if self.hyper_param is None else result['model']
                score, mean, std = self.cv_score(model)
                result['CV_score'] = score
                result['CV_score_mean'] = mean
                result['CV_score_std'] = std

            return result

        return wrapper


class DLManager(ManagerBase):
    def __init__(self, x, y, *, cv=None, hyper_param=None, return_hyper_result=False, return_record=False, is_time_series=False,
                 scoring=None, model_type='classification', keras_callbacks=None):
        super(DLManager, self).__init__(x, y, cv,
                                        hyper_param=hyper_param,
                                        return_hyper_result=return_hyper_result,
                                        return_record=return_record,
                                        is_time_series=is_time_series,
                                        scoring=scoring)

        self.callbacks = keras_callbacks
        if model_type == 'classification':
            self.type = KerasClassifier
        elif model_type == 'regression':
            self.type = KerasRegressor

    def __call__(self, func):
        """
        The model is only for sequential of keras model
        """
        result = {}

        @wraps(func)
        def wrapper(*args, **kwargs):
            if self.hyper_param is not None:
                model = self.type(build_fn=func)
                all_new_model = self.tune_hyper(model, callbacks=self.callbacks)
                if self.return_record is True:
                    
                    result['record'] = self.record_tune_hyper(all_new_model)
                result['model'] = all_new_model.best_estimator_
                if self.return_result is True:
                    result['best_score'] = all_new_model.best_score_
                    result['best_param'] = all_new_model.best_params_

            if self.cv is not None:
                model = self.type(build_fn=func) if self.hyper_param is None else result['model']
                score, mean, std = self.cv_score(model)
                result['CV_score'] = score
                result['CV_score_mean'] = mean
                result['CV_score_std'] = std
            return result
        return wrapper

