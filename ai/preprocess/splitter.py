"""
TODO: use design pattern to reconstruct!
# Split train, validation and test data : spliter.py
    * Naive
    * Time Series
"""
# from abc import ABCMeta, abstractmethod
from typing import Callable, Tuple, Union

import pandas as pd
import numpy as np


  

class Spliter:

    def __init__(self, data: Union[pd.DataFrame, np.array], label: Union[np.array, pd.Series] = None ) -> None:
        
        if not isinstance(data, pd.DataFrame):
            self.data = np.array(data)
        else:
            self.data = data
        # self.data = data.copy()
        self.X = self.data
        self.y = np.array(label) 
    
    def split(self, train_ratio, **kwargs) -> Tuple[np.array, ...]:

        train_len = int(self.data.shape[0] * train_ratio)
        X_train = self.X[:train_len]
        X_test = self.X[train_len:]

        if self.y is None:
            return X_train, X_test

        else:
            y_train, y_test = self.y[:train_len], self.y[train_len:]
            return X_train, X_test, y_train, y_test


class NaiveSpliter(Spliter):
    ...



class TSWindowSpliter(Spliter):


    def split(self, train_ratio, **kwargs) -> Tuple[np.array, ...]:

        self.X, self.y = self._window_data(look_back=kwargs['look_back'])
        X_train, X_test, y_train, y_test = super().split(train_ratio=train_ratio)
        
  
        return X_train, X_test, y_train, y_test
        

    def _window_data(self, look_back):

        _X = []
        _y = []
        for i in range(look_back, len(self.data)):
            _X.append(self.X.values[i - look_back: i])
            if self.y is not None:
                _y.append(self.y[i])

        return np.array(_X), np.array(_y)

      













