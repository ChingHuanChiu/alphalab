from typing import List, Union
import joblib


import numpy as np
# import pandas as pd
# from sklearn.base import TransformerMixin


class Preprocessor:
    """using the method of 'TransformerMixin' from sklearn's api  
    """

    def __init__(self, preprocessor = None, preprocessor_path:str = None) -> None:
        self.preprocessor = preprocessor
        self.preprocessor_path = preprocessor_path

        if self.preprocessor_path is not None:

            self.preprocessor = self.load(self.preprocessor_path)


    def transform(self, X) -> np.array:
        if self.preprocessor_path is None:
            print('Notice: You are fitting the new data!')

            transform_data = self.preprocessor.fit_transform(X)
        else:
            transform_data = self.preprocessor.transform(X)
       
        return transform_data

    def save(self, path:str) -> None:
        joblib.dump(self.preprocessor, path)

    @staticmethod
    def load(path: str):
        return joblib.load(path)
    


