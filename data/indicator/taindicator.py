from typing import Dict, List, Tuple, Union

import pandas as pd
import numpy as np
from talib import abstract
from numba import jit


"""TODO: need to speed up the create method in TaLibIndicator class
"""
class TaLibIndicator:

    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        self.f = None
        self.parameters = None
        self.output_names = None
        self._open = None
        self._close = None
        self._high = None
        self._low = None
        self._volume = None
        self.tickers = None
        

        if self.data:
            self._open: pd.Series = self.data.open
            self._close: pd.Series =self.data.close
            self._high: pd.Series = self.data.high
            self._low: pd.Series = self.data.low
            self._volume: pd.Series = self.data.volume
            self.tickers: np.array  = self.data.ticker.unique()



    def create(self, name:str, timeperiod=None, **parameters) -> Union[pd.DataFrame, Tuple[pd.DataFrame, ...]]:
        # _dict = dict()
        self.f = getattr(abstract, name)
        self.output_names: List[str] = self.f.output_names
        self.parameters = self.f.parameters
        output_names_length: int = len(self.output_names)

        # for t in self.tickers:
        #     OHLCV = self._makeOHLCV(t=t)
        #     indicator_data: Union[List[np.array], np.array] = self.f(OHLCV, timeperiod=timeperiod, **parameters)
            # indicator_data: Union[List[float], List[List[float]]] = pd.to_numeric(indicator_data, errors='coerce')

            # _dict[t] = indicator_data
        
        _dict = {t: self.f(self._makeOHLCV(t=t), timeperiod=timeperiod, **parameters) for t in self.tickers}


        if output_names_length == 1:
            res =  pd.DataFrame(_dict, index=self._close.index)
            return res

        else:
            record_dict = dict() 
            for i in range(output_names_length):
                record_dict[i] = {t: _dict[t][i] for t in _dict.keys()}

            return tuple(pd.DataFrame(record_dict[i], index=self._close.index) for i in range(output_names_length))


    def _makeOHLCV(self, t: str = None) -> Dict[str, Union[pd.Series, pd.DataFrame]]:

        if self.data:
            return {
                    'open': self._open,
                    'high': self._high,
                    'low': self._low,
                    'close':self._close,
                    'volume':self._volume
                            }
        else:
            return {
                'open': self._open[t].values,
                'high': self._high[t].values,
                'low': self._low[t].values,
                'close':self._close[t].values,
                'volume':self._volume[t].values
                        }







