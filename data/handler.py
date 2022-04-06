from typing import Union, Tuple
from functools import lru_cache


import yahoo_fin.stock_info as si
import pandas as pd
from talib import abstract


from data.indicator.taindicator import TaLibIndicator



class Handler:

    def __init__(self) -> None:
        self.taindicator = TaLibIndicator

    def get_data(self):
        pass
    
    @lru_cache(maxsize=None)
    def get(self, field: str = None) -> pd.DataFrame:
        if field is None or field not in ['close', 'open', 'high', 'low', 'volume', 'adjclose']:
            raise ValueError('field must be one of [close, open, high, low, volume, adjclose]')
        return pd.read_pickle(f'./storage/{field}.pkl')


    def indicator(self, name:str, 
                        timeperiod=None, 
                        data:pd.DataFrame = None, 
                        **parameters) -> Union[pd.DataFrame, Tuple[pd.DataFrame, ...]]:
        taindicator_instance = self.taindicator(data=data)

        if data is None:
            for attr in ['open', 'close', 'high', 'low', 'volume']:
                setattr(taindicator_instance, f'_{attr}', self.get(attr))
            setattr(taindicator_instance, 'tickers', self.get('close').columns)

        
        res = taindicator_instance.create(name, 
                                        timeperiod=timeperiod, 
                                        **parameters)
                                 
        return res
    
    
    @staticmethod
    def indicator_info(name):
        f = getattr(abstract, name)
        return {'output_names': f.output_names,
                'parameters': f.parameters
                            } 




class YFData(Handler):
    def __init__(self) -> None:
        super().__init__()

    def get_data(self, ticker, startdate, enddate, interval='1d') -> pd.DataFrame:
        data = si.get_data(ticker, startdate, enddate, interval=interval)
        return data
    


class CsvData(Handler):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.filepath = kwargs['file_path']

    def get_data(self) -> pd.DataFrame:

        return pd.read_csv(self.filepath)
