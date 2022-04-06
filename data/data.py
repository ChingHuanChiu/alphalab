
from typing import Dict

from data.handler import YFData, CsvData, Handler

class _SourceField:

    field: Dict[str, Handler] = {'yf': YFData, 'csv': CsvData}


class Data:
    
    _datasource = None

    def __new__(cls, **kwargs):
        if cls.datasource is None:
            raise ValueError('You must set the data source first')
        else:
            if cls.datasource not in _SourceField.field:
                raise ValueError(f'datasource must be one of {_SourceField.field.keys()}')
            else:
                self = _SourceField.field[cls.datasource](**kwargs)
        return self

    def __init__(self, **kwargs) -> None:

        ...



    @property
    @classmethod
    def datasource(cls):
        return cls._datasource


    @datasource.setter
    @classmethod
    def datasource(cls, source: str = 'yf'):
        """ Parameters:
            ----------------------------
            source: str
                    setting the data source, it must be 'yf' so far
        
        """
        cls._datasource = source








