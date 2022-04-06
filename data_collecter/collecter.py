import os
from typing import Set, List, Dict, Iterator
from datetime import date, timedelta


from tqdm import tqdm
import asyncio
import pandas as pd
import yahoo_fin.stock_info as si

from utility import Log
from logparser import YFLogParser


class Collecter:
    def __init__(self) -> None:
        ...


    def split_data(self, data: List[Dict[str, Dict[str, pd.Series]]]):
        close_dict, open_dict, low_dict, high_dict, volume_dict, adjclose_dict = dict(), dict(), dict(), dict(), dict(), dict()
        
        for i in tqdm(data):
            ticker, sub_data = next(iter(i.items()))
    
            close_dict[ticker] = sub_data['close']
            open_dict[ticker] = sub_data['open']
            low_dict[ticker] = sub_data['low']
            high_dict[ticker] = sub_data['high']
            volume_dict[ticker] = sub_data['volume']
            adjclose_dict[ticker] = sub_data['adjclose']

        return {'close': close_dict, 'open': open_dict, 'low': low_dict, 'high': high_dict, 
                'volume': volume_dict, 'adjclose': adjclose_dict}



    def save_to_pickle(self, data: pd.DataFrame, filename: str) -> None:

        if os.path.isfile(f'../storage/{filename}'):
            origin_data = pd.read_pickle(f'../storage/{filename}')
            new_data = pd.concat([origin_data, data], 0)
        else:
            new_data = data
        new_data.to_pickle(f'../storage/{filename}.pkl')



Logger = Log(log_filename=f'./log/yfdatacollecter_{date.today()}.log', logger_name='YFData')

class YFDataCollecter:
    def __init__(self) -> None:

       if os.path.isfile(f'../storage/close.pkl'):
           self._temp_data = pd.read_pickle('./storage/close.pkl')
           self._exist_ticker = self._temp_data.columns

           self._maxdate = self._temp_data.index.max()
           self._maxdate += timedelta(days=1)
       else:
           self._exist_ticker = []
           self._maxdate = None


    @staticmethod
    def get_alltickers() -> Set[str]:
        # ft_100 = si.tickers_ftse100()
        # ft_250 = si.tickers_ftse250()
        # ibov = si.tickers_ibovespa()
        # nifty = si.tickers_nifty50()
        # niftybank = si.tickers_niftybank()



        nasdaq = si.tickers_nasdaq()
        dow = si.tickers_dow()
        other = si.tickers_other()
        sp500 = si.tickers_sp500()

        all_tickers = dow  + nasdaq + other + sp500
        all_tickers = set(all_tickers)
        return all_tickers


    async def get_data(self, ticker: str,   
                             interval:str) -> Dict[str, Dict[str, pd.Series]]:

        try:
            if ticker not in self._exist_ticker and self._maxdate:
                self._maxdate = None

            data = si.get_data(ticker, start_date=self._maxdate, interval=interval)

            if not data.index.is_unique:
                data = data[~data.index.duplicated(keep='first')]

            close = data.close
            _open = data.open
            high = data.high
            low = data.low
            volume = data.volume
            adjclose = data.adjclose
            

        except Exception as e:

            Logger.write_log('info', msg=f'{ticker}, error is : {e} ')
            close, _open, high, low, volume, adjclose = None, None, None, None, None, None
         
        await asyncio.sleep(0.5)

        return {ticker: {'close': close,
                         'open': _open,
                         'high': high,
                         'low': low,
                         'volume': volume,
                         'adjclose': adjclose
            }}


    async def run(self, tickers: Iterator, interval:str):

        tasks = [self.get_data(t, interval=interval) for t in tickers]
        res = await asyncio.gather(*tasks, return_exceptions=False)
        return res

    
def main(dataclass, interval: str = '1d') -> None:

    datacls = dataclass()
    all_tickers: Set[str] = datacls.get_alltickers()
    delist_tickers: Set[str] = YFLogParser(log_name='yfdatacollecter_2022-03-15.log').find_delist_ticker()
    all_tickers = all_tickers - delist_tickers

    print('Begin collecting data')

    res = asyncio.run(datacls.run(all_tickers, interval))

    print('Begin to split data')
    collector = Collecter()
    data = collector.split_data(data=res)
    for field, _d in data.items():
        df = pd.DataFrame(_d).dropna(1, how='all')
        collector.save_to_pickle(data=df, filename=field)


    

if __name__ == '__main__':

    import time
    s1 = time.time()
    main(dataclass=YFDataCollecter)
    s2 = time.time()
    print('花了', s2 - s1, '秒')

  


