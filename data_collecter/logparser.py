import re
from typing import List

class LogParser:

    def __init__(self, log_name: str) -> None:

        with open(f'./log/{log_name}', 'r') as file:
            self.log_data: List[str] = file.readlines()


    def parse_re(self, pattern: str , data) -> re.Match:
        

        return re.search(pattern, data)

    

class YFLogParser(LogParser):

    def __init__(self, log_name: str) -> None:
        super().__init__(log_name)


    def find_delist_ticker(self) -> List[str]:
        delist_tickers = list()
        for _d in self.log_data:
            if 'No data found, symbol may be delisted' in _d:
                delist_tickers.append(self.parse_re(pattern='- INFO - (.*), error', data=_d).group(1))
        return set(delist_tickers)





