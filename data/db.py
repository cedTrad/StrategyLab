import pandas as pd
import numpy as np
import sqlalchemy

PATH = "C:/Users/Dell/Desktop/CedAlgo/database/"

class connect_db:
    
    def __init__(self, name : str, interval = "1d", path = PATH):
        self.name = name
        self.interval = interval
        self.path = path
        self.engine = sqlalchemy.create_engine('sqlite:///'+self.path+self.name+"_"+self.interval+".db")
    
    def get_data(self, symbol , start = '2017', end = '2023'):
        data = pd.read_sql(symbol+"USDT", self.engine)
        data.set_index('time' , inplace=True)
        data['volume'] = pd.to_numeric(data['volume'])
        data = data[['open', 'high', 'low' , 'close' , 'volume']]
        data = data.loc[start:end].copy()
        return data
    
    

    

