import pandas as pd


def compute_relative_sma_gaps(df, sma_windows, col='close'):
    for i, window in enumerate(sma_windows):
        df[f'sma{i+1}'] = df[col].rolling(window=window).mean()
    for i in range(len(sma_windows) - 1):
        df[f'gap_sma{i+1}_sma{i+2}'] = (df[f'sma{i+1}'] - df[f'sma{i+2}']) / df[f'sma{i+2}']
        


class Momentum2:
    
    def __init__(self, data):
        self.data = data.copy()
        
    def update_params(self, m):
        self.m = m
        
    def preprocess(self):
        self.data['mom'] = self.data["close"].pct_change().rolling(self.m).mean()
    
    def run(self, bar = -1):
        self.preprocess()
        
        if (self.data["mom"].iloc[bar] > 0) and (self.data["mom"].iloc[bar-1] > 0):
            return "LONG"
        elif (self.data["mom"].iloc[bar] < 0) and (self.data["mom"].iloc[bar-1] < 0):
            return "SHORT"
        else:
            return None



class Strateg1:
    
    def __init__(self, data):
        self.data = data
    
    def update_params(self, m):
        self.m = m
    
    def preprocess(self):
        self.data["sma2"] = self.data["close"].rolling(6).mean()
        self.data["sma5"] = self.data["close"].rolling(24).mean()
        self.data["sma6"] = self.data["close"].rolling(24*3).mean()
        
        self.data['gap_sma5_sma6'] = (self.data['sma5'] - self.data['sma6']) * 100 / self.data['sma6']
        self.data['rel_gap_sma2_sma6'] = (self.data['sma2'] - self.data['sma6']) * 100 / self.data['sma6']
    
    def stage1(self):
        self.data
    
    def run(self, bar=-1):
        if "LON":
            return "LONG"
        elif cond :
            return "SHORT"
        else:
            return None
        



class Filter:
    
    def __init__(self) -> None:
        self.data = 0
    





