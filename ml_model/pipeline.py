import os
import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class ReturnFeature(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['return'] = X['close'].pct_change()
        X.columns = X.columns.astype(str)
        return X

class SMACalculator(BaseEstimator, TransformerMixin):
    def __init__(self, window):
        self.window = window

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X[f'SMA_{self.window}'] = X['close'].rolling(window=self.window).mean()
        X.columns = X.columns.astype(str)
        return X
    
class ComputeRelativeSmaGAP(BaseEstimator, TransformerMixin):
    def __init__(self, windows):
        self.windows = windows
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        temp_cols = []
        for i, window in enumerate(self.windows):
            X[f'sma{i+1}'] = X['close'].rolling(window=window).mean()
            temp_cols.append(f'sma{i+1}')
            
        for i in range(len(self.windows) - 1):
            X[f'gap_sma{i+1}_sma{i+2}'] = (X[f'sma{i+1}'] - X[f'sma{i+2}']) * 100 / X[f'sma{i+2}']
            
        for i in range(1, len(self.windows)):
            X[f'rel_gap_sma1_sma{i+1}'] = (X['sma1'] - X[f'sma{i+1}']) * 100 / X[f'sma{i+1}']
            X[f"ret_{i}"] = X['close'].pct_change(i)
        
        X['rel_gap_sma2_sma6'] = (X['sma2'] - X['sma6']) * 100 / X['sma6']
        X['rel_gap_sma2_sma5'] = (X['sma2'] - X['sma5']) * 100 / X['sma5']
        
        X['ret'] = X['close'].pct_change()
        X['stage_1'] = np.where(X['gap_sma5_sma6'] > 0, 1, -1)
        X = X.drop(columns = temp_cols)
        X.columns = X.columns.astype(str)
        return X




class RSICalculator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        delta = X['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        X['RSI'] = 100 - (100 / (1 + rs))
        X.columns = X.columns.astype(str)
        return X

class MACDCalculator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        exp1 = X['close'].ewm(span=12, adjust=False).mean()
        exp2 = X['close'].ewm(span=26, adjust=False).mean()
        X['MACD'] = exp1 - exp2
        X['Signal_line'] = X['MACD'].ewm(span=9, adjust=False).mean()
        X.columns = X.columns.astype(str)
        return X

class HighLowRangeCalculator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X['High_Low_Range'] = X['high'] - X['low']
        X.columns = X.columns.astype(str)
        return X

class OBVCalculator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        obv = (np.sign(X['close'].diff()) * X['volume']).fillna(0).cumsum()
        X['OBV'] = obv
        X.columns = X.columns.astype(str)
        return X

class DropTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X.dropna(inplace=True)
        X = X.drop(columns = ['open', 'high', 'low', 'close', 'volume'])
        X.columns = X.columns.astype(str)
        return X

class ScalerWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, scaler):
        self.scaler = scaler

    def fit(self, X, y=None):
        self.scaler.fit(X)
        self.index = X.index
        self.columns = X.columns
        return self

    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        return pd.DataFrame(X_scaled, index=self.index, columns=self.columns)



# Définir et exporter le pipeline
def get_pipeline():
    return Pipeline([
    ('relative_gap', ComputeRelativeSmaGAP(windows = [3, 6, 12, 18, 24, 72, 24*12])),
    ('drop', DropTransformer())
])

"""
# Définir et exporter le pipeline
def get_pipeline():
    return Pipeline([
    ('return_feature', ReturnFeature()),
    ('sma_10', SMACalculator(window=10)),
    ('sma_20', SMACalculator(window=20)),
    ('rsi', RSICalculator()),
    ('macd', MACDCalculator()),
    ('high_low_range', HighLowRangeCalculator()),
    ('obv', OBVCalculator()),
    ('drop', DropTransformer()),
    ('standard_scaler', ScalerWrapper(StandardScaler()))
])
"""


def save_pipeline(filename):
    pipeline = get_pipeline()
    joblib.dump(pipeline, filename)


