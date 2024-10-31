import numpy as np

from ta.trend import MACD
from ta.volatility import DonchianChannel
from ta.volume import VolumeWeightedAveragePrice


def calculate_vwap(df, window):
    vwap = VolumeWeightedAveragePrice(high=df['high'], low=df['low'],
                                      close=df['close'], volume=df['volume'],
                                      window=window, fillna=False)
    df['vwap'] = vwap.volume_weighted_average_price()


def calculate_vwma(df, window=24):
    price_volume = df['close'] * df['volume']
    vwma = price_volume.rolling(window=window).sum() / df['Volume'].rolling(window=window).sum()
    df['vwma'] = vwma


def calculate_donchian_channel(df, window=20):
    donchian = DonchianChannel(high=df['high'], low=df['low'],
                               close=df['close'], window=window, fillna=False)
    df['dc_high'] = donchian.donchian_channel_hband()
    df['dc_low'] = donchian.donchian_channel_lband()
    df['dc_mid'] = donchian.donchian_channel_mband()


def calculate_macd(df, window_slow, window_fast, window_sign):
    macd = MACD(close=df['close'], window_slow=window_slow,
                window_fast=window_fast, window_sign=window_sign, 
                fillna=False)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()






def compute_relative_sma_gaps(df, sma_windows, col='close'):
    temp_cols = []
    for i, window in enumerate(sma_windows):
        df[f'sma{i+1}'] = df[col].rolling(window=window).mean()
        temp_cols.append(f'sma{i+1}')
        
    for i in range(len(sma_windows) - 1):
        df[f'gap_sma{i+1}_sma{i+2}'] = (df[f'sma{i+1}'] - df[f'sma{i+2}']) * 100 / df[f'sma{i+2}']
    for i in range(1, len(sma_windows)):
        df[f'rel_gap_sma1_sma{i+1}'] = (df['sma1'] - df[f'sma{i+1}']) * 100 / df[f'sma{i+1}']
        df[f"ret_{i}"] = df['close'].pct_change(i)
    
    return temp_cols   



def create_stage_3(data):
    data['stage_1_change'] = (data['stage_1'] != data['stage_1'].shift()).cumsum()
    data['group'] = np.where(data['stage_1'] != 0, data['stage_1_change'], np.nan)
    data['group'] = data['group'].fillna(method='ffill')
    group_confirmed = data.groupby('group')['stage_2_1'].transform('max') == 1
    data['stage_3'] = np.where((group_confirmed) & (data['stage_1'] != 0), data['stage_1'], 0)
    data.drop(columns=['stage_1_change', 'group'], inplace=True)


def create_stage_4(data):
    data['stage_4'] = 0
    # Condition 1 : gap_sma5_sma6 > rel_gap_sma2_sma6 et stage_3 == 1
    condition1 = (data['gap_sma5_sma6'] > data['rel_gap_sma2_sma6']) & (data['stage_3'] == 1)    
    # Condition 2 : gap_sma5_sma6 < rel_gap_sma2_sma6 et stage_3 == -1
    condition2 = (data['gap_sma5_sma6'] < data['rel_gap_sma2_sma6']) & (data['stage_3'] == -1)
    data.loc[condition1 | condition2, 'stage_4'] = 1
    




