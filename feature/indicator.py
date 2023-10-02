import pandas as pd
import numpy as np
import ta
from arch import arch_model


def ema(data, period):
    ema = ta.trend.EMAIndicator(data.close, period)
    return ema.ema_indicator()


def ma(data, period):
    return data.close.rolling(period).mean()


def rsi(data, period):
    RSI = ta.momentum.RSIIndicator(data.close, period)
    return RSI.rsi()


def aroon(data, period):
    """
    Return : Aroon_up, Aroon_down
    """
    Aroon_indicator = ta.trend.AroonIndicator(close = data.close, window=period)
    Aroon_up = Aroon_indicator.aroon_up()
    Aroon_down = Aroon_indicator.aroon_down()
    return Aroon_up, Aroon_down


# ATR
def atr(data, window): 
    Atr = ta.volatility.AverageTrueRange(high = data.high, low = data.low,
                                         close = data.close, window = window)
    return Atr.average_true_range()


def adx(data, window):
    """ 
    Return : ADX.adx , ADX.adx_up, ADX.adx_down
    """
    ADX = ta.trend.ADXIndicator(high = data.high, low = data.low,
                                close = data.close, window =  window)
    
    return ADX.adx(), ADX.adx_neg(), ADX.adx_pos()


# ADX
def adx(data, window):
    """ 
    Return : ADX.adx , ADX.adx_up, ADX.adx_down
    """
    ADX = ta.trend.ADXIndicator(high = data.high, low = data.low,
                                close = data.close, window =  window)
    
    return ADX.adx(), ADX.adx_neg(), ADX.adx_pos()



# SAR
def sar(data, step = 0.2, max_step = 0.2):
    """ 
    Return : SAR.psar, : SAR.psar_up, : SAR.psar_down 
    """
    SAR = ta.trend.PSARIndicator(high = data.high, low = data.low, close = data.close,
                                 step = step, max_step = max_step)
    SAR.psar()
    SAR.psar_down()              # down trend value
    SAR.psar_down_indicator()    # down trend value indicator

    SAR.psar_up()       # up trend value
    SAR.psar_up_indicator() # up trend value indicator
    
    return SAR.psar(), SAR.psar_up(), SAR.psar_down()



def stochastic_oscillator(data, k_period, d_period):
    S_O = ta.momentum.StochasticOscillator(high = data.high, low = data.low,
                                     close = data.close,
                                     window = k_period,
                                     smooth_window = d_period)
    K = S_O.stoch()
    D = S_O.stoch_signal()
    return K, D



def n_day_up(data, period, close = 'close'):
    data['return'] = data[close].pct_change()
    return data['return'].rolling(period).apply(lambda x : np.sum(np.where(x>0, 1, 0))*100/period)


# Bande de bollingers
def bande_bollingers(data, window, wind_dev):
    B_B = ta.volatility.BollingerBands(close = data.close, window = window, 
                                       window_dev = wind_dev)
    B_B.bollinger_hband_indicator()
    """ return 1 or 0 : 1 if close is higher than bollinger_hband, else it return 0 """    
    B_B.bollinger_lband_indicator()
    """ return 1 or 0 : 1 if close is lower than bollinger_lband, else it return 0 """
    
    B_B.bollinger_pband()
    B_B.bollinger_wband()
    return B_B.bollinger_hband(), B_B.bollinger_lband(), B_B.bollinger_mavg()







def macd(data, slow , fast, signal):
    MACD = ta.trend.MACD(close = data.close , window_slow = slow, window_fast = fast,
                         window_sign = signal)
    return MACD.macd() , MACD.macd_diff(), MACD.macd_signal()
    


def GARCH(data):
    ret = data["close"].pct_change().dropna()
    model = arch_model(ret, mean = "Zero", vol = "GARCH", p=1, q=1)
    res = model.fit(update_freq = 5)
    return res.conditional_volatility
    

def GARCH_stochastic(data, freq):
    Min = data['garch'].resample(freq).min()
    Max = data['garch'].resample(freq).max()
    return (data['garch'].resample(freq).last() - Min) / (Max - Min)


def stochastic(data, freq, col = 'close'):
    Min = data[col].resample(freq).min()
    Max = data[col].resample(freq).max()
    return (data[col].resample(freq).last() - Min) / (Max - Min)


