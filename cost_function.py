import pandas as pd 
import numpy as np 


def sharpeRatio():
    ""

def f1(market_ret, strategy_ret):
    cum_market = (market_ret + 1).cumprod()
    cum_strategy = (strategy_ret + 1).cumprod()
    