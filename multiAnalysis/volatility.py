import pandas as pd
import numpy as np 
from arch import arch_model



# GARCH
def GARCH(ret : pd.Series):
    ret = ret.dropna()
    model = arch_model(ret, mean = "Zero", vol = "GARCH", p=1, q=1)
    res = model.fit(update_freq = 5, disp="off")
    print(res.summary())
    return res.conditional_volatility



# DCC-GARCH


