import os
import random
from typing import dict

import gym
import numpy as np
import pandas as pd
from gym import spaces


env_config = {
    "exchange" : "Binance",
    "ticker" : "BTCUSDT",
    "frequency" : "daily",
    "opening_account_balance" : 10000,
    "obervation_horizon_sequence_lengh" : 30
}


