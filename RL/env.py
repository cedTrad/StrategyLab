import random
import enum

env_config = {
    "exchange" : "Binance",
    "symbol" : "BTCUSDT",
    "frequence" : "1d",
    "capital" : 1000,
    "observation_horizon_sequence_length" : 30
}

class action(enum.Enum):
    SKIP = 0
    BUY = 1
    SELL = -1
    CLOSE = 2
    

class observation_space:
    
    def __init__(self, n):
        self.shape = (n,)
        

class action_space:
    
    def __init__(self, n):
        self.n = n
    
    def sample(self):
        return random.randint(0, self.n - 1)



class Env:
    
    def __init__(self, symbol, features):
        self.symbol = symbol
        self.features = features
        
        self.init_capital = self.env_config.get("capital")
        self.frequence = self.env_config.get("frequence")
        
        self.observation_space = observation_space(3)
        self.osn = self.observation_space.shape[0]
        self.action_space = action_space(2)
        self.min_accuracy = 0.475
        
        self.observation_features = ["rets", "mom", "c1", "c2"]
        self.horizon = env_config.get("observation_horizon_sequence_length")
        
        
    def get_data(self):
        return
    
    
    def preprocessing(self):
        self.data['ret'] = self.data["close"].pct_change()
        self.data["log_ret"] = np.log(self.data["close"] / self.data["close"].shift(1))
        self.data["mom"] = self.data["rets"].rolling(3).mean()
        self.data["ma_1"] = self.data["close"].rolling(5).mean()
        self.data["ma_2"] = self.data["close"].rolling(14).mean()
        self.data["ma_3"] = self.data["close"].rolling(21).mean()
        
        self.data["c1"] = self.data["ma_2"] / self.data["ma_1"]
        self.data["c2"] = self.data["ma_3"] / self.data["ma_2"]
        
        return
    
    def render(self):
        return
    
    
    def get_state(self):
        self.data[features].iloc[self.bar - self.osn : self.bar].values

    
    def reset(self):
        self.init_capital = self.env_config.get("capital")
        self.cost = 0
        self.treward = 0
        self.accuracy = 0
        self.bar = self.osn
        state = self.get_state()
    
    
    def get_reward(self):
        reward = self.data["ret"].iloc[self.bar]
        return reward
        
        
    def step(self, action):
        rets = self.data['ret'].iloc[self.bar]
        reward = 1 if rets > 0 else 0
        self.treward += reward
        self.bar += 1
        self.accuracy = self.treward / (self.bar - self.osn)
        
        if self.bar >= len(self.data):
            done = True
        elif reward == 1:
            done = False
        
        elif (self.accuracy < self.min_accuracy) and (self.bar > self.osn +10):
            done = True
        
        else:
            done = False
        
        state = self.get_state()
        info = {}
        return state, reward, done, info
    
        