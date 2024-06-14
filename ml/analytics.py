import numpy as np
import pandas as pd

def exponential_decay(data, lambda_decay):
    n = len(data)
    decay_weights = np.exp(-lambda_decay * np.arange(n))
    return data * decay_weights

def return_attribution(data, model, features):
    predictions = model.predict(data[features])
    weighted_returns = predictions * data['return']
    
    attribution = {}
    for feature in features:
        feature_importance = model.feature_importances_[features.index(feature)]
        attribution[feature] = feature_importance * weighted_returns.sum()
    
    return attribution
