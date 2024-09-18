from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model(X, y):    
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    
    joblib.dump(model, 'random_forest_model.pkl')
    return model