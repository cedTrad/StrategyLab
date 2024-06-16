import joblib

def process_batches(data, batch_size=30):
    pipeline = joblib.load('feature_pipeline.pkl')
    model = joblib.load('random_forest_model.pkl')
    
    num_batches = len(data) // batch_size
    predictions = []
    
    for i in range(num_batches):
        batch_data = data.iloc[i*batch_size:(i+1)*batch_size]
        if len(batch_data) < batch_size:
            break
        X_transformed = pipeline.transform(batch_data)
        batch_predictions = model.predict(X_transformed)
        predictions.extend(batch_predictions)
    
    return predictions
