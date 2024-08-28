import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np

def predict_future_price(model, X, current_price):
    predicted_price = model.predict(X.reshape(1, 1, X.shape[1]))[0][0]
    direction = "UP" if predicted_price > current_price else "DOWN"
    change = abs(predicted_price - current_price)
    return f"{direction} {change:.2f}"

if __name__ == "__main__":
    df = pd.read_csv('featured_stock_data.csv', parse_dates=['time'])
    
    # Load the trained model
    model = load_model('best_stock_lstm_model.h5')
    
    # Example prediction
    current_index = 0 
    
    # Extract current price from the specified row
    current_price = df.iloc[current_index]['price']
    
    # Extract features from the same row
    X_current = df.iloc[current_index][['price', 'hour', 'day', 'month', 'price_lag1', 'price_lag2', 'price_rolling_mean_3', 'price_rolling_std_3']].values
    
    # Make prediction and format the output
    prediction = predict_future_price(model, X_current, current_price)
    print(f"Prediction: {prediction}")
