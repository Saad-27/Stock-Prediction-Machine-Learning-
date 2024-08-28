import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data(df, prediction_horizon=30):
    features = ['price', 'hour', 'day', 'month', 'price_lag1', 'price_lag2', 'price_rolling_mean_3', 'price_rolling_std_3']
    
    # Create the target variable (future price after 5 minutes)
    df['future_price'] = df['price'].shift(-prediction_horizon)
    df.dropna(inplace=True)
    
    X = df[features].values
    y = df['future_price'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Reshape input data for LSTM (samples, time steps, features)
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df = pd.read_csv('featured_stock_data.csv', parse_dates=['time'])
    X_train, X_test, y_train, y_test = prepare_data(df, prediction_horizon=30)
