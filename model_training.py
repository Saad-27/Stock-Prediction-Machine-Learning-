import pandas as pd
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

def create_model(units=100, dropout_rate=0.3, optimizer='adam', activation='relu'):
    model = Sequential()
    model.add(LSTM(units, activation=activation, input_shape=(1, 8)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='linear'))  # Linear activation for regression
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

def train_model(X_train, y_train, X_test, y_test, units=100, dropout_rate=0.3, optimizer='adam', activation='relu', epochs=100, batch_size=20):
    model = create_model(units, dropout_rate, optimizer, activation)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stopping], verbose=1)
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    print(f"Model MSE: {mse}")
    
    return model

if __name__ == "__main__":
    df = pd.read_csv('featured_stock_data.csv', parse_dates=['time'])
    X_train, X_test, y_train, y_test = prepare_data(df, prediction_horizon=30)
    
    # Best parameters found from hyperparameter tuning
    # testing done on multiple parameters, very load heavy on PC
    # param_dist = {
    #    'units': [10, 50, 100, 150, 200],
    #    'dropout_rate': [0.2, 0.3, 0.4, 0.5],
    #    'optimizer': ['adam', 'rmsprop', 'sgd', 'adagrad'],
    #    'activation': ['relu', 'tanh', 'sigmoid'],
    #    'epochs': [50, 100, 150],
    #    'batch_size': [10, 20, 30, 40]
    #}
    best_params = {
        'units': 100,
        'dropout_rate': 0.3,
        'optimizer': 'adam',
        'activation': 'relu',
        'epochs': 100,
        'batch_size': 20
    }
    
    model = train_model(X_train, y_train, X_test, y_test, **best_params)
    
    # Save the trained model
    model.save('best_stock_lstm_model.h5')
