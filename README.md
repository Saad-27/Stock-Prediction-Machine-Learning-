Overview
This project implements a Long Short-Term Memory (LSTM) model to predict future stock prices based on recently recorded data. Predicts the next 5 minutes of the stock, using data sampled every 10 seconds. Out put will be either "UP 00.00" or "Down 00.00"

Project Structure

data_processing.py:
Handles the initial data processing and feature engineering
data_preparation.py:
Prepares the processed data for training, creating the necessary input features
model_training.py:
Builds and trains the LSTM model using the best hyperparameters found from previous tuning. Note this process was very intensive on hardware
make_prediction.py:
Loads the trained LSTM model and makes predictions on whether the stock price will go up or down, including the change in price

Important Notes

Data Sampling Interval:
The model is designed for stock data sampled every 10 seconds. Ensure that the input data follows this format for accurate predictions. Most publically available data does not follow this structure

Prediction Horizon:
The model predicts the stock price 5 minutes (30 data points) into the future.

Model Type:
This is a regression model, meaning it predicts a continuous value (the future stock price), which is then compared to the current price to determine the direction and magnitude of the price.

Feature Engineering:
Key features include lagged prices, rolling mean, and rolling standard deviation, as well as time-based features like the hour, day, and month.

Hyperparameters:
The model uses predefined "best" hyperparameters for training. If you wish to experiment with different parameters, you can modify the model_training.py script accordingly.

Prerequisites

Python 3.7+
TensorFlow 2.x
Pandas
NumPy
Scikit-learn


Run process:

Run the data processing script to generate the required features:
python data_processing.py
This script will generate a featured_stock_data.csv file containing the processed data.

Prepare the data for model training by running:
python data_preparation.py

Train the LSTM model with the prepared data:
python model_training.py
This script will train the model and save it as best_stock_lstm_model.h5.

Make Predictions:
Once the model is trained, you can use it to make predictions:
python make_prediction.py
The prediction output will indicate whether the stock price is expected to go "UP" or "DOWN" in the next 5 minutes, and by how much.

Example Usage

Prediction: UP 12.50
This means that based on the current and past data, the model predicts the stock price will increase by 12.50 units over the next 5 minutes.

Customization

prediction_horizon variable in the data_preparation.py script is the determining variable for how far ahead you want to predict

Experimenting with Hyperparameters:

If you wish to experiment with different hyperparameters, modify the best_params dictionary in model_training.py and retrain the model.
