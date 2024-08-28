import pandas as pd

def process_data(input_csv, output_csv):
    df = pd.read_csv(input_csv, parse_dates=['time'])
    
    # Extracting useful time-based features
    df['hour'] = df['time'].dt.hour
    df['day'] = df['time'].dt.day
    df['month'] = df['time'].dt.month
    
    # Creating lagged features
    df['price_lag1'] = df['price'].shift(1)
    df['price_lag2'] = df['price'].shift(2)
    
    # Creating rolling statistics features
    df['price_rolling_mean_3'] = df['price'].rolling(window=3).mean()
    df['price_rolling_std_3'] = df['price'].rolling(window=3).std()
    
    # Dropping any rows with NaN values (due to lagging/rolling operations)
    df.dropna(inplace=True)
    
    # Saving processed data
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    process_data('raw_stock_data.csv', 'featured_stock_data.csv')
