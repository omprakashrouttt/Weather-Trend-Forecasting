import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

def train_arima(df):
    # Ensure temperature column is numeric
    df['temperature_celsius'] = pd.to_numeric(df['temperature_celsius'], errors='coerce')

    # Handle missing values (forward-fill)
    df['temperature_celsius'].ffill(inplace=True)

    # Train-test split (80-20)
    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]

    # Fit ARIMA model
    model = ARIMA(train['temperature_celsius'], order=(5,1,0))
    model_fit = model.fit()

    # Generate predictions
    start = len(train)
    end = start + len(test) - 1
    predictions = model_fit.predict(start=start, end=end, dynamic=False)

    # Evaluate Model
    mae = mean_absolute_error(test['temperature_celsius'], predictions)
    mse = mean_squared_error(test['temperature_celsius'], predictions)
    print(f"Model Performance: MAE={mae:.4f}, MSE={mse:.4f}")

    # Plot Results
    plt.figure(figsize=(12, 6))
    plt.plot(test.index, test['temperature_celsius'], label="Actual")
    plt.plot(test.index, predictions, label="Predicted", linestyle="dashed")
    plt.legend()
    plt.title("Temperature Forecast using ARIMA")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.grid()
    plt.savefig("outputs/arima_forecast.png")
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv("data/GlobalWeatherRepository.csv", parse_dates=['last_updated'], index_col='last_updated')
    train_arima(df)
