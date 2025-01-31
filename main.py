from data_preprocessing import load_data, clean_data, scale_features
from eda import plot_temperature_trends, plot_correlation
from model import train_arima
from utils import check_data_exists

# Load and clean data
filepath = "data/GlobalWeatherRepository.csv"
check_data_exists(filepath)
df = load_data(filepath)
df = clean_data(df)
df = scale_features(df)

# Run EDA
plot_temperature_trends(df)
plot_correlation(df)

# Train Model
train_arima(df)
