import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_temperature_trends(df):
    plt.figure(figsize=(12, 5))
    sns.lineplot(x=df.index, y=df['temperature_celsius'])
    plt.xticks(rotation=45)
    plt.title("Temperature Trends Over Time")
    plt.savefig("outputs/temp_trends.png")
    plt.show()

def plot_correlation(df):
    """Plots a heatmap of correlation matrix using only numeric columns."""
    numeric_df = df.select_dtypes(include=['number'])  # Select only numeric columns

    plt.figure(figsize=(12, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.savefig("outputs/correlation_heatmap.png")
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv("data/GlobalWeatherRepository.csv", parse_dates=['lastupdated'], index_col='lastupdated')
    plot_temperature_trends(df)
    plot_correlation(df)
