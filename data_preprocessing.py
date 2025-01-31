import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    """Loads the dataset from a CSV file."""
    try:
        df = pd.read_csv(filepath)

        # Debugging: Print available columns
        print("Columns in CSV:", df.columns.tolist())

        # Ensure 'last_updated' exists before proceeding
        if 'last_updated' not in df.columns:
            raise KeyError(f"Column 'last_updated' not found. Available columns: {df.columns.tolist()}")

        df['last_updated'] = pd.to_datetime(df['last_updated'])
        df.set_index('last_updated', inplace=True)
        return df

    except FileNotFoundError:
        print(f"❌ Error: File '{filepath}' not found. Please check the file path.")
        exit(1)
    except pd.errors.EmptyDataError:
        print(f"❌ Error: File '{filepath}' is empty.")
        exit(1)
    except Exception as e:
        print(f"❌ Unexpected error in load_data(): {e}")
        exit(1)

def clean_data(df):
    """Handles missing values and removes outliers."""
    try:
        # Convert necessary columns to numeric
        num_cols = ['temperature_celsius', 'humidity', 'precip_mm']
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to number, replace errors with NaN

        # Forward-fill missing values
        df.ffill(inplace=True)

        # Mean imputation for remaining NaNs
        df.fillna(df.mean(numeric_only=True), inplace=True)

        # Remove outliers using IQR
        Q1 = df[num_cols].quantile(0.25)
        Q3 = df[num_cols].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df[num_cols] < (Q1 - 1.5 * IQR)) | (df[num_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

        return df

    except Exception as e:
        print(f"❌ Unexpected error in clean_data(): {e}")
        exit(1)

def scale_features(df):
    """Normalizes numerical features using StandardScaler."""
    try:
        required_columns = ['temperature_celsius', 'humidity', 'precip_mm']

        # Check if required columns exist
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise KeyError(f"❌ Error: Missing columns: {missing_cols}")

        scaler = StandardScaler()
        df[required_columns] = scaler.fit_transform(df[required_columns])
        return df

    except KeyError as e:
        print(e)
        exit(1)
    except Exception as e:
        print(f"❌ Unexpected error in scale_features(): {e}")
        exit(1)

# Test the functions
if __name__ == "__main__":
    filepath = "data/GlobalWeatherRepository.csv"

    df = load_data(filepath)
    df = clean_data(df)
    df = scale_features(df)

    print("✅ Preprocessing Complete. Data Ready!")
