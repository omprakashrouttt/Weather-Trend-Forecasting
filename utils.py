import os

def check_data_exists(filepath):
    """Checks if the dataset file exists."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

if __name__ == "__main__":
    check_data_exists("data/GlobalWeatherRepository.csv")
    print("File found. Ready to proceed!")

