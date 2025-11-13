import pandas as pd
import os

data_dir = os.path.join(os.getcwd(), "data")

files = {
    "Flights": os.path.join(data_dir, "Flights.csv"),
    "Airlines": os.path.join(data_dir, "Airlines.csv"),
    "Airports": os.path.join(data_dir, "Airports.csv"),
}

for name, path in files.items():
    print(f"\n--- {name} ---")
    try:
        df = pd.read_csv(path, low_memory=False)
        print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        print("Columns:", df.columns.tolist())
        print("\nFirst 3 rows:")
        print(df.head(3).to_string(index=False))
    except Exception as e:
        print(f"Error reading {name}: {e}")
