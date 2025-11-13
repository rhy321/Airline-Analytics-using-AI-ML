import pandas as pd
import os

# Load all datasets
data_dir = os.path.join(os.getcwd(), "data")
flights = pd.read_csv(os.path.join(data_dir, "Flights.csv"))
airlines = pd.read_csv(os.path.join(data_dir, "Airlines.csv"))
airports = pd.read_csv(os.path.join(data_dir, "Airports.csv"))

print("Loaded datasets:")
print(f"Flights: {flights.shape}")
print(f"Airlines: {airlines.shape}")
print(f"Airports: {airports.shape}")

# --- Merge 1: add airline names ---
flights = flights.merge(airlines, left_on="AIRLINE", right_on="IATA_CODE", how="left")
flights.rename(columns={"AIRLINE_y": "AIRLINE_NAME", "AIRLINE_x": "AIRLINE_CODE"}, inplace=True)
flights.drop(columns=["IATA_CODE"], inplace=True)

# --- Merge 2: add origin airport details ---
flights = flights.merge(
    airports.add_suffix("_ORIGIN"),
    left_on="ORIGIN_AIRPORT",
    right_on="IATA_CODE_ORIGIN",
    how="left",
)

# --- Merge 3: add destination airport details ---
flights = flights.merge(
    airports.add_suffix("_DEST"),
    left_on="DESTINATION_AIRPORT",
    right_on="IATA_CODE_DEST",
    how="left",
)

print("After merging:", flights.shape)

# --- Data Cleaning ---
# Drop duplicates
flights.drop_duplicates(inplace=True)

# Drop rows missing critical time data
flights.dropna(subset=["DEPARTURE_TIME", "ARRIVAL_TIME"], inplace=True)

# Remove unrealistic delays
flights = flights[
    flights["DEPARTURE_DELAY"].between(-60, 1000)
    & flights["ARRIVAL_DELAY"].between(-60, 1000)
]

# --- Feature Engineering ---
# Departure hour (convert e.g., 2354 -> 23)
flights["DEPARTURE_HOUR"] = (flights["SCHEDULED_DEPARTURE"] // 100).astype(int)

# Create binary label for delayed flights
flights["DELAYED"] = (flights["ARRIVAL_DELAY"] > 15).astype(int)

# Total delay
flights["TOTAL_DELAY"] = flights["DEPARTURE_DELAY"] + flights["ARRIVAL_DELAY"]

# --- Save cleaned dataset ---
output_path = os.path.join(data_dir, "cleaned_flights.csv")
flights.to_csv(output_path, index=False)
print(f"\n✅ Cleaned dataset saved to: {output_path}")
print("Final shape:", flights.shape)
