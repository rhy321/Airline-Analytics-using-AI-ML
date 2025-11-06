Phase 1 – Data Preparation & Integration
---------------------------------------
Tasks Completed:
• Loaded Flights, Airlines, and Airports datasets.
• Merged them on matching IATA / Airport codes.
• Removed duplicates, missing time data, and unrealistic delays.
• Engineered features:
  - DEPARTURE_HOUR
  - DELAYED (binary target)
  - TOTAL_DELAY (sum of departure + arrival delays)
• Final dataset: 97,378 rows × 49 columns.
• Output file: data/cleaned_flights.csv

Dataset is now ready for Phase 2 (Modeling and Analysis).
