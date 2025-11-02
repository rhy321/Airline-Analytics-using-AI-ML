import pandas as pd

flights = pd.read_csv('data/Flights.csv')
airlines = pd.read_csv('data/Airlines.csv')
airports = pd.read_csv('data/Airports.csv')

flights.info()
flights.head()
flights.isnull().sum()
