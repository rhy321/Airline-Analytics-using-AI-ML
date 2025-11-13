import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

flights = pd.read_csv('cleaned_flights.csv')

print(flights.shape)
print(flights.head())
print(flights.info())
print(flights.describe())
print((flights['ARRIVAL_DELAY'] > 0).mean() * 100)

#delay distribution 
sns.histplot(flights['ARRIVAL_DELAY'], bins=50, kde=True)
plt.title('Distribution of Arrival Delays')
plt.xlabel('Arrival Delay (minutes)')
plt.ylabel('Frequency')
plt.show()

#avg delay per airline
avg_delay_airline = flights.groupby('AIRLINE')['ARRIVAL_DELAY'].mean().sort_values()

sns.barplot(x=avg_delay_airline.index, y=avg_delay_airline.values)
plt.title('Average Arrival Delay per Airline')
plt.xticks(rotation=45)
plt.ylabel('Average Delay (min)')
plt.show()

#avg delay per route
flights['ROUTE'] = flights['ORIGIN_AIRPORT'] + ' → ' + flights['DESTINATION_AIRPORT']

avg_delay_route = flights.groupby('ROUTE')['ARRIVAL_DELAY'].mean().sort_values(ascending=False).head(10)
avg_delay_route.plot(kind='barh', figsize=(8,5))
plt.title('Top 10 Routes with Longest Average Delays')
plt.xlabel('Average Delay (min)')
plt.show()

#avg delay by month and time of day

# By Month
sns.lineplot(x='MONTH', y='ARRIVAL_DELAY', data=flights)
plt.title('Average Delay by Month')

# By Time of Day (e.g., DEP_TIME)
sns.histplot(flights, x='DEP_TIME', y='ARRIVAL_DELAY', bins=50, pmax=0.9)
plt.title('Delay by Time of Day')
plt.show()

# Top 10 airports with most delays
top_delays = flights[flights['ARRIVAL_DELAY'] > 15]['ORIGIN_AIRPORT'].value_counts().head(10)
sns.barplot(x=top_delays.values, y=top_delays.index)
plt.title('Top 10 Airports with Most Delayed Departures')
plt.xlabel('Number of Delays')
plt.show()

# For cancellations
if 'CANCELLED' in flights.columns:
    top_cancel = flights[flights['CANCELLED'] == 1]['ORIGIN_AIRPORT'].value_counts().head(10)
    sns.barplot(x=top_cancel.values, y=top_cancel.index)
    plt.title('Top 10 Airports with Most Cancellations')
    plt.xlabel('Number of Cancellations')
    plt.show()

# Correlation heatmap
numeric = flights.select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=(10,8))
sns.heatmap(numeric.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numerical Features')
plt.show()
