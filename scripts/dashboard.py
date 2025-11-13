import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc

# --- Load data ---
df = pd.read_csv("data/cleaned_flights.csv")
df_clusters = pd.read_csv("unsupervised_clusters_output.csv")

# --- Basic checks ---
print(df["ARRIVAL_DELAY"].dtype)
print(df["ARRIVAL_DELAY"].isna().sum())
print(df.groupby("MONTH")["ARRIVAL_DELAY"].mean())

# --- Dynamic delay trend graph ---
if df["MONTH"].nunique() > 1:
    trend_df = df.groupby("MONTH")["ARRIVAL_DELAY"].mean().reset_index()
    trend_fig = px.line(
        trend_df, x="MONTH", y="ARRIVAL_DELAY",
        title="Average Arrival Delay by Month", markers=True
    )
elif df["DAY"].nunique() > 1:
    trend_df = df.groupby("DAY")["ARRIVAL_DELAY"].mean().reset_index()
    trend_fig = px.line(
        trend_df, x="DAY", y="ARRIVAL_DELAY",
        title="Average Arrival Delay by Day", markers=True
    )
else:
    trend_df = df.groupby("DEPARTURE_HOUR")["ARRIVAL_DELAY"].mean().reset_index()
    trend_fig = px.line(
        trend_df, x="DEPARTURE_HOUR", y="ARRIVAL_DELAY",
        title="Average Arrival Delay by Departure Hour", markers=True
    )

# --- Other EDA figures ---
delay_dist = px.histogram(df, x="ARRIVAL_DELAY", nbins=50, title="Delay Distribution")
airline_avg = px.bar(
    df.groupby("AIRLINE_CODE")["ARRIVAL_DELAY"].mean().reset_index(),
    x="AIRLINE_CODE", y="ARRIVAL_DELAY", title="Average Delay by Airline"
)

cluster_scatter = px.scatter(
    df_clusters, x="DISTANCE", y="ARRIVAL_DELAY",
    color="KMeans_Cluster", title="Clusters (Delay vs Distance)"
)

# --- Dash App ---
app = Dash(__name__)
app.title = "Airline Delay Analytics Dashboard"

app.layout = html.Div([
    html.H1("Airline Analytics Dashboard", style={"textAlign": "center"}),
    dcc.Tabs([
        dcc.Tab(label='EDA Overview', children=[
            dcc.Graph(figure=delay_dist),
            dcc.Graph(figure=airline_avg),
            dcc.Graph(figure=trend_fig)
        ]),
        dcc.Tab(label='Unsupervised Insights', children=[
            dcc.Graph(figure=cluster_scatter)
        ])
    ])
])

if __name__ == "__main__":
    app.run(debug=True)
