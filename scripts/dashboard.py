import pandas as pd
import plotly.express as px
from dash import Dash, html, dcc

# --- Load data ---
df = pd.read_csv("data/cleaned_flights.csv")
df_clusters = pd.read_csv("unsupervised_clusters_output.csv")

# --- Basic checks ---
print("ARRIVAL_DELAY dtype:", df["ARRIVAL_DELAY"].dtype)
print("Missing ARRIVAL_DELAY values:", df["ARRIVAL_DELAY"].isna().sum())

if "MONTH" in df.columns:
    print("Average delay by month:")
    print(df.groupby("MONTH")["ARRIVAL_DELAY"].mean(), "\n")

# --- Dynamic trend graph ---
if "MONTH" in df.columns and df["MONTH"].nunique() > 1:
    trend_df = df.groupby("MONTH")["ARRIVAL_DELAY"].mean().reset_index()
    trend_fig = px.line(
        trend_df, x="MONTH", y="ARRIVAL_DELAY",
        title="Average Arrival Delay by Month", markers=True
    )
elif "DAY" in df.columns and df["DAY"].nunique() > 1:
    trend_df = df.groupby("DAY")["ARRIVAL_DELAY"].mean().reset_index()
    trend_fig = px.line(
        trend_df, x="DAY", y="ARRIVAL_DELAY",
        title="Average Arrival Delay by Day", markers=True
    )
else:
    if "DEPARTURE_HOUR" in df.columns:
        trend_df = df.groupby("DEPARTURE_HOUR")["ARRIVAL_DELAY"].mean().reset_index()
        trend_fig = px.line(
            trend_df, x="DEPARTURE_HOUR", y="ARRIVAL_DELAY",
            title="Average Arrival Delay by Departure Hour", markers=True
        )
    else:
        trend_fig = px.scatter(title="No Time-based Trend Available")

# --- EDA Visuals ---
delay_dist = px.histogram(
    df, x="ARRIVAL_DELAY", nbins=50,
    title="Distribution of Arrival Delays"
)

airline_avg = px.bar(
    df.groupby("AIRLINE_CODE")["ARRIVAL_DELAY"].mean().reset_index(),
    x="AIRLINE_CODE", y="ARRIVAL_DELAY",
    title="Average Arrival Delay by Airline"
)

# --- Unsupervised Cluster Plots ---
cluster_figs = []
if "KMeans_Cluster" in df_clusters.columns and "DISTANCE" in df_clusters.columns:
    cluster_figs.append(
        px.scatter(
            df_clusters, x="DISTANCE", y="ARRIVAL_DELAY",
            color="KMeans_Cluster",
            title="KMeans Clustering: Delay vs Distance"
        )
    )

if "DBSCAN_Cluster" in df_clusters.columns and "DISTANCE" in df_clusters.columns:
    cluster_figs.append(
        px.scatter(
            df_clusters, x="DISTANCE", y="ARRIVAL_DELAY",
            color="DBSCAN_Cluster",
            title="DBSCAN Clustering: Delay vs Distance"
        )
    )

# --- Dash App ---
app = Dash(__name__)
app.title = "✈️ Airline Delay Analytics Dashboard"

app.layout = html.Div([
    html.H1("Airline Delay Analytics Dashboard", style={"textAlign": "center"}),
    dcc.Tabs([
        dcc.Tab(label='EDA Overview', children=[
            dcc.Graph(figure=delay_dist),
            dcc.Graph(figure=airline_avg),
            dcc.Graph(figure=trend_fig)
        ]),
        dcc.Tab(label='Unsupervised Insights', children=[
            *[dcc.Graph(figure=fig) for fig in cluster_figs]
        ])
    ])
])

if __name__ == "__main__":
    app.run(debug=True)
