import dash
from dash import dcc, html
import plotly.graph_objects as go
import numpy as np
import pandas as pd

import os

from flask import Flask


subject_color_map = {
        "language_and_literature": "red",
        "foreign_languages": "purple",
        "mathematics": "blue",
        "natural_science": "green",
        "social_studies": "orange",
        "art": "pink",
        "technology": "cyan",
        "physical_education": "yellow",
        "religious_studies": "brown",
        "informatics": "teal",
        "career_education": "magenta",
        "entrepreneurship_studies": "lime",
    }

base_path = os.path.dirname(__file__)

df = pd.read_csv(os.path.join(base_path, "data/embedding_2d_est_basic_school.csv"))

color_list = [subject_color_map[row["subject"]] for idx, row in df.iterrows()]
hover_texts = [row["text"].replace("\n", "<br>") for idx, row in df.iterrows()]

# Create the scatter plot
scatter_fig = go.Figure()

# Add the scatter trace with hover text
scatter_fig.add_trace(go.Scatter(
    x=df["x"],
    y=df["y"],
    mode='markers',
    marker=dict(
        color=color_list,
        size=6
    ),
    text=hover_texts,  # Add hover text
    hoverinfo='text',  # Display only the hover text
    name="Scatter Points"
))

# Add legend items for each subject
for subject, color in subject_color_map.items():
    scatter_fig.add_trace(go.Scatter(
        x=[None], y=[None],  # Dummy points for legend
        mode='markers',
        marker=dict(size=10, color=color),
        name=subject
    ))

# Update layout
scatter_fig.update_layout(
    title="Scatter Plot with Hover Texts",
    xaxis_title="Dimension 1",
    yaxis_title="Dimension 2",
    legend_title="Subjects",
    legend=dict(x=1.05, y=1),
    margin=dict(t=40, l=0, r=150, b=40),
    width=800,
    height=800, 
    hovermode='closest',  # Ensure hover displays the closest point
    hoverlabel=dict(
        bgcolor="white",  # Background color of hover labels
        font_size=12,
        font_family="Arial",
        align="left"  # Align hover text to the left to avoid truncation
    )
)

# Initialize the Dash app
# app = dash.Dash(__name__)
flask_server = Flask(__name__)
app = dash.Dash(__name__, server=flask_server)
server = app.server

# Define the layout
app.layout = html.Div([
    html.H1("Scatter Plot with Hover Texts", style={'textAlign': 'center'}),
    dcc.Graph(
        id='scatter-plot',
        figure=scatter_fig
    )
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)