import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import os

# Subject color map
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

# Load data
base_path = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(base_path, "data/embedding_2d_est_basic_school.csv"))

# Generate color list and hover texts
color_list = [subject_color_map[row["subject"]] for _, row in df.iterrows()]
hover_texts = [row["text"].replace("\n", "<br>") for _, row in df.iterrows()]

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
    title="Semantic map of paragraphs in Estonian basic school curriculum",
    xaxis_title="Dimension 1 (The axis have no meaning. Only distances of plots matter.)",
    yaxis_title="Dimension 2",
    legend_title="Subjects",
    legend=dict(x=1.05, y=1),
    margin=dict(t=40, l=0, r=150, b=40),
    width=800,
    height=800,
    hovermode='closest',
    hoverlabel=dict(
        bgcolor="white",
        font_size=12,
        font_family="Arial",
        align="left"
    )
)

# Streamlit app
st.title("Semantic Map of Paragraphs in Estonian Basic School Curriculum")
st.plotly_chart(scatter_fig, use_container_width=True)
