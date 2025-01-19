import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd

import weaviate
from weaviate.classes.config import Configure
from weaviate.classes.init import Auth
from weaviate.classes.query import MetadataQuery


import sys
import os
from dotenv import load_dotenv

load_dotenv()

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

weaviate_url = os.environ["WEAVIATE_URL"]
weaviate_api_key = os.environ["WEAVIATE_API_KEY"]
cohere_api_key = os.environ["COHERE_API_KEY"]

def query_vector(query_text):
    class_name = "CurriculumDemo"
        # Connect to Weaviate Cloud
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,                                    # Replace with your Weaviate Cloud URL
        auth_credentials=Auth.api_key(weaviate_api_key),             # Replace with your Weaviate Cloud key
        headers={"X-Cohere-Api-Key": cohere_api_key},           # Replace with your Cohere API key
    )

    collection = client.collections.get("CurriculumDemo")

    response = collection.query.near_text(
        query=query_text,
        limit=30, 
        return_metadata=MetadataQuery(distance=True, certainty=True)
    )

    row_dict_list = []
    for obj in response.objects:
        row_dict = {}
        row_dict["text"] = obj.properties["text"]
        row_dict["subject"] = obj.properties["subject"]
        row_dict["certainty"] = obj.metadata.certainty
        row_dict["distance"] = obj.metadata.distance
        
        row_dict_list.append(row_dict)

    client.close()  # Free up resources

    df_queried = pd.DataFrame(row_dict_list, columns=["text", "subject", "certainty", "distance"])
    df_queried["text"] = df_queried["text"].map(lambda x: x.replace("\n", "<br>") )
    df_queried["dummy"] = df_queried["text"].map(lambda x: "")
    
    return df_queried


# Load data
base_path = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(base_path, "data/embedding_2d_est_basic_school.csv"))

# Generate color list and hover texts
color_list = [subject_color_map[row["subject"]] for _, row in df.iterrows()]
hover_texts = [row["text"].replace("\n", "<br>") for _, row in df.iterrows()]

# # Create the scatter plot
# scatter_fig = go.Figure()

# # Add legend items for each subject
# for subject, color in subject_color_map.items():
#     scatter_fig.add_trace(go.Scatter(
#         x=[None], y=[None],  # Dummy points for legend
#         mode='markers',
#         marker=dict(size=10, color=color),
#         name=subject
#     ))

st.title("Estonian Basic School Curriculum Analyzer Demo")

# Create the subplots with an additional row for the bottom graph
fig = make_subplots(
    rows=2,  # Increase the rows to 2
    cols=2,  # Keep the columns as 2 for the top row
    column_widths=[0.2, 0.8],  # Set column widths for the first row
    row_heights=[0.6, 0.4],  # Adjust row heights (optional)
    subplot_titles=("The most relevant texts", "Searched texts grouped by subjects", "Semantic map of texts"),  # Titles for all subplots
    specs=[
        [{"type": "xy"}, {"type": "domain"}],  # First row specs
        [{"colspan": 2}, None],  # Second row spans both columns
    ]
)

query_text = st.text_input("Write a query to search contents of curriculum:", "")

if query_text:
    df_queried = query_vector(query_text)

    colors = [subject_color_map[cat] for cat in df_queried['subject']]


    # Add bar plot to the first column
    fig.add_trace(
        go.Bar(
            # y=df_queried['text'].map(lambda x: x),
            y=[idx for idx in range(len(df_queried))],
            x=df_queried['certainty'][::-1],
            orientation='h',
            textposition='outside',
            marker=dict(color=colors[::-1])
        ),
        row=1,
        col=1
    )

    fig.update_layout(
        xaxis=dict(
            title='Relavance Score',  # Label for the x-axis
            range=[0.5, 0.8]
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=[idx for idx in range(len(df_queried))],  # Custom y-ticks
            ticktext=df_queried['text'][::-1].map(lambda x: x[:30])  # Use custom labels for y-axis
        )
    )

    unique_subject_list = df_queried["subject"].unique().tolist()

    colors = [subject_color_map[cat] for cat in unique_subject_list + df_queried['subject'].tolist()]

    # Add treemap to the second column
    fig.add_trace(
        go.Treemap(
            labels=unique_subject_list + df_queried["text"].tolist(),
            parents=[""]*len(unique_subject_list) + df_queried["subject"].tolist(),  # Since "path" isn't directly supported, set parents to empty or adjust accordingly.
            values=[0]*len(unique_subject_list) + df_queried["certainty"].tolist(),
            marker=dict(colors=colors)
        ),
        row=1,
        col=2
    )

else:
    # Add bar plot to the first column
    fig.add_trace(
        go.Bar(
        ),
        row=1,
        col=1
    )

    # Add treemap to the second column
    fig.add_trace(
        go.Treemap(
        ),
        row=1,
        col=2
    )



# Add a line plot to the third subplot (row 2, col 1, spanning both columns)
fig.add_trace(
    go.Scatter(
        x=df["x"],
        y=df["y"],
        mode='markers',
        marker=dict(
            color=color_list,
            size=6, 
            opacity=0.5
        ),
        text=hover_texts,  # Add hover text
        hoverinfo='text',  # Display only the hover text
        name="Scatter Points", 
    ),
    row=2,
    col=1
)

fig.update_layout(showlegend=False)
fig.update_layout(height=1000, width=1500)

# Streamlit app
st.plotly_chart(fig, use_container_width=True)
