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
import json
from dotenv import load_dotenv

load_dotenv()

# # Subject color map
# subject_color_map = {
#     "language_and_literature": "red",
#     "foreign_languages": "purple",
#     "mathematics": "blue",
#     "natural_science": "green",
#     "social_studies": "orange",
#     "art": "pink",
#     "technology": "cyan",
#     "physical_education": "yellow",
#     "religious_studies": "brown",
#     "informatics": "teal",
#     "career_education": "magenta",
#     "entrepreneurship_studies": "lime",
# }

# Load from the JSON file
with open("subject_color_map.json", "r") as f:
    subject_color_map = json.load(f)

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


def get_rag_answer_and_sources(query_text, task_text, limit=2):
    class_name = "CurriculumDemo"
        # Connect to Weaviate Cloud
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,                                    # Replace with your Weaviate Cloud URL
        auth_credentials=Auth.api_key(weaviate_api_key),             # Replace with your Weaviate Cloud key
        headers={"X-Cohere-Api-Key": cohere_api_key},           # Replace with your Cohere API key
    )

    collection = client.collections.get("CurriculumDemo")

    response = collection.generate.near_text(
        query=query_text,
        limit=5,
        grouped_task=task_text
    )

    reference_text_list = [obj.properties["text"] for obj in response.objects]

    generated_text = response.generated

    return generated_text, reference_text_list

# Load data
base_path = os.path.dirname(__file__)
df = pd.read_csv(os.path.join(base_path, "data/embedding_2d_est_basic_school.csv"))

# Generate color list and hover texts
color_list = [subject_color_map[row["subject"]] for _, row in df.iterrows()]
hover_texts = [row["text"].replace("\n", "<br>") for _, row in df.iterrows()]

st.title("Estonian Basic School Curriculum Analyzer Demo")

# Create the subplots with an additional row for the bottom graph
fig_1 = make_subplots(
    rows=1,  # Increase the rows to 2
    cols=2,  # Keep the columns as 2 for the top row
    column_widths=[0.3, 0.7],  # Set column widths for the first row
    subplot_titles=(
        "The most relevant texts", 
        "Searched texts grouped by subjects", 
        ),  # Titles for all subplots
    specs=[
        [{"type": "xy"}, {"type": "domain"}],  # First row specs
    ]
)
st.subheader("1. Search curriculum")

query_text = st.text_input("Write a query to search contents of curriculum:", "")

if query_text:
    df_queried = query_vector(query_text)

    colors = [subject_color_map[cat] for cat in df_queried['subject']]


    # Add bar plot to the first column
    fig_1.add_trace(
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


    fig_1.update_layout(
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
    fig_1.add_trace(
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
    fig_1.add_trace(
        go.Bar(
        ),
        row=1,
        col=1
    )

    # Add treemap to the second column
    fig_1.add_trace(
        go.Treemap(
        ),
        row=1,
        col=2
    )

fig_2 = px.scatter(
    df, 
    x="x", 
    y="y", 
    color="subject",  # Color by the subject column
    color_discrete_map=subject_color_map,  # Apply the custom color mapping
    hover_data=['text'],  # Replace with your hover text column name
    title="Scatter Plot with Legend"
)

# Remove x and y axes labels
fig_2.update_layout(
    xaxis_title="",  # Set x-axis label to an empty string
    yaxis_title="",   # Set y-axis label to an empty string
    xaxis=dict(showticklabels=False),  # Remove x-axis tick labels
    yaxis=dict(showticklabels=False)   # Remove y-axis tick labels
)

# Streamlit app
st.plotly_chart(fig_1, use_container_width=True)

st.subheader("2. Ask question of curriuclum saved")

# Input fields for two texts
query_text = st.text_input("Write a query to find reference texts", placeholder="Type something here...")
task_text = st.text_input("Write a task based on the texts queried:", placeholder="Type something here...")

# Button to combine texts
if st.button("Make a prompt"):
    if query_text and task_text:
        # Combine the texts and output the result
        generated_text, ref_list = get_rag_answer_and_sources(query_text, task_text, limit=2)

        st.subheader("Generated Answer:")
        st.write(generated_text)
    else:
        st.warning("Please provide both texts to combine.")

st.subheader("3. Visualize relations of curriculum contents")
st.plotly_chart(fig_2, use_container_width=True)

