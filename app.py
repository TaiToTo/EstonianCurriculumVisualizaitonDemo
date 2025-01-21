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

# Load from the JSON file
with open("subject_color_map.json", "r") as f:
    subject_color_map = json.load(f)

weaviate_url = os.environ["WEAVIATE_URL"]
weaviate_api_key = os.environ["WEAVIATE_API_KEY"]
cohere_api_key = os.environ["COHERE_API_KEY"]

def query_vectors(query_text, search_limit):
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
        limit=search_limit, 
        return_metadata=MetadataQuery(distance=True, certainty=True)
    )

    client.close()  # Free up resources

    row_dict_list = []
    for obj in response.objects:
        row_dict = {}
        row_dict["text"] = obj.properties["text"]
        row_dict["subject"] = obj.properties["subject"]
        row_dict["paragraph_idx"] = int(obj.properties["paragraph_idx"])
        row_dict["certainty"] = obj.metadata.certainty
        row_dict["distance"] = obj.metadata.distance
        
        row_dict_list.append(row_dict)

    df_queried = pd.DataFrame(row_dict_list, columns=["text", "subject", "certainty", "distance", "paragraph_idx"])
    
    df_queried["text"] = df_queried["text"].map(lambda x: x.replace("\n", "<br>") )

    df_queried["hover_text"] = df_queried.apply(
        lambda x: "<b>Relevance:</b> {:.3f}<br><b>Paragraph idx:</b> {}<br><b>Paragraph idx:</b> {}<br>{}".format(x["certainty"], x["subject"], x["paragraph_idx"], x["text"]),
        axis=1
    )

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
        grouped_task=task_text, 
        return_metadata=MetadataQuery(distance=True, certainty=True)
    )

    client.close()  # Free up resources

    row_dict_list = []
    for obj in response.objects:
        row_dict = {}
        row_dict["text"] = obj.properties["text"]
        row_dict["subject"] = obj.properties["subject"]
        row_dict["paragraph_idx"] = int(obj.properties["paragraph_idx"])
        row_dict["certainty"] = obj.metadata.certainty
        row_dict["distance"] = obj.metadata.distance
        
        row_dict_list.append(row_dict)

    df_queried = pd.DataFrame(row_dict_list, columns=["text", "subject", "certainty", "distance", "paragraph_idx"])
    df_queried["text"] = df_queried["text"].map(lambda x: x.replace("\n", "<br>") )
    df_queried["hover_text"] = df_queried.apply(
        lambda x: "<b>Relevance:</b> {:.3f}<br><b>Paragraph idx:</b> {}<br><b>Paragraph idx:</b> {}<br>{}".format(x["certainty"], x["subject"], x["paragraph_idx"], x["text"]),
        axis=1
    )

    generated_text = response.generated

    return generated_text, df_queried


st.set_page_config(layout="wide")

# Load data
base_path = os.path.dirname(__file__)

df = pd.read_csv(os.path.join(base_path, "data/embedding_2d_est_basic_school.csv"))

# Generate color list and hover texts
color_list = [subject_color_map[row["subject"]] for _, row in df.iterrows()]
hover_texts = [row["text"].replace("\n", "<br>") for _, row in df.iterrows()]

st.title("Estonian School Curriculum Analyzer Demo")

# Create the subplots with an additional row for the bottom graph
fig_1 = make_subplots(
    rows=1,  # Increase the rows to 2
    cols=2,  # Keep the columns as 2 for the top row
    column_widths=[0.2, 0.8],  # Set column widths for the first row
    subplot_titles=(
        "The most relevant texts", 
        "Searched texts grouped by subjects", 
        ),  # Titles for all subplots
    specs=[
        [{"type": "xy"}, {"type": "domain"}],  # First row specs
    ]
)

st.header("Search curriculum")

query_text = st.text_input("Write a query to search contents of curriculum:", "")

text_search_limit = st.slider(
    "Select a number of texts to query",  # Label for the slider
    min_value=0,               # Minimum value of the slider
    max_value=100,             # Maximum value of the slider
    value=30,                  # Default value
    step=1                     # Step size
)

if query_text:
    df_queried = query_vectors(query_text, text_search_limit)

    colors = [subject_color_map[cat] for cat in df_queried['subject']]

    # Add bar plot to the first column
    fig_1.add_trace(
        go.Bar(
            # y=df_queried['text'].map(lambda x: x),
            y=[idx for idx in range(len(df_queried))],
            x=df_queried['certainty'][::-1],
            orientation='h',
            textposition='outside',
            marker=dict(color=colors[::-1]), 
            hovertext=df_queried['hover_text'][::-1], 
            hoverinfo='text' 
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
            ticktext=df_queried['text'][::-1].map(lambda x: x[:30] + "...")  # Use custom labels for y-axis
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
            marker=dict(colors=colors), 
            hovertext=unique_subject_list + df_queried['hover_text'].tolist(), 
            hoverinfo='text' 
        ),
        row=1,
        col=2
    )

    fig_1.update_layout(
        height=750,  # Set the height of the figure (increase as needed)
        hoverlabel=dict(
            align="left"  # Left-align the hover text
        )
    )

else:
    pass
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

# Streamlit app
st.plotly_chart(fig_1, use_container_width=True)

st.header("Ask question of curriuclum saved")
rag_search_limit = st.slider(
    "Select a number of texts to use for generating the answer",  # Label for the slider
    min_value=0,               # Minimum value of the slider
    max_value=10,             # Maximum value of the slider
    value=1,                  # Default value
    step=1                     # Step size
)
# Input fields for two texts
query_text = st.text_input("Write a query to find reference texts", placeholder="Type something here...")
task_text = st.text_input("Write a task based on the texts queried:", placeholder="Type something here...")

# Button to combine texts
if st.button("Make a prompt"):
    if query_text and task_text:
        # Combine the texts and output the result
        generated_text, df_ref = get_rag_answer_and_sources(query_text, task_text, limit=rag_search_limit)

        st.markdown("### Generated answer:")
        st.write(generated_text)

        unique_subject_list = df_ref["subject"].unique().tolist()
        colors = [subject_color_map[cat] for cat in unique_subject_list + df_ref['subject'].tolist()]

        fig_2 = go.Figure()

        # Add treemap to the second column
        fig_2.add_trace(
            go.Treemap(
                labels=unique_subject_list + df_ref["hover_text"].tolist(),
                parents=[""]*len(unique_subject_list) + df_ref["subject"].tolist(),  # Since "path" isn't directly supported, set parents to empty or adjust accordingly.
                values=[0]*len(unique_subject_list) + df_ref["certainty"].tolist(),
                marker=dict(colors=colors), 
                hovertext=unique_subject_list + df_ref['hover_text'].tolist(), 
                hoverinfo='text' 
            ),

        )

        fig_2.update_layout(
            height=150,  # Set the height of the figure (increase as needed)
            hoverlabel=dict(
                align="left"  # Left-align the hover text
            ), 
            margin=dict(
                l=10,  # Left margin
                r=10,  # Right margin
                t=10,  # Top margin
                b=10   # Bottom margin
            )
        )

        st.markdown("### Queried texts used for generating the answer:")
        st.plotly_chart(fig_2, use_container_width=True)

    else:
        st.warning("Please provide both texts to combine.")



fig_3 = px.scatter(
    df, 
    x="x", 
    y="y", 
    color="subject",  # Color by the subject column
    color_discrete_map=subject_color_map,  # Apply the custom color mapping
    hover_data=['text'],  # Replace with your hover text column name
    title="Semantic map of texts in the database"
)

# Remove x and y axes labels
fig_3.update_layout(
    xaxis_title="",  # Set x-axis label to an empty string
    yaxis_title="",   # Set y-axis label to an empty string
    xaxis=dict(showticklabels=False),  # Remove x-axis tick labels
    yaxis=dict(showticklabels=False),   # Remove y-axis tick labels
    height=750,  # Set the height of the figure (increase as needed)
    hoverlabel=dict(
            align="left"  # Left-align the hover text
        )
)


st.header("Intermediate results for interdisciplinary analysis")
st.plotly_chart(fig_3, use_container_width=True)

