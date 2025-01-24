import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd

import weaviate
from weaviate.classes.config import Configure
from weaviate.classes.init import Auth
from weaviate.classes.query import MetadataQuery
from weaviate.classes.query import Filter

import sys
import os
import json
import ast

# Needed when you run this locally
from dotenv import load_dotenv
load_dotenv()

# Load from the JSON file
with open("subject_color_map.json", "r") as f:
    subject_color_map = json.load(f)

weaviate_url = os.environ["WEAVIATE_URL"]
weaviate_api_key = os.environ["WEAVIATE_API_KEY"]
cohere_api_key = os.environ["COHERE_API_KEY"]

def query_vectors(query_text, search_limit, selected_subjects):
    class_name = "CurriculumDemo"
        # Connect to Weaviate Cloud
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,                                    # Replace with your Weaviate Cloud URL
        auth_credentials=Auth.api_key(weaviate_api_key),             # Replace with your Weaviate Cloud key
        headers={"X-Cohere-Api-Key": cohere_api_key},           # Replace with your Cohere API key
    )

    collection = client.collections.get("CurriculumDemo")

    object_filter_list = [Filter.by_property("subject").equal(subject) for subject in selected_subjects]

    response = collection.query.near_text(
        query=query_text,
        limit=search_limit, 
        return_metadata=MetadataQuery(distance=True, certainty=True), 
        filters=(
            Filter.any_of(object_filter_list)
        )
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


def get_rag_answer_and_sources(query_text, task_text, search_limit):
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
        limit=search_limit,
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

df_scatter = pd.read_csv(os.path.join(base_path, "data/embedding_2d_est_basic_school.csv"))
# df_scatter["hover_text"] = df_scatter["text"].replace("\n", "<br>")
color_list = [subject_color_map[row["subject"]] for _, row in df_scatter.iterrows()]
hover_texts = [row["text"].replace("\n", "<br>") for _, row in df_scatter.iterrows()]

st.title("AI Curriculum Analyzer Demo")

# Table of Contents
st.sidebar.title("Table of Contents")
st.sidebar.markdown("""
- [Purpose of This Demo](#purpose-of-this-demo)
- [Preparation of Data](#preparation-of-data)
- [Search the Curriculum](#search-the-curriculum)
- [Make a Prompt with Queried Texts](#make-a-prompt-with-queried-texts)
- [Possibilities for Further Analysis](#possibilities-for-further-analysis)
    - [Interdisciplinary Idea Analysis](#interdisciplinary-idea-analysis)
- [Next Steps](#next-steps)
    - [1. Preparing the Data](#1-preparing-the-data)
    - [2. Analyzing Administrative Curricula](#2-analyzing-administrative-curricula)
    - [3. Generating Practical Curricula](#3-generating-practical-curricula)
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 3, 1])  # Create three columns, with the center column being larger
with col2:
    st.image("static/banner.png", use_container_width =True)


with open('stored_samples.txt') as f:
    loaded_data = ast.literal_eval(f.read())


st.header("Purpose of This Demo")
st.write("""
After the rise of **ChatGPT**, the potential of **generative AI** has become a popular topic of discussion, including in the education sector. However, most conversations seem to focus on the **"generative" aspect**—such as chatbots or text generation.
While *content creation* is certainly a key advantage, it's only one part of what generative AI can offer. Beyond its name, the real power of generative AI lies in its ability to **transform how we interact with and retrieve information**.
Think back to the days when people typed precise commands into black screens or navigated files by clicking and dragging. Now, thanks to generative AI, **information can be accessed with fuzzy, conversational prompts in natural language**. 
""")

col1, col2, col3 = st.columns([1, 2, 1])  # Create three columns, with the center column being larger
with col2:
    st.image("static/generative_AI_two_aspects.jpg", use_container_width =True)

st.write("""
Through this demo and presentation, we aim to emphasize the following key points:

1. **Efficient Discovery**: How generative AI can quickly and efficiently locate texts of interest.  
2. **Automated Process**: How the contents of searched texts can be processed automatically with minimal effort.  
3. **Intuitive Visualization**: How AI techniques can visualize the relationships between texts in a more intuitive and meaningful way.  

We hope this demo demonstrates that the seemingly magical behaviors of products with generative AI are, at their core, just **combinations of limited functionalities**:  
- **Searching texts**, and  
- **Executing commands to process the searched texts**.  

By understanding the engineering behind the "magic," we encourage readers to explore **practical methods for curriculum analysis** and unlock new opportunities in education.
""")

# st.write("""We hope this demo shows that seemingly magical behaviors of generative AI are just combinations of the limited funcitonalities: searching texts and conducting commands based on the searched texts. 
# And by knowing the engineering behind the magic, it would be great if the readers could come up with practical methods of curriculum analysis.""")

st.header("Preparation of Data")
st.write("""
To effectively utilize generative AI in administrative curriculum analysis, documents must first be **divided into smaller text units** and then stored in a **database** after being processed by a generative AI model.
Each text is then **converted into a numerical expression**, making it easier for generative AI models to search and process. This numerical representation is called a **vector** or **embedding**.
""")

col1, col2, col3 = st.columns([1, 2, 1])  # Create three columns, with the center column being larger
with col2:
    st.image("static/embedding_split.png", use_container_width =True)

st.write("""
In this demo, the **national curricula** provided by the [Estonian Ministry of Education and Research](https://www.hm.ee/en/national-curricula) are used. 
The texts are **divided roughly by paragraph** and stored in the database with relevant **tags**, such as **"subject"** and **"paragraph number"**, to enable efficient **analysis** and **retrieval**.
The first 10 sample texts stored can be also found below. 
""")

col1, col2, col3, col4, col5 = st.columns([1, 2, 1, 2, 1])  # Create three columns, with the center column being larger
with col2:
    st.image("static/est_basic_school_nat_cur_2014_appendix_1_final-images-0.jpg", use_container_width =True)

with col4:
    st.image("static/est_basic_school_nat_cur_2014_appendix_1_final-images-1.jpg", use_container_width =True)

st.json(loaded_data, expanded=False)


st.header("Search the Curriculum")

col1, col2, col3 = st.columns([1, 2, 1])  # Create three columns, with the center column being larger
with col2:
    st.image("static/text_query.jpg", use_container_width =True)

st.write("""
After the texts are stored in the database, the most **relevant** or **semantically closest** texts to an input query can be found.
The following graphs display the **queried texts** and their **relevance scores** relative to the query. These texts are also **grouped by subject** on the right side for easier analysis.
""")

query_text = st.text_input("Write a query to search contents of curriculum:", "Data literacy")

text_search_limit = st.slider("Select a number of texts to query", 
                              min_value=0, 
                              max_value=100, 
                              value=30, 
                              step=1
                              )

# Multi-select filter
selected_subjects = st.multiselect(
    "Select subjects to filter:",
    options=subject_color_map.keys(),  
    default=subject_color_map.keys(),  
)

if query_text :
    df_queried = query_vectors(query_text, text_search_limit, selected_subjects)

    colors = [subject_color_map[cat] for cat in df_queried['subject']]

    # Create the subplots with an additional row for the bottom graph
    fig_1 = make_subplots(
        rows=1,  # Increase the rows to 2
        cols=2,  # Keep the columns as 2 for the top row
        column_widths=[0.3, 0.7],  # Set column widths for the first row
        subplot_titles=(
            "Relevant texts of the query '{}'".format(query_text), 
            "Searched texts grouped by subjects", 
            ),  # Titles for all subplots
        specs=[
            [{"type": "xy"}, {"type": "domain"}],  # First row specs
        ]
    )

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
            hoverinfo='text' , 
            # title="The most relevant text to the query {}".format(query_text)
        ),
        row=1,
        col=1
    )

    fig_1.update_layout(
        xaxis=dict(
            title='Relavance Score',  # Label for the x-axis
            range=[0.5, 0.8], 
                tickfont=dict(
                size=10  # Adjust the font size as desired
            )
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=[idx for idx in range(len(df_queried))],  # Custom y-ticks
            ticktext=df_queried['text'][::-1].map(lambda x: x[:20] + "...")  # Use custom labels for y-axis
        ), 
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
        height=500,  # Set the height of the figure (increase as needed)
        hoverlabel=dict(
            align="left"  # Left-align the hover text
        ), 
    )

else:
    # Create the subplots with an additional row for the bottom graph
    fig_1 = make_subplots(
        rows=1,  # Increase the rows to 2
        cols=2,  # Keep the columns as 2 for the top row
        column_widths=[0.2, 0.8],  # Set column widths for the first row
        subplot_titles=(
            "The most relevant texts ot the query '{}'".format(), 
            "Searched texts grouped by subjects", 
            ),  # Titles for all subplots
        specs=[
            [{"type": "xy"}, {"type": "domain"}],  # First row specs
        ]
    )

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

st.header("Make a Prompt with Queried Texts")

col1, col2, col3 = st.columns([1, 2, 1])  # Create three columns, with the center column being larger
with col2:
    st.image("static/RAG.jpg", use_container_width =True)

st.write("""
The **queried texts** can be processed using another prompt. In the demo below, a query for searching texts from the database can be set first, followed by the ability to perform various **tasks** on the retrieved texts.
This process is known as **RAG (Retrieval-Augmented Generation)**, and it is already widely used at the **product level**.
""")

rag_search_limit = st.slider("TOP n relevant texts used", min_value=0, max_value=10, value=1, step=1)
# Input fields for two texts
query_text = st.text_input("Write a query to find relevant texts", "Data literacy", placeholder="Type something here...")
task_text = st.text_input("Write a prompt based on the texts queried:", "Extract competency from the text", placeholder="Type something here...")

# Button to combine texts
if st.button("Make a prompt"):
    if query_text and task_text:
        # Combine the texts and output the result
        generated_text, df_ref = get_rag_answer_and_sources(query_text, task_text, rag_search_limit)

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



# fig_3 = px.scatter(
#     df_scatter, 
#     x="x", 
#     y="y", 
#     color="subject",  # Color by the subject column
#     color_discrete_map=subject_color_map,  # Apply the custom color mapping
#     opacity=0.3, 
#     hover_data={"hover_text": True, "paragraph_idx": False, "text": False, "subject":False, "x": False, "y": False},  # Replace with your hover text column name
#     hover_name ="hover_text", 
#     title="Semantic map of texts in the database", 
# )

# fig_3.update_traces(marker=dict(size=7.5,
#                               line=dict(width=.5,
#                                         color='black')),
#                   selector=dict(mode='markers'))

# # Remove x and y axes labels
# fig_3.update_layout(
#     xaxis_title="",  # Set x-axis label to an empty string
#     yaxis_title="",   # Set y-axis label to an empty string
#     xaxis=dict(showticklabels=False),  # Remove x-axis tick labels
#     yaxis=dict(showticklabels=False),   # Remove y-axis tick labels
#     height=600,  # Set the height of the figure (increase as needed)
#     hoverlabel=dict(
#             align="left"  # Left-align the hover text
#         )
# )


# Create the scatter plot
fig_3 = go.Figure()

# Add the scatter trace with hover text
fig_3.add_trace(go.Scatter(
    x=df_scatter["x"],
    y=df_scatter["y"],
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
    fig_3.add_trace(go.Scatter(
        x=[None], y=[None],  # Dummy points for legend
        mode='markers',
        marker=dict(size=10, color=color),
        name=subject
    ))

# Update layout
fig_3.update_layout(
    title="Semantic map of paragraphs in Estonian basic school curriculum",
    legend_title="Subjects",
    legend=dict(x=1.05, y=1),
    margin=dict(t=40, l=0, r=150, b=40),
    width=600,
    height=600,
    hovermode='closest',
    hoverlabel=dict(
        bgcolor="white",
        font_size=12,
        font_family="Arial",
        align="left"
    )
)


st.header("Possibilities for Further Analysis")

st.write("""
In fact, the ideas discussed so far are not entirely new—they have developed rapidly in the **business domain** over the past few years. 
Although **RAG (Retrieval-Augmented Generation)**, as explained in the previous section, is impressive, it essentially replaces manual work on individual texts.
However, the concept of **storing curriculums in a database** and utilizing generative AI may unlock new possibilities for **academically intriguing analysis** of curriculums worldwide. 
Since the texts are treated as **numeric expressions**, the **tendencies** and **relations** within curriculum statements can be analyzed from a more **macro-level perspective**.
""")

st.subheader("Interdisciplinary Idea Analysis")
col1, col2, col3 = st.columns([1, 2, 1])  # Create three columns, with the center column being larger
with col2:
    st.image("static/interdisciplinary_analysis.jpg", use_container_width =True)
st.write("""
One example of this type of analysis is the **detection of interdisciplinary ideas** in curriculums. 
As shown earlier in this demo, generative AI can **calculate semantic relevance** or disparity between texts. 
In other words, **distances between texts** can be calculated, which means some contents may form **clusters**.
The graph below shows an intermediate result of the **interdisciplinary idea analysis**. 
The **numeric expressions** of the texts saved in the database are plotted, with each dot representing a **paragraph** from the curriculums. The distance between two texts reflects the **semantic relevance** or **similarity** of those texts. 
It’s natural for texts of the same **subject** to form a cluster, but interestingly, we also observe that contents from different subjects can be mixed together. 
These mixed clusters, containing texts from multiple subjects, provide valuable **hints of interdisciplinary ideas**.
""")
st.plotly_chart(fig_3, use_container_width=True)

st.header("Next Steps")

st.write("""
I hope this demo has shown the basic functionality of generative AI: searching texts and processing them.
In other words, what texts to store in the database, what texts to search, what to conduct on each texts, and how to visualize them: they are all designed and implemented by humans. 
  
To clarify, let’s break down the types of curricula into two main categories:  
- **Administrative curriculum**: General teaching guidelines provided by educational authorities in different countries or regions.  
- **Practical curriculum**: Specific plans created by school teachers, including schedules and detailed teaching content.  

The following steps could be considered to apply generative AI on curriculum analysis and generation.
"""
)

st.subheader("1. Organizing the Data")  
st.write("""
In this demo, texts are divided paragraph by paragraph. Ideally, though, the data should be organized into sections with labels like:  
- "Subject"  
- "Country"  
- "Language"  
- "Age group"  
- "Goals"  
- "Content"  
         
Since each country has its own curriculum structure, it’s important to study these differences and develop a standardized framework for organizing them.
Also "noises" in texts should be removed, for example page numbers, section numbers. 
""")

st.subheader("2. Analyzing Administrative Curricula")
st.write("""
Analyzing administrative curricula can provide valuable insights. For example, with a well-organized database (as described in step 1), even a simple dashboard could reveal useful trends or patterns.
More advanced analysis—such as those mentioned in the "Possibilities for Further Analysis" section—could offer deeper academic insights.
""")

st.subheader("3. Generating Practical Curricula")
st.write("""
This step focuses on creating usable curriculum plans for teachers. Since this involves fewer risks, it’s already possible to build tools for teachers that are simple and intuitive.
Initially, this doesn’t require the structured data from step 1, so it can be developed independently. Over time, it could be enhanced by combining it with the analysis in step 2. For example, teachers could explore curricula from other subjects or countries to get ideas and then tailor their own plans.
""")