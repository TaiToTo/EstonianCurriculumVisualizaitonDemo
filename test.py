import os
import json
import requests
from dotenv import load_dotenv

import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure

load_dotenv()

# Best practice: store your credentials in environment variables
weaviate_url = os.environ["WEAVIATE_URL"]
weaviate_api_key = os.environ["WEAVIATE_API_KEY"]
cohere_api_key = os.environ["COHERE_API_KEY"]

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=weaviate_url,                                    # Replace with your Weaviate Cloud URL
    auth_credentials=Auth.api_key(weaviate_api_key),             # Replace with your Weaviate Cloud key
    headers={"X-Cohere-Api-Key": cohere_api_key},           # Replace with your Cohere API key
)

# resp = requests.get(
#     "https://raw.githubusercontent.com/weaviate-tutorials/quickstart/main/data/jeopardy_tiny.json"
# )
# data = json.loads(resp.text)

collections = client.collections.get("CurriculumDemo")


for item in collections.iterator():
    if item.properties["paragraph_idx"] <= 10 and item.properties["subject"]=="language_and_literature":
        print(item.properties)



# response = questions.generate.near_text(
#     query="biology",
#     limit=5,
#     grouped_task="Write a tweet with emojis about these facts."
# )

# print(response.generated)
# print(response.objects[0].properties["question"])

client.close()  # Free up resources