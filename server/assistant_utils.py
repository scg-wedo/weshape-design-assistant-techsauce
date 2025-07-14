import json
import os
from langchain_google_vertexai.embeddings import VertexAIEmbeddings
from elasticsearch import Elasticsearch
from dotenv import load_dotenv
load_dotenv()

from io import BytesIO
from google.cloud import storage
from concurrent.futures import ThreadPoolExecutor, as_completed

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(os.environ["GOOGLE_APPLICATION_CREDENTIALS_PATH"])

es_host = os.getenv("ELASTICSEARCH_HOSTS")
elastic_client = Elasticsearch(es_host)
INDEX_NAME = os.getenv("INDEX_NAME")
        
embedding = VertexAIEmbeddings(
    model_name="text-multilingual-embedding-002",
    project=os.getenv("PROJECT_ID"),
    location=os.getenv("LOCATION"),
)

GCS_BUCKET = os.environ.get("GCS_BUCKET_NAME")
bucket = storage.Client().bucket(GCS_BUCKET)
base_url = "https://storage.googleapis.com"


def semantic_search_elasticsearch_langchain(query):
    query_vec = generate_embedding(query)
    response = elastic_client.search(index=INDEX_NAME, body={
    #semantic search or vectorsearch
    "knn": {
        "field": "embedding",
        "query_vector": query_vec,
        "k": 5,
        "num_candidates": 10,
    },
    #keyword search
    "query": {
        "multi_match": {
            "query": query,
            "fields": ["description", 
                    "description.thai", 
                    "description.standard",
                    "style^1.5",
                    "Color",
                    "Surface_applicability",
                    "Species^2"],
            "type": "most_fields"
        }
    },
        "size": 5,
        "_source": ["SKU","Name","Surface_applicability","description", "Color","style","Species",'image_base64', 'image_path']
    }
    )
    # results = retriever.get_relevant_documents(query)
    return response

def generate_embedding(text: str) -> list:
    """
    Generates a dense vector embedding for a given text using VertexAIEmbeddings.
    """
    try:
        # VertexAIEmbeddings.embed_query returns a list of floats (the vector)
        embedding_vector = embedding.embed_query(text)
        return embedding_vector
    except Exception as e:
        print(f"Error generating embedding for text '{text}': {e}")
        return [0.0] * 768 # Return a zero vector or handle as appropriate for error
    
def call_assistant(messages):
    #Variable
    rank = []
    rank_dict = {}
    ori_dict ={}
    tran_dict = {}
    result = []
 
    # print(type(messages))
    search_results = semantic_search_elasticsearch_langchain(elastic_client, messages)
    
    for hit in search_results['hits']['hits']:
        if hit['_score'] >= 1.00:
            # print(f"{hit['_score']:.2f}")
            result.append(hit['_source']['image_path'])    
    return result

def searching(text_list:list) -> list:
    result_list = []
    for text_input in text_list:
        query_vec = generate_embedding(text_input)
        response = elastic_client.search(index=INDEX_NAME, body={
        #semantic search or vectorsearch
        "knn": {
            "field": "embedding",
            "query_vector": query_vec,
            "k": 5,
            "num_candidates": 10,
        },
        #keyword search
        "query": {
            "multi_match": {
                "query": text_input,
                "fields": ["description", 
                        "description.thai", 
                        "description.standard",
                        "style^1.1",
                        "Color^2",
                        "Surface_applicability",
                        "Species^1.5"],
                "type": "most_fields"
            }
        },
            "size": 3,
            "_source": ["SKU","Name","Surface_applicability","description", "Color","style","Species",'image_base64', 'image_path']
        }
        )
        for hit in response['hits']['hits']:
            if hit['_score'] >= 1.00:
                #result depend on what you searching
                result_list.append(hit['_source']['image_path'])
    return result_list    

def get_preset():
    sub_folder = "ai-assistant/preset"
    prefix = sub_folder if sub_folder.endswith("/") else sub_folder + "/"
    client = storage.Client()
    blobs = client.list_blobs(GCS_BUCKET, prefix=prefix)

    all_presets = {}

    def process_blob(blob):
        if blob.name.endswith('.json'):
            # print(blob.name)
            json_data = blob.download_as_text()
            data = json.loads(json_data)
            return (blob.name, data)
        return None

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_blob, blob) for blob in blobs]
        for future in as_completed(futures):
            result = future.result()
            if result:
                blob_name, data = result
                all_presets[data['image_id']] = data

    return all_presets

def process_preset(image_id_list:list) -> list:
    all_presets = get_preset()
    # print("all_presets", all_presets)

    wall_list = []
    floor_list = []
    
    for image_id in image_id_list:
        wall_text = all_presets[image_id]['wall_desc'] + f" style {all_presets[image_id]['style']}"
        floor_text = all_presets[image_id]['floor_desc'] + f" style {all_presets[image_id]['style']}"
        wall_list.append(wall_text)
        floor_list.append(floor_text)
    wall_result = searching(wall_list)
    floor_result = searching(floor_list)
    return wall_result, floor_result