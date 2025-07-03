from langchain.agents.openai_assistant import OpenAIAssistantRunnable
from langchain_elasticsearch import ElasticsearchStore
from elasticsearch import Elasticsearch
from langchain_openai import AzureOpenAIEmbeddings
from langchain_elasticsearch import ElasticsearchRetriever
from langchain_google_vertexai.embeddings import VertexAIEmbeddings
from openai import OpenAI
from langchain_openai import ChatOpenAI
from mimetypes import guess_type
from typing import Literal
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage,HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.documents import Document
from langchain_core.runnables import chain
from langgraph.pregel import Graph
from langgraph.graph import StateGraph
from typing import List
from langchain_openai import OpenAIEmbeddings
from uuid import uuid4
from collections import defaultdict

import base64
import io
import os
import time
import json
from dotenv import load_dotenv
load_dotenv()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(os.environ["GOOGLE_APPLICATION_CREDENTIALS_PATH"])

class QAState(StateGraph):
    question: str
    translated_question: str = None
    original_content:str = None
    generated_content: str = None
    ori_documents: dict = None
    tran_documents: dict = None
    final_answer: str = None
class OpenaiAssistant:
    def __init__(self):
        #self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        #self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.model = ChatOpenAI(
                    model=os.getenv("OPENAI_MODEL"),
                    openai_api_key = os.getenv("OPENAI_API_KEY"),
                    openai_api_base = os.getenv("OPENAI_API_BASE"),
                    temperature=0,
                    max_tokens=None,
                    max_retries=6,
                    stop=None,
        )
        self.index_name = "connector-elastic-cloud-do-cement-std-nonprod-781e"
        self.text_field = "body.text"  
        self.vector_field = "body.inference.chunks.embeddings"
        self.index_source_fields = {
            "gcs-elastic-cloud-do-cap-nonprod": "body"
        }
        
        
        self.elastic_client = Elasticsearch(
                                    hosts=[os.getenv("ELASTICSEARCH_HOSTS")],
                                    api_key=os.getenv("ELASTIC_API_KEY")
                                )
        
        self.embedding = VertexAIEmbeddings(
            model_name="text-multilingual-embedding-002",
            project=os.getenv("PROJECT_ID"),
            location=os.getenv("LOCATION"),
        )

        self.rank = []
    def local_image_to_data_url(self, image_path: str) -> str:
        mime_type, _ = guess_type(image_path)
        if mime_type is None:
            mime_type = 'image/png'
        with open(image_path, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{base64_encoded_data}"
    def semantic_search_elasticsearch_langchain(self,es_client, query):
        query_vec = self.generate_embedding(query)
        response = es_client.search(index=os.getenv("INDEX_NAME"), body={
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
            "_source": ["SKU","Name","Surface_applicability","description", "Color","style","Species",'image_base64']
        }
        )
        # results = retriever.get_relevant_documents(query)
        return response
    
    def generate_embedding(self,text: str) -> list:
        """
        Generates a dense vector embedding for a given text using VertexAIEmbeddings.
        """
        try:
            # VertexAIEmbeddings.embed_query returns a list of floats (the vector)
            embedding_vector = self.embedding.embed_query(text)
            return embedding_vector
        except Exception as e:
            print(f"Error generating embedding for text '{text}': {e}")
            return [0.0] * 768 # Return a zero vector or handle as appropriate for error
        
    def call_assistant(self, messages):
        #Variable
        self.rank = []
        self.rank_dict = {}
        self.ori_dict ={}
        self.tran_dict = {}
        self.result = []

        es_client = Elasticsearch(
            os.getenv("ELASTICSEARCH_HOSTS"),
            api_key=os.getenv("ELASTIC_API_KEY")
        )
        #TODO search message 
        # print(type(messages))
        search_results = self.semantic_search_elasticsearch_langchain(es_client,messages)
        
        for hit in search_results['hits']['hits']:
            if hit['_score'] >= 1.00:
                print(f"{hit['_score']:.2f}")
                self.result.append(hit['_source']['image_base64'])    
        return  self.result
    
    @tool
    def analyze_room_image(self,image_data_url: str) -> str:
        """
        <step>
        1. Understanding room image with boundary marked based on the image data URL.
        2. Describe wall/floor material such as color, material (cement, wood, plain), Surface_applicability(wall or floor).
        3. Analyze room style with 10 styles (Minimalist,Scandinavian,Industrial,Bohemian,Mid-Century Modern,Modern Farmhouse,Japandi,Transitional,Contemporary,Art Deco)
        4. Return as query text starting with "ฉันต้องการ(Surface_applicability)(material)(color)(style)" For example, "ฉันต้องการพื้นกระเบื้องplainสีขาว สไตล์ Minimalist".
        </step>
        
        <style>
        1. Minimalist
        Minimalist design is about "less is more." It features clean lines, open spaces, and neutral colors (white, gray, black) with subtle pops of color. 
        Porcelain tiles in large formats and matte finishes are common, often in neutral tones or wood/stone looks for understated elegance.
        2. Scandinavian
        Scandinavian design is simple, functional, and warm, emphasizing light, natural elements, and coziness. 
        It uses light, neutral colors (whites, soft naturals) with subtle contrasts. Ceramic, porcelain, mosaic, and marble tiles are popular, usually in light colors, sometimes with subtle geometric patterns or wood-look finishes.
        3. Industrial
        Industrial design draws from old factories, showcasing raw, unfinished elements like exposed brick, metal, and concrete. 
        It uses a neutral color palette (gray, black, white) with natural wood and metal tones. Porcelain stoneware is versatile, often replicating concrete or metal textures with raw or matte finishes.
        4. Bohemian (Boho)
        Bohemian style is free-spirited and eclectic, embracing individuality with a rich mix of cultures, colors, and patterns. 
        It features vibrant, diverse patterns and bold, rich jewel tones balanced with earthy hues. Handcrafted ceramic and patterned cement tiles are common, along with ethnic-patterned or black and white porcelain tiles, usually with matte or rough textures.
        5. Mid-Century Modern
        Mid-Century Modern design (1940s-1970s) focuses on simplicity, functionality, and indoor-outdoor connection. 
        It uses clean lines, organic/geometric forms, and a palette of bold hues (mustard, olive, burnt orange) with muted earthy tones. Ceramic tiles (penny, mosaic, subway) are common, along with glass, marble, porcelain, and terrazzo, in both matte and glossy finishes.
        6. Modern Farmhouse
        Modern Farmhouse blends rustic charm with contemporary elements for cozy, sophisticated spaces. 
        It emphasizes comfort, simplicity, open layouts, white walls, and natural wood. The palette includes warm/cool neutrals (white, cream, gray, blue) with earthy accents. Wood-look porcelain and white subway tiles are popular, as are patterned cement and textured hexagon tiles. Natural stone is also used.
        7. Japandi
        Japandi is a serene fusion of Japanese minimalism and Scandinavian simplicity, creating balanced, functional spaces. 
        It focuses on intentionality, natural light, and organic elements. The color palette is soft and neutral (whites, taupe, grays) with deeper grounding shades. Natural stone tiles (marble, slate), Japanese tiles, timber-look, and stone-look porcelain are used, often with honed finishes.
        8. Transitional
        Transitional design blends traditional elegance with contemporary simplicity for a timeless look. 
        It features clean lines, understated decor, and a focus on balance. The palette is predominantly neutral (white, cream, taupe, gray) with subtle accent colors. Natural stone tiles in neutral hues and large-format tiles are common, along with ceramic, marble, and porcelain in various finishes, sometimes with geometric patterns.
        9. Contemporary
        Contemporary design is dynamic, reflecting current trends with open layouts, clean lines, and uncluttered environments. 
        It emphasizes architectural details and indoor-outdoor connection. The color palette is typically neutral (earthy hues, crisp whites, charcoal) with bold pops of color. Bare, smooth flooring like ceramic, cement, granite, marble, mosaic, and porcelain tiles are common, with diverse finishes and large formats.
        10. Art Deco
        Art Deco is a glamorous, luxurious style from the 1920s, known for streamlined forms, symmetry, and opulent materials. 
        It features bold geometric shapes (zigzags, sunbursts) and a rich color palette (deep yellows, reds, blues) balanced with black and metallics. Porcelain, ceramic, and marble tiles are common, often in hexagonal or basket weave patterns, typically with glossy finishes to enhance luxury.
        </style>
        
        
        
        <rule>
        1. Return as Thai sentence with the format specified in the step.
        </rule>
        """
        # นี่คือส่วนที่ควรจะมีการวิเคราะห์รูปภาพจริง
        # โดยจะใช้ image_data_url (ซึ่งเป็น base64 ของรูปภาพ)
        # เพื่อเรียกใช้โมเดล AI ในการวิเคราะห์และดึงข้อมูลที่เกี่ยวข้อง
        # เนื่องจากข้อจำกัดในการเข้าถึงโมเดลวิเคราะห์รูปภาพโดยตรง
        # ในตัวอย่างนี้จะส่งคืนข้อความตัวอย่างตามที่คุณต้องการ
        
        # สมมติว่านี่คือผลลัพธ์ที่ได้จากการวิเคราะห์รูปภาพจริง
        result_text = "ฉันต้องการพื้นกระเบื้องplainสีขาว สไตล์ Minimalist" # ตัวอย่างผลลัพธ์ตามรูปแบบที่ต้องการ
        
        return result_text

    def image_assistant(self,image,mime):
        self.result = []
        es_client = Elasticsearch(
            os.getenv("ELASTICSEARCH_HOSTS"),
            api_key=os.getenv("ELASTIC_API_KEY")
        )
        
        #Image ที่ส่งมาเป็นbyte
        print("Hello from assistant")
        memory = MemorySaver()
        config = {"configurable": {"thread_id": "24245"}}
        tools = [self.analyze_room_image]
        system_message = SystemMessage(content="""คุณเป็นผู้เชี่ยวชาญในการทำความเข้าใจรูปภาพและแปลงเป็นข้อมูลสอบถาม
        หน้าที่ของคุณคือทำความเข้าใจรูปภาพห้องที่มีขอบเขตและใช้วิเคราะห์ห้อง
        หลังจากวิเคราะห์ด้วย tool แล้ว ให้ตอบกลับด้วยผลลัพธ์จาก tool นั้นในรูปแบบข้อความเท่านั้น.
            """)
        
        #TODO agent ของlangchain
        image_base64 = base64.b64encode(image).decode('utf-8')
        graph = create_react_agent(self.model, tools=tools, prompt=system_message)
        image_url = f"data:{mime};base64,{image_base64}"
        # image_url = self.local_image_to_data_url(image)
        image_content = {
        "type": "image_url",
        "image_url": {
            "url": image_url # Assuming 'image' is already a base64 string or a URL
            }
        }
        
        inputs = {"messages": [HumanMessage(content=[image_content])]}

            
        
        # #TODO llm แปลงภาษา

        output = graph.invoke(inputs, config=config, stream_mode="values")
        #print("Raw output from graph.invoke:", output)
        tran = output["messages"][-1].content
        print(tran)
        search_results = self.semantic_search_elasticsearch_langchain(es_client,tran)
        
        for hit in search_results['hits']['hits']:
            if hit['_score'] >= 1.00:
                print(f"{hit['_score']:.2f}")
                self.result.append(hit['_source']['image_base64'])
        return  self.result
