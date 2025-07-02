import os
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Form, Depends
from dotenv import load_dotenv
load_dotenv()
from src.assistant import *

DOMAIN = "0.0.0.0" #"localhost"
PORT = int(os.environ.get("PORT"))

app = FastAPI()
agent = OpenaiAssistant()

@app.post('/assistant')
def handle_message():
    data = request.get_json()
    message = data.get('message')
    

    response = {'response': agent.call_assistant(message)}

    return response

@app.post('/image_assistant')
def handle_image():

    uploaded_file  = request.files['file']
    
    image_bytes = uploaded_file.read()

    file_type = uploaded_file.mimetype

    response = {'response': agent.image_assistant(image_bytes,file_type)}
    
    return response


if __name__ == "__main__":
    uvicorn.run(app, host=DOMAIN, port=PORT)
