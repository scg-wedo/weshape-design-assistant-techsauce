import os
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Form, Depends
from dotenv import load_dotenv
load_dotenv()
from assistant import *

DOMAIN = "0.0.0.0" #"localhost"
PORT = int(os.environ.get("PORT"))

app = FastAPI()
agent = OpenaiAssistant()

@app.post('/assistant')
async def handle_message(reqest_id: str, input_text: str):

    response = {'response': agent.call_assistant(input_text)}

    return response

@app.post('/image_assistant')
async def handle_image(request_id: str, input_image: UploadFile=File(...)):

    image_bytes = await input_image.read()
    file_type = input_image.content_type

    response = {'response': agent.image_assistant(image_bytes, file_type)}
    
    return response


if __name__ == "__main__":
    uvicorn.run(app, host=DOMAIN, port=PORT)
