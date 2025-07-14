import os
import uvicorn
from typing import Annotated, Optional, List
from fastapi import FastAPI, Query, File, UploadFile, HTTPException, Header, Form, Depends
from dotenv import load_dotenv
load_dotenv()
from assistant_utils import *

DOMAIN = "0.0.0.0" #"localhost"
PORT = int(os.environ.get("PORT"))

app = FastAPI()

@app.post('/assistant')
async def handle_message(reqest_id: str, input_text: str):

    response = {'response': call_assistant(input_text)}

    return response

@app.post('/image_assistant')
async def handle_image(request_id: str, image_id_list: List[str] = Query(...)):

    wall_list, floor_list = process_preset(image_id_list)

    response = {'response': {"wall_list": wall_list,
                             "floor_list": floor_list}}
    
    return response


if __name__ == "__main__":
    uvicorn.run(app, host=DOMAIN, port=PORT)
