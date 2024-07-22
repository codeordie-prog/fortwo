import requests
import base64
import json


#encode the image to base64
def encode_image(image_path : str):
    with open(image_path,'rb') as file:
        return base64.b64encode(file.read()).decode('utf-8')

#use GPT4o-mini to describe the image
#returns a string parsed from a json object
def describe_image(image_url, openai_api_key, query):
    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {openai_api_key}"
            }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": query
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_url}"
                }
                }
            ]
            }
        ],
        "max_tokens": 300
        }
    
    response = requests.post("https://api.openai.com/v1/chat/completions",headers=headers,json=payload)

    decoded_response = response.content.decode("utf-8")

    data = json.loads(decoded_response)

    description = data["choices"][0]["message"]["content"]

    return description


#create a txt file, this file will be later stored in a vector database for retrieval
def create_a_textfile(image_description,file_path):

    with open(file_path,'w') as file:
        file.write(image_description)

        return file
    

