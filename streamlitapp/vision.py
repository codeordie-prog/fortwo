import requests
import base64
import json
from langchain.chains.llm import LLMChain
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
import requests
import os
import streamlit as st


""""
This file defines all the functions for vision capabilities and image generation. The model relies on
GPT4o-mini for vision where the description is converted to parsed json string and stored
in a vector database, and dall-e-3 for image generation
"""


image_description_prompt = """
                                in great detail describe the image, start with the Title : 'IMAGE DESCRIPTION' , when presented with an image with a code segment make sure you rewrite the segment in the description, 
                                if the image happens to be an unstructured text with or without images document, aim at reproducing every detail in your description and then describe your observation. 
                                if its a code snippet or a math problem, you should first rewrite it as it is organised properly then proceed to describe it and even to highlight potential errors or fixes that you observe.

                                Your description should have 3 areas:
                                1. Title - which should be 'IMAGE DESCRIPTION'
                                2. Content - Observed contents explained in great detail and listed in bullet format
                                3. Observation - this area has everything you observe from the image, detailed explanation of the contents

                                describe every detail in the description, do not fail to provide assistance with the image provided.

                                """


#encode the image to base64
def encode_image(image_path : str):
    with open(image_path,'rb') as file:
        return base64.b64encode(file.read()).decode('utf-8')

#use GPT4o-mini to describe the image
#returns a string parsed from a json object
def describe_image(image_url, openai_api_key, prompt):
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
                "text": prompt
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
    
#generate image function
def generate_image(description:str, openai_api_key:str):

    try:
        
        llm = OpenAI(temperature=0,api_key=openai_api_key)
        prompt = PromptTemplate(
            input_variables=["image_desc"],
            template="Generate a short but extremely detailed prompt to generate an high definition image given the following description: {description}",

        )
        chain = LLMChain(llm=llm,prompt=prompt)

        return DallEAPIWrapper(model="dall-e-3",api_key=openai_api_key).run(chain.run(description))

    except Exception as e:
        st.write("An error occured while generating the image",e)


#download the generated image
def download_generated_image(image_url: str, image_storage_path:str):

    try:
        response = requests.get(url=image_url)

        file_path = os.path.join(image_storage_path,"image.png")

        if response.status_code == 200:

            with open(file_path,"wb") as file:
                file.write(response.content)

            
            return file_path
                

    except Exception as e:
        st.write("An error occured during image download",e)
    
    

