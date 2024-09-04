import tempfile
from openai import OpenAI



def text_to_speech(input,api_key):
    client = OpenAI(api_key=api_key)
    response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input=input
    )

    return response