import tempfile
from openai import OpenAI


def text_to_speech(input,api_key):
    client = OpenAI(api_key=api_key)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        temp_path = temp_file.name  # Get the name of the temporary file
        response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=input
        )
        with open(temp_path,"wb") as f:
            f.write(response.content)

    return temp_path