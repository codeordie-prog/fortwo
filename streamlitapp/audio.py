
import requests
import tempfile,os

# Replace 'your_api_key' with your actual Hugging Face API key
API_TOKEN = "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"
API_URL = "https://api-inference.huggingface.co/models/facebook/fastspeech2-en-ljspeech"

headers = {"Authorization": f"Bearer {API_TOKEN}"}

def text_to_speech(text):
    payload = {"inputs": text}
    response = requests.post(API_URL, headers=headers, json=payload)

    with tempfile.TemporaryDirectory() as temp_file:
        temp_file = tempfile.TemporaryDirectory()
        temp_path = os.path.join(temp_file.name,"output.wav")


        if response.status_code == 200:
            with open(temp_path, "wb") as f:
                f.write(response.content)

    return temp_path





