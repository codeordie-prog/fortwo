
import requests
import tempfile,os

# Replace 'your_api_key' with your actual Hugging Face API key
def authentication(API_TOKEN):
    return {"Authorization": f"Bearer {API_TOKEN}"}

def text_to_speech(text,hugginface_api):
    try:
        headers= authentication(hugginface_api)
        API_URL="https://api-inference.huggingface.co/models/facebook/fastspeech2-en-ljspeech"
        payload = {"inputs": text}
        response = requests.post(API_URL, headers=headers, json=payload)

        # Create a temporary file that will not be deleted immediately
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_path = temp_file.name  # Get the name of the temporary file

            if response.status_code == 200:
                with open(temp_path, "wb") as f:
                    f.write(response.content)
                return temp_path
            else:
                return None
    except Exception as e:
        return f"an error occured in tts, {e}"





