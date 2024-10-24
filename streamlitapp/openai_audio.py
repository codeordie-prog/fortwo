import tempfile
from openai import OpenAI as opn


def text_to_speech(input,api_key):

    count = len(input)

    if count<4000:


        client = opn(api_key=api_key)
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
    
    else:
        return None


def speech_to_text(audio_file,api_key):
    try:

        client = opn(api_key=api_key)

        transcription = client.audio.transcriptions.create(
                                    model="whisper-1", 
                                    file=audio_file, 
                                    response_format="text"
                                    )
        
        return transcription

    except:
        pass