from RealtimeTTS import TextToAudioStream, AzureEngine
import os, openai # pip install openai  

openai.api_key = os.environ.get("OPENAI_API_KEY")
engine = AzureEngine(os.environ.get("AZURE_SPEECH_KEY"), "eastus")

def generate(prompt):
    for chunk in openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages=[{"role": "user", "content" : prompt}],
            stream=True):
        
        if (text_chunk := chunk["choices"][0]["delta"].get("content")): 
            yield text_chunk

def text_start():
    print("[TEXT START]", end="", flush=True)

def text_stop():
    print("[TEXT STOP]", end="", flush=True)

def audio_start():
    print("[AUDIO START]", end="", flush=True)
          
def audio_stop():
    print("[AUDIO STOP]", end="", flush=True)

text_stream = generate("A three-sentence relaxing speech.")
          
stream = TextToAudioStream(engine, 
                           on_text_stream_start=text_start, 
                           on_text_stream_stop=text_stop, 
                           on_audio_stream_start=audio_start, 
                           on_audio_stream_stop=audio_stop)

stream.feed(text_stream)
stream.play(log_characters=True)