from stream2sentence import generate_sentences
import openai                               # pip install openai  
import os
import logging
import itertools
from RealtimeTTS import TextToAudioStream, SystemEngine, AzureEngine, ElevenlabsEngine

openai.api_key = os.environ.get("OPENAI_API_KEY")

def write(prompt: str):
    for chunk in openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content" : prompt}],
        stream=True
    ):
        if (text_chunk := chunk["choices"][0]["delta"].get("content")) is not None:
            yield text_chunk

text_stream = write("A three-sentence relaxing speech.")

print("Starting to play")

TextToAudioStream(SystemEngine()).feed(text_stream).play()
#TextToAudioStream(AzureEngine(os.environ.get("AZURE_SPEECH_KEY"), "germanywestcentral")).feed(text_stream).play()
#TextToAudioStream(ElevenlabsEngine(os.environ.get("ELEVENLABS_API_KEY"))).feed(text_stream).play()