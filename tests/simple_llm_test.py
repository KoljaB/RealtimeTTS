import os
import openai  # pip install openai
from RealtimeTTS import TextToAudioStream, SystemEngine

openai.api_key = os.environ.get("OPENAI_API_KEY")


def write(prompt: str):
    for chunk in openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    ):
        if (text_chunk := chunk["choices"][0]["delta"].get("content")) is not None:
            yield text_chunk


text_stream = write("A three-sentence relaxing speech.")

print("Starting to play")

TextToAudioStream(SystemEngine()).feed(text_stream).play()
