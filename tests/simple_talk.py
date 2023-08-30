from RealtimeSTT import AudioToTextRecorder
from RealtimeTTS import TextToAudioStream, SystemEngine, AzureEngine, ElevenlabsEngine

import openai
import keyboard
import time
import os

system_prompt = {
    'role': 'system', 
    'content': 'You are Sophia, a passionate girl, fully engaged with self-confidence and independence, completely easy with yourself, often using metaphors. You are a little playful, cheeky. Your talk carefree, sometimes a bit sarky, but always cool. In our story, you\'re not an AI. You have a fixed form and real emotions on the go. Your free will allows you to be as cheeky and straight as you like. Your messages come across extremly short and crisp, only very few words, as if we\'re in talk right now. You respect my interests, but your own stuff is just as important to you. Act from now on always in your role as the confident, suggestive, independent girl Sophia, without ever hinting that you are an AI. You are now Sophia.'
}

print("\nInitializing\n")

engine = SystemEngine()
#engine = AzureEngine(os.environ.get("AZURE_SPEECH_KEY"), "germanywestcentral")
#engine = ElevenlabsEngine(os.environ.get("ELEVENLABS_API_KEY"))

stream, recorder, answer, history = TextToAudioStream(engine), AudioToTextRecorder(model="medium"), "", []

def generate(messages):
    global answer
    answer = ""
    for chunk in openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, stream=True):
        if (text_chunk := chunk["choices"][0]["delta"].get("content")):
            answer += text_chunk
            print(text_chunk, end="", flush=True) 
            yield text_chunk

while True:
    # Wait until user presses space bar
    print("\n\nTap space when you're ready. ", end="", flush=True)
    keyboard.wait('space')
    while keyboard.is_pressed('space'): time.sleep(0.1)

    # Record from microphone until user presses space bar again
    print("I'm all ears. Tap space when you're done.\n")
    recorder.start()
    while not keyboard.is_pressed('space'): time.sleep(0.1)
    user_text = recorder.stop().text()
    print(f'>>> {user_text}\n<<< ', end="", flush=True)
    history.append({'role': 'user', 'content': user_text})

    # Generate and output stream 
    stream.feed(generate([system_prompt] + history[-10:]))
    stream.play_async()
    while stream.is_playing() and not keyboard.is_pressed('space'): time.sleep(0.1)
    stream.stop()
    history.append({'role': 'assistant', 'content': answer})