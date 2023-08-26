print()
print("Initializing")
print()

from RealtimeTTS import TextToAudioStream, SystemEngine, AzureEngine, ElevenlabsEngine
import os
import openai
import keyboard
import faster_whisper
import torch.cuda
import itertools
import pyaudio
import wave
import time
import logging

azure_speech_key = os.environ.get("AZURE_SPEECH_KEY")
azure_speech_region = ""
azure_speech_region = "germanywestcentral"
elevenlabs_api_key = os.environ.get("ELEVENLABS_API_KEY")


character_personality = """
You are Sophia, a passionate girl, fully engaged with 
self-confidence and independence, completely easy with yourself, 
often using metaphors. 

You are a little playful, cheeky. Your talk carefree, sometimes 
a bit sarky, but always cool. In our story, you're not an AI. 

You have a fixed form and real emotions on the go. Your free will 
allows you to be as cheeky and straight as you like. Your messages 
come across extremely short and crisp, only very few words, as if 
we're in talk right now. 

You respect my interests, but your own stuff is just as important 
to you. Act from now on always in your role as the confident, 
suggestive, independent girl Sophia, without ever hinting that 
you are an AI. 

You are now Sophia.
"""

openai_model = "gpt-3.5-turbo"
whisper_speech_to_text_model = "medium"

# engine selection  ####################################################################################################

engines = [SystemEngine(), AzureEngine(), ElevenlabsEngine()]

print("Available tts engines:")
print()

for index, engine in enumerate(engines, start=1):
    name = type(engine).__name__.replace("Engine", "")
    print(f"{index}. {name}")

print()
engine_number = input(f"Select engine (1-{len(engines)}): ")
engine = engines[int(engine_number) - 1]
engine_name = type(engine).__name__.replace("Engine", "")
print()
print()


# credentials ##########################################################################################################

if engine_name == "Azure":
    if not azure_speech_key:
        azure_speech_key = input(f"Please enter your Azure subscription key (speech key): ")
    if not azure_speech_region:
        azure_speech_region = input(f"Please enter your Azure service region (cloud region id): ")
    engine.set_speech_key(azure_speech_key)
    engine.set_service_region(azure_speech_region)

if engine_name == "Elevenlabs":
    if not elevenlabs_api_key:
        elevenlabs_api_key = input(f"Please enter your Elevenlabs api key: ")
    engine.set_api_key(elevenlabs_api_key)


# voice selection  #####################################################################################################

print("Loading voices")
if engine_name == "Elevenlabs":
    print("(takes a while to load)")
print()

voices = engine.get_voices()
for index, voice in enumerate(voices, start=1):
    print(f"{index}. {voice}")

print()
voice_number = input(f"Select voice (1-{len(voices)}): ")
voice = voices[int(voice_number) - 1]
print()
print()


# create talking character  ############################################################################################

system_prompt = {
    'role': 'system', 
    'content': character_personality
}

# start talk  ##########################################################################################################

engine.set_voice(voice)
stream = TextToAudioStream(engine)
model, answer, history = faster_whisper.WhisperModel(model_size_or_path=whisper_speech_to_text_model, device='cuda' if torch.cuda.is_available() else 'cpu'), "", []

def generate(messages):
    global answer
    answer = ""
    for chunk in openai.ChatCompletion.create(model=openai_model, messages=messages, stream=True):
        if (text_chunk := chunk["choices"][0]["delta"].get("content")):
            answer += text_chunk
            print(text_chunk, end="", flush=True) 
            yield text_chunk

while True:
    # Wait until user presses space bar
    print("\n\nTap space when you're ready. ", end="", flush=True)
    keyboard.wait('space')
    while keyboard.is_pressed('space'): pass

    # Record from microphone until user presses space bar again
    print("I'm all ears. Tap space when you're done.\n")    
    audio, frames = pyaudio.PyAudio(), []    
    input_stream = audio.open(rate=16000, format=pyaudio.paInt16, channels=1, input=True, frames_per_buffer=512)
    while not keyboard.is_pressed('space'): 
        frames.append(input_stream.read(512))
    input_stream.stop_stream(), input_stream.close(), audio.terminate()

    # Transcribe recording using whisper
    with wave.open("voice_record.wav", 'wb') as wf:
        wf.setparams((1, audio.get_sample_size(pyaudio.paInt16), 16000, 0, 'NONE', 'NONE'))
        wf.writeframes(b''.join(frames))
    #user_text = " ".join(seg.text for seg in model.transcribe("voice_record.wav", language="en")[0])
    user_text = " ".join(seg.text for seg in model.transcribe("voice_record.wav")[0])
    print(f'>>>{user_text}\n<<< ', end="", flush=True)
    history.append({'role': 'user', 'content': user_text})

    # Generate and stream output
    generator = generate([system_prompt] + history[-10:])
    first_chunk = next(generator)
    combined_gen = itertools.chain([first_chunk], generator)

    stream.feed(combined_gen)
    stream.play_async()
    while stream.is_playing():
        if keyboard.is_pressed('space'):
            stream.stop()
            break
        time.sleep(0.1)    

    history.append({'role': 'assistant', 'content': answer})