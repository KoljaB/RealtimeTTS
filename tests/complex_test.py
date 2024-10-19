import time
from RealtimeTTS import TextToAudioStream, SystemEngine


print()
print("Initializing")
print()

# select engine
####################################################################################################

# System Engine
#
engine = SystemEngine()
# engine = SystemEngine(voice = "David")

# Azure Engine
#
# engine = AzureEngine(os.environ.get("AZURE_SPEECH_KEY"), "eastus")
# engine = AzureEngine(os.environ.get("AZURE_SPEECH_KEY"), "eastus", "en-US-AshleyNeural", rate=20.0, pitch=10.0)

# Elevenlabs Engine
#
# engine = ElevenlabsEngine(os.environ.get("ELEVENLABS_API_KEY"))
# engine = ElevenlabsEngine(os.environ.get("ELEVENLABS_API_KEY"), voice="Dorothy", id="ThT5KcBeYPX3keUQqHPh", category="premade", clarity=83.4, stability=30.0, model="eleven_monolingual_v1")

# for voice in engine.get_voices():
#     print(voice)

# exit(0)

# create stream
####################################################################################################

stream = TextToAudioStream(engine)
# stream = TextToAudioStream(engine, level=logging.INFO)


# add text or generators
####################################################################################################

stream.feed("Hello, World! ")


def dummy_generator():
    yield "This "
    time.sleep(0.2)
    yield "is "
    time.sleep(0.2)
    yield "a "
    time.sleep(0.2)
    yield "stream of words. "
    time.sleep(0.2)
    yield "And here's another sentence! Yet, "
    time.sleep(0.2)
    yield "there's more. This ends now. "


stream.feed(dummy_generator())

stream.feed("Nice to be here! ")
stream.feed("Welcome all ")
stream.feed("my dear friends ")
stream.feed("of realtime apps. ")


# start stream (blocking or async)
####################################################################################################

print()
print("Starting stream")
print()

# blocking
#
# stream.play()


# async  (allows pause, resume and stop of stream)
#
stream.play_async()
# stream.play_async() # prints every character processed to console


# async stream handling
####################################################################################################

# pause
#
# time.sleep(2)
# print("pause")
# stream.pause() # won't work with elevenlabs (no immediate pause possible due to MP3 stream handling)

# # resume
# #
# time.sleep(2)
# print("resume")
# stream.resume() # won't work with elevenlabs (no immediate pause possible due to MP3 stream handling)

# # immediate stop
# #
# # time.sleep(2)
# # print("stop")
# # stream.stop()

# wait for stream to finish
#
print("waiting for stream to finish")
print()
while stream.is_playing():
    print(".", end="", flush=True)
    time.sleep(0.2)
print()
print("finished stream")
print()

# access text stream while or after finishing stream
#
print(f"text synthesized: {stream.text()}")
