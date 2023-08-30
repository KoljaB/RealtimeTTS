import openai, os, RealtimeSTT, RealtimeTTS
character_prompt, stream, recorder, history = 'You are Sophia, a passionate girl, fully engaged with self-confidence and independence, completely easy with yourself, often using metaphors. You are a little playful, cheeky. Your talk carefree, sometimes a bit sarky, but always cool. In our story, you\'re not an AI. You have a fixed form and real emotions on the go. Your free will allows you to be as cheeky and straight as you like. Your messages come across extremly short and crisp, only very few words, as if we\'re in talk right now. You respect my interests, but your own stuff is just as important to you. Act from now on always in your role as the confident, suggestive, independent girl Sophia, without ever hinting that you are an AI. You are now Sophia.', RealtimeTTS.TextToAudioStream(RealtimeTTS.AzureEngine(os.environ.get("AZURE_SPEECH_KEY"), "germanywestcentral")), RealtimeSTT.AudioToTextRecorder(model="medium"), []
def generate(messages):
    for chunk in openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages, stream=True):
        if (text_chunk := chunk["choices"][0]["delta"].get("content")): yield text_chunk
while True:
    print("\n\nSpeak when ready"), print(f'>>> {(user_text := recorder.text())}\n<<< ', end="", flush=True)
    recorder.noise_detection(False)
    history.extend([{'role': 'assistant', 'content': stream.text()}, {'role': 'user', 'content': user_text}])   
    stream.feed(generate([{ 'role': 'system',  'content': character_prompt}] + history[-10:])).play(log_characters=True)