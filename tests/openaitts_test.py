if __name__ == '__main__':
    from RealtimeTTS import TextToAudioStream, OpenAIEngine

    def dummy_generator():
        yield "Hey guys! These here are realtime spoken words based on openai tts text synthesis. "

    stream = TextToAudioStream(OpenAIEngine(model="tts-1-hd"))
    
    print ("Starting to play stream")
    stream.feed(dummy_generator()).play(output_wavfile="synthesis_openai.wav")