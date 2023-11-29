"""
NOTE: Write to file does not work with Elevenlabs Engine due to their api working with mpeg instead of chunks (and also encapsulates their chunk handling)
"""
if __name__ == '__main__':
    from RealtimeTTS import TextToAudioStream, CoquiEngine, AzureEngine, SystemEngine
    import os

    def dummy_generator():
        yield "Hey guys! These here are realtime spoken sentences based on local text synthesis. "
        yield "With a local, neuronal, cloned voice. So every spoken sentence sounds unique."

    # for normal use with minimal logging:
    #engine = CoquiEngine()
    # engine = AzureEngine(os.environ.get("AZURE_SPEECH_KEY"), os.environ.get("AZURE_SPEECH_REGION"))
    # engine = SystemEngine()

    print ("Starting engines")
    coqui_engine = CoquiEngine()
    azure_engine = AzureEngine(os.environ.get("AZURE_SPEECH_KEY"), os.environ.get("AZURE_SPEECH_REGION"))
    system_engine = SystemEngine()

    stream = TextToAudioStream(azure_engine)
    stream.feed(dummy_generator())
    stream.play(output_wavfile=stream.engine.engine_name + "_output.wav")
    coqui_engine.shutdown()

    stream.load_engine(coqui_engine)
    stream.feed(dummy_generator())
    stream.play(output_wavfile=stream.engine.engine_name + "_output.wav")

    stream.load_engine(system_engine)
    stream.feed(dummy_generator())
    stream.play(output_wavfile=stream.engine.engine_name + "_output.wav")

    print("Done synthesizing")

    # if isinstance(engine, CoquiEngine):    
    #     engine.shutdown()


    # # for normal use with minimal logging:
    # engine = CoquiEngine()
    # # engine = AzureEngine(os.environ.get("AZURE_SPEECH_KEY"), os.environ.get("AZURE_SPEECH_REGION"))
    # # engine = SystemEngine()

    # stream = TextToAudioStream(engine)
    # stream.feed(dummy_generator())
    # stream.play(
    #     output_wavfile=engine.engine_name + "_output.wav",
    #     on_sentence_synthesized=lambda sentence: print("Synthesized: " + sentence))

    # print("Done playing")

    # if isinstance(engine, CoquiEngine):    
    #     engine.shutdown()