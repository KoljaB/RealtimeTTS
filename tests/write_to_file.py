"""
NOTE: Write to file does not work with Elevenlabs Engine due to their api working with mpeg instead of chunks (and also encapsulates their chunk handling)
"""

if __name__ == "__main__":
    from RealtimeTTS import TextToAudioStream, CoquiEngine, AzureEngine, SystemEngine
    import os

    def dummy_generator():
        yield "Hey guys! These here are realtime spoken sentences based on local text synthesis. "
        yield "With a local, neuronal, cloned voice. So every spoken sentence sounds unique."

    print("Starting engines")
    coqui_engine = CoquiEngine()
    azure_engine = AzureEngine(
        os.environ.get("AZURE_SPEECH_KEY"), os.environ.get("AZURE_SPEECH_REGION")
    )
    system_engine = SystemEngine()

    stream = TextToAudioStream(azure_engine)
    stream.feed(dummy_generator())
    stream.play(output_wavfile=stream.engine.engine_name + "_output.wav")

    stream.load_engine(coqui_engine)
    stream.feed(dummy_generator())
    stream.play(output_wavfile=stream.engine.engine_name + "_output.wav")
    coqui_engine.shutdown()

    stream.load_engine(system_engine)
    stream.feed(dummy_generator())
    stream.play(output_wavfile=stream.engine.engine_name + "_output.wav")
