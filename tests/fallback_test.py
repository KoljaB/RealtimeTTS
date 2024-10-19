if __name__ == "__main__":
    import os
    from RealtimeTTS import (
        TextToAudioStream,
        AzureEngine,
        CoquiEngine,
        ElevenlabsEngine,
        SystemEngine,
    )

    def dummy_generator():
        yield "Hey guys! These here are realtime spoken sentences based on local text synthesis. "
        yield "With a local, neuronal, cloned voice. So every spoken sentence sounds unique."

    import logging

    logging.basicConfig(level=logging.DEBUG)

    azure_speech_key = os.environ.get("AZURE_SPEECH_KEY")
    azure_speech_region = os.environ.get("AZURE_SPEECH_REGION")
    azure_speech_region = "xyz"  # induce error
    elevenlabs_api_key = os.environ.get("ELEVENLABS_API_KEY")

    azure_engine = AzureEngine(azure_speech_key, azure_speech_region)
    coqui_engine = CoquiEngine(level=logging.DEBUG)
    elevenlabs_engine = ElevenlabsEngine(elevenlabs_api_key)
    system_engine = SystemEngine()

    engines = [
        AzureEngine(azure_speech_key, azure_speech_region),
        coqui_engine,
        system_engine,
    ]

    stream = TextToAudioStream(engines)

    os.system("cls")
    print("Starting to play stream")
    stream.feed(dummy_generator()).play()

    coqui_engine.shutdown()
