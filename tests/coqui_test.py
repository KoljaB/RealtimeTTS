if __name__ == '__main__':
    import os
    from RealtimeTTS import TextToAudioStream, CoquiEngine

    def dummy_generator():
        yield "Hey guys! These here are realtime spoken sentences based on local text synthesis. "
        yield "With a local, neuronal, cloned voice. So every spoken sentence sounds unique."

    # import logging
    # logging.basicConfig(level=logging.DEBUG)    
    # engine = CoquiEngine(cloning_reference_wav="female.wav", level=logging.DEBUG)
    engine = CoquiEngine(cloning_reference_wav="female.wav")
    stream = TextToAudioStream(engine)
    
    os.system('cls')
    print ("Starting to play stream")
    stream.feed(dummy_generator()).play()

    engine.shutdown()