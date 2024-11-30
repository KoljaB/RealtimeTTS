import argparse

if __name__ == "__main__":
    print_voices = False
    print_detailed_voices = False

    from RealtimeTTS import TextToAudioStream, EdgeEngine, EdgeVoice
    import time
    import signal
    import keyboard

    parser = argparse.ArgumentParser(description="Realtime TTS script with pause/resume functionality.")
    parser.add_argument("-l", "-voices", "-list", "--list-voices", "--listvoices", action="store_true", help="List all available voices and exit.")
    parser.add_argument("-dl", "-detailedvoices", "--detailed-voices", action="store_true", help="List all available voices with detailed information and exit.")
    parser.add_argument("-t", "-text", "--text", type=str, default="", help="Text to convert to speech")
    parser.add_argument("-v", "--voice", type=str, default="en-US-JennyNeural", help="Voice to use for speech synthesis")
    args = parser.parse_args()

    text = args.text
    voice = args.voice

    if args.list_voices:
        print_voices = True
    
    if args.detailed_voices:
        print_detailed_voices = True

    def dummy_generator():
        if text:
            yield text
            return
        yield "This is a longer test with multiple sentences to demonstrate pause and resume functionality. "
        yield "Press spacebar to pause or resume playback. Press Q to quit. "
        yield "Quick technical note: When you press pause, there may be a brief delay before the audio stops with EdgeEngine. "
        yield "This happens because the audio player (MPV) pre-buffers some audio data for smooth playback - like water still flowing from a pipe even after you close the tap. "

    engine = EdgeEngine(rate=5, pitch=10)


    if print_voices or print_detailed_voices:
        all_voices = engine.get_voices()
        print("Available voices:")
        for i, voice in enumerate(all_voices):
            gender = str(voice.gender)
            gender_lower = gender.lower()

            if print_detailed_voices:
                print(f"Voice {i + 1}:\n{repr(voice)}")
            else:
                print(f"{i + 1}. {voice}")
        exit(0)

    stream = TextToAudioStream(engine)
    if voice:
        engine.set_voice(voice)

    def signal_handler(sig, frame):
        print("\nCtrl+C detected. Cleaning up...")
        stream.stop()
        import sys
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    print("Starting stream - Press SPACE to pause/resume, Q to quit")
    stream.feed(dummy_generator()).play_async(log_synthesized_text=True)

    is_playing = True
    while True:
        if not stream.is_playing():
            print("\nStream finished.")
            break

        if keyboard.is_pressed('space'):
            if is_playing:
                stream.pause()
                print("Paused")
            else:
                stream.resume()
                print("Resumed")
            is_playing = not is_playing
            time.sleep(0.1)  # Debounce
            
        if keyboard.is_pressed('q'):
            print("\nQuitting...")
            break
            
        time.sleep(0.1)
    
    stream.stop()
