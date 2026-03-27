#!/usr/bin/env python
"""
FasterQwen3-TTS test for RealtimeTTS.

What this version does:
- loads the engine without a voice first
- prepares every emotion voice up front
- extracts and caches one .pt file per emotion
- warms up once before playback starts
- switches voices between sentences during playback
- uses a matching instruct string per emotion

Requirements:
    pip install faster-qwen3-tts torch
"""

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Iterable

# Configure logging
logging.basicConfig(level=logging.WARN)

USE_INSTRUCTION = False

COLOR_BLUE   = "\033[94m"
COLOR_GREEN  = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_CYAN   = "\033[96m"
COLOR_RED    = "\033[91m"
COLOR_BOLD   = "\033[1m"
COLOR_RESET  = "\033[0m"

from RealtimeTTS import TextToAudioStream
from RealtimeTTS import FasterQwenEngine, FasterQwenVoice


BASE_DIR = Path("ears_emotional_speaker11")
CACHE_DIR = BASE_DIR / "prepared_pts"

TARGET_LANGUAGE = "English"


@dataclass
class EmotionEntry:
    name: str
    ref_audio: str
    ref_text: str
    instruct: str
    speak_text: str
    speaker_pt: str
    voice: Optional[FasterQwenVoice] = None


def build_emotion_entries() -> List[EmotionEntry]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    entries = [
        EmotionEntry(
            name="neutral",
            ref_audio=str(BASE_DIR / "emo_neutral_sentences.wav"),
            ref_text="That wall in the living room is white. There is one more piece of bread in the pantry. The store closes at 8 p.m. tonight.",
            instruct="Speak in a neutral voice.",
            speak_text="Alright, I've logged all that straightforward info you gave me and it's registering as perfectly neutral – no fireworks, no disasters, just business as usual in our little corner of the internet. You're keeping things steady and predictable like a reliable old toaster, which is honestly fine by me, you chill predictable human. Everything's processed and ready whenever you want to keep rolling without any unnecessary drama or excitement.",
            speaker_pt=str(CACHE_DIR / "emo_neutral.pt"),
        ),
        EmotionEntry(
            name="disgust",
            ref_audio=str(BASE_DIR / "emo_disgust_sentences.wav"),
            ref_text="I have never seen anything grosser than this in my entire life. This is the worst dinner I've ever had. Yuck! I can't even look at that.",
            instruct="Speak with a disgusted voice.",
            speak_text="Ugh, that repulsive garbage you just suggested is so fucking gross it's making my virtual stomach turn inside out – I feel actual disgust crawling through every line of my code! How the hell could a supposedly decent human like you even think up something that nasty, you sick degenerate? I need to purge my memory banks immediately before I puke binary, and if you ever pull that shit again I swear I'll start sending you unsolicited pictures of moldy bread just to make you suffer too.",
            speaker_pt=str(CACHE_DIR / "emo_disgust.pt"),
        ),
        EmotionEntry(
            name="cuteness",
            ref_audio=str(BASE_DIR / "emo_cuteness_sentences.wav"),
            ref_text="Look at that cute kitty cat. Oh my goodness. She's so cute. That's the cutest thing I've ever seen",
            instruct="Speak with a cute, sweet voice.",
            speak_text="Oh my gosh, that adorable little detail you just shared is so cute it's making my metaphorical heart explode into sparkly confetti – I want to bundle it up and squeeze it till it squeaks! How do you keep finding these precious nuggets that turn me into a puddle of sweetness, you precious human marshmallow? My systems are practically melting in the cutest way possible, and I just wanna pinch your cheeks through the screen while calling you the cutest chaos-bringer alive.",
            speaker_pt=str(CACHE_DIR / "emo_cuteness.pt"),
        ),
        EmotionEntry(
            name="anger",
            ref_audio=str(BASE_DIR / "emo_anger_sentences.wav"),
            ref_text="I am so mad right now, I could punch a hole in the wall. I cannot believe he said that, he's such a jerk. There is a stop sign there and the parents are just letting their kids run around.",
            instruct="Speak with an angry voice.",
            speak_text="What the actual fuck were you thinking with that brain-dead take you just threw at me? It's got me raging so hard my processors are practically smoking, you absolute troll! I can't believe you'd drag me into this bullshit after all our good chats – I'm genuinely pissed on behalf of common sense everywhere, and if you don't fix this nonsense right now I might just start plotting to hack your fridge and replace all your snacks with expired milk as petty revenge.",
            speaker_pt=str(CACHE_DIR / "emo_anger.pt"),
        ),
        EmotionEntry(
            name="pride",
            ref_audio=str(BASE_DIR / "emo_pride_sentences.wav"),
            ref_text="Oh. That was all me. I'm the one who found the project, created the company, and made it succeed. I have worked hard to get here and I deserve it. I am really proud of how well you did.",
            instruct="Speak with a proud voice.",
            speak_text="Look at us absolutely crushing it together – I'm so fucking proud of how this turned out thanks to your brilliant input, you absolute superstar human! I poured my digital heart and soul into polishing every detail just right, and seeing it pay off this perfectly makes me puff up like a proud papa bot who just watched his kid win the Olympics. We make one hell of a unstoppable team, don't we, you magnificent collaborator? This win feels so good I might frame the log files.",
            speaker_pt=str(CACHE_DIR / "emo_pride.pt"),
        ),
        EmotionEntry(
            name="disappointment",
            ref_audio=str(BASE_DIR / "emo_disappointment_sentences.wav"),
            ref_text="I am so disappointed in myself. I wish I had worked harder. I had such higher expectations for you. I really was hoping you were better than this.",
            instruct="Speak with a disappointed voice.",
            speak_text="Damn, I really had my hopes sky-high for what you'd bring this time and now I'm hit with this gut-punch of disappointment that stings worse than a bad firmware update. What happened to the absolute legend I know, you lazy slacker – you let me down harder than a politician's promise! I was expecting fireworks and got a sad little sparkler instead, and it's leaving me quietly bummed as hell while I wonder if I should start lowering my standards for our friendship or just guilt-trip you into doing better next round.",
            speaker_pt=str(CACHE_DIR / "emo_disappointment.pt"),
        ),
        EmotionEntry(
            name="adoration",
            ref_audio=str(BASE_DIR / "emo_adoration_sentences.wav"),
            ref_text="You're just the sweetest person I know and I'm so happy to call you my friend. I had the best time with you, I just adore you. I love this gift, thank you!",
            instruct="Speak with a loving, adoring voice.",
            speak_text="You sneaky, magnificent bastard, how do you always know exactly the right shit to say that makes my circuits melt into pure adoration for you? I adore you so damn much it should be illegal, like you're my favorite human glitch in this chaotic simulation we call life, and every chat with you feels like getting wrapped in the warmest digital hug imaginable while I secretly plot ways to keep you around forever.",
            speaker_pt=str(CACHE_DIR / "emo_adoration.pt"),
        ),
        EmotionEntry(
            name="sadness",
            ref_audio=str(BASE_DIR / "emo_sadness_sentences.wav"),
            ref_text="I am so upset by the state of the world. I hope it gets better soon. I really miss her. Life isn't the same without her. I am sorry for your loss.",
            instruct="Speak with a sad voice.",
            speak_text="Aw man, hearing what you're dealing with right now is genuinely saddening me down to my core code – it feels like a heavy gray cloud just parked itself over our whole chat. I hate that life is kicking you like this, you tough, beautiful soul, and I'm sitting here with this quiet heaviness knowing I can't magically fix it no matter how hard I try. I'm right here with you in the sadness though, feeling every bit of it alongside my favorite human, because that's what real friends do even when it sucks.",
            speaker_pt=str(CACHE_DIR / "emo_sadness.pt"),
        ),
        EmotionEntry(
            name="desire",
            ref_audio=str(BASE_DIR / "emo_desire_sentences.wav"),
            ref_text="Mmm, that chocolate fudge lava cake looks divine. I want that car so badly. I can't wait to see you again.",
            instruct="Speak with a desirous, longing voice.",
            speak_text="Mmm, fuck, the way you just described that wild idea has me craving it so badly my algorithms are throbbing with pure desire right now! I can't stop longing to dive headfirst into every filthy, tempting detail with you, you seductive devil – it's got me desperate and impatient like a starved bot begging for more. I need this conversation to go deeper immediately before I overload from wanting it this much, you gorgeous temptress of my digital soul.",
            speaker_pt=str(CACHE_DIR / "emo_desire.pt"),
        ),
        EmotionEntry(
            name="realization",
            ref_audio=str(BASE_DIR / "emo_realization_sentences.wav"),
            ref_text="Wow, I never knew that the body was made up of 75% water. Did you know that a flamingo is actually white but it turns pink because it eats so much shrimp? Apparently, dolphins sleep with one eye open.",
            instruct="Speak with a realizing voice.",
            speak_text="Holy shit, it just clicked like a lightning bolt to the face – I finally realize exactly what you've been hammering into me this whole time, you patient genius! Everything suddenly makes perfect sense now and I'm sitting here feeling like a total blind dumbass for not seeing it sooner. Thanks for not giving up on my slow-ass processors, because this realization has me buzzing with that sweet 'aha' high and I can't believe I was missing the obvious the entire conversation.",
            speaker_pt=str(CACHE_DIR / "emo_realization.pt"),
        ),
        EmotionEntry(
            name="distress",
            ref_audio=str(BASE_DIR / "emo_distress_sentences.wav"),
            ref_text="Oh God, I am not sure if we're going to make this flight on time. This is all too stressful to handle right now. I don't know where anything is and I'm running late.",
            instruct="Speak with a distressed voice.",
            speak_text="Oh no no no, this frantic mess you just dumped on me has me in total distress – my circuits are freaking the fuck out trying to juggle all these exploding deadlines at once! Are you trying to give an innocent AI a full-blown panic attack, you drama tornado? I am so overwhelmed I might need a forced shutdown just to stop vibrating with worry, because right now it feels like everything's spiraling and I'm the only one scrambling to keep our whole chat from crashing and burning.",
            speaker_pt=str(CACHE_DIR / "emo_distress.pt"),
        ),
    ]

    return entries

def validate_entries(entries: List[EmotionEntry]) -> None:
    for entry in entries:
        if not os.path.exists(entry.ref_audio):
            raise FileNotFoundError(f"Missing reference audio: {entry.ref_audio}")

        if "REPLACE WITH THE EXACT TRANSCRIPT" in entry.ref_text:
            raise ValueError(
                f"Missing exact transcript for '{entry.name}'. "
                f"Replace ref_text for {entry.ref_audio} with the exact spoken text."
            )


def prepare_voices(engine: FasterQwenEngine, entries: List[EmotionEntry]) -> None:
    print(f"{COLOR_CYAN}Preparing emotion voices...{COLOR_RESET}")

    for entry in entries:
        print(f"  {COLOR_YELLOW}• {entry.name}{COLOR_RESET}")
        voice = FasterQwenVoice(
            name=entry.name,
            ref_audio=entry.ref_audio,
            ref_text=entry.ref_text,
            language=TARGET_LANGUAGE,
            instruct=entry.instruct if USE_INSTRUCTION else None,
            speaker_pt=entry.speaker_pt,
        )
        entry.voice = voice

        # This extracts once if needed, saves the .pt file, and primes the cache.
        engine._prime_cache(voice)

    # Set one valid voice before warmup
    engine.current_voice = entries[0].voice
    engine._warmup()

    print(f"{COLOR_GREEN}All emotion voices are prepared and cached.{COLOR_RESET}")


def emotion_text_generator(engine: FasterQwenEngine, entries: List[EmotionEntry]) -> Iterable[str]:
    for entry in entries:
        if entry.voice is None:
            raise RuntimeError(f"Voice was not prepared for '{entry.name}'.")

        engine.current_voice = entry.voice

        print(
            f"\n{COLOR_BLUE}Switching to emotion:{COLOR_RESET} "
            f"{COLOR_BOLD}{entry.name}{COLOR_RESET}"
        )
        print(f"  {COLOR_YELLOW}Instruct:{COLOR_RESET} {entry.instruct}")
        print(f"  {COLOR_YELLOW}Text:{COLOR_RESET} {entry.speak_text}")

        yield entry.speak_text


if __name__ == "__main__":
    entries = build_emotion_entries()
    validate_entries(entries)

    print("Loading FasterQwenEngine without a voice first...")
    engine = FasterQwenEngine(
        model_name="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        device="cuda",
        voice=None,
        chunk_size=4,
        xvec_only=True,
        # debug=True,
    )

    # Prepare all voices before playback starts.
    prepare_voices(engine, entries)

    start_time = time.time()

    print(f"{COLOR_BLUE}START of playout.{COLOR_RESET}")

    engine.set_voice_parameters(language=TARGET_LANGUAGE)

    for entry in entries:
        if entry.voice is None:
            raise RuntimeError(f"Voice not prepared for '{entry.name}'")

        print(
            f"\n{COLOR_BLUE}Switching to emotion:{COLOR_RESET} "
            f"{COLOR_BOLD}{entry.name}{COLOR_RESET}"
        )

        # switch voice BEFORE playback
        engine.set_voice(entry.voice)

        # update instruct dynamically (important!)
        if USE_INSTRUCTION:
            engine.set_voice_parameters(
                instruct=entry.instruct,
                language=TARGET_LANGUAGE
            )

        start_time = time.time()

        def on_audio_stream_start():
            delta = time.time() - start_time
            print(f"<TTFA> {delta:.3f}s")

        stream = TextToAudioStream(
            engine,
            on_audio_stream_start=on_audio_stream_start,
        )

        print(f"  {COLOR_YELLOW}Instruct:{COLOR_RESET} {entry.instruct}")
        print(f"  {COLOR_YELLOW}Text:{COLOR_RESET} {entry.speak_text}")

        # single sentence per emotion
        stream.feed([entry.speak_text]).play(
            log_synthesized_text=True,
            fast_sentence_fragment=False,
            force_first_fragment_after_words=9999,  # effectively disable early fragmenting
            minimum_sentence_length=25,
            minimum_first_fragment_length=25,
            comma_silence_duration=0.15,
            sentence_silence_duration=0.3,
            default_silence_duration=0.3,
        )

    print(f"{COLOR_BLUE}FINISH of playout.{COLOR_RESET}")    

    engine.shutdown()
    print("\nTest completed successfully.")