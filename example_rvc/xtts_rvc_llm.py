from openai import OpenAI

xtts_model = "models/xtts/Lasinya"
xtts_voice = "Lasinya_Reference.json"
rvc_model = "models/rvc/Lasinya"


if __name__ == "__main__":
    from xtts_rvc_synthesizer import XTTSRVCSynthesizer

    print("Starting synthesizer")
    tts = XTTSRVCSynthesizer(
        xtts_voice=xtts_voice, rvc_model=rvc_model, rvc_sample_rate=40000
    )

    print("Synthesizer ready, starting LLM")
    client = OpenAI()

    stream = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "A three-sentence relaxing speech."}],
        stream=True,
    )

    for chunk in stream:
        token = chunk.choices[0].delta.content
        if token is not None:
            print(token, end="", flush=True)
            tts.push_text(token)

    print()
    tts.synthesize()
    tts.wait_playing()
    tts.shutdown()
