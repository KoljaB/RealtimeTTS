from openai import OpenAI
from RealtimeTTS import (
    TextToAudioStream,
    CoquiEngine,
)


if __name__ == "__main__":
    client = OpenAI()

    stream = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": "A three-sentence relaxing speech."}],
        stream=True,
    )

    TextToAudioStream(CoquiEngine(), log_characters=True).feed(stream).play()
