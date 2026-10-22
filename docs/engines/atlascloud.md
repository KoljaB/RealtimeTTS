# Atlas Cloud Engine

`AtlasCloudEngine` submits speech generation to Atlas Cloud and feeds the
resulting MP3 data into RealtimeTTS playback.

## Install

```bash
pip install "realtimetts[atlascloud]"
```

Set `ATLASCLOUD_API_KEY`, then create the optional engine:

```python
from RealtimeTTS import AtlasCloudEngine, TextToAudioStream

engine = AtlasCloudEngine(
    model="minimax/speech-2.6-turbo",
    voice="English_expressive_narrator",
)
stream = TextToAudioStream(engine)
stream.feed("Hello from Atlas Cloud.")
stream.play()
```

The engine uses Atlas Cloud's asynchronous audio API. It submits each text
segment once, polls the prediction endpoint until completion, then streams the
downloaded MP3 response into the playback queue. `speed`, `volume`, `pitch`,
and `sample_rate` can be set on the engine constructor.
