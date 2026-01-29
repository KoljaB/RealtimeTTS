import pyaudio
from cartesia import Cartesia, OutputFormatParams
from .base_engine import BaseEngine

class CartesiaEngine(BaseEngine):
  def __init__(
      self,
      api_key: str = "",
      model_id: str = "sonic-2",
      voice_id: str = "",
      output_format: OutputFormatParams = {
         encoding: "pcm_s16le",
         container: "raw",
         sample_rate: 16000
      },
      debug: bool = False
  ):
    """
    Initializes a Cartesia realtime text to speech engine object.

    Args:
      api_key (str, optional): The API key to use to connect to Cartesia
      model_id (str, required): The model to use for speech synthesis.
        Available models: sonic, sonic-2, sonic-turbo
      voice_id (str, required): The ID of the voice to use for speech synthesis.
      output_format (OutputFormatParams, optional): Parameters defining certain aspects of the generated audio (data type, sample rate, encoding)
        Defaults to raw pcm_s16le at 16kHz
      debug (bool, optional): If True, prints debugging information.
    """
    self.model_id = model_id
    self.voice_id = voice_id
    self.output_format = output_format
    self.debug = debug

    if not api_key:
        api_key = os.environ.get("CARTESIA_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing Cartesia API key. Either:\n"
            "1. Pass the API key as a parameter\n"
            "2. Set CARTESIA_API_KEY environment variable"
        )

    self.client = Cartesia(api_key=api_key).tts.websocket()

  def post_init(self):
      self.engine_name = "cartesia"

  def synthesize(self, text: str) -> bool:
    stream = self.client.send(
      model_id=self.model_id,
      transcript=text,
      output_format=self.output_format,
      voice={
        mode: "id",
        id: self.voice_id
      },
      stream=True
    )

    for chunk in stream:
      self.queue.put(chunk.audio)

    return True

  def get_stream_info(self):
    return self._encoding_to_format, 1, self.output_format.sample_rate

  def shutdown(self):
    self.client.close()

  def set_voice(self, voice):
    self.voice_id = voice

  def _encoding_to_format(self):
    if self.output_format.encoding == "pcm_f32le":
        return pyaudio.paFloat32
    if self.output_format.encoding == "pcm_s16le":
        return pyaudio.paInt16
    if self.output_format.encoding == "pcm_mulaw":
        return pyaudio.paInt16
    if self.output_format.encoding == "pcm_alaw":
        return pyaudio.paInt16
    else:
      raise "Invalid"