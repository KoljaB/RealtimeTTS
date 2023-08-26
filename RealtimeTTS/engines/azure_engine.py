from .base_engine import BaseEngine
import azure.cognitiveservices.speech as tts
import pyaudio
import requests
import logging
from typing import Union

class PushAudioOutputStreamSampleCallback(tts.audio.PushAudioOutputStreamCallback):
    """
    This class provides a callback mechanism to handle audio output streams for Azure's Text-to-Speech (TTS) service.
    It allows you to capture synthesized audio data in real-time and push it to a buffer.

    Attributes:
        buffer: A buffer or queue where the audio stream data will be stored.
    """    
    def __init__(self, buffer):
        self.buffer = buffer

    def write(self, audio_buffer: memoryview) -> int:
        self.buffer.put(audio_buffer.tobytes())
        return audio_buffer.nbytes


class AzureVoice:
    def __init__(self, name, locale, gender):
        self.full_name = name
        self.name = AzureVoice._extract_voice_name(self.full_name)
        self.locale = locale
        self.language = AzureVoice._extract_voice_language(self.locale)
        self.gender = gender

    def __repr__(self):
        return f"{self.name} ({self.gender}, {self.locale})"
        #return f"<Voice(name={self.name}, locale={self.locale}, gender={self.gender})>"

    @staticmethod
    def _extract_voice_name(full_name):
        # Extracts the real voice name from the full "Name" string
        start_index = full_name.rfind(", ") + 2
        end_index = full_name.rfind(")")
        return full_name[start_index:end_index]     

    @staticmethod
    def _extract_voice_language(locale):
        # Extracts the language from the locale string
        #return locale
        end_index = locale.find("-")
        return locale[:end_index]

class AzureEngine(BaseEngine):

    def __init__(self, 
                 speech_key: str = "", 
                 service_region: str = "", 
                 voice: str = "en-US-AshleyNeural", 
                 rate: float = 20.0,
                 pitch: float = 0.0):
        """
        Initializes an azure voice realtime text to speech engine object.

        Args:
            speech_key (str): Azure subscription key. (TTS API key)
            service_region (str): Azure service region. (Cloud Region ID)
            voice (str, optional): Voice name. Defaults to "en-US-AshleyNeural".
            rate (float, optional): Speech speed. Defaults to "0.0".
            pitch (float, optional): Speech pitch. Defaults to "0.0".
        """

        self.speech_key = speech_key
        self.service_region = service_region
        self.language = voice[:5]
        self.voice_name = voice
        self.rate = rate
        self.pitch = pitch

    def get_stream_info(self):
        """
        Returns the audio stream configuration information suitable for PyAudio.

        Returns:
            tuple: A tuple containing the audio format, number of channels, and the sample rate.
                  - Format (int): The format of the audio stream. pyaudio.paInt16 represents 16-bit integers.
                  - Channels (int): The number of audio channels. 1 represents mono audio.
                  - Sample Rate (int): The sample rate of the audio in Hz. 16000 represents 16kHz sample rate.
        """        
        return pyaudio.paInt16, 1, 16000

    def synthesize(self, 
                   text: str):
        """
        Synthesizes text to audio stream.

        Args:
            text (str): Text to synthesize.
        """

        # Set up the Azure TTS configuration
        speech_config = tts.SpeechConfig(subscription=self.speech_key, region=self.service_region)
        stream_callback = PushAudioOutputStreamSampleCallback(self.queue)
        push_stream = tts.audio.PushAudioOutputStream(stream_callback)
        stream_config = tts.audio.AudioOutputConfig(stream=push_stream)
        speech_synthesizer = tts.SpeechSynthesizer(speech_config=speech_config, audio_config=stream_config)

        # Construct the SSML string
        ssml_string = f"""
        <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{self.language}">
            <voice name="{self.voice_name}">
                <prosody rate="{self.rate}%" pitch="{self.pitch}%">
                    {text}
                </prosody>
            </voice>
        </speak>
        """

        logging.debug(f"SSML:\n{ssml_string}")

        # Convert the SSML to audio stream
        result = speech_synthesizer.speak_ssml_async(ssml_string).get()

        # Check for an error
        if result.reason != tts.ResultReason.SynthesizingAudioCompleted:
            print(f"Speech synthesis failed, check speech_key and service_region: {result.reason}")
            raise RuntimeError(f"Speech synthesis failed: {result}")
        
    def set_speech_key(self, speech_key: str):
        """
        Sets the azure subscription key. 

        Args:
            speech_key (str): Azure subscription key. (TTS API key)
        """        
        self.speech_key = speech_key

    def set_service_region(self, service_region: str):
        """
        Sets the azure service region. 

        Args:
            service_region (str): Azure service region. (Cloud Region ID)
        """        
        self.service_region = service_region

    def get_voices(self):
        """
        Retrieves the installed voices available for this Azure Speech engine.

        Sends a request to the Azure Speech API to fetch the list of available voices.
        The method uses the `service_region` and `speech_key` attributes of the instance to authenticate 
        and get the list of voices.

        Returns:
            list[AzureVoice]: A list containing AzureVoice objects representing the available voices. 
                            Each AzureVoice object encapsulates information like the real name, locale, 
                            and gender of the voice. If the API call fails, an empty list is returned.

        Raises:
            May raise exceptions related to the `requests` module like ConnectionError, Timeout, etc.

        Side Effects:
            Makes HTTP requests to the Azure Speech API. Prints an error message to stdout if the 
            request fails.

        Note:
            Ensure that `self.service_region` and `self.speech_key` are correctly set before calling 
            this method. 
        """        
        token_endpoint  = f'https://{self.service_region}.api.cognitive.microsoft.com/sts/v1.0/issueToken'
        headers = {
            'Ocp-Apim-Subscription-Key': self.speech_key
        }
        response = requests.post(token_endpoint, headers=headers)
        access_token = str(response.text)

        fetch_voices_endpoint = f'https://{self.service_region}.tts.speech.microsoft.com/cognitiveservices/voices/list'
        voice_headers = {
            'Authorization': 'Bearer ' + access_token
        }
        response = requests.get(fetch_voices_endpoint, headers=voice_headers)
        
        voice_objects = []
        
        if response.status_code == 200:
            voices = response.json()
            for voice in voices:
                real_name = voice['Name']
                locale = voice['Locale']
                gender = voice.get('Gender', 'N/A')
                voice_object = AzureVoice(real_name, locale, gender)
                voice_objects.append(voice_object)
            
            return voice_objects
        else:
            print(f"Error {response.status_code}: {response.text}")
            return []

    def set_voice(self, voice: Union[str, AzureVoice]):
        """
        Sets the voice to be used for speech synthesis.

        Args:
            voice (Union[str, AzureVoice]): The voice to be used for speech synthesis.
        """
        if isinstance(voice, AzureVoice):
            self.voice_name = voice.full_name
            self.language = self.voice_name[:5]
        else:
            installed_voices = self.get_voices()
            for installed_voice in installed_voices:
                if voice.name in installed_voice.full_name:
                    self.voice_name = installed_voice.full_name
                    self.language = self.voice_name[:5]