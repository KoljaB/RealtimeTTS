import azure.cognitiveservices.speech as tts
from .base_engine import BaseEngine
from typing import Union
import requests
import pyaudio
import logging

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

    @staticmethod
    def _extract_voice_name(full_name):
        # Extracts the real voice name from the full "Name" string
        start_index = full_name.rfind(", ") + 2
        end_index = full_name.rfind(")")
        return full_name[start_index:end_index]     

    @staticmethod
    def _extract_voice_language(locale):
        # Extracts the language from the locale string
        end_index = locale.find("-")
        return locale[:end_index]


class AzureEngine(BaseEngine):

    def __init__(self,
                 speech_key: str = "", 
                 service_region: str = "", 
                 voice: str = "en-US-AshleyNeural",
                 rate: float = 0.0,
                 pitch: float = 0.0):
        """
        Initializes an azure voice realtime text to speech engine object.

        Args:
            speech_key (str): Azure subscription key. (TTS API key)
            service_region (str): Azure service region. (Cloud Region ID)
            voice (str, optional): Voice name. Defaults to "en-US-AshleyNeural".
            rate (float, optional): Speech speed as a percentage. Defaults to "0.0". Indicating the relative change.
            pitch (float, optional): Speech pitch as a percentage. Defaults to "0.0". Indicating the relative change.
        """

        self.speech_key = speech_key
        self.service_region = service_region
        self.language = voice[:5]
        self.voice_name = voice
        self.rate = rate
        self.pitch = pitch
        self.emotion = "neutral"
        self.emotion_degree = 1.0
        self.emotion_role = "YoungAdultFemale"
        self.emotion_roles = [
            "Girl",
            "Boy",
            "YoungAdultFemale",
            "YoungAdultMale",
            "OlderAdultFemale",
            "OlderAdultMale",
            "SeniorFemale",
            "SeniorMale"
        ]
        self.emotions = [
            "advertisement_upbeat", "affectionate", "angry", "assistant",
            "calm", "chat", "cheerful", "customerservice", "depressed",
            "disgruntled", "documentary-narration", "embarrassed",
            "empathetic", "envious", "excited", "fearful", "friendly",
            "gentle", "hopeful", "lyrical", "narration-professional",
            "narration-relaxed", "neutral", "newscast", "newscast-casual",
            "newscast-formal", "poetry-reading", "sad", "serious", "shouting",
            "sports_commentary", "sports_commentary_excited", "whispering",
            "terrified", "unfriendly"
        ]

    def post_init(self):
        self.engine_name = "azure"

    def get_stream_info(self):
        """
        Returns the PyAudio stream configuration information suitable for Azure Engine.

        Returns:
            tuple: A tuple containing the audio format, number of channels, and the sample rate.
                  - Format (int): The format of the audio stream. pyaudio.paInt16 represents 16-bit integers.
                  - Channels (int): The number of audio channels. 1 represents mono audio.
                  - Sample Rate (int): The sample rate of the audio in Hz. 16000 represents 16kHz sample rate.
        """        
        return pyaudio.paInt16, 1, 16000

    def synthesize(self, 
                   text: str) -> bool:
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

        emotion_start_tag = f'<mstts:express-as style="{self.emotion}" styledegree="{self.emotion_degree}" role="{self.emotion_role}">'
        emotion_end_tag = "</mstts:express-as>"
        if self.emotion not in self.emotions or self.emotion == "neutral":
            emotion_start_tag = ""
            emotion_end_tag = ""

        ssml_string = f"""
        <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="https://www.w3.org/2001/mstts" xml:lang="{self.language}">
            <voice name="{self.voice_name}">
                {emotion_start_tag}
                    <prosody rate="{self.rate}%" pitch="{self.pitch}%">
                        {text}
                    </prosody>
                {emotion_end_tag}
            </voice>
        </speak>
        """

        logging.debug(f"SSML:\n{ssml_string}")

        # Convert the SSML to audio stream
        result = speech_synthesizer.speak_ssml_async(ssml_string).get()

        if result.reason == tts.ResultReason.SynthesizingAudioCompleted:
            logging.debug(f"Speech synthesized")
            return True
        elif result.reason == tts.ResultReason.Canceled:
            cancellation_details = result.cancellation_details
            print(f"Speech synthesis canceled, check speech_key and service_region: {result.reason}")
            print("Cancellation details: {}".format(cancellation_details.reason))
            print("SSLM:")
            print(ssml_string)
            if cancellation_details.reason == tts.CancellationReason.Error:
                print("Error details: {}".format(cancellation_details.error_details))        
        else:
            print(f"Speech synthesis failed: {result.reason}")
            print(f"Result: {result}")

    def set_emotion(
            self,
            emotion: str,
            emotion_role: str = "YoungAdultFemale",
            emotion_degree: float = 1.0,
            ):
        """
        Sets the emotion to be used for speech synthesis.

        Args:
            emotion (str): The emotion to be used for speech synthesis.
            emotion_degree (float, optional): The degree of the emotion. Defaults to 1.0.
            emotion_role (str, optional): The role of the emotion. Defaults to "YoungAdultFemale".
        """
        # https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-synthesis-markup-voice
        self.emotion = emotion
        self.emotion_degree = emotion_degree
        self.emotion_role = emotion_role

    def get_emotions(self):
        """
        Retrieves the available emotions for the Azure Speech engine.

        Returns:
            list[str]: A list containing the available emotions for the Azure Speech engine.
        """
        return self.emotions

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
        Retrieves the installed voices available for the Azure Speech engine.

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
                if voice in installed_voice.full_name:
                    self.voice_name = installed_voice.full_name
                    self.language = self.voice_name[:5]

    def set_voice_parameters(self, **voice_parameters):
        """
        Sets the voice parameters to be used for speech synthesis.

        Args:
            **voice_parameters: The voice parameters to be used for speech synthesis.
        """
        if 'rate' in voice_parameters:
            self.rate = voice_parameters['rate']
        if 'pitch' in voice_parameters:
            self.pitch = voice_parameters['pitch']