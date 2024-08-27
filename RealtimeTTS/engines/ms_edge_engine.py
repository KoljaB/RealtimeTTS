import edge_tts
import pyaudio
from typing import Union
from .base_engine import BaseEngine

class MSEdgeVoice:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"{self.name}"

class MSEdgeEngine(BaseEngine):

    def __init__(self,
                 voice: str = "en-GB-SoniaNeural"):
        """
        Initializes a msedge text to speech engine object.

        Args:
            voice (str, optional): The voice to use for speech synthesis.
              Available voices: af-ZA-AdriNeural, am-ET-MekdesNeural, ar-EG-SalmaNeural, ar-SA-ZariyahNeural to name a few. Run edge-tts --list-voices to get the whole list.
              Defaults to "en-GB-SoniaNeural".
        """

        self.voices = [
            "af-ZA-AdriNeural", "af-ZA-WillemNeural", "am-ET-AmehaNeural", "am-ET-MekdesNeural",
            "ar-AE-FatimaNeural", "ar-AE-HamdanNeural", "ar-BH-AliNeural", "ar-BH-LailaNeural",
            "ar-DZ-AminaNeural", "ar-DZ-IsmaelNeural", "ar-EG-SalmaNeural", "ar-EG-ShakirNeural",
            "ar-IQ-BasselNeural", "ar-IQ-RanaNeural", "ar-JO-SanaNeural", "ar-JO-TaimNeural",
            "ar-KW-FahedNeural", "ar-KW-NouraNeural", "ar-LB-LaylaNeural", "ar-LB-RamiNeural",
            "ar-LY-ImanNeural", "ar-LY-OmarNeural", "ar-MA-JamalNeural", "ar-MA-MounaNeural",
            "ar-OM-AbdullahNeural", "ar-OM-AyshaNeural", "ar-QA-AmalNeural", "ar-QA-MoazNeural",
            "ar-SA-HamedNeural", "ar-SA-ZariyahNeural", "ar-SY-AmanyNeural", "ar-SY-LaithNeural",
            "ar-TN-HediNeural", "ar-TN-ReemNeural", "ar-YE-MaryamNeural", "ar-YE-SalehNeural",
            "az-AZ-BabekNeural", "az-AZ-BanuNeural", "bg-BG-BorislavNeural", "bg-BG-KalinaNeural",
            "bn-BD-NabanitaNeural", "bn-BD-PradeepNeural", "bn-IN-BashkarNeural", "bn-IN-TanishaaNeural",
            "bs-BA-GoranNeural", "bs-BA-VesnaNeural", "ca-ES-EnricNeural", "ca-ES-JoanaNeural",
            "cs-CZ-AntoninNeural", "cs-CZ-VlastaNeural", "cy-GB-AledNeural", "cy-GB-NiaNeural",
            "da-DK-ChristelNeural", "da-DK-JeppeNeural", "de-AT-IngridNeural", "de-AT-JonasNeural",
            "de-CH-JanNeural", "de-CH-LeniNeural", "de-DE-AmalaNeural", "de-DE-ConradNeural",
            "de-DE-FlorianMultilingualNeural", "de-DE-KatjaNeural", "de-DE-KillianNeural",
            "de-DE-SeraphinaMultilingualNeural", "el-GR-AthinaNeural", "el-GR-NestorasNeural",
            "en-AU-NatashaNeural", "en-AU-WilliamNeural", "en-CA-ClaraNeural", "en-CA-LiamNeural",
            "en-GB-LibbyNeural", "en-GB-MaisieNeural", "en-GB-RyanNeural", "en-GB-SoniaNeural",
            "en-GB-ThomasNeural", "en-HK-SamNeural", "en-HK-YanNeural", "en-IE-ConnorNeural",
            "en-IE-EmilyNeural", "en-IN-NeerjaExpressiveNeural", "en-IN-NeerjaNeural",
            "en-IN-PrabhatNeural", "en-KE-AsiliaNeural", "en-KE-ChilembaNeural", "en-NG-AbeoNeural",
            "en-NG-EzinneNeural", "en-NZ-MitchellNeural", "en-NZ-MollyNeural", "en-PH-JamesNeural",
            "en-PH-RosaNeural", "en-SG-LunaNeural", "en-SG-WayneNeural", "en-TZ-ElimuNeural",
            "en-TZ-ImaniNeural", "en-US-AnaNeural", "en-US-AndrewMultilingualNeural",
            "en-US-AndrewNeural", "en-US-AriaNeural", "en-US-AvaMultilingualNeural",
            "en-US-AvaNeural", "en-US-BrianMultilingualNeural", "en-US-BrianNeural",
            "en-US-ChristopherNeural", "en-US-EmmaMultilingualNeural", "en-US-EmmaNeural",
            "en-US-EricNeural", "en-US-GuyNeural", "en-US-JennyNeural", "en-US-MichelleNeural",
            "en-US-RogerNeural", "en-US-SteffanNeural", "en-ZA-LeahNeural", "en-ZA-LukeNeural",
            "es-AR-ElenaNeural", "es-AR-TomasNeural", "es-BO-MarceloNeural", "es-BO-SofiaNeural",
            "es-CL-CatalinaNeural", "es-CL-LorenzoNeural", "es-CO-GonzaloNeural", "es-CO-SalomeNeural",
            "es-CR-JuanNeural", "es-CR-MariaNeural", "es-CU-BelkysNeural", "es-CU-ManuelNeural",
            "es-DO-EmilioNeural", "es-DO-RamonaNeural", "es-EC-AndreaNeural", "es-EC-LuisNeural",
            "es-ES-AlvaroNeural", "es-ES-ElviraNeural", "es-ES-XimenaNeural", "es-GQ-JavierNeural",
            "es-GQ-TeresaNeural", "es-GT-AndresNeural", "es-GT-MartaNeural", "es-HN-CarlosNeural",
            "es-HN-KarlaNeural", "es-MX-DaliaNeural", "es-MX-JorgeNeural", "es-NI-FedericoNeural",
            "es-NI-YolandaNeural", "es-PA-MargaritaNeural", "es-PA-RobertoNeural", "es-PE-AlexNeural",
            "es-PE-CamilaNeural", "es-PR-KarinaNeural", "es-PR-VictorNeural", "es-PY-MarioNeural",
            "es-PY-TaniaNeural", "es-SV-LorenaNeural", "es-SV-RodrigoNeural", "es-US-AlonsoNeural",
            "es-US-PalomaNeural", "es-UY-MateoNeural", "es-UY-ValentinaNeural", "es-VE-PaolaNeural",
            "es-VE-SebastianNeural", "et-EE-AnuNeural", "et-EE-KertNeural", "fa-IR-DilaraNeural",
            "fa-IR-FaridNeural", "fi-FI-HarriNeural", "fi-FI-NooraNeural", "fil-PH-AngeloNeural",
            "fil-PH-BlessicaNeural", "fr-BE-CharlineNeural", "fr-BE-GerardNeural", "fr-CA-AntoineNeural",
            "fr-CA-JeanNeural", "fr-CA-SylvieNeural", "fr-CA-ThierryNeural", "fr-CH-ArianeNeural",
            "fr-CH-FabriceNeural", "fr-FR-DeniseNeural", "fr-FR-EloiseNeural", "fr-FR-HenriNeural",
            "fr-FR-RemyMultilingualNeural", "fr-FR-VivienneMultilingualNeural", "ga-IE-ColmNeural",
            "ga-IE-OrlaNeural", "gl-ES-RoiNeural", "gl-ES-SabelaNeural", "gu-IN-DhwaniNeural",
            "gu-IN-NiranjanNeural", "he-IL-AvriNeural", "he-IL-HilaNeural", "hi-IN-MadhurNeural",
            "hi-IN-SwaraNeural", "hr-HR-GabrijelaNeural", "hr-HR-SreckoNeural", "hu-HU-NoemiNeural",
            "hu-HU-TamasNeural", "id-ID-ArdiNeural", "id-ID-GadisNeural", "is-IS-GudrunNeural",
            "is-IS-GunnarNeural", "it-IT-DiegoNeural", "it-IT-ElsaNeural", "it-IT-GiuseppeNeural",
            "it-IT-IsabellaNeural", "ja-JP-KeitaNeural", "ja-JP-NanamiNeural", "jv-ID-DimasNeural",
            "jv-ID-SitiNeural", "ka-GE-EkaNeural", "ka-GE-GiorgiNeural", "kk-KZ-AigulNeural",
            "kk-KZ-DauletNeural", "km-KH-PisethNeural", "km-KH-SreymomNeural", "kn-IN-GaganNeural",
            "kn-IN-SapnaNeural", "ko-KR-HyunsuNeural", "ko-KR-InJoonNeural", "ko-KR-SunHiNeural",
            "lo-LA-ChanthavongNeural", "lo-LA-KeomanyNeural", "lt-LT-LeonasNeural", "lt-LT-OnaNeural",
            "lv-LV-EveritaNeural", "lv-LV-NilsNeural", "mk-MK-AleksandarNeural", "mk-MK-MarijaNeural",
            "ml-IN-MidhunNeural", "ml-IN-SobhanaNeural", "mn-MN-BataaNeural", "mn-MN-YesuiNeural",
            "mr-IN-AarohiNeural", "mr-IN-ManoharNeural", "ms-MY-OsmanNeural", "ms-MY-YasminNeural",
            "mt-MT-GraceNeural", "mt-MT-JosephNeural", "my-MM-NilarNeural", "my-MM-ThihaNeural",
            "nb-NO-FinnNeural", "nb-NO-PernilleNeural", "ne-NP-HemkalaNeural", "ne-NP-SagarNeural",
            "nl-BE-ArnaudNeural", "nl-BE-DenaNeural", "nl-NL-ColetteNeural", "nl-NL-FennaNeural",
            "nl-NL-MaartenNeural", "pl-PL-MarekNeural", "pl-PL-ZofiaNeural", "ps-AF-GulNawazNeural",
            "ps-AF-LatifaNeural", "pt-BR-AntonioNeural", "pt-BR-FranciscaNeural", "pt-BR-ThalitaNeural",
            "pt-PT-DuarteNeural", "pt-PT-RaquelNeural", "ro-RO-AlinaNeural", "ro-RO-EmilNeural",
            "ru-RU-DmitryNeural", "ru-RU-SvetlanaNeural", "si-LK-SameeraNeural", "si-LK-ThiliniNeural",
            "sk-SK-LukasNeural", "sk-SK-ViktoriaNeural", "sl-SI-PetraNeural", "sl-SI-RokNeural",
            "so-SO-MuuseNeural", "so-SO-UbaxNeural", "sq-AL-AnilaNeural", "sq-AL-IlirNeural",
            "sr-RS-NicholasNeural", "sr-RS-SophieNeural", "su-ID-JajangNeural", "su-ID-TutiNeural",
            "sv-SE-MattiasNeural", "sv-SE-SofieNeural", "sw-KE-RafikiNeural", "sw-KE-ZuriNeural",
            "sw-TZ-DaudiNeural", "sw-TZ-RehemaNeural", "ta-IN-PallaviNeural", "ta-IN-ValluvarNeural",
            "ta-LK-KumarNeural", "ta-LK-SaranyaNeural", "ta-MY-KaniNeural", "ta-MY-SuryaNeural",
            "ta-SG-AnbuNeural", "ta-SG-VenbaNeural", "te-IN-MohanNeural", "te-IN-ShrutiNeural",
            "th-TH-NiwatNeural", "th-TH-PremwadeeNeural", "tr-TR-AhmetNeural", "tr-TR-EmelNeural",
            "uk-UA-OstapNeural", "uk-UA-PolinaNeural", "ur-IN-GulNeural", "ur-IN-SalmanNeural",
            "ur-PK-AsadNeural", "ur-PK-UzmaNeural", "uz-UZ-MadinaNeural", "uz-UZ-SardorNeural",
            "vi-VN-HoaiMyNeural", "vi-VN-NamMinhNeural", "zh-CN-XiaoxiaoNeural", "zh-CN-XiaoyiNeural",
            "zh-CN-YunjianNeural", "zh-CN-YunxiNeural", "zh-CN-YunxiaNeural", "zh-CN-YunyangNeural",
            "zh-CN-liaoning-XiaobeiNeural", "zh-CN-shaanxi-XiaoniNeural", "zh-HK-HiuGaaiNeural",
            "zh-HK-HiuMaanNeural", "zh-HK-WanLungNeural", "zh-TW-HsiaoChenNeural", "zh-TW-HsiaoYuNeural",
            "zh-TW-YunJheNeural", "zu-ZA-ThandoNeural", "zu-ZA-ThembaNeural"
        ]
        self.voice = voice

    def post_init(self):
        self.engine_name = "ms-edge-tts"

    def get_stream_info(self):
        """
        Returns the PyAudio stream configuration information suitable
        for MSEdge Engine.

        Returns:
            tuple: A tuple containing the audio format, number of channels,
              and the sample rate.
                  - Format (int): The format of the audio stream.
                    pyaudio.paInt16 represents 16-bit integers.
                  - Channels (int): The number of audio channels.
                    1 represents mono audio.
                  - Sample Rate (int): The sample rate of the audio in Hz.
                    24000 represents 24kHz sample rate.
        """
        return pyaudio.paCustomFormat, 1, 24000

    def synthesize(self,
                   text: str) -> bool:
        """
        Synthesizes text to audio stream.

        Args:
            text (str): Text to synthesize.
        """
        response = edge_tts.Communicate(text, self.voice)

        for chunk in response.stream_sync():
            if chunk["type"] == "audio":
                self.queue.put(chunk["data"])

        return True

    def get_voices(self):
        """
        Retrieves the installed voices available for the MSEdge TTS engine.
        """
        voice_objects = []
        for voice in self.voices:
            voice_objects.append(MSEdgeVoice(voice))
        return voice_objects

    def set_voice(self, voice: Union[str, object]):
        """
        Sets the voice to be used for speech synthesis.

        Args:
            voice (Union[str, SystemVoice]): The voice to be used for speech
              synthesis.
        """
        if isinstance(voice, MSEdgeVoice):
            self.voice = voice.name
        else:
            installed_voices = self.get_voices()
            for installed_voice in installed_voices:
                if voice in installed_voice.name:
                    self.voice = installed_voice.name

    def set_voice_parameters(self, **voice_parameters):
        """
        Sets the voice parameters to be used for speech synthesis.

        Args:
            **voice_parameters: The voice parameters to be used for speech
              synthesis.
        """
        pass