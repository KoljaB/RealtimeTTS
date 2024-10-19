from RealtimeTTS import (
    TextToAudioStream,
    SystemEngine,
    AzureEngine,
    ElevenlabsEngine,
    CoquiEngine,
)
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QComboBox,
    QTextEdit,
    QLabel,
)
from PyQt6.QtCore import pyqtSlot
import time
import sys
import os


class TTSApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize TTS engines
        print("Initializing TTS Engines...")
        self.engine_system = SystemEngine(voice="Huihui")
        self.engine_azure = AzureEngine(
            os.environ.get("AZURE_SPEECH_KEY"),
            os.environ.get("AZURE_SPEECH_REGION"),
            voice="zh-CN-XiaoxiaoNeural",
        )
        self.engine_elevenlabs = ElevenlabsEngine(os.environ.get("ELEVENLABS_API_KEY"))
        self.engine_coqui = CoquiEngine(voice="female_chinese.wav", language="zh")
        print("TTS Engines initialized.")

        # Add a dictionary to map engine names to engine instances
        self.engines = {
            "System Engine": self.engine_system,
            "Azure Engine": self.engine_azure,
            "Elevenlabs Engine": self.engine_elevenlabs,
            "Coqui Engine": self.engine_coqui,
        }

        # Initialize TTS Stream
        self.stream = TextToAudioStream(
            self.engine_system, on_audio_stream_start=self.on_audio_stream_start
        )

        # Main widget and layout
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        # Dropdown for TTS Engine Selection
        self.tts_engine_dropdown = QComboBox(self)
        self.tts_engine_dropdown.addItems(
            ["System Engine", "Azure Engine", "Elevenlabs Engine", "Coqui Engine"]
        )
        self.tts_engine_dropdown.currentTextChanged.connect(self.tts_engine_changed)
        self.layout.addWidget(self.tts_engine_dropdown)

        # Big Input Text Control
        self.text_input = QTextEdit(self)
        self.text_input.textChanged.connect(self.text_pasted)
        self.layout.addWidget(self.text_input)

        # Label for Latency Display
        self.latency_label = QLabel("Latency: N/A", self)
        self.layout.addWidget(self.latency_label)

        self.setWindowTitle("TTS Synthesis Speed Test")

    @pyqtSlot()
    def tts_engine_changed(self):
        selected_engine_name = self.tts_engine_dropdown.currentText()
        selected_engine = self.engines[selected_engine_name]
        self.stream.load_engine(selected_engine)
        print(f"TTS Engine selected: {selected_engine_name}")

    @pyqtSlot()
    def text_pasted(self):
        pasted_text = self.text_input.toPlainText()
        print(f"Text pasted: {pasted_text}")

        self.time_pasted = time.time()
        self.stream.feed(pasted_text)
        filename = "synthesis_chinese_" + self.stream.engine.engine_name
        self.stream.play_async(
            minimum_sentence_length=2,
            minimum_first_fragment_length=2,
            output_wavfile=f"{filename}.wav",
            on_sentence_synthesized=lambda sentence: print("Synthesized: " + sentence),
            tokenizer="stanza",
            language="zh",
            context_size=2,
        )

    def on_audio_stream_start(self):
        self.time_started = time.time()
        latency = self.time_started - self.time_pasted
        self.latency_label.setText("Latency: {:.2f} seconds".format(latency))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWin = TTSApp()
    mainWin.show()
    sys.exit(app.exec())
