# Konfiguration

## Initialisierungsparameter für TextToAudioStream

Bei der Initialisierung der TextToAudioStream-Klasse haben Sie verschiedene Möglichkeiten, deren Verhalten anzupassen. Hier sind die verfügbaren Parameter:

### `engine` (BaseEngine)
- Typ: `BaseEngine`
- Erforderlich: Ja
- Beschreibung: Die zugrunde liegende Engine, die für die Text-zu-Audio-Synthese verantwortlich ist. Sie müssen eine Instanz von BaseEngine oder deren Unterklasse bereitstellen, um die Audio-Synthese zu ermöglichen.

### `on_text_stream_start` (callable)
- Typ: `Callable function`
- Erforderlich: Nein
- Beschreibung: Diese optionale Callback-Funktion wird ausgelöst, wenn der Textstream beginnt. Verwenden Sie sie für beliebige Setup- oder Logging-Aufgaben.

### `on_text_stream_stop` (callable)
- Typ: `Callable function`
- Erforderlich: Nein
- Beschreibung: Diese optionale Callback-Funktion wird aktiviert, wenn der Textstream endet. Sie können diese für Aufräumarbeiten oder Logging verwenden.

### `on_audio_stream_start` (callable)
- Typ: `Callable function`
- Erforderlich: Nein
- Beschreibung: Diese optionale Callback-Funktion wird aufgerufen, wenn der Audiostream startet. Nützlich für UI-Aktualisierungen oder Event-Logging.

### `on_audio_stream_stop` (callable)
- Typ: `Callable function`
- Erforderlich: Nein
- Beschreibung: Diese optionale Callback-Funktion wird aufgerufen, wenn der Audiostream stoppt. Ideal für Ressourcenbereinigung oder Nachbearbeitungsaufgaben.

### `on_character` (callable)
- Typ: `Callable function`
- Erforderlich: Nein
- Beschreibung: Diese optionale Callback-Funktion wird aufgerufen, wenn ein einzelnes Zeichen verarbeitet wird.

### `output_device_index` (int)
- Typ: `Integer`
- Erforderlich: Nein
- Standard: `None`
- Beschreibung: Gibt den zu verwendenden Ausgabegeräte-Index an. None verwendet das Standardgerät.

### `tokenizer` (string)
- Typ: `String`
- Erforderlich: Nein
- Standard: `nltk`
- Beschreibung: Tokenizer für die Satztrennung (derzeit werden "nltk" und "stanza" unterstützt).

### `language` (string)
- Typ: `String`
- Erforderlich: Nein
- Standard: `en`
- Beschreibung: Sprache für die Satztrennung.

### `muted` (bool)
- Typ: `Bool`
- Erforderlich: Nein
- Standard: `False`
- Beschreibung: Globaler Stummschaltungsparameter. Wenn True, wird kein pyAudio-Stream geöffnet. Deaktiviert die Audiowiedergabe über lokale Lautsprecher.

### `level` (int)
- Typ: `Integer`
- Erforderlich: Nein
- Standard: `logging.WARNING`
- Beschreibung: Legt den Logging-Level für den internen Logger fest.

### Beispielverwendung:

```python
engine = YourEngine()  # Substitute with your engine
stream = TextToAudioStream(
    engine=engine,
    on_text_stream_start=my_text_start_func,
    on_text_stream_stop=my_text_stop_func,
    on_audio_stream_start=my_audio_start_func,
    on_audio_stream_stop=my_audio_stop_func,
    level=logging.INFO
)
```
## Methoden

### play und play_async

Diese Methoden sind für die Ausführung der Text-zu-Audio-Synthese und das Abspielen des Audio-Streams verantwortlich. Der Unterschied besteht darin, dass play eine blockierende Funktion ist, während play_async in einem separaten Thread läuft, wodurch andere Operationen fortgesetzt werden können.

#### Parameter:

##### fast_sentence_fragment (bool)
- **Default**: True
- **Beschreibung**: Wenn auf True gesetzt, priorisiert die Methode die Geschwindigkeit und generiert und spielt Satzfragmente schneller ab. Dies ist nützlich für Anwendungen, bei denen die Latenz wichtig ist.

##### fast_sentence_fragment_allsentences (bool)
- **Default**: False
- **Beschreibung**: Wenn auf True gesetzt, wird die schnelle Satzfragmentverarbeitung auf alle Sätze angewendet, nicht nur auf den ersten.

##### fast_sentence_fragment_allsentences_multiple (bool)
- **Default**: False
- **Beschreibung**: Wenn auf True gesetzt, ermöglicht es die Ausgabe mehrerer Satzfragmente anstelle von nur einem.

##### buffer_threshold_seconds (float)
- **Default**: 0.0
- **Beschreibung**: Gibt die Zeit in Sekunden für den Puffer-Schwellenwert an, der die Gleichmäßigkeit und Kontinuität der Audiowiedergabe beeinflusst.

 - **Funktionsweise**: Bevor ein neuer Satz synthetisiert wird, prüft das System, ob mehr Audiomaterial im Puffer verbleibt als die durch buffer_threshold_seconds angegebene Zeit. Wenn ja, ruft es einen weiteren Satz vom Textgenerator ab, unter der Annahme, dass es diesen neuen Satz innerhalb des Zeitfensters, das durch das verbleibende Audio im Puffer bereitgestellt wird, abrufen und synthetisieren kann. Dieser Prozess ermöglicht es der Text-to-Speech-Engine, mehr Kontext für eine bessere Synthese zu haben und verbessert dadurch das Benutzererlebnis.

 Ein höherer Wert sorgt für mehr vorgepuffertes Audio und reduziert die Wahrscheinlichkeit von Stille oder Lücken während der Wiedergabe. Wenn Sie Unterbrechungen oder Pausen bemerken, erhöhen Sie diesen Wert.

##### minimum_sentence_length (int)
- **Default**: 10
- **Beschreibung**: Legt die minimale Zeichenlänge fest, ab der ein String als zu synthetisierender Satz betrachtet wird. Dies beeinflusst, wie Textabschnitte verarbeitet und abgespielt werden.

##### minimum_first_fragment_length (int)
- **Default**: 10
- **Beschreibung**: Die minimale Anzahl von Zeichen, die für das erste Satzfragment erforderlich sind, bevor es ausgegeben wird.

##### log_synthesized_text (bool)
- **Default**: False
- **Beschreibung**: Wenn aktiviert, protokolliert es die Textabschnitte während ihrer Synthese zu Audio. Hilfreich für Überprüfung und Debugging.

##### reset_generated_text (bool)
- **Default**: True
- **Beschreibung**: Wenn True, wird der generierte Text vor der Verarbeitung zurückgesetzt.

##### output_wavfile (str)
- **Default**: None
- **Beschreibung**: Wenn gesetzt, wird das Audio in der angegebenen WAV-Datei gespeichert.

##### on_sentence_synthesized (callable)
- **Default**: None
- **Beschreibung**: Eine Callback-Funktion, die aufgerufen wird, nachdem ein einzelnes Satzfragment synthetisiert wurde.

##### before_sentence_synthesized (callable)
- **Default**: None
- **Beschreibung**: Eine Callback-Funktion, die aufgerufen wird, bevor ein einzelnes Satzfragment synthetisiert wird.

##### on_audio_chunk (callable)
- **Default**: None
- **Beschreibung**: Callback-Funktion, die aufgerufen wird, wenn ein einzelner Audio-Chunk bereit ist.

##### tokenizer (str)
- **Default**: "nltk"
- **Beschreibung**: Tokenizer für die Satztrennung. Unterstützt derzeit "nltk" und "stanza".

##### tokenize_sentences (callable)
- **Default**: None
- **Beschreibung**: Eine benutzerdefinierte Funktion, die Sätze aus dem Eingabetext tokenisiert. Sie können Ihren eigenen leichtgewichtigen Tokenizer bereitstellen, wenn Sie mit nltk und stanza unzufrieden sind. Die Funktion sollte Text als String entgegennehmen und getrennte Sätze als Liste von Strings zurückgeben.

##### language (str)
- **Default**: "en"
- **Beschreibung**: Sprache für die Satztrennung.

##### context_size (int)
- **Default**: 12
- **Beschreibung**: Die Anzahl der Zeichen, die verwendet werden, um den Kontext für die Satzerkennung festzulegen. Ein größerer Kontext verbessert die Genauigkeit der Satzerkennung.

##### context_size_look_overhead (int)
- **Default**: 12
- **Beschreibung**: Zusätzliche Kontextgröße für den Vorausblick bei der Satzerkennung.

##### muted (bool)
- **Default**: False
- **Beschreibung**: Wenn True, wird die Audiowiedergabe über lokale Lautsprecher deaktiviert. Nützlich, wenn Sie in eine Datei synthetisieren oder Audio-Chunks verarbeiten möchten, ohne sie abzuspielen.

##### sentence_fragment_delimiters (str)
- **Default**: ".?!;:,\n…)]}。-"
- **Beschreibung**: Eine Zeichenkette von Zeichen, die als Satztrennzeichen betrachtet werden.

##### force_first_fragment_after_words (int)
- **Default**: 15
- **Beschreibung**: Die Anzahl der Wörter, nach denen das erste Satzfragment erzwungen ausgegeben wird.