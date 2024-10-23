## Configuration

### Initialization Parameters for `TextToAudioStream`

When you initialize the `TextToAudioStream` class, you have various options to customize its behavior. Here are the available parameters:

#### `engine` (BaseEngine)
- **Type**: BaseEngine
- **Required**: Yes
- **Description**: The underlying engine responsible for text-to-audio synthesis. You must provide an instance of `BaseEngine` or its subclass to enable audio synthesis.

#### `on_text_stream_start` (callable)
- **Type**: Callable function
- **Required**: No
- **Description**: This optional callback function is triggered when the text stream begins. Use it for any setup or logging you may need.

#### `on_text_stream_stop` (callable)
- **Type**: Callable function
- **Required**: No
- **Description**: This optional callback function is activated when the text stream ends. You can use this for cleanup tasks or logging.

#### `on_audio_stream_start` (callable)
- **Type**: Callable function
- **Required**: No
- **Description**: This optional callback function is invoked when the audio stream starts. Useful for UI updates or event logging.

#### `on_audio_stream_stop` (callable)
- **Type**: Callable function
- **Required**: No
- **Description**: This optional callback function is called when the audio stream stops. Ideal for resource cleanup or post-processing tasks.

#### `on_character` (callable)
- **Type**: Callable function
- **Required**: No
- **Description**: This optional callback function is called when a single character is processed.

#### `output_device_index` (int)
- **Type**: Integer
- **Required**: No
- **Default**: None
- **Description**: Specifies the output device index to use. None uses the default device.

#### `tokenizer` (string)
- **Type**: String
- **Required**: No
- **Default**: nltk
- **Description**: Tokenizer to use for sentence splitting (currently "nltk" and "stanza" are supported).

#### `language` (string)
- **Type**: String
- **Required**: No
- **Default**: en
- **Description**: Language to use for sentence splitting.

#### `muted` (bool)
- **Type**: Bool
- **Required**: No
- **Default**: False
- **Description**: Global muted parameter. If True, no pyAudio stream will be opened. Disables audio playback via local speakers (in case you want to synthesize to file or process audio chunks) and overrides the play parameters muted setting.

#### `level` (int)
- **Type**: Integer
- **Required**: No
- **Default**: `logging.WARNING`
- **Description**: Sets the logging level for the internal logger. This can be any integer constant from Python's built-in `logging` module.

#### Example Usage:

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

### Methods

#### `play` and `play_async`

These methods are responsible for executing the text-to-audio synthesis and playing the audio stream. The difference is that `play` is a blocking function, while `play_async` runs in a separate thread, allowing other operations to proceed.

##### Parameters:

###### `fast_sentence_fragment` (bool)
- **Default**: `True`
- **Description**: When set to `True`, the method will prioritize speed, generating and playing sentence fragments faster. This is useful for applications where latency matters.

###### `fast_sentence_fragment_allsentences` (bool)
- **Default**: `False`
- **Description**: When set to `True`, applies the fast sentence fragment processing to all sentences, not just the first one.

###### `fast_sentence_fragment_allsentences_multiple` (bool)
- **Default**: `False`
- **Description**: When set to `True`, allows yielding multiple sentence fragments instead of just a single one.

###### `buffer_threshold_seconds` (float)
- **Default**: `0.0`
- **Description**: Specifies the time in seconds for the buffering threshold, which impacts the smoothness and continuity of audio playback.

  - **How it Works**: Before synthesizing a new sentence, the system checks if there is more audio material left in the buffer than the time specified by `buffer_threshold_seconds`. If so, it retrieves another sentence from the text generator, assuming that it can fetch and synthesize this new sentence within the time window provided by the remaining audio in the buffer. This process allows the text-to-speech engine to have more context for better synthesis, enhancing the user experience.

  A higher value ensures that there's more pre-buffered audio, reducing the likelihood of silence or gaps during playback. If you experience breaks or pauses, consider increasing this value.

###### `minimum_sentence_length` (int)
- **Default**: `10`
- **Description**: Sets the minimum character length to consider a string as a sentence to be synthesized. This affects how text chunks are processed and played.

###### `minimum_first_fragment_length` (int)
- **Default**: `10`
- **Description**: The minimum number of characters required for the first sentence fragment before yielding.

###### `log_synthesized_text` (bool)
- **Default**: `False`
- **Description**: When enabled, logs the text chunks as they are synthesized into audio. Helpful for auditing and debugging.

###### `reset_generated_text` (bool)
- **Default**: `True`
- **Description**: If True, reset the generated text before processing.

###### `output_wavfile` (str)
- **Default**: `None`
- **Description**: If set, save the audio to the specified WAV file.

###### `on_sentence_synthesized` (callable)
- **Default**: `None`
- **Description**: A callback function that gets called after a single sentence fragment was synthesized.

###### `before_sentence_synthesized` (callable)
- **Default**: `None`
- **Description**: A callback function that gets called before a single sentence fragment gets synthesized.

###### `on_audio_chunk` (callable)
- **Default**: `None`
- **Description**: Callback function that gets called when a single audio chunk is ready.

###### `tokenizer` (str)
- **Default**: `"nltk"`
- **Description**: Tokenizer to use for sentence splitting. Currently supports "nltk" and "stanza".

###### `tokenize_sentences` (callable)
- **Default**: `None`
- **Description**: A custom function that tokenizes sentences from the input text. You can provide your own lightweight tokenizer if you are unhappy with nltk and stanza. It should take text as a string and return split sentences as a list of strings.

###### `language` (str)
- **Default**: `"en"`
- **Description**: Language to use for sentence splitting.

###### `context_size` (int)
- **Default**: `12`
- **Description**: The number of characters used to establish context for sentence boundary detection. A larger context improves the accuracy of detecting sentence boundaries.

###### `context_size_look_overhead` (int)
- **Default**: `12`
- **Description**: Additional context size for looking ahead when detecting sentence boundaries.

###### `muted` (bool)
- **Default**: `False`
- **Description**: If True, disables audio playback via local speakers. Useful when you want to synthesize to a file or process audio chunks without playing them.

###### `sentence_fragment_delimiters` (str)
- **Default**: `".?!;:,\n…)]}。-"`
- **Description**: A string of characters that are considered sentence delimiters.

###### `force_first_fragment_after_words` (int)
- **Default**: `15`
- **Description**: The number of words after which the first sentence fragment is forced to be yielded.

