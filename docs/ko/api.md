## 구성

### `TextToAudioStream` 초기화 매개변수

`TextToAudioStream` 클래스를 초기화할 때, 그 동작을 사용자 정의할 수 있는 다양한 옵션이 있습니다. 사용 가능한 매개변수는 다음과 같습니다:

#### `engine` (BaseEngine)
- **유형**: BaseEngine
- **필수**: 네 텍스트를 오디오로 변환하는 데 책임이 있는 기본 엔진. 오디오 합성을 활성화하려면 `BaseEngine` 또는 그 하위 클래스의 인스턴스를 제공해야 합니다.

#### `on_text_stream_start` (호출 가능)
- **유형**: 호출 가능한 함수
- **필수**: 아니요 이 선택적 콜백 함수는 텍스트 스트림이 시작될 때 호출됩니다. 필요한 설정이나 로깅에 사용하세요.

#### `on_text_stream_stop` (호출 가능)
- **유형**: 호출 가능한 함수
- **필수**: 아니요 이 선택적 콜백 함수는 텍스트 스트림이 끝날 때 활성화됩니다. 이것을 정리 작업이나 로깅에 사용할 수 있습니다.

#### `on_audio_stream_start` (호출 가능)
- **유형**: 호출 가능한 함수
- **필수**: 아니요 이 선택적 콜백 함수는 오디오 스트림이 시작될 때 호출됩니다. UI 업데이트나 이벤트 로깅에 유용합니다.

#### `on_audio_stream_stop` (호출 가능)
- **유형**: 호출 가능한 함수
- **필수**: 아니요 이 선택적 콜백 함수는 오디오 스트림이 중지될 때 호출됩니다. 리소스 정리나 후처리 작업에 적합합니다.

#### `on_character` (callable)
- **유형**: 호출 가능한 함수
- **필수**: 아니요 이 선택적 콜백 함수는 단일 문자가 처리될 때 호출됩니다.

#### `output_device_index` (int)
- **유형**: 정수
- **필수**: 아니요 사용할 출력 장치 인덱스를 지정합니다. 아무도 기본 장치를 사용하지 않습니다.

#### `tokenizer` (string)
- **유형**: 문자열
- **필수**: 아니요
- **기본값**: nltk
- **설명**: 문장 분할에 사용할 토크나이저 (currently "nltk" and "stanza" are supported).

#### `language` (문자열)
- **유형**: 문자열
- **필수**: 아니요 문장 분할에 사용할 언어.

#### `muted` (bool)
- **유형**: Bool
- **필수**: 아니오
- **기본값**: False
- **설명**: 전역 음소거 매개변수. 참이면, pyAudio 스트림이 열리지 않습니다. 로컬 스피커를 통한 오디오 재생을 비활성화합니다 (파일로 합성하거나 오디오 청크를 처리하려는 경우) 및 재생 매개변수의 음소거 설정을 무시합니다.

#### `level` (int)
- **Type**: 정수
- **Required**: 아니요
- **기본값**: `logging.WARNING`
- **설명**: 내부 로거의 로깅 수준을 설정합니다. 이는 Python의 내장 `logging` 모듈에서 제공하는 정수 상수일 수 있습니다.

#### 예시 사용법:

```python
engine = YourEngine()  # 엔진을 당신의 엔진으로 대체하세요
stream = TextToAudioStream(
    engine=engine,
    on_text_stream_start=my_text_start_func,
    on_text_stream_stop=my_text_stop_func,
    on_audio_stream_start=my_audio_start_func,
    on_audio_stream_stop=my_audio_stop_func,
    level=logging.INFO
)
```

### 방법

#### `play` 및 `play_async`

이 방법들은 텍스트-오디오 합성을 실행하고 오디오 스트림을 재생하는 역할을 합니다. 차이점은 `play`가 블로킹 함수인 반면, `play_async`는 별도의 스레드에서 실행되어 다른 작업이 진행될 수 있다는 것입니다.

##### 매개변수:

###### `fast_sentence_fragment` (bool)
- **기본값**: `True`
- **설명**: `True`로 설정하면, 이 방법은 속도를 우선시하여 문장 조각을 더 빨리 생성하고 재생합니다. 이는 지연 시간이 중요한 애플리케이션에 유용합니다.

###### `fast_sentence_fragment_allsentences` (bool)
- **기본값**: `False`
- **설명**: `True`로 설정하면 첫 번째 문장뿐만 아니라 모든 문장에 대해 빠른 문장 조각 처리를 적용합니다.

###### `fast_sentence_fragment_allsentences_multiple` (bool)
- **기본값**: `False`
- **설명**: `True`로 설정하면 단일 문장 조각 대신 여러 문장 조각을 생성할 수 있습니다.

###### `buffer_threshold_seconds` (float)
- **기본값**: `0.0`
- **설명**: 버퍼링 임계값을 초 단위로 지정하며, 이는 오디오 재생의 부드러움과 연속성에 영향을 미칩니다.

  - **작동 방식**: 새로운 문장을 합성하기 전에 시스템은 버퍼에 남아 있는 오디오 자료가 `buffer_threshold_seconds`로 지정된 시간보다 더 많은지 확인합니다. 그렇다면, 텍스트 생성기에서 또 다른 문장을 가져오는데, 이는 버퍼에 남아 있는 오디오의 시간 창 내에서 이 새로운 문장을 가져오고 합성할 수 있다고 가정합니다. 이 과정은 텍스트 음성 변환 엔진이 더 나은 합성을 위해 더 많은 맥락을 갖도록 하여 사용자 경험을 향상시킵니다.

  더 높은 값은 더 많은 미리 버퍼링된 오디오를 보장하여 재생 중 침묵이나 간격이 발생할 가능성을 줄여줍니다. 중단이나 일시 정지가 발생하면 이 값을 늘려보세요.

###### `minimum_sentence_length` (int)
- **기본값**: `10`
- **설명**: 문자열을 합성할 문장으로 간주하기 위한 최소 문자 길이를 설정합니다. 이것은 텍스트 청크가 처리되고 재생되는 방식에 영향을 미칩니다.

###### `minimum_first_fragment_length` (int)
- **기본값**: `10`
- **설명**: 양보하기 전에 첫 번째 문장 조각에 필요한 최소 문자 수.

###### `log_synthesized_text` (bool)
- **기본값**: `False`
- **설명**: 활성화되면, 텍스트 조각이 오디오로 합성될 때 로그를 기록합니다. 감사 및 디버깅에 유용합니다.

###### `reset_generated_text` (bool)
- **기본값**: `True`
- **설명**: 참이면, 처리하기 전에 생성된 텍스트를 재설정하세요.

###### `output_wavfile` (str)
- **기본값**: `None`
- **설명**: 설정된 경우, 오디오를 지정된 WAV 파일로 저장합니다.

###### `on_sentence_synthesized` (호출 가능)
- **기본값**: `None`
- **설명**: 단일 문장 조각이 합성된 후 호출되는 콜백 함수.

###### `before_sentence_synthesized` (호출 가능)
- **기본값**: `없음`
- **설명**: 단일 문장 조각이 합성되기 전에 호출되는 콜백 함수.

###### `on_audio_chunk` (호출 가능)
- **기본값**: `None`
- **설명**: 단일 오디오 청크가 준비되면 호출되는 콜백 함수.

###### `tokenizer` (str)
- **기본값**: `"nltk"`
- **설명**: 문장 분리를 위한 토크나이저. 현재 "nltk"와 "stanza"를 지원합니다.

###### `tokenize_sentences` (호출 가능)
- **기본값**: `None`
- **설명**: 입력 텍스트에서 문장을 토큰화하는 사용자 정의 함수. nltk와 stanza에 만족하지 않으면 자신만의 경량 토크나이저를 제공할 수 있습니다. 텍스트를 문자열로 받아서 문장으로 나눈 후 문자열 목록으로 반환해야 합니다.

###### `language` (str)
- **기본값**: `"en"`
- **설명**: 문장 분할에 사용할 언어.

###### `context_size` (int)
- **기본값**: `12`
- **설명**: 문장 경계 감지를 위한 컨텍스트를 설정하는 데 사용되는 문자 수. 더 넓은 맥락이 문장 경계를 감지하는 정확성을 높입니다.

###### `context_size_look_overhead` (int)
- **기본값**: `12`
- **설명**: 문장 경계를 감지할 때 미리 보기 위한 추가 컨텍스트 크기.

###### `muted` (bool)
- **기본값**: `False`
- **설명**: 참이면, 로컬 스피커를 통한 오디오 재생을 비활성화합니다. 파일로 합성하거나 오디오 청크를 재생하지 않고 처리할 때 유용합니다.

###### `sentence_fragment_delimiters` (str)
- **기본값**: `".?!;:,\n…)]}。-"` 문장 구분자로 간주되는 문자 문자열.

###### `force_first_fragment_after_words` (int)
- **기본값**: `15`
- **설명**: 첫 번째 문장 조각이 강제로 생성되는 단어 수.

