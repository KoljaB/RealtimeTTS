# TextToAudioStream - Documentación en Español

## Configuración

### Parámetros de Inicialización para `TextToAudioStream`

Cuando inicializa la clase `TextToAudioStream`, tiene varias opciones para personalizar su comportamiento. Aquí están los parámetros disponibles:

### Parámetros Principales

#### `engine` (BaseEngine)
* **Tipo**: BaseEngine
* **Requerido**: Sí
* **Descripción**: El motor subyacente responsable de la síntesis de texto a audio. Debe proporcionar una instancia de `BaseEngine` o su subclase para habilitar la síntesis de audio.

#### `on_text_stream_start` (callable)
* **Tipo**: Función callable
* **Requerido**: No
* **Descripción**: Esta función de callback opcional se activa cuando comienza el flujo de texto. Utilícela para cualquier configuración o registro que pueda necesitar.

#### `on_text_stream_stop` (callable)
* **Tipo**: Función callable
* **Requerido**: No
* **Descripción**: Esta función de callback opcional se activa cuando finaliza el flujo de texto. Puede utilizarla para tareas de limpieza o registro.

#### `on_audio_stream_start` (callable)
* **Tipo**: Función callable
* **Requerido**: No
* **Descripción**: Esta función de callback opcional se invoca cuando comienza el flujo de audio. Útil para actualizaciones de UI o registro de eventos.

#### `on_audio_stream_stop` (callable)
* **Tipo**: Función callable
* **Requerido**: No
* **Descripción**: Esta función de callback opcional se llama cuando se detiene el flujo de audio. Ideal para limpieza de recursos o tareas de post-procesamiento.

#### `on_character` (callable)
* **Tipo**: Función callable
* **Requerido**: No
* **Descripción**: Esta función de callback opcional se llama cuando se procesa un solo carácter.

#### `output_device_index` (int)
* **Tipo**: Entero
* **Requerido**: No
* **Valor predeterminado**: None
* **Descripción**: Especifica el índice del dispositivo de salida a utilizar. None usa el dispositivo predeterminado.

#### `tokenizer` (string)
* **Tipo**: String
* **Requerido**: No
* **Valor predeterminado**: nltk
* **Descripción**: Tokenizador a utilizar para la división de oraciones (actualmente se admiten "nltk" y "stanza").

#### `language` (string)
* **Tipo**: String
* **Requerido**: No
* **Valor predeterminado**: en
* **Descripción**: Idioma a utilizar para la división de oraciones.

#### `muted` (bool)
* **Tipo**: Bool
* **Requerido**: No
* **Valor predeterminado**: False
* **Descripción**: Parámetro global de silencio. Si es True, no se abrirá ningún flujo pyAudio. Deshabilita la reproducción de audio a través de los altavoces locales.

#### `level` (int)
* **Tipo**: Entero
* **Requerido**: No
* **Valor predeterminado**: `logging.WARNING`
* **Descripción**: Establece el nivel de registro para el registrador interno. Puede ser cualquier constante entera del módulo `logging` incorporado de Python.

### Ejemplo de Uso

```python
engine = YourEngine()  # Sustituya con su motor
stream = TextToAudioStream(
    engine=engine,
    on_text_stream_start=my_text_start_func,
    on_text_stream_stop=my_text_stop_func,
    on_audio_stream_start=my_audio_start_func,
    on_audio_stream_stop=my_audio_stop_func,
    level=logging.INFO
)
```

## Métodos

### `play` y `play_async`

Estos métodos son responsables de ejecutar la síntesis de texto a audio y reproducir el flujo de audio. La diferencia es que `play` es una función bloqueante, mientras que `play_async` se ejecuta en un hilo separado, permitiendo que otras operaciones continúen.

### Parámetros de Reproducción

#### `fast_sentence_fragment` (bool)
* **Valor predeterminado**: `True`
* **Descripción**: Cuando se establece en `True`, el método priorizará la velocidad, generando y reproduciendo fragmentos de oraciones más rápidamente.

#### `fast_sentence_fragment_allsentences` (bool)
* **Valor predeterminado**: `False`
* **Descripción**: Cuando se establece en `True`, aplica el procesamiento rápido de fragmentos de oraciones a todas las oraciones.

#### `fast_sentence_fragment_allsentences_multiple` (bool)
* **Valor predeterminado**: `False`
* **Descripción**: Cuando se establece en `True`, permite generar múltiples fragmentos de oraciones.

#### `buffer_threshold_seconds` (float)
* **Valor predeterminado**: `0.0`
* **Descripción**: Especifica el tiempo en segundos para el umbral de búfer.

  **Cómo funciona**: Antes de sintetizar una nueva oración, el sistema verifica si queda más material de audio en el búfer que el tiempo especificado. Un valor más alto asegura que haya más audio pre-almacenado en el búfer.

#### `minimum_sentence_length` (int)
* **Valor predeterminado**: `10`
* **Descripción**: Establece la longitud mínima de caracteres para considerar una cadena como una oración.

#### `minimum_first_fragment_length` (int)
* **Valor predeterminado**: `10`
* **Descripción**: El número mínimo de caracteres requeridos para el primer fragmento de oración.

#### `log_synthesized_text` (bool)
* **Valor predeterminado**: `False`
* **Descripción**: Cuando está habilitado, registra los fragmentos de texto sintetizados.

#### `reset_generated_text` (bool)
* **Valor predeterminado**: `True`
* **Descripción**: Si es True, reinicia el texto generado antes del procesamiento.

#### `output_wavfile` (str)
* **Valor predeterminado**: `None`
* **Descripción**: Si se establece, guarda el audio en el archivo WAV especificado.

#### Funciones de Callback

#### `on_sentence_synthesized` (callable)
* **Valor predeterminado**: `None`
* **Descripción**: Se llama después de sintetizar un fragmento de oración.

#### `before_sentence_synthesized` (callable)
* **Valor predeterminado**: `None`
* **Descripción**: Se llama antes de sintetizar un fragmento de oración.

#### `on_audio_chunk` (callable)
* **Valor predeterminado**: `None`
* **Descripción**: Se llama cuando un fragmento de audio está listo.

### Configuración de Tokenización

#### `tokenizer` (str)
* **Valor predeterminado**: `"nltk"`
* **Descripción**: Tokenizador para la división de oraciones. Admite "nltk" y "stanza".

#### `tokenize_sentences` (callable)
* **Valor predeterminado**: `None`
* **Descripción**: Función personalizada para tokenizar oraciones del texto de entrada.

#### `language` (str)
* **Valor predeterminado**: `"en"`
* **Descripción**: Idioma para la división de oraciones.

### Parámetros de Contexto

#### `context_size` (int)
* **Valor predeterminado**: `12`
* **Descripción**: Caracteres utilizados para establecer el contexto de límites de oraciones.

#### `context_size_look_overhead` (int)
* **Valor predeterminado**: `12`
* **Descripción**: Tamaño de contexto adicional para mirar hacia adelante.

### Otros Parámetros

#### `muted` (bool)
* **Valor predeterminado**: `False`
* **Descripción**: Deshabilita la reproducción de audio local si es True.

#### `sentence_fragment_delimiters` (str)
* **Valor predeterminado**: `".?!;:,\n…)]}。-"`
* **Descripción**: Caracteres considerados como delimitadores de oraciones.

#### `force_first_fragment_after_words` (int)
* **Valor predeterminado**: `15`
* **Descripción**: Número de palabras después de las cuales se fuerza el primer fragmento.