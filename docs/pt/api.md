## Configuração

### Parâmetros de Inicialização para `TextToAudioStream`

Quando você inicializa a classe `TextToAudioStream`, você tem várias opções para personalizar seu comportamento. Aqui estão os parâmetros disponíveis:

#### `engine` (BaseEngine)
- **Tipo**: BaseEngine
- **Obrigatório**: Sim
- **Descrição**: O mecanismo subjacente responsável pela síntese de texto para áudio. Você deve fornecer uma instância de `BaseEngine` ou sua subclasse para habilitar a síntese de áudio.

#### `on_text_stream_start` (callable)
- **Tipo**: Função chamável
- **Obrigatório**: Não
- **Descrição**: Esta função de callback opcional é acionada quando o fluxo de texto começa. Use-o para qualquer configuração ou registro que você possa precisar.

#### `on_text_stream_stop` (callable)
- **Tipo**: Função chamável
- **Obrigatório**: Não
- **Descrição**: Esta função de retorno de chamada opcional é ativada quando o fluxo de texto termina. Você pode usar isso para tarefas de limpeza ou registro.

#### `on_audio_stream_start` (callable)
- **Tipo**: Função chamável
- **Obrigatório**: Não
- **Descrição**: Esta função de callback opcional é invocada quando o fluxo de áudio começa. Útil para atualizações de UI ou registro de eventos.

#### `on_audio_stream_stop` (callable)
- **Tipo**: Função chamável
- **Obrigatório**: Não
- **Descrição**: Esta função de callback opcional é chamada quando o fluxo de áudio para. Ideal para limpeza de recursos ou tarefas de pós-processamento.

#### `on_character` (callable)
- **Tipo**: Função chamável Não
- **Descrição**: Esta função de retorno de chamada opcional é chamada quando um único caractere é processado.

#### `output_device_index` (int)
- **Tipo**: Inteiro
- **Obrigatório**: Não
- **Padrão**: Nenhum
- **Descrição**: Especifica o índice do dispositivo de saída a ser usado. Nenhum usa o dispositivo padrão.

#### `tokenizer` (string)
- **Tipo**: String
- **Obrigatório**: Não
- **Padrão**: nltk
- **Descrição**: Tokenizador a ser usado para divisão de sentenças (currently "nltk" and "stanza" are supported).

#### `language` (string)
- **Tipo**: String
- **Obrigatório**: Não
- **Padrão**: en
- **Descrição**: Idioma a usar para a divisão de frases.

#### `muted` (bool)
- **Tipo**: Bool
- **Obrigatório**: Não
- **Padrão**: Falso
- **Descrição**: Parâmetro global de mudo. Se True, nenhum fluxo pyAudio será aberto. Desativa a reprodução de áudio pelos alto-falantes locais (caso você queira sintetizar para um arquivo ou processar pedaços de áudio) e substitui a configuração de silenciamento dos parâmetros de reprodução.

#### `level` (int)
- **Tipo**: Inteiro
- **Requerido**: Não
- **Padrão**: `logging.WARNING`
- **Descrição**: Define o nível de registro para o logger interno. Isso pode ser qualquer constante inteira do módulo `logging` embutido do Python.

#### Exemplo de Uso:

```python
engine = YourEngine()  # Substitua pelo seu motor
stream = TextToAudioStream(
    engine=engine,
    on_text_stream_start=my_text_start_func,
    on_text_stream_stop=my_text_stop_func,
    on_audio_stream_start=my_audio_start_func,
    on_audio_stream_stop=my_audio_stop_func,
    level=logging.INFO
)
```

### Métodos

#### `play` e `play_async`

Esses métodos são responsáveis por executar a síntese de texto para áudio e reproduzir o fluxo de áudio. A diferença é que `play` é uma função bloqueante, enquanto `play_async` é executada em uma thread separada, permitindo que outras operações prossigam.

##### Parâmetros:

###### `fast_sentence_fragment` (bool)
- **Padrão**: `True`
- **Descrição**: Quando definido como `True`, o método priorizará a velocidade, gerando e reproduzindo fragmentos de frases mais rapidamente. Isso é útil para aplicações onde a latência é importante.

###### `fast_sentence_fragment_allsentences` (bool)
- **Padrão**: `False`
- **Descrição**: Quando definido como `True`, aplica o processamento rápido de fragmentos de sentenças a todas as sentenças, não apenas à primeira.

###### `fast_sentence_fragment_allsentences_multiple` (bool)
- **Padrão**: `False`
- **Descrição**: Quando definido como `True`, permite gerar múltiplos fragmentos de frase em vez de apenas um único.

###### `buffer_threshold_seconds` (float)
- **Padrão**: `0.0`
- **Descrição**: Especifica o tempo em segundos para o limite de buffer, o que impacta a suavidade e a continuidade da reprodução de áudio.

  - **Como Funciona**: Antes de sintetizar uma nova frase, o sistema verifica se há mais material de áudio restante no buffer do que o tempo especificado por `buffer_threshold_seconds`. Se for o caso, ele recupera outra frase do gerador de texto, assumindo que pode buscar e sintetizar essa nova frase dentro da janela de tempo fornecida pelo áudio restante no buffer. Este processo permite que o mecanismo de texto-para-fala tenha mais contexto para uma melhor síntese, melhorando a experiência do usuário.

  Um valor mais alto garante que haja mais áudio pré-bufferizado, reduzindo a probabilidade de silêncio ou lacunas durante a reprodução. Se você experimentar quebras ou pausas, considere aumentar este valor.

###### `minimum_sentence_length` (int)
- **Padrão**: `10`
- **Descrição**: Define o comprimento mínimo de caracteres para considerar uma string como uma frase a ser sintetizada. Isso afeta como os trechos de texto são processados e reproduzidos.

###### `minimum_first_fragment_length` (int)
- **Padrão**: `10`
- **Descrição**: O número mínimo de caracteres exigido para o primeiro fragmento de frase antes de ceder.

###### `log_synthesized_text` (bool)
- **Padrão**: `False`
- **Descrição**: Quando ativado, registra os trechos de texto à medida que são sintetizados em áudio. Útil para auditoria e depuração.

###### `reset_generated_text` (bool)
- **Padrão**: `True`
- **Descrição**: Se verdadeiro, redefina o texto gerado antes de processar.

###### `output_wavfile` (str)
- **Padrão**: `None`
- **Descrição**: Se definido, salve o áudio no arquivo WAV especificado.

###### `on_sentence_synthesized` (callable)
- **Padrão**: `Nenhum`
- **Descrição**: Uma função de callback que é chamada após um único fragmento de sentença ser sintetizado.

###### `before_sentence_synthesized` (callable)
- **Padrão**: `Nenhum`
- **Descrição**: Uma função de callback que é chamada antes que um único fragmento de frase seja sintetizado.

###### `on_audio_chunk` (callable)
- **Padrão**: `Nenhum`
- **Descrição**: Função de callback que é chamada quando um único pedaço de áudio está pronto.

###### `tokenizer` (str)
- **Padrão**: `"nltk"`
- **Descrição**: Tokenizer a ser usado para a divisão de sentenças. Atualmente suporta "nltk" e "stanza".

###### `tokenize_sentences` (callable)
- **Padrão**: `None`
- **Descrição**: Uma função personalizada que tokeniza sentenças a partir do texto de entrada. Você pode fornecer seu próprio tokenizador leve se não estiver satisfeito com o nltk e o stanza. Deve receber o texto como uma string e retornar as sentenças divididas como uma lista de strings.

###### `language` (str)
- **Padrão**: `"pt"`
- **Descrição**: Idioma a ser usado para a divisão de frases.

###### `context_size` (int)
- **Padrão**: `12`
- **Descrição**: O número de caracteres usados para estabelecer o contexto para a detecção de limites de sentenças. Um contexto maior melhora a precisão na detecção de limites de sentenças.

###### `context_size_look_overhead` (int)
- **Padrão**: `12`
- **Descrição**: Tamanho adicional do contexto para olhar à frente ao detectar limites de sentenças.

###### `muted` (bool)
- **Padrão**: `Falso`
- **Descrição**: Se verdadeiro, desativa a reprodução de áudio pelos alto-falantes locais. Útil quando você quer sintetizar para um arquivo ou processar trechos de áudio sem reproduzi-los.

###### `sentence_fragment_delimiters` (str)
- **Padrão**: `".?!;:,\n…)]}。-"`
- **Descrição**: Uma sequência de caracteres que são considerados delimitadores de frases.

###### `force_first_fragment_after_words` (int)
- **Padrão**: `15`
- **Descrição**: O número de palavras após o qual o primeiro fragmento de frase é forçado a ser gerado.


