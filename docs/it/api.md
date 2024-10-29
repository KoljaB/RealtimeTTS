## Configurazione

### Parametri di Inizializzazione per `TextToAudioStream`

Quando si inizializza la classe `TextToAudioStream`, sono disponibili diverse opzioni per personalizzare il suo comportamento. Ecco i parametri disponibili:

#### `engine` (BaseEngine)
- **Tipo**: BaseEngine
- **Obbligatorio**: Sì
- **Descrizione**: Il motore sottostante responsabile della sintesi da testo ad audio. È necessario fornire un'istanza di `BaseEngine` o della sua sottoclasse per abilitare la sintesi audio.

#### `on_text_stream_start` (callable)
- **Tipo**: Funzione callable
- **Obbligatorio**: No
- **Descrizione**: Questa funzione di callback opzionale viene attivata quando inizia lo stream di testo. Utilizzala per qualsiasi configurazione o registrazione necessaria.

#### `on_text_stream_stop` (callable)
- **Tipo**: Funzione callable
- **Obbligatorio**: No
- **Descrizione**: Questa funzione di callback opzionale viene attivata quando termina lo stream di testo. Puoi utilizzarla per attività di pulizia o registrazione.

#### `on_audio_stream_start` (callable)
- **Tipo**: Funzione callable
- **Obbligatorio**: No
- **Descrizione**: Questa funzione di callback opzionale viene invocata quando inizia lo stream audio. Utile per aggiornamenti dell'interfaccia utente o registrazione eventi.

#### `on_audio_stream_stop` (callable)
- **Tipo**: Funzione callable
- **Obbligatorio**: No
- **Descrizione**: Questa funzione di callback opzionale viene chiamata quando lo stream audio si ferma. Ideale per la pulizia delle risorse o attività di post-elaborazione.

#### `on_character` (callable)
- **Tipo**: Funzione callable
- **Obbligatorio**: No
- **Descrizione**: Questa funzione di callback opzionale viene chiamata quando viene elaborato un singolo carattere.

#### `output_device_index` (int)
- **Tipo**: Intero
- **Obbligatorio**: No
- **Predefinito**: None
- **Descrizione**: Specifica l'indice del dispositivo di output da utilizzare. None usa il dispositivo predefinito.

#### `tokenizer` (string)
- **Tipo**: Stringa
- **Obbligatorio**: No
- **Predefinito**: nltk
- **Descrizione**: Tokenizer da utilizzare per la divisione delle frasi (attualmente sono supportati "nltk" e "stanza").

#### `language` (string)
- **Tipo**: Stringa
- **Obbligatorio**: No
- **Predefinito**: en
- **Descrizione**: Lingua da utilizzare per la divisione delle frasi.

#### `muted` (bool)
- **Tipo**: Bool
- **Obbligatorio**: No
- **Predefinito**: False
- **Descrizione**: Parametro globale di silenziamento. Se True, non verrà aperto alcuno stream pyAudio. Disabilita la riproduzione audio attraverso gli altoparlanti locali (nel caso in cui si desideri sintetizzare su file o elaborare chunk audio) e sovrascrive l'impostazione muted dei parametri di riproduzione.

#### `level` (int)
- **Tipo**: Intero
- **Obbligatorio**: No
- **Predefinito**: `logging.WARNING`
- **Descrizione**: Imposta il livello di logging per il logger interno. Può essere qualsiasi costante intera dal modulo `logging` integrato di Python.

#### Esempio di Utilizzo:

```python
engine = YourEngine()  # Sostituire con il proprio motore
stream = TextToAudioStream(
    engine=engine,
    on_text_stream_start=my_text_start_func,
    on_text_stream_stop=my_text_stop_func,
    on_audio_stream_start=my_audio_start_func,
    on_audio_stream_stop=my_audio_stop_func,
    level=logging.INFO
)
```

### Metodi

#### `play` e `play_async`

Questi metodi sono responsabili dell'esecuzione della sintesi testo-audio e della riproduzione dello stream audio. La differenza è che `play` è una funzione bloccante, mentre `play_async` viene eseguito in un thread separato, permettendo ad altre operazioni di procedere.

##### Parametri:

###### `fast_sentence_fragment` (bool)
- **Predefinito**: `True`
- **Descrizione**: Quando impostato su `True`, il metodo darà priorità alla velocità, generando e riproducendo i frammenti di frase più rapidamente. Questo è utile per applicazioni dove la latenza è importante.

###### `fast_sentence_fragment_allsentences` (bool)
- **Predefinito**: `False`
- **Descrizione**: Quando impostato su `True`, applica l'elaborazione rapida dei frammenti di frase a tutte le frasi, non solo alla prima.

###### `fast_sentence_fragment_allsentences_multiple` (bool)
- **Predefinito**: `False`
- **Descrizione**: Quando impostato su `True`, permette di produrre più frammenti di frase invece di uno solo.

###### `buffer_threshold_seconds` (float)
- **Predefinito**: `0.0`
- **Descrizione**: Specifica il tempo in secondi per la soglia di buffering, che influenza la fluidità e la continuità della riproduzione audio.

  - **Come Funziona**: Prima di sintetizzare una nuova frase, il sistema controlla se nel buffer rimane più materiale audio del tempo specificato da `buffer_threshold_seconds`. In caso affermativo, recupera un'altra frase dal generatore di testo, assumendo che possa recuperare e sintetizzare questa nuova frase entro la finestra temporale fornita dall'audio rimanente nel buffer. Questo processo consente al motore di sintesi vocale di avere più contesto per una migliore sintesi, migliorando l'esperienza utente.

  Un valore più alto assicura che ci sia più audio pre-bufferizzato, riducendo la probabilità di silenzi o interruzioni durante la riproduzione. Se si verificano interruzioni o pause, considera di aumentare questo valore.

###### `minimum_sentence_length` (int)
- **Predefinito**: `10`
- **Descrizione**: Imposta la lunghezza minima in caratteri per considerare una stringa come una frase da sintetizzare. Questo influisce su come vengono elaborati e riprodotti i chunk di testo.

###### `minimum_first_fragment_length` (int)
- **Predefinito**: `10`
- **Descrizione**: Il numero minimo di caratteri richiesti per il primo frammento di frase prima della produzione.

###### `log_synthesized_text` (bool)
- **Predefinito**: `False`
- **Descrizione**: Quando abilitato, registra i chunk di testo mentre vengono sintetizzati in audio. Utile per il controllo e il debugging.

###### `reset_generated_text` (bool)
- **Predefinito**: `True`
- **Descrizione**: Se True, reimposta il testo generato prima dell'elaborazione.

###### `output_wavfile` (str)
- **Predefinito**: `None`
- **Descrizione**: Se impostato, salva l'audio nel file WAV specificato.

###### `on_sentence_synthesized` (callable)
- **Predefinito**: `None`
- **Descrizione**: Una funzione di callback che viene chiamata dopo che un singolo frammento di frase è stato sintetizzato.

###### `before_sentence_synthesized` (callable)
- **Predefinito**: `None`
- **Descrizione**: Una funzione di callback che viene chiamata prima che un singolo frammento di frase venga sintetizzato.

###### `on_audio_chunk` (callable)
- **Predefinito**: `None`
- **Descrizione**: Funzione di callback che viene chiamata quando un singolo chunk audio è pronto.

###### `tokenizer` (str)
- **Predefinito**: `"nltk"`
- **Descrizione**: Tokenizer da utilizzare per la divisione delle frasi. Attualmente supporta "nltk" e "stanza".

###### `tokenize_sentences` (callable)
- **Predefinito**: `None`
- **Descrizione**: Una funzione personalizzata che tokenizza le frasi dal testo di input. Puoi fornire il tuo tokenizer leggero se non sei soddisfatto di nltk e stanza. Dovrebbe prendere il testo come stringa e restituire le frasi divise come lista di stringhe.

###### `language` (str)
- **Predefinito**: `"en"`
- **Descrizione**: Lingua da utilizzare per la divisione delle frasi.

###### `context_size` (int)
- **Predefinito**: `12`
- **Descrizione**: Il numero di caratteri utilizzati per stabilire il contesto per il rilevamento dei confini della frase. Un contesto più ampio migliora la precisione nel rilevare i confini delle frasi.

###### `context_size_look_overhead` (int)
- **Predefinito**: `12`
- **Descrizione**: Dimensione del contesto aggiuntiva per guardare avanti durante il rilevamento dei confini delle frasi.

###### `muted` (bool)
- **Predefinito**: `False`
- **Descrizione**: Se True, disabilita la riproduzione audio attraverso gli altoparlanti locali. Utile quando si desidera sintetizzare su file o elaborare chunk audio senza riprodurli.

###### `sentence_fragment_delimiters` (str)
- **Predefinito**: `".?!;:,\n…)]}。-"`
- **Descrizione**: Una stringa di caratteri che sono considerati delimitatori di frase.

###### `force_first_fragment_after_words` (int)
- **Predefinito**: `15`
- **Descrizione**: Il numero di parole dopo il quale viene forzata la produzione del primo frammento di frase.