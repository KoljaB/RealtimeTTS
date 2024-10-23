## Configuration

### Paramètres d'initialisation pour `TextToAudioStream

Lorsque vous initialisez la classe `TextToAudioStream`, vous disposez de diverses options pour personnaliser son comportement. Voici les paramètres disponibles :

###`(BaseEngine)
- **Type**: BaseEngine
- **Obligatoire**: Oui
- **Description** : Le moteur sous-jacent responsable de la synthèse texte-audio. Vous devez fournir une instance de `ine` ou sa sous-classe pour permettre la synthèse audio.

####`_text_stream_start` (appelable)
- **Type**: Fonction appelable
- **Obligatoire**: Non
- **Description** : Cette fonction de rappel optionnelle est déclenchée lorsque le flux de texte commence. Utilisez-le pour toute configuration ou journalisation dont vous pourriez avoir besoin.

####`_text_stream_stop` (appelable)
- **Type**: Fonction appelable
- **Obligatoire**: Non
- **Description** : Cette fonction de rappel optionnelle est activée à la fin du flux de texte. Vous pouvez l'utiliser pour des tâches de nettoyage ou de journalisation.

###` `_audio_stream_start` (appelable)
- **Type**: Fonction appelable
- **Obligatoire**: Non
- **Description** : Cette fonction de rappel facultative est invoquée au démarrage du flux audio. Utile pour les mises à jour de l'interface utilisateur ou la journalisation des événements.

####`_audio_stream_stop` (appelable)
- **Type**: Fonction appelable
- **Obligatoire**: Non
- **Description** : Cette fonction de rappel optionnelle est appelée lorsque le flux audio s'arrête. Idéal pour les tâches de nettoyage des ressources ou de post-traitement.

###` `on_character` (appelable)
- **Type**: Fonction appelable
- **Obligatoire**: Non
- **Description** : Cette fonction de rappel optionnelle est appelée lorsqu'un seul caractère est traité.

####`_device_index` (int)
- **Type**: Entier
- **Obligatoire**: Non
- **Par défaut**: Aucun
- **Description** : Spécifie l'index du périphérique de sortie à utiliser. Aucun n'utilise le périphérique par défaut.

###`(tokenizer`(chaîne)
- **Type**: Chaîne
- **Obligatoire**: Non
- **Par défaut**: nltk
- **Description** : Tokenizer à utiliser pour le fractionnement des phrases (actuellement « nltk » et « stroza » sont pris en charge).

###`language(chaîne)
- **Type**: Chaîne
- **Obligatoire**: Non
- **Par défaut**: fr
- **Description** : Langue à utiliser pour le fractionnement des phrases.

###`muted`(bool)
- **Type**: Bool
- **Obligatoire**: Non
- **Par défaut**: Faux
- **Description** : Paramètre global coupé. Si True, aucun flux pyAudio ne sera ouvert. Désactive la lecture audio via des haut-parleurs locaux (au cas où vous souhaitez synthétiser dans un fichier ou traiter des morceaux audio) et remplace le paramètre de mise en sourdine des paramètres de lecture.

###`level` (int)
- **Type**: Entier
- **Obligatoire**: Non
- **Défaut**:`logging.AVERTISSEMENT`
- **Description** : Définit le niveau de journalisation pour l'enregistreur interne. Cela peut être n'importe quelle constante entière du module `ging` intégré de Python.

#### Exemple d'utilisation :

````(`python
moteur = YourEngine () # Remplacez-vous par votre moteur
flux = TextToAudioStream(
    moteur=engine,
    on_text_stream_start=my_text_start_func,
    on_text_stream_stop=my_text_stop_func,
    on_audio_stream_start=my_audio_start_func,
    on_audio_stream_stop=my_audio_stop_func,
    niveau=logging.INFO
)
``
### Méthodes

###`play et`play_async`

Ces méthodes sont responsables de l'exécution de la synthèse texte-audio et de la lecture du flux audio. La différence est que `play` est une fonction de blocage, tandis que `play_async` s'exécute dans un thread séparé, permettant à d'autres opérations de se poursuivre.

##### Paramètres :

###### fast`_sentence_fragment` (bool)
- **Par défaut**: `True`
- **Description** : Lorsqu'elle est définie sur `True`, la méthode donnera la priorité à la vitesse, générant et jouant plus rapidement des fragments de phrases. Ceci est utile pour les applications où la latence est importante.

###### fast`_sentence_fragment_allsentences`(bool)
- **Par défaut**: `False`
- **Description** : Lorsqu'il est défini sur `True`, applique le traitement rapide des fragments de phrase à toutes les phrases, pas seulement à la première.

###### fast`_sentence_fragment_allsentences_multiple` (bool)
- **Par défaut**: `False`
- **Description** : Lorsqu'il est défini sur `True`, permet de produire plusieurs fragments de phrase au lieu d'un seul.

###### `_threshold_seconds` (flotteur)
- **Par défaut**: `0.0`
- **Description** : Spécifie le temps en secondes pour le seuil de mise en mémoire tampon, ce qui a un impact sur la douceur et la continuité de la lecture audio.

  - **Comment ça marche** : Avant de synthétiser une nouvelle phrase, le système vérifie s'il reste plus de matériel audio dans le tampon que le temps spécifié par `buffer_threshold_seconds`. Si tel est le cas, il récupère une autre phrase du générateur de texte, en supposant qu'il peut récupérer et synthétiser cette nouvelle phrase dans la fenêtre temporelle fournie par l'audio restant dans le tampon. Ce processus permet au moteur de synthèse vocale d'avoir plus de contexte pour une meilleure synthèse, améliorant ainsi l'expérience utilisateur.

  Une valeur plus élevée garantit qu'il y a plus d'audio pré-tamponné, réduisant ainsi le risque de silence ou de lacunes pendant la lecture. Si vous rencontrez des pauses ou des pauses, envisagez d'augmenter cette valeur.

###### `_sentence_length` (int)
- **Par défaut**: `10`
- **Description** : Définit la longueur minimale des caractères pour considérer une chaîne comme une phrase à synthétiser. Cela affecte la façon dont les morceaux de texte sont traités et lus.

###### `_first_fragment_length`(int)
- **Par défaut**: `10`
- **Description** : Le nombre minimum de caractères requis pour le premier fragment de phrase avant de céder.

###### `_synthesized_text` (bool)
- **Par défaut**: `False`
- **Description** : Lorsqu'il est activé, enregistre les morceaux de texte au fur et à mesure de leur synthèse en audio. Utile pour l'audit et le débogage.

##### #reset_generated_text` (bool)
- **Par défaut**: `True`
- **Description** : Si Vrai, réinitialisez le texte généré avant le traitement.

###### `_wavfile` (str)
- **Par défaut**: `None`
- **Description** : Si défini, enregistrez l'audio dans le fichier WAV spécifié.

###### `_sentence_synthesized (appelable)
- **Par défaut**: `None`
- **Description** : Une fonction de rappel appelée après un seul fragment de phrase a été synthétisée.

###### before`_sentence_synthesized (appelable)
- **Par défaut**: `None`
- **Description** : Une fonction de rappel qui est appelée avant qu'un seul fragment de phrase ne soit synthétisé.

###### `_audio_chunk` (appelable)
- **Par défaut**: `None`
- **Description** : Fonction de rappel qui est appelée lorsqu'un seul morceau audio est prêt.

###### ```(str)
- **Par défaut**:`"nltk"`
- **Description** : Tokenizer à utiliser pour le fractionnement des phrases. Prend actuellement en charge « nltk » et « stroza ».

###### `_sentences` (appelable)
- **Par défaut**: `None`
- **Description** : Une fonction personnalisée qui tokenise les phrases du texte saisi. Vous pouvez fournir votre propre tokenizer léger si vous n'êtes pas satisfait de nltk et stanza. Il doit prendre du texte comme chaîne et renvoyer des phrases divisées comme liste de chaînes.

###### `angu`(str)
- **Par défaut**:`"en"`
- **Description** : Langue à utiliser pour le fractionnement des phrases.

###### `_size`(int)
- **Par défaut**: `12`
- **Description** : Le nombre de caractères utilisés pour établir le contexte pour la détection des limites de phrase. Un contexte plus large améliore la précision de la détection des limites des phrases.

###### `_size_look_overhead` (int)
- **Par défaut**: `12`
- **Description** : Taille de contexte supplémentaire pour regarder vers l'avenir lors de la détection des limites des phrases.

###### `mute` (bool)
- **Par défaut**: `False`
- **Description** : Si vrai, désactive la lecture audio via des haut-parleurs locaux. Utile lorsque vous souhaitez synthétiser dans un fichier ou traiter des morceaux audio sans les lire.

###### `ence_fragment_delimiters` (str)
- **Par défaut**:`"?!;::\n...)]}-`
- **Description** : Une chaîne de caractères qui sont considérés comme des délimiteurs de phrases.

###### `_first_fragment_after_`words (int)
- **Par défaut**: `15`
- **Description** : Le nombre de mots après lesquels le fragment de la première phrase est forcé d'être donné.