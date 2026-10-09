# Pronunciation exchanges

The compatible RealtimeTTS HTTP speech endpoint and WebSocket `config` accept
two optional dictionaries. Both default to empty, so other clients are unchanged.

```json
{
  "voice": "mira",
  "language": "Auto",
  "exchange": {"API": "A P I"},
  "exchange_by_language": {
    "en": {"miep": "meep", "Miep": "Meep"},
    "de": {"API": "Ah Peh Ih"}
  }
}
```

Add `"input": "Hello, Miep!"` for `POST /v1/audio/speech`, or `"type": "config"`
for `/v1/audio/speech-stream`.

Matching is case-sensitive and literal, with Unicode word boundaries:
letters, digits, combining accents and underscores prevent a match on either side. Whitespace,
punctuation and text edges permit one. Thus `Miep,`, `miep!` and `Miep's`
match, while `smiep`, `mieps`, `miep2` and `miep_test` do not. Regex syntax
in keys has no special meaning. Keys and values must be non-empty strings.

Global entries apply to every language. The selected language's entries
override identical global keys. Language keys accept supported names and
aliases such as `English`, `en` and `en-US`; duplicate aliases are rejected.
`auto` is not a language-specific key.

The existing language decision uses the original text. Exchanges run once
after that decision, immediately before synthesis. Auto with language-specific
entries requires the server's language router; otherwise the request is rejected. Streaming exchanges use
the server's existing complete fragments, so keys split across incoming
tokens still match without a new buffer or language detector. Multiword
keys cannot cross synthesis-fragment or explicit flush boundaries. Overlaps
use the longest matching key; replacement output is never exchanged again.

The server advertises `features.text_exchange`. When exchanges change a
fragment, its `fragment_start` carries spoken `text`, original `source_text`
and `source_ranges`: one original UTF-16 [start, end) range per spoken Python
character. Alignment events still describe the spoken text. Clients can use
the ranges to highlight the original spelling; length-changing exchanges
distribute source characters proportionally and are approximate within a word.
PCM, completion and trimming timestamps are unchanged.
