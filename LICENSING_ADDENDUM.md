**ADDITIONAL CLARIFICATION ON SCOPE AND THIRD-PARTY COMPONENTS:**

The "Software" licensed under the Apache License 2.0 (see [LICENSE](LICENSE)
and [NOTICE](NOTICE)) refers specifically to the source code of the
RealtimeTTS library itself, authored by Kolja Beigel and the RealtimeTTS
contributors. Releases up to and including 0.8.10 were published under the MIT
License.

This library (the "Software") is designed to interface with and utilize various
third-party text-to-speech (TTS) engines, models and services ("Third-Party
Engines"). These Third-Party Engines are **not** part of the "Software"
licensed under Apache-2.0, are **not** distributed with it (unless explicitly
stated otherwise for a specific bundled component, which would be under its own
license), and are governed by their **own respective licenses and terms of
service**. This includes their model weights, voices, voice latents, reference
audio, datasets and hosted APIs.

The Apache-2.0 license granted for the "Software" **does not** extend to, nor
provide any rights for, the use of these Third-Party Engines, and it does not
release you from checking and complying with their terms. Users of the
"Software" are solely responsible for:
1.  Understanding the license terms and any usage restrictions (including those
    related to commercial use, attribution or share-alike) of any Third-Party
    Engine they choose to use via this "Software".
2.  Obtaining any necessary licenses or permissions directly from the providers
    of such Third-Party Engines.

For a summary of known Third-Party Engine licenses and links to their terms,
please consult the project's README file and the engine pages under
`docs/engines/`. That information is for guidance only and does not constitute
legal advice; users should always refer to the official licensing documentation
of each Third-Party Engine, for the exact package, model and revision they
ship.
