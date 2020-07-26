# ebur128rs

This library contains an implementation of EBU R-128 loudness measurement.

It exposes functions to calculate the loudess of a chunk of audio,
plus a higher-level interface that calculates loudness over time
according the restrictions defined in EBU R-128.

TODO:
* Support sample rates other than 48kHz
* Become agnostic to audio chunk size
