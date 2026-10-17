import io

import numpy as np
import pytest

from RealtimeTTS.alignment.decoder import MP3Decoder


av = pytest.importorskip("av")


def _mp3_fixture(sample_rate=24_000):
    samples = np.sin(
        np.arange(sample_rate, dtype=np.float32) * 2.0 * np.pi * 440.0 / sample_rate
    )
    output = io.BytesIO()
    container = av.open(output, mode="w", format="mp3")
    stream = container.add_stream("mp3", rate=sample_rate)
    stream.layout = "mono"
    frame = av.AudioFrame.from_ndarray(
        samples.reshape(1, -1), format="fltp", layout="mono"
    )
    frame.sample_rate = sample_rate
    for packet in stream.encode(frame):
        container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return output.getvalue()


def _decode_in_chunks(payload, sizes):
    decoder = MP3Decoder(24_000)
    pcm = []
    cursor = 0
    index = 0
    while cursor < len(payload):
        size = sizes[index % len(sizes)]
        pcm.extend(decoder.feed(payload[cursor:cursor + size]))
        cursor += size
        index += 1
    pcm.extend(decoder.finish())
    return np.frombuffer(b"".join(pcm), dtype="<f4")


def test_mp3_decoder_handles_split_id3_header_and_arbitrary_packet_boundaries():
    payload = _mp3_fixture()
    assert payload.startswith(b"ID3")

    whole = _decode_in_chunks(payload, [len(payload)])
    split = _decode_in_chunks(payload, [1])
    irregular = _decode_in_chunks(payload, [2, 17, 3, 41, 5, 97])

    assert len(whole) > 12_000
    assert np.max(np.abs(whole)) > 0.01
    np.testing.assert_allclose(split, whole, rtol=0, atol=1e-6)
    np.testing.assert_allclose(irregular, whole, rtol=0, atol=1e-6)


def test_mp3_decoder_rejects_audio_after_finish():
    decoder = MP3Decoder(24_000)
    list(decoder.finish())
    with pytest.raises(RuntimeError, match="finished"):
        list(decoder.feed(b""))
