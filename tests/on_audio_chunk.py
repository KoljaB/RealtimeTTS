import wave
import numpy as np
import torch

# Accumulators for all chunks
all_bytes = bytearray()  # Raw byte data
all_numpy_chunks = []    # List of NumPy arrays
all_tensor_chunks = []   # List of PyTorch tensors

def process_chunk(chunk_bytes):
    """
    Processes each chunk and accumulates the data for writing later.
    """
    # Convert bytes to NumPy array
    audio_data_numpy = np.frombuffer(chunk_bytes, dtype=np.int16)

    # Convert NumPy array to PyTorch tensor
    audio_pytorch_tensor = torch.from_numpy(audio_data_numpy)

    # Accumulate data for later writing
    all_bytes.extend(chunk_bytes)  # Add raw bytes
    all_numpy_chunks.append(audio_data_numpy)  # Add NumPy array
    all_tensor_chunks.append(audio_pytorch_tensor)  # Add PyTorch tensor

if __name__ == "__main__":
    from RealtimeTTS import TextToAudioStream, CoquiEngine

    # Initialize the engine and stream
    engine = CoquiEngine()
    stream = TextToAudioStream(engine, muted=True)

    # Feed the text and play the audio
    stream.feed("Hello World")
    stream.play(on_audio_chunk=process_chunk, muted=True)

    # Retrieve audio parameters
    format, channels, sample_rate = engine.get_stream_info()

    # Write the accumulated raw byte data to a wave file
    with wave.open("audio_data_from_bytes.wav", 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # int16 => 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(all_bytes)

    # Write the accumulated NumPy data to a wave file
    combined_numpy = np.concatenate(all_numpy_chunks)  # Combine all NumPy arrays
    with wave.open("audio_data_from_numpy.wav", 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(combined_numpy.tobytes())

    # Write the accumulated PyTorch tensor data to a wave file
    combined_tensor = torch.cat(all_tensor_chunks)  # Combine all tensors
    with wave.open("audio_data_from_tensor.wav", 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(combined_tensor.numpy().tobytes())

    engine.shutdown()

