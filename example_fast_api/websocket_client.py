"""
WebSocket client for testing the TTS WebSocket endpoint.
Connects to the server, sends text, and plays received audio.
"""

import asyncio
import websockets
import json
import base64
import pyaudio
import wave
import io
import sys


class TTSWebSocketClient:
    def __init__(self, uri="ws://localhost:8000/ws"):
        self.uri = uri
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.wav_header = None
        self.sample_rate = None
        self.audio_buffer = []
        
    async def connect_and_test(self, text_messages):
        """Connect to WebSocket and send text messages"""
        try:
            async with websockets.connect(self.uri) as websocket:
                print(f"Connected to {self.uri}")
                
                # Create tasks for sending and receiving
                receive_task = asyncio.create_task(self.receive_audio(websocket))
                send_task = asyncio.create_task(self.send_messages(websocket, text_messages))
                
                # Wait for both tasks to complete
                await asyncio.gather(send_task, receive_task)
                
        except ConnectionRefusedError:
            print(f"Error: Could not connect to {self.uri}")
            print("Make sure the server is running (python async_server.py)")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.cleanup()
    
    async def send_messages(self, websocket, text_messages):
        """Send text messages to the server"""
        try:
            for text in text_messages:
                print(f"\n📤 Sending: '{text}'")
                await websocket.send(text)
                # Wait a bit between messages to allow processing
                await asyncio.sleep(0.5)
            
            # Send stop signal
            print("\n📤 Sending stop signal")
            await websocket.send("stop")
            
        except Exception as e:
            print(f"Error sending messages: {e}")
    
    async def receive_audio(self, websocket):
        """Receive and play audio from the server"""
        try:
            message_count = 0
            current_audio_chunks = []
            
            while True:
                try:
                    # Receive message with timeout
                    message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                    message_count += 1
                    
                    data = json.loads(message)
                    
                    # Handle audio output
                    if "audioOutput" in data:
                        audio_data = base64.b64decode(data["audioOutput"]["audio"])
                        
                        # If this is the very first chunk of a new sentence, it contains a WAV header.
                        if len(current_audio_chunks) == 0 and audio_data.startswith(b'RIFF'):
                            try:
                                # Dynamically extract sample rate from the header
                                with wave.open(io.BytesIO(audio_data), 'rb') as wf:
                                    self.sample_rate = wf.getframerate()
                                
                                # Calculate header size correctly
                                data_idx = audio_data.find(b'data')
                                if data_idx != -1:
                                    header_size = data_idx + 8
                                    audio_data = audio_data[header_size:]
                                else:
                                    # Fallback if chunk was cut off before 'data' marker
                                    audio_data = audio_data[44:] 
                                    
                            except Exception as e:
                                print(f"⚠️ Error parsing WAV header, using fallback: {e}")
                                # Fallback to 44 bytes if wave.open fails on a streaming header
                                audio_data = audio_data[44:]

                        # Initialize stream on first chunk AFTER sample rate is dynamically set
                        if self.stream is None:
                            self.setup_audio_stream()

                        # Audio chunk
                        current_audio_chunks.append(audio_data)
                        if self.stream:
                            self.stream.write(audio_data)
                    
                    # Handle completion signal
                    if "finalOutput" in data:
                        if data["finalOutput"].get("isFinal", False):
                            print(f"✅ Audio complete ({len(current_audio_chunks)} chunks received)")
                            self.audio_buffer.extend(current_audio_chunks)
                            current_audio_chunks = []

                except asyncio.TimeoutError:
                    print("⏱️  Timeout waiting for audio")
                    break
                except websockets.exceptions.ConnectionClosed:
                    print("🔌 Connection closed by server")
                    break
                    
        except Exception as e:
            print(f"Error receiving audio: {e}")
    
    def setup_audio_stream(self):
        """Setup PyAudio stream for playback"""
        try:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            
            self.stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                output=True
            )
            print(f"🔊 Audio stream ready (sample rate: {self.sample_rate}Hz)")
        except Exception as e:
            print(f"Error setting up audio stream: {e}")

    def save_audio_to_file(self, filename="output.wav"):
        """Save received audio to a WAV file"""
        if not self.audio_buffer:
            print("No audio data to save")
            return

        # Fallback sample rate if somehow not set
        if not self.sample_rate:
            self.sample_rate = 24000

        try:
            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
                # self.audio_buffer now contains pure PCM data
                wav_file.writeframes(b''.join(self.audio_buffer))
            
            print(f"💾 Audio saved to {filename}")
        except Exception as e:
            print(f"Error saving audio: {e}")

    def cleanup(self):
        """Cleanup audio resources"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.audio:
            self.audio.terminate()


async def main():
    # Default test messages
    test_messages = [
        "Hello, this is a test of the WebSocket text to speech endpoint.",
        "The quick brown fox jumps over the lazy dog.",
        "WebSocket connections allow for real-time bidirectional communication."
    ]
    
    # Check if custom messages provided via command line
    if len(sys.argv) > 1:
        test_messages = sys.argv[1:]
    
    print("=" * 60)
    print("TTS WebSocket Client Demo")
    print("=" * 60)
    print(f"Testing with {len(test_messages)} message(s):\n")
    for i, msg in enumerate(test_messages, 1):
        print(f"  {i}. {msg}")
    print("\n" + "=" * 60 + "\n")
    
    client = TTSWebSocketClient()
    await client.connect_and_test(test_messages)
    
    # Optionally save audio to file
    save_option = input("\n💾 Save audio to file? (y/n): ").strip().lower()
    if save_option == 'y':
        filename = input("Enter filename (default: output.wav): ").strip() or "output.wav"
        client.save_audio_to_file(filename)
    
    print("\n✨ Demo complete!")


if __name__ == "__main__":
    print("\n🚀 Starting WebSocket TTS Client...")
    print("📝 Make sure the server is running on http://localhost:8000\n")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
