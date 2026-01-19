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
                        
                        # Check if this is a header
                        if data["audioOutput"].get("isHeader", False):
                            print(f"📥 Received WAV header")
                            self.sample_rate = data["audioOutput"].get("sampleRate", 24000)
                            self.wav_header = audio_data
                            self.setup_audio_stream()
                        else:
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
        if not self.audio_buffer or not self.wav_header:
            print("No audio data to save")
            return
        
        try:
            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(self.sample_rate)
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
