from .engines import BaseEngine
from .threadsafe_generators import CharIterator, AccumulatingThreadSafeGenerator
from .stream_player import StreamPlayer, AudioConfiguration

import stream2sentence as s2s
from typing import Iterator

import threading
import logging
from typing import Union, Iterator


class TextToAudioStream:

    def __init__(self, 
                 engine: BaseEngine,
                 level=logging.WARNING ,
                 ):
        """Initializes the TextToAudioStream.

        Args:
            engine: The engine used for text to audio synthesis.
            level: Logging level.
        """

        # Initialize the logging configuration with the specified level
        logging.basicConfig(format='RealTimeTTS: %(message)s', level=level)

        # Store the engine instance (responsible for text-to-audio conversion)
        self.engine = engine

        # Extract stream information (format, channels, rate) from the engine
        format, channels, rate = self.engine.get_stream_info()

        # Create a CharIterator instance for managing individual characters
        self.char_iter = CharIterator()

        # Create a thread-safe version of the char iterator
        self.thread_safe_char_iter = AccumulatingThreadSafeGenerator(self.char_iter)

        # Check if the engine doesn't support consuming generators directly
        if not self.engine.can_consume_generators:
            
            # Initialize a StreamPlayer for managing audio playback if the engine can't handle generators directly
            self.player = StreamPlayer(self.engine.queue, AudioConfiguration(format, channels, rate))
        
        # Initialize the play_thread attribute (used for playing audio in a separate thread)
        self.play_thread = None

        # Initialize an attribute to store generated text
        self.generated_text = ""

        # A flag to indicate if the audio stream is currently running or not
        self.stream_running = False

    def feed(self, 
             text_or_iterator: Union[str, Iterator[str]]):
        """Feeds text or an iterator to the stream.

        Args:
            text_or_iterator: Text or iterator to be fed.

        Returns:
            Self instance.
        """
        self.char_iter.add(text_or_iterator)
        return self        

    def play_async(self,   
                   fast_sentence_fragment: bool = False,
                   buffer_threshold_seconds: float = 2.0,
                   minimum_sentence_length: int = 3, 
                   log_characters: bool = False,
                   log_synthesized_text = False             
                   ):
        """
        Handles the synthesis of text to audio.
        Initiates the audio stream playback in a separate thread, allowing the main thread to continue without waiting for the playback to finish.
        If the engine can't consume generators, it utilizes a player.

        Args:
        - fast_sentence_fragment: Determines if sentence fragments should be quickly yielded. Useful when a faster response is desired even if a sentence isn't complete.
        - buffer_threshold_seconds: Time in seconds to determine the buffering threshold. Helps to decide when to generate more audio based on buffered content.
        - minimum_sentence_length: Minimum characters required to treat content as a sentence.
        - log_characters: If True, logs the characters processed for synthesis.
        - log_synthesized_text: If True, logs the synthesized text chunks.
        """
        
        self.stream_running = True
        self.play_thread = threading.Thread(target=self.play, args=(fast_sentence_fragment, buffer_threshold_seconds, minimum_sentence_length, log_characters, log_synthesized_text))
        self.play_thread.start()        

    def play(self,
            fast_sentence_fragment: bool = False,
            buffer_threshold_seconds: float = 2.0,
            minimum_sentence_length: int = 3,
            log_characters: bool = False,
            log_synthesized_text = False):
        """
        Handles the synthesis of text to audio.
        Plays the audio stream and waits until it is finished playing.
        If the engine can't consume generators, it utilizes a player.

        Args:
        - fast_sentence_fragment: Determines if sentence fragments should be quickly yielded. Useful when a faster response is desired even if a sentence isn't complete.
        - buffer_threshold_seconds: Time in seconds to determine the buffering threshold. Helps to decide when to generate more audio based on buffered content.
        - minimum_sentence_length: Minimum characters required to treat content as a sentence.
        - log_characters: If True, logs the characters processed for synthesis.
        - log_synthesized_text: If True, logs the synthesized text chunks.
        """

        # Log the start of the stream
        logging.info(f"stream start")

        # Set the stream_running flag to indicate the stream is active
        self.stream_running = True

        # Initialize the generated_text variable
        self.generated_text = ""

        # Check if the engine can handle generators directly
        if self.engine.can_consume_generators:

            try:
                # Directly synthesize audio using the character iterator
                self.engine.synthesize(self.char_iter)

            finally:
                # Once done, set the stream running flag to False and log the stream stop
                self.stream_running = False
                logging.info("stream stop")

                # Accumulate the generated text and reset the character iterators
                self.generated_text = self.thread_safe_char_iter.accumulated_text()

                self.char_iter = CharIterator()
                self.thread_safe_char_iter = AccumulatingThreadSafeGenerator(self.char_iter)
        else:
            try:
                # Start the audio player to handle playback
                self.player.start()

                # Generate sentences from the characters
                generate_sentences = s2s.generate_sentences(self.thread_safe_char_iter, minimum_sentence_length=minimum_sentence_length, quick_yield_single_sentence_fragment=fast_sentence_fragment, cleanup_text_links=True, cleanup_text_emojis=True, log_characters=log_characters)
                
                # Create the synthesis chunk generator with the given sentences
                chunk_generator = self.synthesis_chunk_generator(generate_sentences, buffer_threshold_seconds, log_synthesized_text)

                # Iterate through the synthesized chunks and feed them to the engine for audio synthesis
                for sentence in chunk_generator:

                    sentence = sentence.strip()
                    if log_synthesized_text:
                        logging.info(f"synthesizing: {sentence}")

                    self.engine.synthesize(sentence)

                    # Break out of the loop if the stream is no longer running
                    if not self.stream_running:
                        break
            finally:
                # Once done, stop the player, set the stream running flag to False, and log the stream stop
                self.player.stop()
                self.stream_running = False
                logging.info("stream stop")

                # Accumulate the generated text and reset the character iterators
                self.generated_text = self.thread_safe_char_iter.accumulated_text()

                self.char_iter = CharIterator()
                self.thread_safe_char_iter = AccumulatingThreadSafeGenerator(self.char_iter)

    def pause(self):
        """Pauses playback of the synthesized audio stream (won't work properly with elevenlabs)."""
        if self.is_playing():
            logging.info("stream pause")
            if self.engine.can_consume_generators:
                self.engine.pause()
            else:
                self.player.pause()

    def resume(self):
        """Resumes a previously paused playback of the synthesized audio stream (won't work properly with elevenlabs)."""
        if self.is_playing():
            logging.info("stream resume")
            if self.engine.can_consume_generators:
                self.engine.resume()
            else:
                self.player.resume()

    def stop(self):
        """Stops the playback of the synthesized audio stream immediately."""
        if self.is_playing():
            self.char_iter.stop()
            if self.engine.can_consume_generators:
                if self.engine.stop():
                    self.stream_running = False
            else:
                self.player.stop(immediate=True)
                self.stream_running = False
    
    def text(self):
        """Retrieves the text that has been fed into the stream.
        Returns:
            The accumulated text.
        """        
        if self.generated_text:
            return self.generated_text
        return self.thread_safe_char_iter.accumulated_text()

    def is_playing(self):
        """Checks if the stream is currently playing.
        Returns:
            Boolean indicating if the stream is playing.
        """ 
        return self.stream_running

    def synthesis_chunk_generator(self,
                                  generator: Iterator[str],
                                  buffer_threshold_seconds: float = 2.0,
                                  log_synthesis_chunks: bool = False) -> Iterator[str]:
        """
        Generates synthesis chunks based on buffered audio length.

        The function buffers chunks of synthesis until the buffered audio seconds fall below the provided threshold. 
        Once the threshold is crossed, the buffered synthesis chunk is yielded.

        Args:
            generator: Input iterator that provides chunks for synthesis.
            buffer_threshold_seconds: Time in seconds to specify how long audio data should be buffered 
                                    before yielding the synthesis chunk.
            log_synthesis_chunks: Boolean flag that, if set to True, logs the synthesis chunks to the logging system.

        Returns:
            Iterator of synthesis chunks.
        """     

        # Initializes an empty string to accumulate chunks of synthesis
        synthesis_chunk = ""
        
        # Iterates over each chunk from the provided generator
        for chunk in generator:

            # Fetch the total seconds of buffered audio
            buffered_audio_seconds = self.player.get_buffered_seconds()
            
            # Append the current chunk (and a space) to the accumulated synthesis_chunk
            synthesis_chunk += chunk + " "
            
            # Check if the buffered audio is below the specified threshold
            if buffered_audio_seconds < buffer_threshold_seconds:
                # If the log_synthesis_chunks flag is True, log the current synthesis_chunk
                if log_synthesis_chunks:
                    logging.info(f"-- \"{synthesis_chunk}\"")
                
                # Yield the current synthesis_chunk and reset it for the next set of accumulations
                yield synthesis_chunk
                synthesis_chunk = ""

        # After iterating over all chunks, check if there's any remaining data in synthesis_chunk
        if synthesis_chunk:
            # If the log_synthesis_chunks flag is True, log the remaining synthesis_chunk
            if log_synthesis_chunks:
                logging.info(f"-- \"{synthesis_chunk}\"")
            
            # Yield the remaining synthesis_chunk
            yield synthesis_chunk