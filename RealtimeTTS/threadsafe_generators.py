# """
# threadsafe_generators.py

# This file contains a collection of classes aimed at providing thread-safe operations over generators 
# and iterables. 

# The utility of this module can be mainly seen in multi-threaded environments where generators or iterables 
# need to be consumed across threads without race conditions. Additionally, functionalities like character-based 
# iteration and accumulation are provided for enhanced flexibility.
# """


import threading
from collections import deque
from typing import Union, Iterator

class CharIterator:
    """
    An iterator that allows iteration over characters of strings or string iterators. 

    This class provides an interface for adding either strings or string iterators. When iterated upon, 
    it will yield characters from the added items. Additionally, it provides functionalities to stop 
    the iteration and accumulate iterated text.

    Attributes:
        items (List[Union[str, Iterator[str]]]): The list of strings or string iterators added to the CharIterator.
        _index (int): The current index in the items list that is being iterated.
        _char_index (int, optional): The current character index in the current string being iterated. None if currently iterating over an iterator.
        _current_iterator (Iterator[str], optional): The current iterator being consumed. None if currently iterating over a string.
        immediate_stop (threading.Event): An event signaling if the iteration should be stopped immediately.
        iterated_text (str): Accumulates the characters that have been iterated over.

    Methods:
        add(item: Union[str, Iterator[str]]): Adds a string or a string iterator to the items list.
        stop(): Stops the iterator immediately on the next iteration.
    """

    def __init__(self, 
                 log_characters: bool = False):
        """
        Initialize the CharIterator instance.

        Args:
        - log_characters: If True, logs the characters processed.
        """
        self.items = []
        self._index = 0
        self._char_index = None
        self._current_iterator = None
        self.immediate_stop = threading.Event()
        self.iterated_text = ""
        self.log_characters = log_characters

    def add(self, item: Union[str, Iterator[str]]) -> None:
        """
        Add a string or a string iterator to the list of items.

        Args:
            item (Union[str, Iterator[str]]): The string or string iterator to add.
        """
        self.items.append(item)

    def stop(self):
        """
        Signal the iterator to stop immediately during the next iteration.
        """
        self.immediate_stop.set()        

    def __iter__(self) -> "CharIterator":
        """
        Returns the iterator object itself.

        Returns:
            CharIterator: The instance of CharIterator.
        """
        return self

    def __next__(self) -> str:
        """
        Fetch the next character from the current string or string iterator in the items list.

        If the current item is a string, it will yield characters from the string until it's exhausted. 
        If the current item is a string iterator, it will yield characters from the iterator until it's exhausted.

        Returns:
            str: The next character.

        Raises:
            StopIteration: If there are no more characters left or the immediate_stop event is set.
        """

        # Check if the stop event has been triggered, if so, end the iteration immediately
        if self.immediate_stop.is_set():
            raise StopIteration
        
        # Continue while there are items left to iterate over
        while self._index < len(self.items):

            # Get the current item (either a string or an iterator)
            item = self.items[self._index]

            # Check if the item is a string
            if isinstance(item, str):

                # If we haven't started iterating over this string yet, initialize the character index
                if self._char_index is None:
                    self._char_index = 0

                # If there are characters left in the string to yield
                if self._char_index < len(item):
                    char = item[self._char_index]
                    self._char_index += 1

                    # Accumulate the iterated character to the iterated_text attribute
                    self.iterated_text += char
                    if self.log_characters:
                        print(char, end="", flush=True)
                    return char
                
                else:
                    # If the string is exhausted, reset the character index and move on to the next item
                    self._char_index = None
                    self._index += 1

            else:
                # The item is a string iterator

                # If we haven't started iterating over this iterator yet, initialize it
                if self._current_iterator is None:
                    self._current_iterator = iter(item)
                
                # Try to fetch the next character from the iterator
                try:
                    char = next(self._current_iterator)

                    # Accumulate the iterated character to the iterated_text attribute
                    self.iterated_text += char
                    if self.log_characters:
                        print(char, end="", flush=True)
                    return char
                
                except StopIteration:
                    
                    # If the iterator is exhausted, reset it and move on to the next item
                    self._current_iterator = None
                    self._index += 1

        # If all items are exhausted, raise the StopIteration exception to signify end of iteration
        raise StopIteration

import threading

class AccumulatingThreadSafeGenerator:
    """
    A thread-safe generator that accumulates the iterated tokens into a text.
    """

    def __init__(self, gen_func, on_first_text_chunk=None, on_last_text_chunk=None):
        """
        Initialize the AccumulatingThreadSafeGenerator instance.

        Args:
            gen_func: The generator function to be used.
            on_first_text_chunk: Callback function to be executed after the first chunk of text is received.
            on_last_text_chunk: Callback function to be executed after the last chunk of text is received.
        """
        self.lock = threading.Lock()
        self.generator = gen_func
        self.exhausted = False
        self.iterated_text = ""
        self.on_first_text_chunk = on_first_text_chunk
        self.on_last_text_chunk = on_last_text_chunk
        self.first_chunk_received = False

    def __iter__(self):
        """
        Returns the iterator object itself.

        Returns:
            AccumulatingThreadSafeGenerator: The instance of AccumulatingThreadSafeGenerator.
        """
        return self

    def __next__(self):
        """
        Fetch the next token from the generator in a thread-safe manner.

        Returns:
            The next item from the generator.

        Raises:
            StopIteration: If there are no more items left in the generator.
        """
        with self.lock:
            try:
                token = next(self.generator) 
                self.iterated_text += str(token)

                if not self.first_chunk_received and self.on_first_text_chunk:
                    self.on_first_text_chunk()
                    self.first_chunk_received = True

                return token

            except StopIteration:
                if self.on_last_text_chunk:
                    self.on_last_text_chunk()
                self.exhausted = True
                raise

    def is_exhausted(self):
        """
        Check if the generator has been exhausted.

        Returns:
            bool: True if the generator is exhausted, False otherwise.
        """
        with self.lock:
            return self.exhausted

    def accumulated_text(self):
        """
        Retrieve the accumulated text from the iterated tokens.

        Returns:
            str: The accumulated text.
        """
        with self.lock:
            return self.iterated_text