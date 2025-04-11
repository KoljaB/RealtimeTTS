"""
Module: threadsafe_generators.py

This module offers tools for safe, thread-friendly iteration over generators and iterables.

Classes:

1. CharIterator:
   - Iterates over characters from strings or string iterators.
   - Logs each character and triggers callbacks for the first and last text chunks.
   - Can be stopped instantly with a threading event.

2. AccumulatingThreadSafeGenerator:
   - Wraps a generator for safe multi-threaded token consumption.
   - Accumulates tokens into a full text.
   - Uses locks to avoid race conditions and supports first/last token callbacks.
"""


from typing import Union, Iterator, Callable, Optional
import threading
from dataclasses import dataclass, field


@dataclass
class CharIterator:
    """
    An iterator that allows iteration over characters of strings or string iterators.
    
    Attributes:
        items (List[Union[str, Iterator[str]]]): The list of strings or string iterators.
        _index (int): Current index in the items list being iterated.
        _char_index (Optional[int]): Current character index in the current string.
        _current_iterator (Optional[Iterator[str]]): Current iterator being consumed.
        immediate_stop (threading.Event): Event signaling to stop iteration.
        iterated_text (str): Accumulates the characters that have been iterated over.
        log_characters (bool): If True, logs processed characters.
        on_character (Callable): Callback on each character processed.
        on_first_text_chunk (Callable): Callback on receiving the first text chunk.
        on_last_text_chunk (Callable): Callback on receiving the last text chunk.
        first_chunk_received (bool): Flag indicating if the first chunk was processed.
    """

    log_characters: bool = False
    on_character: Optional[Callable[[str], None]] = None
    on_first_text_chunk: Optional[Callable[[], None]] = None
    on_last_text_chunk: Optional[Callable[[], None]] = None

    items: list = field(default_factory=list)
    _index: int = 0
    _char_index: Optional[int] = None
    _current_iterator: Optional[Iterator[str]] = None
    immediate_stop: threading.Event = field(default_factory=threading.Event)
    iterated_text: str = ""
    first_chunk_received: bool = False

    def add(self, item: Union[str, Iterator[str]]) -> None:
        """Add a string or a string iterator to the list of items."""
        self.items.append(item)

    def stop(self) -> None:
        """Signal the iterator to stop immediately during the next iteration."""
        self.immediate_stop.set()

    def __iter__(self) -> "CharIterator":
        """Return the iterator object itself."""
        return self

    def _log_and_trigger(self, char: str) -> None:
        """Log character and trigger associated callbacks."""
        self.iterated_text += char
        if self.log_characters:
            print(char, end="", flush=True)
        if self.on_character:
            self.on_character(char)
        if not self.first_chunk_received and self.on_first_text_chunk:
            self.on_first_text_chunk()
            self.first_chunk_received = True

    def __next__(self) -> str:
        """Fetch the next character from the current string or string iterator."""
        if self.immediate_stop.is_set():
            raise StopIteration

        while self._index < len(self.items):
            item = self.items[self._index]

            if isinstance(item, str):
                if self._char_index is None:
                    self._char_index = 0

                if self._char_index < len(item):
                    char = item[self._char_index]
                    self._char_index += 1
                    self._log_and_trigger(char)
                    return char
                else:
                    self._char_index = None
                    self._index += 1

            else:  # item is an iterator
                if self._current_iterator is None:
                    self._current_iterator = iter(item)

                if self._char_index is None:
                    try:
                        self._current_str = next(self._current_iterator)
                        if hasattr(self._current_str, "choices"):
                            self._current_str = str(self._current_str.choices[0].delta.content) or ""
                    except StopIteration:
                        self._char_index = None
                        self._current_iterator = None
                        self._index += 1
                        continue

                    self._char_index = 0

                if self._char_index < len(self._current_str):
                    char = self._current_str[self._char_index]
                    self._char_index += 1
                    self._log_and_trigger(char)
                    return char
                else:
                    self._char_index = None

        if self.iterated_text and self.on_last_text_chunk:
            self.on_last_text_chunk()

        raise StopIteration


class AccumulatingThreadSafeGenerator:
    """
    A thread-safe generator that accumulates the iterated tokens into a text.
    """

    def __init__(self, gen_func: Iterator[str], on_first_text_chunk: Optional[Callable[[], None]] = None, on_last_text_chunk: Optional[Callable[[], None]] = None):
        """
        Initialize the AccumulatingThreadSafeGenerator instance.
        
        Args:
            gen_func (Iterator[str]): The generator function to be used.
            on_first_text_chunk (Optional[Callable]): Callback for the first chunk of text.
            on_last_text_chunk (Optional[Callable]): Callback for the last chunk of text.
        """
        self.lock = threading.Lock()
        self.generator = gen_func
        self.exhausted = False
        self.iterated_text = ""
        self.on_first_text_chunk = on_first_text_chunk
        self.on_last_text_chunk = on_last_text_chunk
        self.first_chunk_received = False

    def __iter__(self) -> "AccumulatingThreadSafeGenerator":
        """Return the iterator object itself."""
        return self

    def __next__(self) -> str:
        """Fetch the next token from the generator in a thread-safe manner."""
        with self.lock:
            try:
                token = next(self.generator)
                self.iterated_text += str(token)

                if not self.first_chunk_received and self.on_first_text_chunk:
                    self.on_first_text_chunk()

                self.first_chunk_received = True
                return token

            except StopIteration:
                if self.iterated_text and self.on_last_text_chunk:
                    self.on_last_text_chunk()
                self.exhausted = True
                raise

    def is_exhausted(self) -> bool:
        """Check if the generator has been exhausted."""
        with self.lock:
            return self.exhausted

    def accumulated_text(self) -> str:
        """Retrieve the accumulated text from the iterated tokens."""
        with self.lock:
            return self.iterated_text
