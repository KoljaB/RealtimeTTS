import queue
import threading
import uuid
from typing import Generator, List, Any


class BufferStream:
    def __init__(self):
        self.items: queue.Queue = queue.Queue()
        self._stop_event: threading.Event = threading.Event()
        self.stopped: bool = False
        self.stream_id: str = str(uuid.uuid4())

    def add(self, item: Any) -> None:
        """Add an item to the buffer."""
        self.items.put(item)

    def stop(self) -> None:
        """Signal to stop the buffer stream."""
        self._stop_event.set()

    def snapshot(self) -> List[Any]:
        """Take a snapshot of all items in the buffer without exhausting it."""
        with self.items.mutex:
            return list(self.items.queue)

    def gen(self) -> Generator[Any, None, None]:
        """Generate items from the buffer, yielding them one at a time."""
        while not self._stop_event.is_set() or not self.items.empty():
            try:
                yield self.items.get(timeout=0.1)
            except queue.Empty:
                continue
        self.stopped = True
