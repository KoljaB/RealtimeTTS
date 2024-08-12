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

# import queue
# import threading
# import uuid


# class BufferStream:
#     """
#     A class for buffering and streaming data items.

#     Attributes:
#         items (queue.Queue): A thread-safe queue for storing data items.
#         _stop_event (threading.Event): An event to signal stopping the generator.
#     """
#     def __init__(self):
#         self.items = queue.Queue()
#         self._stop_event = threading.Event()
#         self.stopped = False
#         self.stream_id = str(uuid.uuid4())

#     def add(self, item: str) -> None:
#         """Add an item to the buffer."""
#         self.items.put(item)

#     def stop(self) -> None:
#         """Signal to stop the buffer stream."""
#         self._stop_event.set()

#     def snapshot(self) -> list:
#         """Take a snapshot of all items in the buffer without exhausting it.

#         Returns:
#             list: A list of all items currently in the buffer.
#         """
#         all_items = []
#         temp_storage = []

#         # Temporarily dequeue items to snapshot them.
#         while not self.items.empty():
#             item = self.items.get_nowait()
#             all_items.append(item)
#             temp_storage.append(item)

#         # Re-queue the items.
#         for item in temp_storage:
#             self.items.put(item)

#         return all_items

#     def gen(self):
#         """
#         Generate items from the buffer, yielding them one at a time.

#         Continues yielding items until the buffer is empty and stop has been signaled.
#         """
#         while not self._stop_event.is_set() or not self.items.empty():
#             try:
#                 items = self.items.get(timeout=0.1)
#                 yield items
#             except queue.Empty:
#                 continue

#         self.stopped = True

# import queue
# import threading
# import uuid


# class BufferStream:
#     """
#     A class for buffering and streaming data items.

#     Attributes:
#         items (queue.Queue): A thread-safe queue for storing data items.
#         _stop_event (threading.Event): An event to signal stopping the generator.
#     """
#     def __init__(self):
#         self.items = queue.Queue()
#         self._stop_event = threading.Event()
#         self.stopped = False
#         self.stream_id = str(uuid.uuid4())
#         print(f"*** creating new bufferstream {self.stream_id}")

#     def add(self, item: str) -> None:
#         """Add an item to the buffer."""
#         print(f"*** add to bufferstream {self.stream_id}")
#         self.items.put(item)

#     def stop(self) -> None:
#         """Signal to stop the buffer stream."""
#         print("*** setting stop event")
#         self._stop_event.set()
#         print(f"*** status of self._stop_event is now: {self._stop_event.is_set()}")         

#     def snapshot(self) -> list:
#         """Take a snapshot of all items in the buffer without exhausting it.

#         Returns:
#             list: A list of all items currently in the buffer.
#         """
#         all_items = []
#         temp_storage = []

#         # Temporarily dequeue items to snapshot them.
#         while not self.items.empty():
#             item = self.items.get_nowait()
#             all_items.append(item)
#             temp_storage.append(item)

#         # Re-queue the items.
#         for item in temp_storage:
#             self.items.put(item)

#         return all_items

#     def gen(self):
#         """
#         Generate items from the buffer, yielding them one at a time.

#         Continues yielding items until the buffer is empty and stop has been signaled.
#         """
#         print(f"*** gen for bufferstream {self.stream_id}")
#         while not self._stop_event.is_set() or not self.items.empty():
#             print(f"*** queue empty: {self.items.empty()}, status of self._stop_event is now: {self._stop_event.is_set()}") 
#             try:
#                 items = self.items.get(timeout=0.1)
#                 print(".", end="", flush=True)
#                 if items:
#                     print(f"Items: [{items}]")
#                 yield items
#             except queue.Empty:
                
#                 #print(f"*** queue empty, status of self._stop_event is now: {self._stop_event.is_set()}") 
#                 # if not self._stop_event.is_set():
#                 #     print(f"*** status of self._stop_event is now: {self._stop_event.is_set()}") 
#                 #     print("X", end="", flush=True)
#                 # else:
#                 #     print("#", end="", flush=True)
#                 continue

#         self.stopped = True
#         print("*** buffer exhausted")


