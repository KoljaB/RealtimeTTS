"""The ASGI thread must not wait for a slow terminal or log file."""
import logging
import threading

import RealtimeTTS.qwen_server as server


def test_slow_log_sink_cannot_block_websocket_accept_logging():
    entered = threading.Event()
    release = threading.Event()
    returned = threading.Event()
    messages = []

    class BlockedSink(logging.Handler):
        def emit(self, record):
            entered.set()
            release.wait(2)
            messages.append(record.getMessage())

    root = logging.getLogger()
    old_handlers = list(root.handlers)
    old_level = root.level
    sink = BlockedSink()
    root.handlers = [sink]
    try:
        with server._queued_server_logging("info"):
            def log_accept():
                logging.getLogger("uvicorn.error").info("websocket accepted")
                returned.set()

            producer = threading.Thread(target=log_accept)
            producer.start()
            try:
                assert entered.wait(1)
                assert returned.wait(0.2), "slow output blocked event-loop caller"
                logging.getLogger("uvicorn.error").warning("warning retained")
            finally:
                release.set()
                producer.join(2)
        assert root.handlers == [sink]
        assert root.level == old_level
        assert messages == ["websocket accepted", "warning retained"]
    finally:
        release.set()
        root.handlers = old_handlers
        root.setLevel(old_level)
        sink.close()
