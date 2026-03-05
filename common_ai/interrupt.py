import threading
import time

class InterruptController:
    def __init__(self):
        self.stop_event = threading.Event()

    def stop(self):
        self.stop_event.set()

    def clear(self):
        self.stop_event.clear()

    def should_stop(self) -> bool:
        return self.stop_event.is_set()