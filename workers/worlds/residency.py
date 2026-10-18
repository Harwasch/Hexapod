"""At most one idle inference process, reusable only after an acknowledged reset."""
import os
import threading


class WarmResidentPool:
    def __init__(self, idle_seconds=None):
        self.idle_seconds = int(os.environ.get("WORLD_WARM_MODEL_SECONDS", "180")) if idle_seconds is None else idle_seconds
        if not 0 <= self.idle_seconds <= 900:
            raise ValueError("WORLD_WARM_MODEL_SECONDS must be between 0 and 900")
        self.lock = threading.Lock()
        self.warm = None
        self.timer = None
        self.closed = False

    def acquire(self, factory):
        with self.lock:
            if self.closed:
                raise RuntimeError("Adapter is shutting down")
            if self.timer:
                self.timer.cancel()
                self.timer = None
            resident, self.warm = self.warm, None
        if resident:
            if resident.process.poll() is None and not resident.closed.is_set():
                return resident
            resident.close()
        return factory()

    def release(self, resident, clean):
        if not clean or not self.idle_seconds or resident.closed.is_set() or resident.process.poll() is not None:
            resident.close()
            return
        try:
            resident.reset()
        except Exception:
            resident.close()
            return
        with self.lock:
            if self.closed or self.warm is not None:
                resident.close()
                return
            self.warm = resident
            self.timer = threading.Timer(self.idle_seconds, self.expire, args=(resident,))
            self.timer.daemon = True
            self.timer.start()

    def expire(self, resident):
        with self.lock:
            if self.warm is not resident:
                return
            self.warm = None
            self.timer = None
            # Do not let a new session load weights while the old idle process
            # still owns VRAM during bounded termination.
            resident.close()

    def close(self):
        with self.lock:
            self.closed = True
            if self.timer:
                self.timer.cancel()
                self.timer = None
            resident, self.warm = self.warm, None
        if resident:
            resident.close()
