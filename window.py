"""
The labeler's on-screen window, as the rest of the program sees it.

The window itself is a resizable pygame window running in its own process
(window_server.py). OpenCV's own windows aren't used: on macOS their mouse
positions are wrong. pygame can't share a process with OpenCV either: both
bundle different SDL2 builds, whose Objective-C classes collide. So this module
never imports pygame; it starts the window process, hands it each canvas through
shared memory, and receives key presses and mouse events back in canvas pixels.

The interface mirrors cv2.imshow / cv2.waitKey: show() a canvas, then poll()
for up to some milliseconds to get the next key.
"""

import atexit
import json
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from multiprocessing import shared_memory

import numpy as np

# poll() result when no key was pressed (as cv2.waitKey).
KEY_NONE = 255
# Key reported once the window has been closed or its process has gone away
# (Esc: every screen treats it as cancel / quit).
CLOSE_KEY = 27
# Arrow keys, which have no character code; outside the 0-255 range of other keys.
KEY_LEFT = 1001
KEY_RIGHT = 1002

_SERVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "window_server.py")


class Window:
    """A resizable window showing fixed-size BGR canvases."""

    def __init__(self, title, canvas_w, canvas_h):
        shape = (canvas_h, canvas_w, 3)
        # Two canvas slots, so the next frame can be written while the window
        # process is still reading the previous one.
        self._shm = shared_memory.SharedMemory(create=True, size=2 * int(np.prod(shape)))
        self._slots = np.ndarray((2,) + shape, dtype=np.uint8, buffer=self._shm.buf)
        self._next_slot = 0
        self._proc = subprocess.Popen(
            [sys.executable, _SERVER, self._shm.name, str(canvas_w), str(canvas_h), title],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self._messages = queue.Queue()
        threading.Thread(target=self._read_messages, daemon=True).start()
        self._closed = False
        self.mouse_pos = None         # latest pointer position, canvas pixels
        self.mouse_events = deque()   # ("down" | "up", (x, y)) in canvas pixels
        self.wheel_events = deque()   # (dy, (x, y)): scroll amount (+ = up) and pointer position
        atexit.register(self.close)

    def show(self, canvas_bgr):
        """Display a BGR canvas (must be the size given at construction)."""
        slot = self._next_slot
        self._next_slot ^= 1
        self._slots[slot] = canvas_bgr
        self._send(f"show {slot}")

    def poll(self, wait_ms):
        """
        Wait up to wait_ms for input (like cv2.waitKey). Mouse presses, releases,
        movement and scrolling are recorded on mouse_events / mouse_pos /
        wheel_events.

        Returns:
            The first key pressed (cv2.waitKey-style code), CLOSE_KEY if the
            window is gone, or KEY_NONE.
        """
        deadline = time.perf_counter() + max(wait_ms, 0) / 1000.0
        while True:
            if self._proc.poll() is not None and self._messages.empty():
                return CLOSE_KEY
            try:
                remaining = deadline - time.perf_counter()
                message = (self._messages.get(timeout=remaining) if remaining > 0
                           else self._messages.get_nowait())
            except queue.Empty:
                return KEY_NONE
            if "key" in message:
                return message["key"]
            pos = (message["x"], message["y"])
            self.mouse_pos = pos
            if "wheel" in message:
                self.wheel_events.append((message["wheel"], pos))
            elif message["mouse"] in ("down", "up"):
                self.mouse_events.append((message["mouse"], pos))

    def post_test_event(self, spec):
        """Test hook: inject a resize or mouse event (see window_server._test_event)."""
        self._send("post " + json.dumps(spec))

    @property
    def closed(self):
        """True once the window has been closed (its process has exited)."""
        return self._proc.poll() is not None

    def close(self):
        """Close the window and release the shared memory. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        self._send("close")
        try:
            self._proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        del self._slots
        self._shm.close()
        self._shm.unlink()

    def _send(self, line):
        """Send one command; ignored if the window process has already exited."""
        try:
            self._proc.stdin.write(line + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass

    def _read_messages(self):
        """Background thread: queue the window process's messages."""
        for line in self._proc.stdout:
            try:
                self._messages.put(json.loads(line))
            except ValueError:
                continue   # not one of ours (e.g. a library printing to stdout)
