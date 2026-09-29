"""
The window process: a resizable pygame window, driven by window.py.

Started by window.Window as

    python -m tmwt.ui.window_server <shared_memory_name> <width> <height> <title>

and never imported by the labeler itself. It must not import OpenCV: OpenCV's
wheel bundles its own SDL2 (through FFmpeg), and loading two different SDL2
builds in one macOS process makes their Objective-C classes collide ("Class
SDLWindow is implemented in both ..."), which can crash. Running pygame in its
own process avoids that, and keeps the window responsive (resizing, repainting)
while the labeler is busy.

The window shows a fixed-size canvas scaled to fit however large the user makes
it, centred with black bars where the aspect ratios differ. Mouse positions are
converted back through that scaling, so window.py only ever sees canvas pixels.

Protocol, one line per message:
  stdin (from window.py):
    show <slot>    a new canvas is in shared-memory slot 0 or 1
    close          quit
    post <json>    test hook: inject a mouse or resize event, given in canvas
                   pixels, as if the user did it (see _test_event)
  stdout (to window.py), JSON:
    {"key": code}                                  key press, cv2.waitKey-style code
    {"mouse": "down" | "up" | "move", "x": x, "y": y}   left button / pointer, canvas pixels
    {"wheel": dy, "x": x, "y": y}                  scroll (positive = up / back), pointer position

Closing the window sends {"key": 27} (Esc, which every screen treats as cancel)
and exits.
"""

import json
import os
import queue
import sys
import threading
from multiprocessing import resource_tracker, shared_memory

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np  # noqa: E402
import pygame  # noqa: E402

from tmwt.ui.window import CLOSE_KEY, KEY_LEFT, KEY_RIGHT  # noqa: E402  (constants only; no OpenCV)

# Key codes sent for special keys; printable keys send their character code.
_SPECIAL_KEYS = {
    pygame.K_RETURN: 13,
    pygame.K_KP_ENTER: 13,
    pygame.K_ESCAPE: 27,
    pygame.K_BACKSPACE: 8,
    pygame.K_DELETE: 127,
    pygame.K_SPACE: 32,
    pygame.K_TAB: 9,
    pygame.K_LEFT: KEY_LEFT,
    pygame.K_RIGHT: KEY_RIGHT,
}

# Events that mean the window needs repainting at its (possibly new) size.
_REPAINT_EVENTS = {pygame.VIDEORESIZE, pygame.VIDEOEXPOSE,
                   pygame.WINDOWSIZECHANGED, pygame.WINDOWEXPOSED}

# Event-loop period.
_LOOP_MS = 5


class _Display:
    """The pygame window: shows a canvas scaled to fit, maps positions back."""

    def __init__(self, title, canvas_w, canvas_h):
        pygame.display.init()
        pygame.display.set_caption(title)
        self.screen = pygame.display.set_mode((canvas_w, canvas_h), pygame.RESIZABLE)
        self._frame = None            # last canvas, as a surface (for repaints)
        self.scale = 1.0              # canvas -> window scale
        self.offset = (0, 0)          # window position of the canvas's top-left corner
        self.pointer = (0, 0)         # last pointer position, window pixels

    def set_canvas(self, canvas_bgr):
        """Display a BGR canvas."""
        rgb = np.ascontiguousarray(canvas_bgr[:, :, ::-1])
        h, w = rgb.shape[:2]
        self._frame = pygame.image.frombuffer(rgb.tobytes(), (w, h), "RGB")
        self.present()

    def present(self):
        """Draw the last canvas scaled to fit the current window size."""
        if self._frame is None:
            return
        win_w, win_h = self.screen.get_size()
        cw, ch = self._frame.get_size()
        s = min(win_w / cw, win_h / ch)
        dw, dh = max(1, int(cw * s)), max(1, int(ch * s))
        self.scale = s
        self.offset = ((win_w - dw) // 2, (win_h - dh) // 2)
        frame = self._frame if (dw, dh) == (cw, ch) else pygame.transform.smoothscale(self._frame, (dw, dh))
        self.screen.fill((0, 0, 0))
        self.screen.blit(frame, self.offset)
        pygame.display.flip()

    def to_canvas(self, pos):
        """Window pixel position -> canvas pixel position."""
        ox, oy = self.offset
        return (round((pos[0] - ox) / self.scale), round((pos[1] - oy) / self.scale))

    def to_window(self, pos):
        """Canvas pixel position -> window pixel position."""
        ox, oy = self.offset
        return (int(pos[0] * self.scale + ox), int(pos[1] * self.scale + oy))

    def message_for(self, event):
        """The message to send window.py for a pygame event, or None."""
        if event.type in _REPAINT_EVENTS:
            self.present()
        elif event.type == pygame.KEYDOWN:
            if event.key in _SPECIAL_KEYS:
                return {"key": _SPECIAL_KEYS[event.key]}
            if len(event.unicode) == 1 and ord(event.unicode) < 256:
                return {"key": ord(event.unicode)}
        elif event.type == pygame.MOUSEMOTION:
            return self._mouse("move", event.pos)
        elif event.type == pygame.MOUSEWHEEL:
            dy = getattr(event, "precise_y", event.y)   # fractional on trackpads
            x, y = self.to_canvas(self.pointer)
            return {"wheel": dy, "x": x, "y": y}
        elif event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP) and event.button == 1:
            return self._mouse("down" if event.type == pygame.MOUSEBUTTONDOWN else "up", event.pos)
        return None

    def _mouse(self, kind, pos):
        self.pointer = pos
        x, y = self.to_canvas(pos)
        return {"mouse": kind, "x": x, "y": y}


def _test_event(spec, display):
    """
    Test hook for the "post" command: a resize, or a mouse event at a canvas
    position (converted to window pixels, so the real mapping back is exercised).
    """
    if spec["type"] == "quit":
        return pygame.event.Event(pygame.QUIT)
    if spec["type"] == "wheel":
        display.pointer = display.to_window((spec["x"], spec["y"]))
        return pygame.event.Event(pygame.MOUSEWHEEL, x=0, y=int(spec["dy"]),
                                  precise_x=0.0, precise_y=float(spec["dy"]), flipped=False)
    if spec["type"] == "resize":
        display.screen = pygame.display.set_mode((spec["w"], spec["h"]), pygame.RESIZABLE)
        return pygame.event.Event(pygame.VIDEORESIZE, size=(spec["w"], spec["h"]),
                                  w=spec["w"], h=spec["h"])
    pos = display.to_window((spec["x"], spec["y"]))
    if spec["type"] == "move":
        return pygame.event.Event(pygame.MOUSEMOTION, pos=pos, rel=(0, 0), buttons=(0, 0, 0))
    kind = pygame.MOUSEBUTTONDOWN if spec["type"] == "down" else pygame.MOUSEBUTTONUP
    return pygame.event.Event(kind, pos=pos, button=1)


def _read_commands(commands):
    """Background thread: queue stdin lines; "close" if window.py goes away."""
    for line in sys.stdin:
        commands.put(line.strip())
    commands.put("close")


def _send(message):
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def main(argv):
    shm_name, w, h, title = argv[1], int(argv[2]), int(argv[3]), argv[4]
    shm = shared_memory.SharedMemory(name=shm_name)
    # window.py owns the shared memory; stop this process's resource tracker
    # from deleting it when we exit.
    resource_tracker.unregister(shm._name, "shared_memory")
    slots = np.ndarray((2, h, w, 3), dtype=np.uint8, buffer=shm.buf)
    display = _Display(title, w, h)

    commands = queue.Queue()
    threading.Thread(target=_read_commands, args=(commands,), daemon=True).start()
    try:
        while True:
            latest = None
            while not commands.empty():
                cmd = commands.get_nowait()
                if cmd == "close":
                    return
                if cmd.startswith("show "):
                    latest = int(cmd.split()[1])   # only the newest frame matters
                elif cmd.startswith("post "):
                    pygame.event.post(_test_event(json.loads(cmd[5:]), display))
            if latest is not None:
                display.set_canvas(slots[latest])

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    _send({"key": CLOSE_KEY})
                    return
                message = display.message_for(event)
                if message is not None:
                    _send(message)
            pygame.time.wait(_LOOP_MS)
    finally:
        del slots
        shm.close()
        pygame.display.quit()


if __name__ == "__main__":
    main(sys.argv)
