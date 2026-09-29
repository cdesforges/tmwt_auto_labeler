"""
The labeler's on-screen window, using pygame.

OpenCV's own windows (cv2.imshow) aren't used here. On macOS, OpenCV's Cocoa
backend works out mouse positions from the global cursor position through a
deprecated conversion, so clicks land in the wrong place, and its window
geometry queries are unreliable. pygame (SDL) reports mouse positions relative
to the window and supports resizing properly.

The window shows a fixed-size canvas (a BGR numpy image) scaled to fit however
large the user makes the window, centred with black bars where the aspect
ratios differ. Mouse positions are converted back through that scaling, so
callers always work in canvas pixels and never see the window's real size.
"""

import os
import time
from collections import deque

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np  # noqa: E402
import pygame  # noqa: E402

# Key codes are returned in cv2.waitKey style: the character code for printable
# keys, these codes for special keys, and KEY_NONE when nothing was pressed.
KEY_NONE = 255
_SPECIAL_KEYS = {
    pygame.K_RETURN: 13,
    pygame.K_KP_ENTER: 13,
    pygame.K_ESCAPE: 27,
    pygame.K_BACKSPACE: 8,
    pygame.K_DELETE: 127,
    pygame.K_SPACE: 32,
    pygame.K_TAB: 9,
}
# Closing the window with its close button is reported as this key (Esc), which
# every screen treats as cancel / quit.
CLOSE_KEY = 27

# Events that mean the window needs repainting at its (possibly new) size.
_REPAINT_EVENTS = {pygame.VIDEORESIZE, pygame.VIDEOEXPOSE,
                   pygame.WINDOWSIZECHANGED, pygame.WINDOWEXPOSED}


class Window:
    """A resizable window showing a canvas, with input in canvas coordinates."""

    def __init__(self, title, canvas_w, canvas_h):
        pygame.display.init()
        pygame.display.set_caption(title)
        self._screen = pygame.display.set_mode((canvas_w, canvas_h), pygame.RESIZABLE)
        self._frame = None            # last canvas shown, as a surface (for repaints)
        self._scale = 1.0             # canvas -> window scale
        self._offset = (0, 0)         # window position of the canvas's top-left corner
        self.mouse_pos = None         # latest pointer position, canvas pixels
        self.mouse_events = deque()   # ("down" | "up", (x, y)) in canvas pixels

    def show(self, canvas_bgr):
        """Display a BGR canvas, scaled to fit the window."""
        rgb = np.ascontiguousarray(canvas_bgr[:, :, ::-1])
        h, w = rgb.shape[:2]
        self._frame = pygame.image.frombuffer(rgb.tobytes(), (w, h), "RGB")
        self._present()

    def poll(self, wait_ms):
        """
        Handle window events for up to wait_ms (like cv2.waitKey). Mouse presses,
        releases and movement are recorded on mouse_events / mouse_pos.

        Returns:
            The first key pressed (cv2.waitKey-style code), or KEY_NONE.
        """
        deadline = time.perf_counter() + max(wait_ms, 0) / 1000.0
        while True:
            for event in pygame.event.get():
                key = self._handle(event)
                if key is not None:
                    return key
            remaining_ms = (deadline - time.perf_counter()) * 1000.0
            if remaining_ms <= 0:
                return KEY_NONE
            pygame.time.wait(int(min(remaining_ms, 5)))

    def close(self):
        pygame.display.quit()

    # --- Internals -------------------------------------------------------------

    def _present(self):
        """Draw the last canvas scaled to fit the current window size."""
        if self._frame is None:
            return
        win_w, win_h = self._screen.get_size()
        cw, ch = self._frame.get_size()
        s = min(win_w / cw, win_h / ch)
        dw, dh = max(1, int(cw * s)), max(1, int(ch * s))
        self._scale = s
        self._offset = ((win_w - dw) // 2, (win_h - dh) // 2)
        frame = self._frame if (dw, dh) == (cw, ch) else pygame.transform.smoothscale(self._frame, (dw, dh))
        self._screen.fill((0, 0, 0))
        self._screen.blit(frame, self._offset)
        pygame.display.flip()

    def _to_canvas(self, pos):
        """Window pixel position -> canvas pixel position."""
        ox, oy = self._offset
        return (round((pos[0] - ox) / self._scale), round((pos[1] - oy) / self._scale))

    def _handle(self, event):
        """Process one pygame event; returns a key code if it was a key press."""
        if event.type == pygame.QUIT:
            return CLOSE_KEY
        if event.type in _REPAINT_EVENTS:
            self._present()
        elif event.type == pygame.MOUSEMOTION:
            self.mouse_pos = self._to_canvas(event.pos)
        elif event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP) and event.button == 1:
            self.mouse_pos = self._to_canvas(event.pos)
            kind = "down" if event.type == pygame.MOUSEBUTTONDOWN else "up"
            self.mouse_events.append((kind, self.mouse_pos))
        elif event.type == pygame.KEYDOWN:
            if event.key in _SPECIAL_KEYS:
                return _SPECIAL_KEYS[event.key]
            if len(event.unicode) == 1 and ord(event.unicode) < 256:
                return ord(event.unicode)
        return None
