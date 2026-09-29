"""
Drawing building blocks for the labeler's window (labeler_ui.py) and players:
layout sizes, colours, key codes, text and image helpers, buttons (text or
icon), and the seek bar.

Everything is drawn with OpenCV into the fixed-size main-area canvas
(MAIN_W x MAIN_H); window.py scales it to the real window. Buttons are described
by specs, (text, value, keys) or (text, value, keys, icon): the label, the value
returned when chosen, the key codes that also choose it, and an optional icon
(see ICONS) drawn instead of the text.
"""

import cv2
import numpy as np

# Main area (left of the sidebar), in canvas pixels.
MAIN_W, MAIN_H = 960, 720
FONT = cv2.FONT_HERSHEY_SIMPLEX

WHITE = (255, 255, 255)
GREY = (150, 150, 150)
DIM = (90, 90, 90)
YELLOW = (0, 255, 255)
GREEN = (0, 200, 0)
ORANGE = (0, 150, 255)
RED = (60, 60, 255)
BLUE = (255, 0, 0)

# Key codes, as returned by Window.poll (cv2.waitKey style); arrow keys are in
# window.py (KEY_LEFT, KEY_RIGHT).
KEY_ESC = 27
KEY_ENTER = (13, 10)
KEY_BACKSPACE = (8, 127)
KEY_SPACE = ord(" ")

# Layout.
BTN_H = 44
BAR_H = 64            # bottom button bar on image screens
HEADER_H = 84         # instruction strip above the frame when picking endpoints
_BTN_GAP = 16
_BTN_MIN_W = 150


# --- Text and image helpers ----------------------------------------------------

def put_centered(img, text, y, scale, color, thickness=1):
    """Draw `text` horizontally centred on `img` at baseline y."""
    (tw, _), _ = cv2.getTextSize(text, FONT, scale, thickness)
    x = max(10, (img.shape[1] - tw) // 2)
    cv2.putText(img, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def truncate(text, max_w, scale, thickness=1):
    """Shorten `text` with a trailing '...' until it fits in max_w pixels."""
    if cv2.getTextSize(text, FONT, scale, thickness)[0][0] <= max_w:
        return text
    while text and cv2.getTextSize(text + "...", FONT, scale, thickness)[0][0] > max_w:
        text = text[:-1]
    return text + "..."


def fit(img, w, h):
    """
    Scale `img` to fit inside w x h (keeping aspect) and centre it on black.

    Returns:
        (canvas, scale, x0, y0): image pixel (x, y) lands at canvas pixel
        (x * scale + x0, y * scale + y0).
    """
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    if img is None:
        return canvas, 1.0, 0, 0
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * s)), max(1, int(ih * s))
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = cv2.resize(img, (nw, nh), interpolation=interp)
    return canvas, s, x0, y0


def dimmed(img, brightness):
    """`img` fitted to the main area and darkened, as a background for text."""
    return (fit(img, MAIN_W, MAIN_H)[0] * brightness).astype(np.uint8)


def frame_screen(img, top=0, bottom=BAR_H):
    """
    Main-area canvas with `img` fitted between a `top` strip and a `bottom`
    strip (the button bar, plus the seek bar if there is one). Returns
    (canvas, scale, x0, y0) like _fit, in canvas pixels.
    """
    main = np.zeros((MAIN_H, MAIN_W, 3), dtype=np.uint8)
    fitted, s, x0, y0 = fit(img, MAIN_W, MAIN_H - top - bottom)
    main[top:MAIN_H - bottom] = fitted
    cv2.rectangle(main, (0, MAIN_H - bottom), (MAIN_W, MAIN_H), (20, 20, 20), -1)
    return main, s, x0, y0 + top


# --- Seek bar ------------------------------------------------------------------

# Seek bar (show_frame's `seek`): its strip height, track ends and hit margin.
SEEK_H = 34
_SEEK_X0, _SEEK_X1 = 24, MAIN_W - 150
_SEEK_Y = MAIN_H - BAR_H - SEEK_H // 2
_SEEK_GRAB = 12     # how far above / below the track a press still grabs it


def seek_fraction(x):
    """Position along the seek bar (0-1) for canvas x."""
    return min(1.0, max(0.0, (x - _SEEK_X0) / (_SEEK_X1 - _SEEK_X0)))


def on_seek_bar(pt):
    return (pt is not None and _SEEK_X0 - _SEEK_GRAB <= pt[0] <= _SEEK_X1 + _SEEK_GRAB
            and abs(pt[1] - _SEEK_Y) <= _SEEK_GRAB)


def draw_seek_bar(img, fraction, markers, text, active):
    """
    The seek bar: track, played part, markers [(fraction, colour)], a handle at
    `fraction` (bigger while hovered or dragged) and `text` at its right.
    """
    y = _SEEK_Y
    x = int(_SEEK_X0 + fraction * (_SEEK_X1 - _SEEK_X0))
    cv2.line(img, (_SEEK_X0, y), (_SEEK_X1, y), (70, 70, 70), 4, cv2.LINE_AA)
    cv2.line(img, (_SEEK_X0, y), (x, y), (200, 200, 200), 4, cv2.LINE_AA)
    for f, color in markers:
        mx = int(_SEEK_X0 + f * (_SEEK_X1 - _SEEK_X0))
        cv2.line(img, (mx, y - 8), (mx, y + 8), color, 2, cv2.LINE_AA)
    cv2.circle(img, (x, y), 9 if active else 7, WHITE, -1, cv2.LINE_AA)
    cv2.putText(img, text, (_SEEK_X1 + 18, y + 5), FONT, 0.5, GREY, 1, cv2.LINE_AA)


# --- Buttons -------------------------------------------------------------------

class Button:
    """
    A clickable button drawn with OpenCV, behaving like a standard UI button:
    it highlights on hover, looks pushed in (inset shadow, label shifted) while
    held, and only counts as clicked if the mouse is released over it. Dragging
    off before releasing cancels the click; dragging back on re-arms it.
    """

    # Fill / border colours per visual state.
    _STYLES = {
        "normal": ((48, 48, 48), (95, 95, 95)),
        "hover": ((66, 66, 66), (170, 170, 170)),
        "pressed": ((36, 36, 36), YELLOW),
    }
    # Inset shadow lines along the top and left edges when pressed, outermost first.
    _SHADOW = ((0, 0, 0), (6, 6, 6), (12, 12, 12), (18, 18, 18), (24, 24, 24), (30, 30, 30))

    def __init__(self, rect, text, value, keys=(), key_label="", icon=None):
        self.x, self.y, self.w, self.h = rect
        self.text = text
        self.value = value          # returned when chosen
        self.keys = keys            # key codes that also choose it
        self.key_label = key_label  # shortcut shown at the left; text is centred if empty
        self.icon = icon            # draw this icon (see ICONS) instead of the text

    def contains(self, pt):
        return (pt is not None and self.x <= pt[0] < self.x + self.w
                and self.y <= pt[1] < self.y + self.h)

    def draw(self, img, state):
        """Draw in `state`: "normal", "hover" or "pressed"."""
        fill, border = self._STYLES[state]
        x0, y0, x1, y1 = self.x, self.y, self.x + self.w - 1, self.y + self.h - 1
        cv2.rectangle(img, (x0, y0), (x1, y1), fill, -1)
        shift = 0
        if state == "pressed":
            for k, shade in enumerate(self._SHADOW):
                cv2.line(img, (x0 + k, y0 + k), (x1, y0 + k), shade, 1)
                cv2.line(img, (x0 + k, y0 + k), (x0 + k, y1), shade, 1)
            shift = 2
        cv2.rectangle(img, (x0, y0), (x1, y1), border, 1)

        if self.icon:
            draw_icon(img, self.icon, (x0 + self.w // 2 + shift, y0 + self.h // 2 + shift))
            return
        base_y = y0 + self.h // 2 + 7 + shift
        if self.key_label:
            cv2.putText(img, self.key_label, (x0 + 16 + shift, base_y), FONT, 0.6, YELLOW, 2, cv2.LINE_AA)
            text_x = x0 + 80
        else:
            (tw, _), _ = cv2.getTextSize(self.text, FONT, 0.6, 1)
            text_x = x0 + (self.w - tw) // 2
        cv2.putText(img, self.text, (text_x + shift, base_y), FONT, 0.6, WHITE, 1, cv2.LINE_AA)


def button_row(specs, y):
    """
    Buttons for `specs`, side by side and centred at height y. A spec is
    (text, value, keys), or (text, value, keys, icon) for an icon button (see
    ICONS), which is drawn as that icon at a fixed width.
    """
    buttons = []
    for spec in specs:
        text, value, keys = spec[:3]
        icon = spec[3] if len(spec) > 3 else None
        w = _ICON_BTN_W if icon else max(_BTN_MIN_W, cv2.getTextSize(text, FONT, 0.6, 1)[0][0] + 48)
        buttons.append(Button((0, y, w, BTN_H), text, value, keys, icon=icon))
    x = (MAIN_W - sum(b.w for b in buttons) - _BTN_GAP * (len(buttons) - 1)) // 2
    for b in buttons:
        b.x = x
        x += b.w + _BTN_GAP
    return buttons


# Icon buttons: media-player symbols drawn with shapes (the font has none).
ICONS = ("play", "pause", "prev_frame", "next_frame", "prev_video", "next_video")
_ICON_BTN_W = 72


def draw_icon(img, icon, center, size=11, color=WHITE):
    """Draw one of ICONS centred at `center`, about 2*size pixels tall."""
    cx, cy = center
    s = size

    def triangle(tip_x, facing):
        # A filled triangle whose tip is at tip_x, pointing right (+1) or left (-1).
        base_x = tip_x - facing * int(1.6 * s)
        pts = np.array([(tip_x, cy), (base_x, cy - s), (base_x, cy + s)], np.int32)
        cv2.fillPoly(img, [pts], color, cv2.LINE_AA)

    def bar(x):
        cv2.rectangle(img, (x - 2, cy - s), (x + 1, cy + s), color, -1)

    if icon == "play":
        triangle(cx + int(0.8 * s), +1)
    elif icon == "pause":
        cv2.rectangle(img, (cx - s + 2, cy - s), (cx - 3, cy + s), color, -1)
        cv2.rectangle(img, (cx + 3, cy - s), (cx + s - 2, cy + s), color, -1)
    elif icon == "next_frame":                      # ▶|
        triangle(cx + 4, +1)
        bar(cx + 8)
    elif icon == "prev_frame":                      # |◀
        triangle(cx - 4, -1)
        bar(cx - 8)
    elif icon == "next_video":                      # ▶▶|
        triangle(cx - 2, +1)
        triangle(cx + 14, +1)
        bar(cx + 18)
    elif icon == "prev_video":                      # |◀◀
        triangle(cx + 2, -1)
        triangle(cx - 14, -1)
        bar(cx - 18)


def bar_buttons(specs):
    """Button row in the bottom bar of an image screen."""
    return button_row(specs, MAIN_H - BAR_H + (BAR_H - BTN_H) // 2)
