"""
Drawing building blocks for the labeler's window (labeler_ui.py) and players:
layout sizes, colours, key codes, text and image helpers, buttons, icons,
and tooltips. (The seek bar is in seek_bar.py.)

Everything is drawn with OpenCV into the fixed-size main-area canvas
(MAIN_W x MAIN_H); window.py scales it to the real window. Buttons are described
by specs, (text, value, keys) or (text, value, keys, icon): the label, the value
returned when chosen, the key codes that also choose it, and an optional icon
(see ICONS) drawn instead of the text.

Buttons share one base class (Button: hit-testing and the normal / hover /
pressed look) and differ only in what they draw on top: TextButton (centred
label), KeyedButton (shortcut at the left, then the label) and IconButton (an
icon; its text is shown as a tooltip).
"""

import unicodedata

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


def ascii_text(text):
    """
    `text` in characters OpenCV's font can draw: accents are dropped
    (e.g. "é" -> "e") and anything else unprintable becomes "?".
    """
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return "".join(c if " " <= c <= "~" else "?" for c in text)


def truncate_middle(text, max_w, scale, thickness=1):
    """
    Shorten `text` by replacing its middle with '...' until it fits in max_w
    pixels, keeping the start and the end (where names usually differ, e.g.
    "control_vids_cluster_version_test2" -> "control_vi...ersion_test2").
    Returns "" if not even '...' fits.
    """
    def width(t):
        return cv2.getTextSize(t, FONT, scale, thickness)[0][0]

    if width(text) <= max_w:
        return text
    if width("...") > max_w:
        return ""
    # Keep the longest total of characters that fits, split evenly with the
    # extra one at the start.
    lo, hi = 0, len(text) - 1
    while lo < hi:
        keep = (lo + hi + 1) // 2
        head, tail = (keep + 1) // 2, keep // 2
        if width(text[:head] + "..." + text[len(text) - tail:]) <= max_w:
            lo = keep
        else:
            hi = keep - 1
    head, tail = (lo + 1) // 2, lo // 2
    return text[:head] + "..." + text[len(text) - tail:]


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


# --- Buttons -------------------------------------------------------------------

class Button:
    """
    Base class for clickable buttons drawn with OpenCV, behaving like a standard
    UI button: it highlights on hover, looks pushed in (inset shadow, content
    shifted) while held, and only counts as clicked if the mouse is released
    over it. Dragging off before releasing cancels the click; dragging back on
    re-arms it.

    Subclasses draw their content in _draw_content.
    """

    # Fill / border colours per visual state.
    _STYLES = {
        "normal": ((48, 48, 48), (95, 95, 95)),
        "hover": ((66, 66, 66), (170, 170, 170)),
        "pressed": ((36, 36, 36), YELLOW),
    }
    # Inset shadow lines along the top and left edges when pressed, outermost first.
    _SHADOW = ((0, 0, 0), (6, 6, 6), (12, 12, 12), (18, 18, 18), (24, 24, 24), (30, 30, 30))

    def __init__(self, rect, text, value, keys=()):
        self.x, self.y, self.w, self.h = rect
        self.text = text
        self.value = value          # returned when chosen
        self.keys = keys            # key codes that also choose it

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
        self._draw_content(img, shift)

    def _draw_content(self, img, shift):
        """Draw what's on the button, moved down and right by `shift` while pressed."""
        raise NotImplementedError

    def _baseline(self, shift):
        return self.y + self.h // 2 + 7 + shift


class TextButton(Button):
    """A button with its label centred."""

    def _draw_content(self, img, shift):
        (tw, _), _ = cv2.getTextSize(self.text, FONT, 0.6, 1)
        x = self.x + (self.w - tw) // 2 + shift
        cv2.putText(img, self.text, (x, self._baseline(shift)), FONT, 0.6, WHITE, 1, cv2.LINE_AA)


class KeyedButton(Button):
    """A button with its keyboard shortcut at the left (e.g. "[1]") and the label after it."""

    def __init__(self, rect, text, value, keys=(), key_label=""):
        super().__init__(rect, text, value, keys)
        self.key_label = key_label

    def _draw_content(self, img, shift):
        y = self._baseline(shift)
        cv2.putText(img, self.key_label, (self.x + 16 + shift, y), FONT, 0.6, YELLOW, 2, cv2.LINE_AA)
        cv2.putText(img, self.text, (self.x + 80 + shift, y), FONT, 0.6, WHITE, 1, cv2.LINE_AA)


class IconButton(Button):
    """A button showing one of ICONS; its text is the tooltip (see draw_tooltip)."""

    def __init__(self, rect, text, value, keys=(), icon="play", icon_size=11):
        super().__init__(rect, text, value, keys)
        self.icon = icon
        self.icon_size = icon_size

    def _draw_content(self, img, shift):
        center = (self.x + self.w // 2 + shift, self.y + self.h // 2 + shift)
        draw_icon(img, self.icon, center, self.icon_size)


def button_state(button, mouse, armed):
    """
    How `button` should look: "pressed" while the mouse is held down on it
    (`armed` is its value) and still over it, "hover" while the mouse is over
    it and nothing is held, else "normal". `mouse` is in the button's pixels.
    """
    over = button.contains(mouse)
    if armed == button.value:
        return "pressed" if over else "normal"
    return "hover" if over and armed is None else "normal"


def draw_buttons(img, buttons, mouse, armed):
    """Draw `buttons` on `img`, each in its state (see button_state)."""
    for b in buttons:
        b.draw(img, button_state(b, mouse, armed))


def draw_badge(img, text, pos, color, scale=0.55, thickness=1, center=False):
    """
    `text` in `color` on a dark box with a border in the same colour, so it
    stays readable over a video frame. `pos` is the box's top-left corner, or
    with `center` its top-centre. Returns the box's bottom y.
    """
    pad = 8
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = pos
    if center:
        x -= tw // 2 + pad
    x1, y1 = x + tw + 2 * pad, y + th + 2 * pad
    cv2.rectangle(img, (x, y), (x1, y1), (20, 20, 20), -1)
    cv2.rectangle(img, (x, y), (x1, y1), color, 1)
    cv2.putText(img, text, (x + pad, y + pad + th), FONT, scale, color, thickness, cv2.LINE_AA)
    return y1


def draw_tooltip(img, text, anchor, above=False):
    """
    A small label with `text` just below `anchor` (x, y) — or, with `above`,
    just above it — e.g. by a hovered icon button, kept inside `img`.
    """
    scale, pad = 0.45, 6
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, 1)
    x = min(max(2, anchor[0]), img.shape[1] - tw - 2 * pad - 2)
    y = anchor[1] - th - 2 * pad - 4 if above else anchor[1] + 4
    cv2.rectangle(img, (x, y), (x + tw + 2 * pad, y + th + 2 * pad), (15, 15, 15), -1)
    cv2.rectangle(img, (x, y), (x + tw + 2 * pad, y + th + 2 * pad), DIM, 1)
    cv2.putText(img, text, (x + pad, y + pad + th), FONT, scale, WHITE, 1, cv2.LINE_AA)


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
        if icon:
            buttons.append(IconButton((0, y, _ICON_BTN_W, BTN_H), text, value, keys, icon=icon))
        else:
            w = max(_BTN_MIN_W, cv2.getTextSize(text, FONT, 0.6, 1)[0][0] + 48)
            buttons.append(TextButton((0, y, w, BTN_H), text, value, keys))
    x = (MAIN_W - sum(b.w for b in buttons) - _BTN_GAP * (len(buttons) - 1)) // 2
    for b in buttons:
        b.x = x
        x += b.w + _BTN_GAP
    return buttons


# Icons, drawn with shapes (the font has no symbols): media-player controls,
# the walk start / stop marks (a green / red dot, as on the seek bar), jumping to
# the previous / next flagged pose frame (an arrow and an orange dot), and close
# (an X) and save (a floppy disk) for the top bar.
ICONS = ("play", "pause", "prev_frame", "next_frame", "prev_video", "next_video",
         "mark_start", "mark_stop", "prev_flag", "next_flag", "close", "save")
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
    elif icon in ("mark_start", "mark_stop"):       # ● in the mark's colour
        dot = GREEN if icon == "mark_start" else RED
        cv2.circle(img, (cx, cy), int(0.8 * s), dot, -1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), int(0.8 * s), color, 1, cv2.LINE_AA)
    elif icon in ("prev_flag", "next_flag"):        # ◀●  /  ●▶
        facing = -1 if icon == "prev_flag" else 1
        triangle(cx + facing * 2 * s, facing)
        cv2.circle(img, (cx - facing * int(0.8 * s), cy), int(0.55 * s), ORANGE, -1, cv2.LINE_AA)
    elif icon == "close":                           # X
        d = int(0.75 * s)
        cv2.line(img, (cx - d, cy - d), (cx + d, cy + d), color, 2, cv2.LINE_AA)
        cv2.line(img, (cx - d, cy + d), (cx + d, cy - d), color, 2, cv2.LINE_AA)
    elif icon == "save":                            # floppy disk
        # Body with a clipped top-right corner, the metal shutter at the top
        # and the label at the bottom.
        body = np.array([(cx - s, cy - s), (cx + s - 4, cy - s), (cx + s, cy - s + 4),
                         (cx + s, cy + s), (cx - s, cy + s)], np.int32)
        cv2.fillPoly(img, [body], color, cv2.LINE_AA)
        dark = (40, 40, 40)
        cv2.rectangle(img, (cx - s + 5, cy - s), (cx + s - 7, cy - s + 7), dark, -1)
        cv2.rectangle(img, (cx + s - 12, cy - s + 1), (cx + s - 9, cy - s + 5), color, -1)
        cv2.rectangle(img, (cx - s + 4, cy + 1), (cx + s - 4, cy + s - 3), dark, -1)
    else:
        raise ValueError(f"unknown icon {icon!r}")


def bar_buttons(specs):
    """Button row in the bottom bar of an image screen."""
    return button_row(specs, MAIN_H - BAR_H + (BAR_H - BTN_H) // 2)
