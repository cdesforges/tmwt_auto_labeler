"""
The top bar across the labeler window:

  [X] [save]            <folder name>            <RNA Institute logo>

  - X (always there): quit. During a review, it first asks whether to save the
    review progress (labeler_ui.py handles the click).
  - Save (a floppy disk; only while a review is open): save the review progress
    and quit, to continue later.
  - The title (e.g. the folder being worked on), centred on the window and
    shortened in the middle if it doesn't fit (fit_title).
  - The RNA Institute logo (assets/rna_institute_logo.png: white lettering with
    transparency, for this dark bar), or nothing if the file is missing.
"""

import os

import cv2
import numpy as np

from tmwt.ui.panel import Panel
from tmwt.ui.widgets import DIM, FONT, WHITE, IconButton, ascii_text, draw_buttons, truncate_middle

TOPBAR_H = 48
# Button values, kept distinct from any screen's own buttons.
CLOSE = "topbar:close"
SAVE = "topbar:save"

LOGO_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         "assets", "rna_institute_logo.png")

_BG = (32, 32, 32)
_PAD = 8                   # gap between the bar's edge, its buttons and the logo
_BTN = 36                  # icon buttons are _BTN x _BTN
_LOGO_H = TOPBAR_H - 14
_TITLE_GAP = 24            # least space between the title and the buttons or logo
_TITLE_SCALE = 0.65
_TITLE_THICKNESS = 2


def load_logo(path, height):
    """
    The logo at `path` scaled to `height` pixels tall, as (BGR, alpha 0-1), or
    None if it can't be read. An image without transparency is used as it is.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED) if path and os.path.isfile(path) else None
    if img is None or img.ndim != 3:
        return None
    w = max(1, round(img.shape[1] * height / img.shape[0]))
    img = cv2.resize(img, (w, height), interpolation=cv2.INTER_AREA)
    if img.shape[2] == 4:
        return img[..., :3], img[..., 3:].astype(np.float32) / 255.0
    return img, np.ones((height, w, 1), np.float32)


def folder_title(path):
    """A folder's name for the title, e.g. "media/control_vids/" -> "control_vids"."""
    return os.path.basename(os.path.normpath(os.path.abspath(path)))


def fit_title(title, max_w, scale=_TITLE_SCALE, thickness=_TITLE_THICKNESS):
    """
    `title` as it's drawn in `max_w` pixels: in characters the font can draw
    (see widgets.ascii_text), on one line, shortened in the middle if needed.
    """
    return truncate_middle(ascii_text(" ".join(str(title).split())), max(0, max_w), scale, thickness)


class TopBar(Panel):
    """The bar at the top of the window (see module docs)."""

    def __init__(self, width, title="", logo_path=LOGO_PATH):
        super().__init__(0, 0, width, TOPBAR_H)
        self.title = title
        self.show_save = False      # show the save button (while a review is open)
        y = (TOPBAR_H - _BTN) // 2
        self.close_button = IconButton((_PAD, y, _BTN, _BTN), "Quit", CLOSE, icon="close", icon_size=10)
        self.save_button = IconButton((2 * _PAD + _BTN, y, _BTN, _BTN), "Save progress & quit",
                                      SAVE, icon="save", icon_size=10)
        self._logo = load_logo(logo_path, _LOGO_H)

    @property
    def buttons(self):
        """The buttons showing now."""
        return [self.close_button, self.save_button] if self.show_save else [self.close_button]

    def title_width(self):
        """
        Room for the title. It's centred on the whole bar, so it gets twice the
        space to the nearer side's widest item (both buttons, or the logo).
        """
        left = self.save_button.x + self.save_button.w
        right = (self._logo[0].shape[1] + _PAD) if self._logo else 0
        return self.w - 2 * (max(left, right) + _TITLE_GAP)

    def render(self, mouse, armed):
        img = np.full((self.h, self.w, 3), _BG, np.uint8)
        cv2.line(img, (0, self.h - 1), (self.w, self.h - 1), DIM, 1)
        draw_buttons(img, self.buttons, mouse, armed)

        text = fit_title(self.title, self.title_width())
        if text:
            (tw, th), _ = cv2.getTextSize(text, FONT, _TITLE_SCALE, _TITLE_THICKNESS)
            cv2.putText(img, text, ((self.w - tw) // 2, (self.h + th) // 2), FONT, _TITLE_SCALE,
                        WHITE, _TITLE_THICKNESS, cv2.LINE_AA)

        if self._logo:
            bgr, alpha = self._logo
            lh, lw = bgr.shape[:2]
            x, y = self.w - _PAD - lw, (self.h - lh) // 2
            region = img[y:y + lh, x:x + lw].astype(np.float32)
            img[y:y + lh, x:x + lw] = (region * (1 - alpha) + bgr * alpha).astype(np.uint8)
        return img

    def hovered_button(self, mouse):
        """The button under `mouse` (bar pixels), or None; for its tooltip."""
        return next((b for b in self.buttons if b.contains(mouse)), None)
