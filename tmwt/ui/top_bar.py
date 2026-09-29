"""
The top bar across the labeler window:

  [X] [save]              <folder name>              <UR shield | RNA rosette>

  - X (always there): quit. During a review, it first asks whether to save the
    review progress (labeler_ui.py handles the click).
  - Save (a floppy disk; only while a review is open): save the review progress
    and quit, to continue later.
  - The title (e.g. the folder being worked on), centred on the window and
    shortened in the middle if it doesn't fit (fit_title).
  - The logos (LOGO_PATHS): the University of Rochester's shield and the RNA
    Institute's rosette, without their names (assets/*_shield.png,
    *_rosette.png, with transparency for this dark bar), side by side with a
    thin divider and some space around them; hovering over one shows its name.
    Any file that's missing is left out, and in a window too narrow for them
    all the leftmost are left out first (they never overlap the buttons).
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

_ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                       "assets")
# The University of Rochester's shield, cut from the white "knockout" version of
# its primary logo (brand.rochester.edu/visual-identity/logos/), and the RNA
# Institute's rosette, cut from its logo (albany.edu/rna).
UR_LOGO_PATH = os.path.join(_ASSETS, "university_of_rochester_shield.png")
RNA_LOGO_PATH = os.path.join(_ASSETS, "rna_institute_rosette.png")
# The logos shown at the right of the bar, left to right, and the names shown
# when the pointer is over them.
LOGO_PATHS = (UR_LOGO_PATH, RNA_LOGO_PATH)
LOGO_NAMES = {UR_LOGO_PATH: "University of Rochester",
              RNA_LOGO_PATH: "The RNA Institute, University at Albany"}

_BG = (32, 32, 32)
_PAD = 8                   # gap between the bar's edge and its buttons
_BTN = 36                  # icon buttons are _BTN x _BTN
_LOGO_H = TOPBAR_H - 20    # logos are this tall, with room above and below
_LOGO_MARGIN = 16          # between the last logo and the bar's right edge
_TITLE_GAP = 24            # least space between the title and the buttons or logos
_LOGO_GAP = 24             # between two logos (a thin divider sits in the middle)
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


def _logos_width(logos):
    """Width of (bgr, alpha) logos side by side, with the gaps between them."""
    return sum(bgr.shape[1] for bgr, _ in logos) + _LOGO_GAP * max(0, len(logos) - 1)


class TopBar(Panel):
    """The bar at the top of the window (see module docs)."""

    def __init__(self, width, title="", logo_paths=LOGO_PATHS):
        super().__init__(0, 0, width, TOPBAR_H)
        self.title = title
        self.show_save = False      # show the save button (while a review is open)
        y = (TOPBAR_H - _BTN) // 2
        self.close_button = IconButton((_PAD, y, _BTN, _BTN), "Quit", CLOSE, icon="close", icon_size=10)
        self.save_button = IconButton((2 * _PAD + _BTN, y, _BTN, _BTN), "Save progress & quit",
                                      SAVE, icon="save", icon_size=10)
        # Loaded logos, left to right (any that can't be read are left out),
        # and their names (LOGO_NAMES, else the file's name).
        loaded = [(p, load_logo(p, _LOGO_H)) for p in logo_paths]
        self._logos = [logo for _, logo in loaded if logo]
        self._logo_names = [LOGO_NAMES.get(p, os.path.splitext(os.path.basename(p))[0])
                            for p, logo in loaded if logo]

    @property
    def buttons(self):
        """The buttons showing now."""
        return [self.close_button, self.save_button] if self.show_save else [self.close_button]

    def title_width(self):
        """
        Room for the title. It's centred on the whole bar, so it gets twice the
        space to the nearer side's widest item (both buttons, or the logo).
        """
        left = self._buttons_right()
        right = self.logos_width() + _LOGO_MARGIN if self.shown_logos() else 0
        return self.w - 2 * (max(left, right) + _TITLE_GAP)

    def _buttons_right(self):
        """Right edge of the buttons (the save button's place, shown or not, so the layout doesn't jump)."""
        return self.save_button.x + self.save_button.w

    def shown_logos(self):
        """
        The logos that fit right of the buttons (with _TITLE_GAP between):
        in a narrow window the leftmost are left out first, so the far-right
        logo stays longest.
        """
        logos = list(self._logos)
        while logos and self.w - _LOGO_MARGIN - _logos_width(logos) < self._buttons_right() + _TITLE_GAP:
            logos.pop(0)
        return logos

    def logos_width(self):
        """Width of the logos shown, together with the gaps between them."""
        return _logos_width(self.shown_logos())

    def logo_places(self):
        """[(x, y, w, h, name)] of the logos shown, in bar pixels, left to right."""
        shown = self.shown_logos()
        names = self._logo_names[len(self._logos) - len(shown):]
        places = []
        x = self.w - _LOGO_MARGIN - _logos_width(shown)
        for (bgr, _), name in zip(shown, names):
            h, w = bgr.shape[:2]
            places.append((x, (self.h - h) // 2, w, h, name))
            x += w + _LOGO_GAP
        return places

    def hovered_logo(self, mouse):
        """The (x, y, w, h, name) of the logo under `mouse` (bar pixels), or None; for its label."""
        if mouse is None:
            return None
        return next((p for p in self.logo_places()
                     if p[0] <= mouse[0] < p[0] + p[2] and p[1] <= mouse[1] < p[1] + p[3]), None)

    def render(self, mouse, armed):
        img = np.full((self.h, self.w, 3), _BG, np.uint8)
        cv2.line(img, (0, self.h - 1), (self.w, self.h - 1), DIM, 1)
        draw_buttons(img, self.buttons, mouse, armed)

        text = fit_title(self.title, self.title_width())
        if text:
            (tw, th), _ = cv2.getTextSize(text, FONT, _TITLE_SCALE, _TITLE_THICKNESS)
            cv2.putText(img, text, ((self.w - tw) // 2, (self.h + th) // 2), FONT, _TITLE_SCALE,
                        WHITE, _TITLE_THICKNESS, cv2.LINE_AA)

        for n, ((bgr, alpha), (x, y, lw, lh, _)) in enumerate(zip(self.shown_logos(), self.logo_places())):
            if n:   # a thin divider between two logos
                mid = x - _LOGO_GAP // 2
                top = (self.h - _LOGO_H) // 2
                cv2.line(img, (mid, top), (mid, top + _LOGO_H - 1), DIM, 1)
            region = img[y:y + lh, x:x + lw].astype(np.float32)
            img[y:y + lh, x:x + lw] = (region * (1 - alpha) + bgr * alpha).astype(np.uint8)
        return img

    def hovered_button(self, mouse):
        """The button under `mouse` (bar pixels), or None; for its tooltip."""
        return next((b for b in self.buttons if b.contains(mouse)), None)
