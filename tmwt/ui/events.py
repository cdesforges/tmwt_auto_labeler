"""
How the user leaves a screen other than through its own buttons. Raised by the
window (base_window.py) from whatever screen is showing, and handled by the
caller (review_session.py, processing.py, view.py):

  - JumpTo: a file in the sidebar's `review_targets` was clicked.
  - UserQuit and its subclasses: the user asked to stop —
      WindowClosed      the window was closed (or the top bar's X clicked
                        outside a review);
      SaveAndQuit       "save progress and quit", from the top bar (save icon,
                        or X then "Save progress & quit") during a review;
      QuitWithoutSaving the top bar's X, then "Quit without saving".
"""


class JumpTo(Exception):
    """A file in the sidebar was clicked; `index` is which one."""

    def __init__(self, index):
        super().__init__(index)
        self.index = index


class UserQuit(Exception):
    """Base class: the user asked to stop, from any screen."""


class WindowClosed(UserQuit):
    """The window was closed (or the top bar's X clicked outside a review)."""


class SaveAndQuit(UserQuit):
    """During a review: save the review progress and stop, to continue later."""


class QuitWithoutSaving(UserQuit):
    """During a review: stop and discard the review progress."""
