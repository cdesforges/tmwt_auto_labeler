"""
Panel: a rectangular region of the labeler window's canvas.

The canvas (labeler_ui.py) is split into panels: the top bar (top_bar.py)
across the top, and below it the main area on the left and the sidebar
(sidebar.py) on the right. A panel knows where it sits, so mouse positions,
which the window reports in canvas pixels, can be turned into the panel's own
pixels; drawing and hit-testing inside a panel all use its own pixels.

Panels that draw themselves (TopBar, Sidebar) subclass Panel and implement
render().
"""


class Panel:
    """A region of the canvas at (x, y), w x h pixels."""

    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h

    def contains(self, pt):
        """True if canvas point `pt` is inside this panel."""
        return (pt is not None and self.x <= pt[0] < self.x + self.w
                and self.y <= pt[1] < self.y + self.h)

    def to_local(self, pt):
        """Canvas point `pt` in this panel's pixels (it may lie outside); None stays None."""
        return None if pt is None else (pt[0] - self.x, pt[1] - self.y)

    def local_if_inside(self, pt):
        """Canvas point `pt` in this panel's pixels, or None if it isn't inside the panel."""
        return self.to_local(pt) if self.contains(pt) else None

    def render(self, mouse, armed):
        """
        Draw the panel.

        Args:
            mouse: the pointer in this panel's pixels, or None if it's elsewhere.
            armed: what the mouse is held down on (a button value or a row),
                for pressed looks; None if nothing.

        Returns:
            A BGR image h x w.
        """
        raise NotImplementedError
