"""
Picking screens (a family of LabelerUI's screens): the rope endpoints on the
first frame, and the walking subject among several people. Built on
BaseWindow's _interact / _new_screen, so they get the top bar, sidebar and
button handling like every other screen.
"""

import cv2

from tmwt.pose import pose_common
from tmwt.ui.base_window import HANDLE_RADIUS
from tmwt.ui.widgets import (BLUE, FONT, HEADER_H, KEY_BACKSPACE, KEY_ENTER, KEY_ESC, ORANGE, RED,
                             WHITE, YELLOW, bar_buttons, frame_screen)

# Colours for telling people apart on the "pick the walker" screen.
PERSON_COLORS = [(0, 255, 0), (255, 160, 0), (255, 0, 255), (0, 200, 255), (60, 60, 255)]


class EndpointPicking:
    """
    The state of picking the rope endpoints (pick_endpoints), without any
    drawing: the START and FINISH points (frame pixels, or None), which one a
    click places next (`placing`: "start", "finish" or None when both are
    set), and the buttons to offer.

      - A click places the point being placed, then moves on to the next one
        missing.
      - Dragging a point moves it (dropping the point being placed counts as
        placing it).
      - Move start point / Move endpoint: a click places that point again;
        Keep start point / Keep endpoint leaves it as it was.
      - Auto start point (when the start isn't the detected standing spot):
        puts it back there.
      - Confirm (Enter) once both are set; Cancel (Esc) any time.
    """

    PROMPTS = {
        "start": "Click the START point (where the walk begins)",
        "finish": "Click the FINISH point (end of the course)",
        None: "Check the line, then Confirm. Drag a point to move it.",
    }

    def __init__(self, start=None, finish=None, auto_start=None, move=None):
        self.start, self.finish, self.auto_start = start, finish, auto_start
        self.start_moved = start is not None and start != auto_start
        self.placing = self._missing() or move

    def _missing(self):
        """The first point not placed yet, or None."""
        return "start" if self.start is None else ("finish" if self.finish is None else None)

    def prompt(self):
        return self.PROMPTS[self.placing]

    def specs(self):
        """Button specs for the bar, for the current state."""
        cancel = ("Cancel", "cancel", (KEY_ESC,))
        move_start = ("Move start point", "move_start", ())
        auto = ([("Auto start point", "auto_start", ())]
                if self.auto_start is not None and self.start != self.auto_start else [])
        if self.placing == "start":
            keep = [("Keep start point", "keep", ())] if self.start is not None else []
            return keep + auto + [cancel]
        if self.placing == "finish":
            keep = [("Keep endpoint", "keep", ())] if self.finish is not None else []
            return [move_start] + auto + keep + [cancel]
        return [move_start, ("Move endpoint", "move_finish", KEY_BACKSPACE)] + auto + [
            ("Confirm (Enter)", "confirm", KEY_ENTER), cancel]

    def press(self, value):
        """Apply a button (other than Confirm / Cancel)."""
        if value == "move_start":
            self.placing = "start"
        elif value == "move_finish":
            self.placing = "finish"
        elif value == "keep":
            self.placing = self._missing()
        elif value == "auto_start" and self.auto_start is not None:
            self.start, self.start_moved = self.auto_start, False
            self.placing = self._missing()

    def place(self, point):
        """A click at `point` (frame pixels): places the point being placed, if any."""
        if self.placing is None:
            return
        self._set(self.placing, point)
        self.placing = self._missing()

    def drag(self, which, point, done):
        """Point `which` dragged to `point`; on release, placing moves on if it was that point."""
        self._set(which, point)
        if done and self.placing == which:
            self.placing = self._missing()

    def _set(self, which, point):
        if which == "start":
            self.start, self.start_moved = point, point != self.auto_start
        else:
            self.finish = point

    def result(self):
        """(start, finish, start_moved), for pick_endpoints."""
        return self.start, self.finish, self.start_moved


class PickerScreens:
    """
    Mixin for BaseWindow subclasses: pick_endpoints and pick_person. Uses the
    window's _interact and _new_screen.
    """

    def pick_endpoints(self, frame, reason=None, start=None, finish=None, move=None, auto_start=None):
        """
        Let the user set the rope endpoints on `frame`: the START point (far
        endpoint, where the walk begins) and the FINISH point (near endpoint).
        Points are placed by clicking, or moved by dragging them; the buttons
        and what a click places follow EndpointPicking.

        Args:
            frame: the video's first frame (the endpoints are in its pixels).
            reason: optional line explaining why clicks are needed.
            start, finish: points to pre-place (e.g. where the subject was
                detected standing), or None.
            move: "start" or "finish" to begin by moving that point.
            auto_start: the detected standing spot, for "Auto start point".

        Returns:
            (start, finish, start_moved) in frame pixels — start_moved is True
            if the start isn't the automatic one (the user placed it) — or
            None if cancelled.
        """
        base, scale, ox, oy = frame_screen(frame, top=HEADER_H)
        fh, fw = frame.shape[:2]

        def to_screen(p):
            return (int(p[0] * scale + ox), int(p[1] * scale + oy))

        def to_frame(pt):
            """A main-area point in frame pixels, kept inside the frame."""
            return (int(round(min(max((pt[0] - ox) / scale, 0), fw - 1))),
                    int(round(min(max((pt[1] - oy) / scale, 0), fh - 1))))

        state = EndpointPicking(start, finish, auto_start, move)
        self._new_screen()
        while True:
            handles = {k: to_screen(p) for k, p in (("start", state.start), ("finish", state.finish))
                       if p is not None}
            # The point being dragged, else the one under the pointer (ringed).
            if self._drag is not None:
                active = self._drag[1]
            else:
                grab = self._grab(self._mouse(self.main), None, handles)
                active = grab[1] if grab else None
            main = base.copy()
            if state.start is not None and state.finish is not None:
                cv2.line(main, to_screen(state.start), to_screen(state.finish), YELLOW, 2)
            for key, color, label in (("start", BLUE, "START"), ("finish", RED, "FINISH")):
                if key in handles:
                    sp = handles[key]
                    cv2.circle(main, sp, 7, color, -1)
                    if key == active:   # hovered or being dragged
                        cv2.circle(main, sp, HANDLE_RADIUS, WHITE, 2, cv2.LINE_AA)
                    cv2.putText(main, label, (sp[0] + 10, sp[1] - 8), FONT, 0.5, color, 2, cv2.LINE_AA)
            y = 32
            if reason:
                cv2.putText(main, reason, (16, y), FONT, 0.55, ORANGE, 1, cv2.LINE_AA)
                y += 30
            cv2.putText(main, state.prompt(), (16, y), FONT, 0.65, WHITE, 2, cv2.LINE_AA)

            value, _, clicks = self._interact(main, bar_buttons(state.specs()), 30, handles=handles)
            if value is None:
                value = self._live_drag_value()
            if value == "cancel":
                return None
            if value == "confirm":
                return state.result()
            if isinstance(value, tuple) and value[0] in ("point_drag", "point_drop"):
                state.drag(value[1], to_frame(value[2]), done=value[0] == "point_drop")
            elif value is not None:
                state.press(value)
            for cx, cy in clicks:
                fx, fy = (cx - ox) / scale, (cy - oy) / scale
                if 0 <= fx < fw and 0 <= fy < fh:
                    state.place((int(round(fx)), int(round(fy))))

    def pick_person(self, frame, poses, reason):
        """
        Let the user click the walking subject among `poses` (people in `frame`),
        each drawn in its own colour with a number. Returns the index of the
        chosen pose, or None if cancelled.
        """
        img = frame.copy()
        fh, fw = frame.shape[:2]
        boxes = []
        for k, pose in enumerate(poses):
            color = PERSON_COLORS[k % len(PERSON_COLORS)]
            pose_common.draw_pose(img, pose, color=color, point_radius=5, line_thickness=3)
            xs = [lm.x * fw for lm in pose if lm is not None]
            ys = [lm.y * fh for lm in pose if lm is not None]
            pad = 0.15 * (max(ys) - min(ys)) + 10
            boxes.append((min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad))
            cv2.putText(img, str(k + 1), (int(min(xs)), int(min(ys) - pad)), FONT, 1.0, color, 3, cv2.LINE_AA)

        base, scale, ox, oy = frame_screen(img, top=HEADER_H)
        cv2.putText(base, reason, (16, 32), FONT, 0.55, ORANGE, 1, cv2.LINE_AA)
        cv2.putText(base, "Click the person doing the walk test", (16, 62), FONT, 0.65, WHITE, 2, cv2.LINE_AA)
        buttons = bar_buttons([("Cancel", "cancel", (KEY_ESC,))])
        self._new_screen()
        while True:
            value, _, clicks = self._interact(base, buttons, 30)
            if value == "cancel":
                return None
            for cx, cy in clicks:
                fx, fy = (cx - ox) / scale, (cy - oy) / scale
                hits = [k for k, (x0, y0, x1, y1) in enumerate(boxes) if x0 <= fx <= x1 and y0 <= fy <= y1]
                if hits:
                    # Overlapping boxes: take the person whose centre is closest.
                    return min(hits, key=lambda k: abs((boxes[k][0] + boxes[k][2]) / 2 - fx))
