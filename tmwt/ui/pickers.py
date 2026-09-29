"""
Picking screens (a family of LabelerUI's screens): the rope endpoints on the
first frame, and the walking subject among several people. Built on
BaseWindow's _interact / _new_screen, so they get the top bar, sidebar and
button handling like every other screen.
"""

import cv2

from tmwt.pose import pose_common
from tmwt.ui.widgets import (BLUE, FONT, HEADER_H, KEY_BACKSPACE, KEY_ENTER, KEY_ESC, ORANGE, RED,
                             WHITE, YELLOW, bar_buttons, frame_screen)

# Colours for telling people apart on the "pick the walker" screen.
PERSON_COLORS = [(0, 255, 0), (255, 160, 0), (255, 0, 255), (0, 200, 255), (60, 60, 255)]


class PickerScreens:
    """
    Mixin for BaseWindow subclasses: pick_endpoints and pick_person. Uses the
    window's _interact and _new_screen.
    """

    def pick_endpoints(self, frame, reason=None, start=None, finish=None):
        """
        Let the user set the rope endpoints on `frame`: the START point (far
        endpoint, where the walk begins) and the FINISH point (near endpoint).

        A given `start` (e.g. where the subject was detected standing) is
        pre-placed, so usually only the finish is clicked; "Move start point"
        lets the user click a new one. A given `finish` is pre-placed too.
        Once both are placed: Confirm, Redo finish point, Move start point, or
        Cancel.

        Args:
            frame: the video's first frame (the endpoints are in its pixels).
            reason: optional line explaining why clicks are needed.

        Returns:
            (start, finish, start_moved) in frame pixels — start_moved is True
            if the user placed the start point themselves — or None if cancelled.
        """
        base, scale, ox, oy = frame_screen(frame, top=HEADER_H)
        fh, fw = frame.shape[:2]

        def to_screen(p):
            return (int(p[0] * scale + ox), int(p[1] * scale + oy))

        prompts = {
            "start": "Click the START point (where the walk begins)",
            "finish": "Click the FINISH point (end of the course)",
            None: "Check the line, then Confirm",
        }
        cancel = ("Cancel", "cancel", (KEY_ESC,))
        move = ("Move start point", "move", ())
        placing = "start" if start is None else ("finish" if finish is None else None)
        moved = False
        self._new_screen()
        while True:
            main = base.copy()
            if start is not None and finish is not None:
                cv2.line(main, to_screen(start), to_screen(finish), YELLOW, 2)
            for p, color, label in ((start, BLUE, "START"), (finish, RED, "FINISH")):
                if p is not None:
                    sp = to_screen(p)
                    cv2.circle(main, sp, 7, color, -1)
                    cv2.putText(main, label, (sp[0] + 10, sp[1] - 8), FONT, 0.5, color, 2, cv2.LINE_AA)
            y = 32
            if reason:
                cv2.putText(main, reason, (16, y), FONT, 0.55, ORANGE, 1, cv2.LINE_AA)
                y += 30
            cv2.putText(main, prompts[placing], (16, y), FONT, 0.65, WHITE, 2, cv2.LINE_AA)

            if placing == "start":
                specs = [("Keep start point", "keep", ()), cancel] if start is not None else [cancel]
            elif placing == "finish":
                specs = [move, cancel]
            else:
                specs = [move, ("Redo finish point", "redo", KEY_BACKSPACE),
                         ("Confirm", "confirm", KEY_ENTER), cancel]
            value, _, clicks = self._interact(main, bar_buttons(specs), 30)

            if value == "cancel":
                return None
            if value == "confirm":
                return start, finish, moved
            if value == "move":
                placing = "start"
            elif value == "keep":
                placing = "finish" if finish is None else None
            elif value == "redo":
                finish, placing = None, "finish"
            for cx, cy in clicks:
                fx, fy = (cx - ox) / scale, (cy - oy) / scale
                if placing is None or not (0 <= fx < fw and 0 <= fy < fh):
                    continue
                point = (int(round(fx)), int(round(fy)))
                if placing == "start":
                    start, moved = point, True
                    placing = "finish" if finish is None else None
                else:
                    finish, placing = point, None

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
