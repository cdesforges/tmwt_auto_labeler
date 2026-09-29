"""Tests for finding the finish line from the ArUco marker (tmwt/detection/endpoints.py)."""

import unittest

import cv2
import numpy as np

from tmwt.detection import endpoints

SIDE = 140
DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)


def scene(markers, dictionary=DICT, shape=(480, 640)):
    """A white BGR image with each (marker id, x, y) drawn with its top-left corner at (x, y)."""
    img = np.full(shape + (3,), 255, np.uint8)
    for marker_id, x, y in markers:
        img[y:y + SIDE, x:x + SIDE] = cv2.aruco.generateImageMarker(dictionary, marker_id, SIDE)[..., None]
    return img


def near(point, expected, tol=2):
    return abs(point[0] - expected[0]) <= tol and abs(point[1] - expected[1]) <= tol


class DetectNearEndpointTest(unittest.TestCase):
    def test_one_marker_gives_its_bottom_right_corner(self):
        point = endpoints.detect_near_endpoint(scene([(7, 200, 100)]))
        self.assertTrue(near(point, (200 + SIDE - 1, 100 + SIDE - 1)), point)
        self.assertIsInstance(point[0], int)
        self.assertIsInstance(point[1], int)

    def test_corner_follows_the_marker_not_the_image(self):
        # Upside down, the marker's own bottom-right corner is at the image's top-left.
        img = cv2.rotate(scene([(7, 200, 100)]), cv2.ROTATE_180)
        h, w = img.shape[:2]
        x0, y0 = w - (200 + SIDE), h - (100 + SIDE)
        self.assertTrue(near(endpoints.detect_near_endpoint(img), (x0, y0)))

    def test_any_marker_id(self):
        self.assertIsNotNone(endpoints.detect_near_endpoint(scene([(0, 50, 50)])))
        self.assertIsNotNone(endpoints.detect_near_endpoint(scene([(49, 50, 50)])))

    def test_greyscale_frame(self):
        grey = cv2.cvtColor(scene([(7, 200, 100)]), cv2.COLOR_BGR2GRAY)
        self.assertTrue(near(endpoints.detect_near_endpoint(grey), (200 + SIDE - 1, 100 + SIDE - 1)))

    def test_no_marker(self):
        self.assertIsNone(endpoints.detect_near_endpoint(scene([])))
        self.assertIsNone(endpoints.detect_near_endpoint(np.zeros((480, 640, 3), np.uint8)))

    def test_two_markers_are_ambiguous(self):
        self.assertIsNone(endpoints.detect_near_endpoint(scene([(7, 50, 50), (3, 400, 300)])))
        self.assertIsNone(endpoints.detect_near_endpoint(scene([(7, 50, 50), (7, 400, 300)])))

    def test_marker_from_another_dictionary_is_ignored(self):
        other = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.assertIsNone(endpoints.detect_near_endpoint(scene([(7, 200, 100)], other)))

    def test_marker_cut_off_by_the_frame_edge(self):
        img = scene([(7, 200, 100)])[:, :270]           # half the marker is out of frame
        self.assertIsNone(endpoints.detect_near_endpoint(img))


if __name__ == "__main__":
    unittest.main()
