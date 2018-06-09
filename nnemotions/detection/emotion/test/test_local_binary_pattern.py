import unittest
import numpy as np
from nnemotions.detection.emotion.local_binary_pattern import BinaryPatternAnalysis, BinaryPattern


class LBPTestCase(unittest.TestCase):
    """Tests for 'local_binary_pattern.py'."""

    def test_pattern(self):
        img = np.random.rand(100, 100)
        bpa = BinaryPatternAnalysis(img, (6, 5))
        self.assertEqual(17 * 20 * 59, len(bpa.get_histogram()))

    def test_pattern_detection(self):
        img = np.array([[0.1, 0.3, 0.2], [0.1, 0.3, 0.03], [0.1, 0.0, 0.4]])
        bp = BinaryPattern(img)
        self.assertTrue(bp.is_uniform())
        self.assertEqual(int(bp), 16)


if __name__ == '__main__':
    unittest.main()
