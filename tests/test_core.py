import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from palette import PINDOU_PALETTE
from pindou_node import (
    PALETTE_CODES,
    image_to_bead_grid,
    render_pattern_sheet,
    rgb_to_lab,
)


class PindouCoreTests(unittest.TestCase):
    def test_reference_palette_is_complete(self):
        self.assertEqual(len(PINDOU_PALETTE), 221)
        self.assertEqual(PINDOU_PALETTE["A1"], "#FAF5CC")
        self.assertEqual(PINDOU_PALETTE["M15"], "#767F7C")

    def test_lab_reference_points(self):
        black, white = rgb_to_lab(
            np.array([[0, 0, 0], [255, 255, 255]], dtype=np.uint8)
        )
        self.assertAlmostEqual(float(black[0]), 0.0, places=3)
        self.assertAlmostEqual(float(white[0]), 100.0, places=2)

    def test_solid_image_produces_one_physical_color(self):
        source = Image.new("RGB", (40, 20), (242, 55, 60))
        grid_rgb, indices = image_to_bead_grid(
            source,
            bead_width=12,
            max_colors=8,
            resize_method="面积平均（推荐）",
            dither="关闭（推荐）",
        )
        self.assertEqual(grid_rgb.shape, (6, 12, 3))
        self.assertEqual(indices.shape, (6, 12))
        self.assertEqual(len(np.unique(indices)), 1)
        code = str(PALETTE_CODES[int(indices[0, 0])])
        self.assertIn(code[0], ("A", "F"))

    def test_long_board_is_safely_capped(self):
        source = Image.new("RGB", (10, 1000), "white")
        _, indices = image_to_bead_grid(
            source,
            bead_width=100,
            max_colors=4,
            resize_method="最近邻",
            dither="关闭（推荐）",
        )
        self.assertLessEqual(max(indices.shape), 300)

    def test_sheet_contains_grid_and_legend(self):
        source = Image.new("RGB", (30, 20), (30, 100, 180))
        grid_rgb, indices = image_to_bead_grid(
            source,
            bead_width=6,
            max_colors=4,
            resize_method="面积平均（推荐）",
            dither="关闭（推荐）",
        )
        sheet, counts = render_pattern_sheet(
            grid_rgb,
            indices,
            cell_size=20,
            show_symbols=True,
            show_coordinates=True,
            show_legend=True,
            title="测试图纸",
        )
        self.assertGreaterEqual(sheet.width, 640)
        self.assertGreater(sheet.height, indices.shape[0] * 20)
        self.assertEqual(sum(counts.values()), int(indices.size))


if __name__ == "__main__":
    unittest.main()
