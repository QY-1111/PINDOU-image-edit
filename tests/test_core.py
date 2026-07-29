import hashlib
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from palette import PINDOU_PALETTE
from pindou_node import (
    BLACK_PALETTE_INDEX,
    PALETTE_CODES,
    PALETTE_RGB,
    PindouMosaicPattern,
    image_to_bead_grid,
    render_pattern_sheet,
    rgb_to_lab,
)


class PindouCoreTests(unittest.TestCase):
    def test_node_exposes_optional_mask_and_new_controls(self):
        inputs = PindouMosaicPattern.INPUT_TYPES()
        self.assertIn("mask", inputs["optional"])
        self.assertFalse(inputs["required"]["enhance_outer_edge"][1]["default"])
        self.assertEqual(
            inputs["required"]["grid_line_opacity"][1]["default"],
            0.35,
        )

    def test_reference_palette_is_complete(self):
        self.assertEqual(len(PINDOU_PALETTE), 221)
        self.assertEqual(PINDOU_PALETTE["A1"], "#FAF4C8")
        self.assertEqual(PINDOU_PALETTE["B22"], "#0B3C43")
        self.assertEqual(PINDOU_PALETTE["C5"], "#01ACEB")
        self.assertEqual(PINDOU_PALETTE["D13"], "#B90095")
        self.assertEqual(PINDOU_PALETTE["E3"], "#FFB7E7")
        self.assertEqual(PINDOU_PALETTE["F11"], "#5A2121")
        self.assertEqual(PINDOU_PALETTE["G8"], "#753832")
        self.assertEqual(PINDOU_PALETTE["H7"], "#000000")
        self.assertEqual(PINDOU_PALETTE["M15"], "#757D78")

    def test_standard_palette_fingerprint(self):
        canonical = "\n".join(
            f"{code}={hex_value}" for code, hex_value in PINDOU_PALETTE.items()
        )
        self.assertEqual(
            hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            "7224f95876edcd8ec47bb994d5ee7d0ac3ceb0adcd02f3896ac6316c7b8d290a",
        )

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

    def test_quantized_pixels_only_use_standard_palette(self):
        pixels = np.arange(18 * 24 * 3, dtype=np.uint16).reshape(18, 24, 3)
        source = Image.fromarray((pixels % 256).astype(np.uint8), "RGB")
        grid_rgb, _ = image_to_bead_grid(
            source,
            bead_width=24,
            max_colors=16,
            resize_method="面积平均（推荐）",
            dither="Floyd-Steinberg",
        )
        standard_colors = {tuple(color) for color in PALETTE_RGB.tolist()}
        output_colors = {tuple(color) for color in grid_rgb.reshape(-1, 3).tolist()}
        self.assertTrue(output_colors.issubset(standard_colors))

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

    def test_optional_mask_enhances_only_its_inner_outer_edge(self):
        source = Image.new("RGB", (80, 80), (242, 55, 60))
        mask = Image.new("L", (80, 80), 0)
        mask_pixels = np.asarray(mask).copy()
        mask_pixels[20:60, 20:60] = 255
        mask = Image.fromarray(mask_pixels, "L")

        _, baseline_indices = image_to_bead_grid(
            source,
            bead_width=8,
            max_colors=4,
            resize_method="最近邻",
            dither="关闭（推荐）",
        )
        _, enhanced_indices = image_to_bead_grid(
            source,
            bead_width=8,
            max_colors=4,
            resize_method="最近邻",
            dither="关闭（推荐）",
            edge_mask=mask,
            enhance_outer_edge=True,
        )

        self.assertFalse(np.any(baseline_indices == BLACK_PALETTE_INDEX))
        expected_edge = np.zeros((8, 8), dtype=bool)
        expected_edge[2:6, 2:6] = True
        expected_edge[3:5, 3:5] = False
        self.assertTrue(
            np.array_equal(
                enhanced_indices == BLACK_PALETTE_INDEX,
                expected_edge,
            )
        )

    def test_edge_switch_without_mask_preserves_previous_result(self):
        source = Image.new("RGB", (40, 20), (30, 100, 180))
        baseline = image_to_bead_grid(
            source,
            bead_width=12,
            max_colors=8,
            resize_method="面积平均（推荐）",
            dither="关闭（推荐）",
        )
        without_mask = image_to_bead_grid(
            source,
            bead_width=12,
            max_colors=8,
            resize_method="面积平均（推荐）",
            dither="关闭（推荐）",
            enhance_outer_edge=True,
        )
        np.testing.assert_array_equal(baseline[0], without_mask[0])
        np.testing.assert_array_equal(baseline[1], without_mask[1])

    def test_sheet_background_is_white_and_grid_opacity_is_adjustable(self):
        source = Image.new("RGB", (20, 20), (242, 55, 60))
        grid_rgb, indices = image_to_bead_grid(
            source,
            bead_width=2,
            max_colors=2,
            resize_method="最近邻",
            dither="关闭（推荐）",
        )
        sheet_without_grid, _ = render_pattern_sheet(
            grid_rgb,
            indices,
            cell_size=20,
            show_symbols=False,
            show_coordinates=False,
            show_legend=False,
            title="测试图纸",
            grid_line_opacity=0.0,
        )
        sheet_with_grid, _ = render_pattern_sheet(
            grid_rgb,
            indices,
            cell_size=20,
            show_symbols=False,
            show_coordinates=False,
            show_legend=False,
            title="测试图纸",
            grid_line_opacity=1.0,
        )

        self.assertEqual(sheet_without_grid.getpixel((639, 0)), (255, 255, 255))
        self.assertEqual(sheet_with_grid.getpixel((639, 0)), (255, 255, 255))
        grid_position = (16, 66)
        self.assertNotEqual(
            sheet_without_grid.getpixel(grid_position),
            sheet_with_grid.getpixel(grid_position),
        )


if __name__ == "__main__":
    unittest.main()
