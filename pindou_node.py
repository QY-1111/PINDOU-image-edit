"""ComfyUI node that turns an IMAGE into a symbol-labelled fuse-bead pattern."""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from .palette import PINDOU_PALETTE
except ImportError:  # Allows the core renderer to be tested as a standalone module.
    from palette import PINDOU_PALETTE


PALETTE_CODES = np.array(list(PINDOU_PALETTE.keys()))
PALETTE_RGB = np.array(
    [
        tuple(int(value[i : i + 2], 16) for i in (1, 3, 5))
        for value in PINDOU_PALETTE.values()
    ],
    dtype=np.uint8,
)

RESAMPLE_METHODS = {
    "面积平均（推荐）": Image.Resampling.BOX,
    "高质量 Lanczos": Image.Resampling.LANCZOS,
    "最近邻": Image.Resampling.NEAREST,
}


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB values in [0, 255] to CIE Lab (D65), vectorized."""
    values = np.asarray(rgb, dtype=np.float32) / 255.0
    linear = np.where(
        values <= 0.04045,
        values / 12.92,
        ((values + 0.055) / 1.055) ** 2.4,
    )
    xyz = linear @ np.array(
        [
            [0.4124564, 0.2126729, 0.0193339],
            [0.3575761, 0.7151522, 0.1191920],
            [0.1804375, 0.0721750, 0.9503041],
        ],
        dtype=np.float32,
    )
    xyz /= np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    delta = 6.0 / 29.0
    transformed = np.where(
        xyz > delta**3,
        np.cbrt(xyz),
        xyz / (3 * delta**2) + 4.0 / 29.0,
    )
    x, y, z = transformed[..., 0], transformed[..., 1], transformed[..., 2]
    return np.stack(
        (116.0 * y - 16.0, 500.0 * (x - y), 200.0 * (y - z)),
        axis=-1,
    )


PALETTE_LAB = rgb_to_lab(PALETTE_RGB)
BLACK_PALETTE_INDEX = int(np.where(PALETTE_CODES == "H7")[0][0])


def _nearest_indices(
    colors_lab: np.ndarray, candidates_lab: np.ndarray, chunk_size: int = 16384
) -> np.ndarray:
    """Return nearest candidate for each Lab color without a large peak allocation."""
    flat = colors_lab.reshape(-1, 3)
    result = np.empty(flat.shape[0], dtype=np.int32)
    for start in range(0, flat.shape[0], chunk_size):
        part = flat[start : start + chunk_size]
        distance = np.sum(
            (part[:, None, :] - candidates_lab[None, :, :]) ** 2, axis=2
        )
        result[start : start + len(part)] = np.argmin(distance, axis=1)
    return result.reshape(colors_lab.shape[:-1])


def _choose_palette(grid_rgb: np.ndarray, max_colors: int) -> np.ndarray:
    """Choose a compact set of real bead colors using deterministic Lab k-means."""
    max_colors = int(np.clip(max_colors, 1, len(PALETTE_CODES)))
    pixels = rgb_to_lab(grid_rgb).reshape(-1, 3)

    # Palette selection does not need every pixel on very large boards.
    if len(pixels) > 12000:
        step = int(math.ceil(len(pixels) / 12000))
        sample = pixels[::step]
    else:
        sample = pixels

    rounded_unique = np.unique(np.round(sample, 1), axis=0)
    cluster_count = min(max_colors, len(rounded_unique))
    if cluster_count == 0:
        return np.array([0], dtype=np.int32)

    mean = sample.mean(axis=0)
    first = sample[np.argmin(np.sum((sample - mean) ** 2, axis=1))]
    centers = [first]
    min_distance = np.sum((sample - first) ** 2, axis=1)
    for _ in range(1, cluster_count):
        next_index = int(np.argmax(min_distance))
        if min_distance[next_index] < 1e-8:
            break
        next_center = sample[next_index]
        centers.append(next_center)
        min_distance = np.minimum(
            min_distance, np.sum((sample - next_center) ** 2, axis=1)
        )

    centers_array = np.asarray(centers, dtype=np.float32)
    labels = np.zeros(len(sample), dtype=np.int32)
    for _ in range(12):
        new_labels = _nearest_indices(sample, centers_array)
        new_centers = centers_array.copy()
        for index in range(len(centers_array)):
            members = sample[new_labels == index]
            if len(members):
                new_centers[index] = members.mean(axis=0)
        if np.array_equal(labels, new_labels) and np.allclose(
            centers_array, new_centers, atol=0.02
        ):
            centers_array = new_centers
            labels = new_labels
            break
        centers_array = new_centers
        labels = new_labels

    populations = np.bincount(labels, minlength=len(centers_array))
    selected: list[int] = []
    for center_index in np.argsort(-populations):
        distances = np.sum((PALETTE_LAB - centers_array[center_index]) ** 2, axis=1)
        for palette_index in np.argsort(distances):
            candidate = int(palette_index)
            if candidate not in selected:
                selected.append(candidate)
                break
    return np.asarray(selected, dtype=np.int32)


def _floyd_steinberg(
    grid_rgb: np.ndarray, selected_palette: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Floyd-Steinberg error diffusion against the selected physical palette."""
    work = grid_rgb.astype(np.float32).copy()
    height, width = work.shape[:2]
    palette_float = PALETTE_RGB[selected_palette].astype(np.float32)
    output_rgb = np.empty_like(grid_rgb)
    output_global_indices = np.empty((height, width), dtype=np.int32)

    for y in range(height):
        x_range: Iterable[int] = range(width) if y % 2 == 0 else range(width - 1, -1, -1)
        direction = 1 if y % 2 == 0 else -1
        for x in x_range:
            old = np.clip(work[y, x], 0, 255)
            old_lab = rgb_to_lab(old[None, :])[0]
            nearest_local = int(
                np.argmin(np.sum((PALETTE_LAB[selected_palette] - old_lab) ** 2, axis=1))
            )
            new = palette_float[nearest_local]
            output_rgb[y, x] = new.astype(np.uint8)
            output_global_indices[y, x] = selected_palette[nearest_local]
            error = old - new
            nx = x + direction
            if 0 <= nx < width:
                work[y, nx] += error * (7.0 / 16.0)
            if y + 1 < height:
                if 0 <= x - direction < width:
                    work[y + 1, x - direction] += error * (3.0 / 16.0)
                work[y + 1, x] += error * (5.0 / 16.0)
                if 0 <= nx < width:
                    work[y + 1, nx] += error * (1.0 / 16.0)
    return output_rgb, output_global_indices


def image_to_bead_grid(
    image: Image.Image,
    bead_width: int,
    max_colors: int,
    resize_method: str,
    dither: str,
    max_board_side: int = 300,
    edge_mask: Image.Image | None = None,
    enhance_outer_edge: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Resize, select palette colors, and quantize a PIL image."""
    image = image.convert("RGB")
    grid_width = max(1, int(bead_width))
    grid_height = max(1, int(round(image.height * grid_width / image.width)))
    longest = max(grid_width, grid_height)
    if longest > max_board_side:
        scale = max_board_side / longest
        grid_width = max(1, int(round(grid_width * scale)))
        grid_height = max(1, int(round(grid_height * scale)))

    resample = RESAMPLE_METHODS.get(resize_method, Image.Resampling.BOX)
    resized = image.resize((grid_width, grid_height), resample)
    grid_rgb = np.asarray(resized, dtype=np.uint8)
    selected = _choose_palette(grid_rgb, max_colors)

    if dither == "Floyd-Steinberg":
        output_rgb, global_indices = _floyd_steinberg(grid_rgb, selected)
    else:
        grid_lab = rgb_to_lab(grid_rgb)
        local_indices = _nearest_indices(grid_lab, PALETTE_LAB[selected])
        global_indices = selected[local_indices]
        output_rgb = PALETTE_RGB[global_indices]

    if enhance_outer_edge and edge_mask is not None:
        mask = np.asarray(
            edge_mask.convert("L").resize(
                (grid_width, grid_height), Image.Resampling.BOX
            ),
            dtype=np.uint8,
        )
        foreground = mask >= 128
        padded = np.pad(foreground, 1, constant_values=False)
        eroded = np.ones_like(foreground)
        for y_offset in range(3):
            for x_offset in range(3):
                eroded &= padded[
                    y_offset : y_offset + grid_height,
                    x_offset : x_offset + grid_width,
                ]
        outer_edge = foreground & ~eroded
        global_indices[outer_edge] = BLACK_PALETTE_INDEX
        output_rgb = PALETTE_RGB[global_indices]

    return output_rgb, global_indices


def _font_candidates(bold: bool) -> list[str]:
    if bold:
        names = [
            "C:/Windows/Fonts/msyhbd.ttc",
            "C:/Windows/Fonts/arialbd.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        ]
    else:
        names = [
            "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        ]
    return names


def _get_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    for path in _font_candidates(bold):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, max(8, int(size)))
            except OSError:
                pass
    try:
        return ImageFont.load_default(size=max(8, int(size)))
    except TypeError:
        return ImageFont.load_default()


def _text_color(rgb: np.ndarray) -> tuple[int, int, int]:
    red, green, blue = (float(value) / 255.0 for value in rgb)
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return (20, 20, 20) if luminance > 0.58 else (255, 255, 255)


def render_mosaic_preview(grid_rgb: np.ndarray, cell_size: int) -> Image.Image:
    height, width = grid_rgb.shape[:2]
    effective_cell = max(4, min(int(cell_size), max(4, 7200 // max(width, height))))
    return Image.fromarray(grid_rgb, "RGB").resize(
        (width * effective_cell, height * effective_cell), Image.Resampling.NEAREST
    )


def render_pattern_sheet(
    grid_rgb: np.ndarray,
    grid_indices: np.ndarray,
    cell_size: int,
    show_symbols: bool,
    show_coordinates: bool,
    show_legend: bool,
    title: str,
    grid_line_opacity: float = 0.35,
    symbol_font_scale: float = 0.40,
) -> tuple[Image.Image, Counter]:
    """Render a printable board with cell symbols, coordinates, and color counts."""
    rows, columns = grid_indices.shape
    cell = max(8, min(int(cell_size), max(8, 7200 // max(rows, columns))))
    counts = Counter(int(value) for value in grid_indices.reshape(-1))
    coordinate_gutter = 34 if show_coordinates else 0
    header_height = 66
    x0 = 16 + coordinate_gutter
    y0 = header_height + (24 if show_coordinates else 0)
    grid_width_px = columns * cell
    grid_height_px = rows * cell
    canvas_width = max(640, x0 + grid_width_px + 18)

    legend_height = 0
    legend_columns = max(2, canvas_width // 190)
    legend_rows = math.ceil(len(counts) / legend_columns)
    if show_legend:
        legend_height = 52 + legend_rows * 34 + 22
    canvas_height = y0 + grid_height_px + 18 + legend_height

    paper = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
    draw = ImageDraw.Draw(paper)
    title_font = _get_font(24, bold=True)
    meta_font = _get_font(13)
    coord_font = _get_font(max(9, min(13, cell // 2)), bold=True)
    symbol_scale = float(np.clip(symbol_font_scale, 0.20, 0.80))
    symbol_font_size = max(8, int(cell * symbol_scale))
    symbol_fonts: dict[str, ImageFont.ImageFont] = {}
    legend_font = _get_font(14, bold=True)
    legend_count_font = _get_font(13)

    clean_title = (title or "拼豆图纸").strip()
    draw.text((16, 12), clean_title, fill=(30, 30, 30), font=title_font)
    draw.text(
        (16, 43),
        f"{columns} × {rows} 格   ·   {columns * rows} 颗   ·   {len(counts)} 色",
        fill=(80, 80, 80),
        font=meta_font,
    )

    for y in range(rows):
        for x in range(columns):
            color = tuple(int(channel) for channel in grid_rgb[y, x])
            left = x0 + x * cell
            top = y0 + y * cell
            draw.rectangle(
                (left, top, left + cell - 1, top + cell - 1),
                fill=color,
            )
            if show_symbols:
                code = str(PALETTE_CODES[grid_indices[y, x]])
                symbol_font = symbol_fonts.get(code)
                if symbol_font is None:
                    symbol_font = _get_font(symbol_font_size, bold=True)
                    left_bound, top_bound, right_bound, bottom_bound = (
                        symbol_font.getbbox(code)
                    )
                    text_width = max(1, right_bound - left_bound)
                    text_height = max(1, bottom_bound - top_bound)
                    fit_scale = min(
                        1.0,
                        (cell - 2) / text_width,
                        (cell - 2) / text_height,
                    )
                    if fit_scale < 1.0:
                        symbol_font = _get_font(
                            max(8, int(symbol_font_size * fit_scale)),
                            bold=True,
                        )
                    symbol_fonts[code] = symbol_font
                draw.text(
                    (left + cell / 2, top + cell / 2),
                    code,
                    fill=_text_color(grid_rgb[y, x]),
                    font=symbol_font,
                    anchor="mm",
                    stroke_width=0,
                )

    # Blend grid lines over each cell so the slider can soften them without
    # making pale bead colors look muddy.
    line_alpha = float(np.clip(grid_line_opacity, 0.0, 1.0))
    fine_line = (155, 155, 155)
    strong_line = (70, 70, 70)
    line_overlay = Image.new("RGBA", paper.size, (0, 0, 0, 0))
    line_draw = ImageDraw.Draw(line_overlay)

    # Fine lines plus a stronger line every five beads make counting easier.
    for x in range(columns + 1):
        position = x0 + x * cell
        strong = x % 5 == 0
        line_draw.line(
            (position, y0, position, y0 + grid_height_px),
            fill=(
                *(strong_line if strong else fine_line),
                round(255 * line_alpha),
            ),
            width=2 if strong else 1,
        )
    for y in range(rows + 1):
        position = y0 + y * cell
        strong = y % 5 == 0
        line_draw.line(
            (x0, position, x0 + grid_width_px, position),
            fill=(
                *(strong_line if strong else fine_line),
                round(255 * line_alpha),
            ),
            width=2 if strong else 1,
        )
    paper.paste(line_overlay, (0, 0), line_overlay)
    draw = ImageDraw.Draw(paper)

    if show_coordinates:
        for x in range(4, columns, 5):
            draw.text(
                (x0 + (x + 0.5) * cell, y0 - 13),
                str(x + 1),
                fill=(45, 45, 45),
                font=coord_font,
                anchor="mm",
            )
        for y in range(4, rows, 5):
            draw.text(
                (x0 - 8, y0 + (y + 0.5) * cell),
                str(y + 1),
                fill=(45, 45, 45),
                font=coord_font,
                anchor="rm",
            )

    if show_legend:
        legend_top = y0 + grid_height_px + 28
        draw.line(
            (16, legend_top - 10, canvas_width - 16, legend_top - 10),
            fill=(190, 185, 170),
            width=1,
        )
        draw.text(
            (16, legend_top),
            "色号与用量",
            fill=(35, 35, 35),
            font=_get_font(17, bold=True),
        )
        sorted_indices = sorted(
            counts,
            key=lambda index: (
                str(PALETTE_CODES[index])[0],
                int(str(PALETTE_CODES[index])[1:]),
            ),
        )
        card_width = (canvas_width - 32) / legend_columns
        item_top = legend_top + 32
        for item, palette_index in enumerate(sorted_indices):
            column = item % legend_columns
            row = item // legend_columns
            left = 16 + column * card_width
            top = item_top + row * 34
            color = tuple(int(channel) for channel in PALETTE_RGB[palette_index])
            draw.rounded_rectangle(
                (left, top, left + 24, top + 24),
                radius=3,
                fill=color,
                outline=(135, 135, 135),
                width=1,
            )
            draw.text(
                (left + 34, top + 12),
                str(PALETTE_CODES[palette_index]),
                fill=(25, 25, 25),
                font=legend_font,
                anchor="lm",
            )
            draw.text(
                (left + 82, top + 12),
                f"× {counts[palette_index]}",
                fill=(80, 80, 80),
                font=legend_count_font,
                anchor="lm",
            )
    return paper, counts


def _pad_pil_batch(images: list[Image.Image]) -> np.ndarray:
    max_width = max(image.width for image in images)
    max_height = max(image.height for image in images)
    batch = np.ones((len(images), max_height, max_width, 3), dtype=np.float32)
    for index, image in enumerate(images):
        array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        batch[index, : image.height, : image.width] = array
    return batch


class PindouMosaicPattern:
    """ComfyUI-facing wrapper around the standalone renderer."""

    CATEGORY = "图像/拼豆 PINDOU"
    FUNCTION = "generate"
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("马赛克预览", "带色号图纸", "色号统计")
    DESCRIPTION = "把输入图像转换为匹配 A-H/M 实体色卡的拼豆图纸。"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "bead_width": (
                    "INT",
                    {"default": 48, "min": 4, "max": 300, "step": 1},
                ),
                "max_colors": (
                    "INT",
                    {"default": 24, "min": 2, "max": 128, "step": 1},
                ),
                "resize_method": (list(RESAMPLE_METHODS.keys()),),
                "dither": (["关闭（推荐）", "Floyd-Steinberg"],),
                "cell_size": (
                    "INT",
                    {"default": 24, "min": 8, "max": 64, "step": 1},
                ),
                "show_symbols": ("BOOLEAN", {"default": True}),
                "show_coordinates": ("BOOLEAN", {"default": True}),
                "show_legend": ("BOOLEAN", {"default": True}),
                "title": (
                    "STRING",
                    {"default": "拼豆图纸", "multiline": False},
                ),
                "enhance_outer_edge": ("BOOLEAN", {"default": False}),
                "grid_line_opacity": (
                    "FLOAT",
                    {
                        "default": 0.35,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "display": "slider",
                    },
                ),
                "symbol_font_scale": (
                    "FLOAT",
                    {
                        "default": 0.40,
                        "min": 0.20,
                        "max": 0.80,
                        "step": 0.05,
                        "display": "slider",
                    },
                ),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    def generate(
        self,
        image,
        bead_width,
        max_colors,
        resize_method,
        dither,
        cell_size,
        show_symbols,
        show_coordinates,
        show_legend,
        title,
        enhance_outer_edge=False,
        grid_line_opacity=0.35,
        symbol_font_scale=0.40,
        mask=None,
    ):
        import torch

        if image.ndim != 4 or image.shape[-1] < 3:
            raise ValueError("IMAGE 输入必须是 [批次, 高, 宽, 通道]，且至少有 RGB 三通道。")

        previews: list[Image.Image] = []
        sheets: list[Image.Image] = []
        statistics = []

        source_batch = image.detach().cpu().numpy()
        mask_batch = None if mask is None else mask.detach().cpu().numpy()
        for batch_index, source in enumerate(source_batch):
            source_rgb = np.clip(source[..., :3] * 255.0, 0, 255).astype(np.uint8)
            pil_image = Image.fromarray(source_rgb, "RGB")
            pil_mask = None
            if mask_batch is not None:
                mask_source = mask_batch[min(batch_index, len(mask_batch) - 1)]
                mask_uint8 = np.clip(mask_source * 255.0, 0, 255).astype(np.uint8)
                pil_mask = Image.fromarray(mask_uint8, "L")
            grid_rgb, grid_indices = image_to_bead_grid(
                pil_image,
                bead_width=bead_width,
                max_colors=max_colors,
                resize_method=resize_method,
                dither=dither,
                edge_mask=pil_mask,
                enhance_outer_edge=enhance_outer_edge,
            )
            preview = render_mosaic_preview(grid_rgb, cell_size)
            sheet, counts = render_pattern_sheet(
                grid_rgb,
                grid_indices,
                cell_size=cell_size,
                show_symbols=show_symbols,
                show_coordinates=show_coordinates,
                show_legend=show_legend,
                title=title,
                grid_line_opacity=grid_line_opacity,
                symbol_font_scale=symbol_font_scale,
            )
            previews.append(preview)
            sheets.append(sheet)
            rows, columns = grid_indices.shape
            statistics.append(
                {
                    "image": batch_index + 1,
                    "board": f"{columns}x{rows}",
                    "total_beads": int(columns * rows),
                    "colors": [
                        {
                            "code": str(PALETTE_CODES[index]),
                            "hex": PINDOU_PALETTE[str(PALETTE_CODES[index])],
                            "count": int(count),
                        }
                        for index, count in sorted(
                            counts.items(),
                            key=lambda item: (
                                str(PALETTE_CODES[item[0]])[0],
                                int(str(PALETTE_CODES[item[0]])[1:]),
                            ),
                        )
                    ],
                }
            )

        preview_tensor = torch.from_numpy(_pad_pil_batch(previews))
        sheet_tensor = torch.from_numpy(_pad_pil_batch(sheets))
        summary = json.dumps(
            statistics[0] if len(statistics) == 1 else statistics,
            ensure_ascii=False,
            indent=2,
        )
        return preview_tensor, sheet_tensor, summary
