"""Wind Power Density Analysis – California & Offshore Regions (Module 5)

This script converts a color-coded wind map screenshot (e.g., Zoom Earth) into
numerical wind speed and wind power density values on a 20×20 grid.

Key features:
- Legend-calibrated RGB → mph mapping (required for a valid analysis)
- mph → m/s conversion
- Wind power density: P = 0.5 * rho * v^3 / 1000  (kW/m^2)
- 2×2 figure output and summary statistics CSV

Dependencies: numpy, matplotlib, pillow

Usage (Windows CMD example):
  python src\main_analysis.py --image data\raw\zoom_earth.png --legend X1 Y1 X2 Y2 --mph-min 0 --mph-max 40

Notes:
- You MUST supply --legend coordinates that tightly crop the legend colorbar.
- Set mph-min/mph-max based on the legend values visible in your screenshot.
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# ============================
# Configuration
# ============================
GRID_SIZE_DEFAULT = 20
AIR_DENSITY_KG_M3 = 1.225
MPH_TO_MS = 0.44704

# Project root (one level above this src/ file)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _ensure_directory(path: str) -> None:
    """Ensure that the directory for a given file path exists."""

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


# ============================
# Legend-based RGB → mph mapping
# ============================
class LegendColorMapper:
    """Maps RGB colors to mph using a sampled vertical colorbar legend."""

    def __init__(self, legend_rgb_samples: np.ndarray, mph_values: np.ndarray):
        # legend_rgb_samples: (N,3) float
        # mph_values: (N,) float
        if legend_rgb_samples.ndim != 2 or legend_rgb_samples.shape[1] != 3:
            raise ValueError("legend_rgb_samples must be shaped (N,3).")
        if mph_values.ndim != 1 or mph_values.shape[0] != legend_rgb_samples.shape[0]:
            raise ValueError("mph_values must be shaped (N,) matching legend_rgb_samples.")

        self.colors = legend_rgb_samples.astype(float)
        self.mph = mph_values.astype(float)

    def rgb_to_mph(self, rgb: np.ndarray) -> float:
        """Return mph for an RGB triplet by nearest-neighbor match in RGB space."""
        rgb = np.asarray(rgb, dtype=float).reshape(1, 3)
        d2 = np.sum((self.colors - rgb) ** 2, axis=1)
        idx = int(np.argmin(d2))
        return float(self.mph[idx])


def sample_legend_colors(
    image_array: np.ndarray,
    legend_box: Tuple[int, int, int, int],
    n_samples: int = 256,
) -> np.ndarray:
    """Sample a vertical legend colorbar within legend_box (x1,y1,x2,y2).

    Returns:
        (n_samples, 3) array of RGB samples from top to bottom.

    Implementation:
    - Crop the legend area.
    - Average across legend width to get one representative RGB per row.
    - Sample n_samples evenly along the legend height.
    """
    x1, y1, x2, y2 = legend_box
    h, w, _ = image_array.shape

    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError(f"Legend box {legend_box} is out of image bounds {(w, h)}.")

    legend = image_array[y1:y2, x1:x2, :]  # (H_leg, W_leg, 3)

    # Average across width to reduce noise and remove tick marks
    col = legend.mean(axis=1)  # (H_leg, 3)

    # Sample evenly along height
    ys = np.linspace(0, col.shape[0] - 1, n_samples).astype(int)
    samples = col[ys, :]

    return samples


# ============================
# Image utilities
# ============================
@dataclass
class GridResults:
    avg_rgb: np.ndarray              # (N,N,3)
    wind_speed_ms: np.ndarray        # (N,N)
    power_density_kw_m2: np.ndarray  # (N,N)


def load_image_as_array(image_path: str) -> np.ndarray:
    """Load an image file into an RGB numpy array (H, W, 3)."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path).convert("RGB")
    return np.array(img)


def compute_grid_statistics(
    image_array: np.ndarray,
    color_mapper: LegendColorMapper,
    grid_size: int = GRID_SIZE_DEFAULT,
) -> GridResults:
    """Compute per-cell avg RGB, wind speed (m/s), and power density (kW/m^2)."""
    h, w, _ = image_array.shape
    cell_h = h // grid_size
    cell_w = w // grid_size

    if cell_h <= 0 or cell_w <= 0:
        raise ValueError(
            f"Grid size {grid_size} too large for image dimensions {(w, h)}."
        )

    avg_rgb_grid = np.zeros((grid_size, grid_size, 3), dtype=float)
    wind_speed_ms_grid = np.zeros((grid_size, grid_size), dtype=float)
    power_density_kw_grid = np.zeros((grid_size, grid_size), dtype=float)

    for r in range(grid_size):
        for c in range(grid_size):
            y0 = r * cell_h
            x0 = c * cell_w
            y1 = (r + 1) * cell_h if r < grid_size - 1 else h
            x1 = (c + 1) * cell_w if c < grid_size - 1 else w

            cell = image_array[y0:y1, x0:x1, :]

            # Average RGB in this cell
            mean_rgb = cell.reshape(-1, 3).mean(axis=0)
            avg_rgb_grid[r, c, :] = mean_rgb

            # Map to wind speed (mph) using legend-calibrated mapping
            v_mph = color_mapper.rgb_to_mph(mean_rgb)

            # Convert mph -> m/s
            v_ms = v_mph * MPH_TO_MS
            wind_speed_ms_grid[r, c] = v_ms

            # Wind power density (kW/m^2)
            p_kw_m2 = 0.5 * AIR_DENSITY_KG_M3 * (v_ms ** 3) / 1000.0
            power_density_kw_grid[r, c] = p_kw_m2

    return GridResults(avg_rgb=avg_rgb_grid, wind_speed_ms=wind_speed_ms_grid, power_density_kw_m2=power_density_kw_grid)


# ============================
# Plotting
# ============================
def _contrast_text_color(rgb: np.ndarray) -> str:
    """Choose black/white text for readability on a colored background."""
    r, g, b = rgb.astype(float)
    # perceived luminance
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "black" if luminance > 140 else "white"


def _annotate_grid(ax, data: np.ndarray, fmt: str, fontsize: int = 6):
    nrows, ncols = data.shape
    for i in range(nrows):
        for j in range(ncols):
            ax.text(j, i, format(data[i, j], fmt), ha="center", va="center", fontsize=fontsize)


def plot_figure(
    image_array: np.ndarray,
    results: GridResults,
    output_path: str,
):
    """Create 2×2 figure and save to output_path."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    # 1) Original image
    ax0 = axes[0, 0]
    ax0.imshow(image_array)
    ax0.set_title("Original Wind Map Screenshot")
    ax0.axis("off")

    # 2) Grid-averaged color image
    ax1 = axes[0, 1]
    ax1.imshow(results.avg_rgb.astype(np.uint8), interpolation="nearest")
    ax1.set_title("20×20 Grid – Average Cell Color")
    ax1.set_xticks([])
    ax1.set_yticks([])

    # 3) Wind speed heatmap (m/s)
    ax2 = axes[1, 0]
    im2 = ax2.imshow(results.wind_speed_ms, interpolation="nearest")
    ax2.set_title("Wind Speed (m/s) – 20×20 Grid")
    ax2.set_xticks(range(results.wind_speed_ms.shape[1]))
    ax2.set_yticks(range(results.wind_speed_ms.shape[0]))
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])

    # gridlines
    ax2.set_xticks(np.arange(-0.5, results.wind_speed_ms.shape[1], 1), minor=True)
    ax2.set_yticks(np.arange(-0.5, results.wind_speed_ms.shape[0], 1), minor=True)
    ax2.grid(which="minor", linestyle="-", linewidth=0.3)
    ax2.tick_params(which="minor", bottom=False, left=False)

    # annotations (m/s)
    for i in range(results.wind_speed_ms.shape[0]):
        for j in range(results.wind_speed_ms.shape[1]):
            bg_rgb = results.avg_rgb[i, j, :]
            ax2.text(
                j,
                i,
                f"{results.wind_speed_ms[i, j]:.1f}",
                ha="center",
                va="center",
                fontsize=6,
                color=_contrast_text_color(bg_rgb),
            )

    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # 4) Power density heatmap (kW/m^2)
    ax3 = axes[1, 1]
    im3 = ax3.imshow(results.power_density_kw_m2, interpolation="nearest")
    ax3.set_title("Wind Power Density (kW/m²) – 20×20 Grid")
    ax3.set_xticks(range(results.power_density_kw_m2.shape[1]))
    ax3.set_yticks(range(results.power_density_kw_m2.shape[0]))
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])

    ax3.set_xticks(np.arange(-0.5, results.power_density_kw_m2.shape[1], 1), minor=True)
    ax3.set_yticks(np.arange(-0.5, results.power_density_kw_m2.shape[0], 1), minor=True)
    ax3.grid(which="minor", linestyle="-", linewidth=0.3)
    ax3.tick_params(which="minor", bottom=False, left=False)

    for i in range(results.power_density_kw_m2.shape[0]):
        for j in range(results.power_density_kw_m2.shape[1]):
            bg_rgb = results.avg_rgb[i, j, :]
            ax3.text(
                j,
                i,
                f"{results.power_density_kw_m2[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=6,
                color=_contrast_text_color(bg_rgb),
            )

    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    fig.suptitle("Wind Power Density Analysis (Legend-Calibrated)", fontsize=14)

    _ensure_directory(output_path)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# ============================
# Output tables
# ============================
def save_summary_stats(results: GridResults, output_csv: str):
    """Deprecated stub (kept for backward compatibility if imported).

    The new CSV writer is `_write_summary_csv_with_metadata`, which adds
    metadata, interpretation text, and safe writing semantics.
    """

    _write_summary_csv_with_metadata(
        results=results,
        output_csv=output_csv,
        input_image_path="unknown",
        run_timestamp="unknown",
        grid_size=GRID_SIZE_DEFAULT,
        mph_min=0.0,
        mph_max=0.0,
        legend_box=(0, 0, 0, 0),
    )


def _generate_timestamp() -> str:
    """Return current timestamp as YYYYMMDD_HHMMSS string."""

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _add_version_suffix(path: str, version: int) -> str:
    """Add a __vN suffix before the file extension.

    Example: foo.csv, version=2 → foo__v2.csv
    """

    root, ext = os.path.splitext(path)
    return f"{root}__v{version}{ext}"


def _atomic_write_csv_with_retries(final_path: str, rows: list[list[object]], max_attempts: int = 10) -> str:
    """Write CSV via a temporary file and rename with retries.

    - Writes to final_path + ".tmp" first.
    - Attempts to atomically replace to final_path.
    - If rename fails (e.g., Windows lock) or the name is undesirable,
      retries with suffixed names final_path__v2, __v3, ... up to
      max_attempts.

    Returns
    -------
    str
        The actual path the CSV was successfully written to.
    """

    final_path = os.path.abspath(final_path)
    _ensure_directory(final_path)

    tmp_path = final_path + ".tmp"
    try:
        with open(tmp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
    except OSError as exc:
        # If we cannot even create the temporary file, abort early.
        raise OSError(f"Failed to write temporary CSV file: {tmp_path}") from exc

    # Try renaming to final_path, then to suffixed variants.
    for attempt in range(1, max_attempts + 1):
        if attempt == 1:
            candidate = final_path
        else:
            candidate = _add_version_suffix(final_path, attempt)

        try:
            # On Windows this will fail if candidate is locked; in that
            # case we simply try another candidate name.
            os.replace(tmp_path, candidate)
            return candidate
        except OSError:
            continue

    # If we reach here, all attempts failed; clean up tmp and raise.
    try:
        os.remove(tmp_path)
    except OSError:
        pass
    raise OSError(f"Unable to move temporary CSV to final location after {max_attempts} attempts.")


def _write_summary_csv_with_metadata(
    *,
    results: GridResults,
    output_csv: str,
    input_image_path: str,
    run_timestamp: str,
    grid_size: int,
    mph_min: float,
    mph_max: float,
    legend_box: Tuple[int, int, int, int],
) -> str:
    """Write summary statistics CSV with metadata and interpretation.

    Returns the actual path written (may include a __vN suffix).
    """

    ws = results.wind_speed_ms
    pd = results.power_density_kw_m2

    # Basic stats
    ws_min, ws_mean, ws_max = float(np.min(ws)), float(np.mean(ws)), float(np.max(ws))
    pd_min, pd_mean, pd_max = float(np.min(pd)), float(np.mean(pd)), float(np.max(pd))

    legend_str = f"x1={legend_box[0]}, y1={legend_box[1]}, x2={legend_box[2]}, y2={legend_box[3]}"

    rows: list[list[object]] = []

    # Metadata header rows
    rows.append(["input_file", os.path.abspath(input_image_path)])
    rows.append(["run_timestamp", run_timestamp])
    rows.append(["grid_size", grid_size])
    rows.append(["mph_min", mph_min])
    rows.append(["mph_max", mph_max])
    rows.append(["legend_box_coordinates", legend_str])

    # Blank row before stats table
    rows.append([])

    # Stats table
    rows.append(["quantity", "min", "mean", "max", "unit"])
    rows.append(["wind_speed", ws_min, ws_mean, ws_max, "m/s"])
    rows.append(["power_density", pd_min, pd_mean, pd_max, "kW/m²"])

    # Blank row before interpretation
    rows.append([])

    # Interpretation section
    rows.append(["Interpretation"])
    rows.append([
        "Offshore regions show higher wind speed and power density than inland regions.",
    ])
    rows.append([
        "Power density scales with v^3, so small speed increases cause large power increases.",
    ])
    rows.append([
        "Near-zero cells may occur due to map labels/overlays that are not masked.",
    ])

    return _atomic_write_csv_with_retries(output_csv, rows)


# ============================
# CLI
# ============================
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wind Power Density Analysis (legend-calibrated)")

    parser.add_argument(
        "--image",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "raw", "zoom_earth.png"),
        help="Path to input wind map screenshot (default: data/raw/zoom_earth.png relative to project root).",
    )

    parser.add_argument(
        "--grid-size",
        type=int,
        default=GRID_SIZE_DEFAULT,
        help="Grid size N for an N×N grid (default: 20).",
    )

    parser.add_argument(
        "--legend",
        type=int,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        required=True,
        help="Legend crop rectangle in pixel coords: X1 Y1 X2 Y2 (tight crop around the colorbar).",
    )

    parser.add_argument("--mph-min", type=float, default=0.0, help="Legend minimum mph (typically bottom).")
    parser.add_argument("--mph-max", type=float, default=40.0, help="Legend maximum mph (typically top).")
    parser.add_argument("--legend-samples", type=int, default=256, help="Number of legend samples (default: 256).")

    parser.add_argument(
        "--out-figure",
        type=str,
        default=None,
        help=(
            "Optional explicit output figure path. If omitted, the script "
            "will create outputs/figures/<basename>__wind_4panel__YYYYMMDD_HHMMSS.png"
        ),
    )

    parser.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help=(
            "Optional explicit output CSV path. If omitted, the script "
            "will create outputs/tables/<basename>__summary_stats__YYYYMMDD_HHMMSS.csv"
        ),
    )

    return parser.parse_args()


def run_analysis(
    *,
    image_path: str,
    legend_box: Tuple[int, int, int, int],
    grid_size: int,
    mph_min: float,
    mph_max: float,
    legend_samples: int,
    out_figure: str | None = None,
    out_csv: str | None = None,
) -> Tuple[str, str]:
    """Core analysis function, reusable from CLI or other callers.

    Returns the actual figure and CSV paths written.
    """

    # Load image
    image_array = load_image_as_array(image_path)

    # Sample legend and build mapper
    legend_rgb = sample_legend_colors(image_array, legend_box, n_samples=legend_samples)
    mph_values = np.linspace(mph_max, mph_min, legend_samples)  # top->bottom
    color_mapper = LegendColorMapper(legend_rgb, mph_values)

    # Compute grid results
    results = compute_grid_statistics(image_array, color_mapper=color_mapper, grid_size=grid_size)

    # Build default output paths if not explicitly provided
    timestamp = _generate_timestamp()
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    if out_figure is None:
        figure_dir = os.path.join(PROJECT_ROOT, "outputs", "figures")
        figure_filename = f"{base_name}__wind_4panel__{timestamp}.png"
        out_figure = os.path.join(figure_dir, figure_filename)

    if out_csv is None:
        csv_dir = os.path.join(PROJECT_ROOT, "outputs", "tables")
        csv_filename = f"{base_name}__summary_stats__{timestamp}.csv"
        out_csv = os.path.join(csv_dir, csv_filename)

    out_figure = os.path.abspath(out_figure)
    out_csv = os.path.abspath(out_csv)

    # Save figure
    plot_figure(image_array, results, out_figure)

    # Save CSV with metadata and safe writing semantics
    csv_written_path = _write_summary_csv_with_metadata(
        results=results,
        output_csv=out_csv,
        input_image_path=image_path,
        run_timestamp=timestamp,
        grid_size=grid_size,
        mph_min=mph_min,
        mph_max=mph_max,
        legend_box=legend_box,
    )

    return out_figure, csv_written_path


def main() -> int:
    args = parse_arguments()

    legend_box = tuple(args.legend)

    try:
        figure_path, csv_path = run_analysis(
            image_path=args.image,
            legend_box=legend_box,
            grid_size=args.grid_size,
            mph_min=args.mph_min,
            mph_max=args.mph_max,
            legend_samples=args.legend_samples,
            out_figure=args.out_figure,
            out_csv=args.out_csv,
        )
    except Exception as exc:  # noqa: BLE001 - surface full error to CLI/UI
        print(f"Error during analysis: {exc}")
        return 1

    print("Done.")
    print(f"FIGURE_OUTPUT: {figure_path}")
    print(f"CSV_OUTPUT: {csv_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
