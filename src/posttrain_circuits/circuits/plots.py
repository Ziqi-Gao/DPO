"""Static circuit-analysis plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def write_attribution_patching_calibration(
    rows: list[dict[str, Any]],
    *,
    spearman: float,
    output_prefix: Path,
) -> dict[str, str]:
    if len(rows) < 2:
        raise ValueError("calibration plot requires at least two points")
    x = np.array([float(row["attribution_score"]) for row in rows])
    y = np.array([float(row["exact_patching_score"]) for row in rows])
    figure, axis = plt.subplots(
        figsize=(5.2, 4.2),
        constrained_layout=True,
    )
    axis.scatter(
        x,
        y,
        s=24,
        alpha=0.75,
        color="#0072B2",
        edgecolors="white",
        linewidths=0.4,
        rasterized=True,
    )
    if float(np.ptp(x)) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        domain = np.linspace(float(x.min()), float(x.max()), 100)
        axis.plot(
            domain,
            slope * domain + intercept,
            color="#D55E00",
            linewidth=1.5,
            label="Least-squares fit",
        )
        axis.legend(frameon=False)
    axis.axhline(0.0, color="0.65", linewidth=0.7)
    axis.axvline(0.0, color="0.65", linewidth=0.7)
    axis.set_xlabel("EAP-IG attribution score")
    axis.set_ylabel("Exact patching effect")
    axis.set_title(f"Attribution-patching calibration (Spearman rho={spearman:.3f})")
    axis.grid(True, alpha=0.2, linewidth=0.6)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png = output_prefix.with_suffix(".png")
    pdf = output_prefix.with_suffix(".pdf")
    figure.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(
        pdf,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)
    return {"png": str(png), "pdf": str(pdf)}
