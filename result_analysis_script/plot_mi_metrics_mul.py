#!/usr/bin/env python3
"""Generate multiplication MI plots from a ``mi_metrics_mul.csv`` file.

Usage:
  python result_analysis_script/plot_mi_metrics_mul.py /path/to/mi_metrics_mul.csv

This script is analogous to ``plot_mi_metrics.py`` (addition), but it parses
place-indexed multiplication columns dynamically, e.g.:
  - mi_mul/p0-z, mi_mul/p0-z-base
  - mi_mul/p0-carries, mi_mul/p0-carries-base
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import FuncFormatter


FIGSIZE = (9, 5)
LINEWIDTH = 2.0
ALPHA_BASE = 0.55
LEGEND_FONTSIZE = 10
LABEL_FONTSIZE = 12
TICK_FONTSIZE = 11
MAX_X_DISPLAY = 200_000
SAVE_KW = dict(bbox_inches="tight", pad_inches=0.02)
HGRID_KW = dict(axis="y", linestyle=":", linewidth=0.5, alpha=0.18, color="gray")


def k_formatter(x_val, pos=None):
    if x_val == 0:
        return "0"
    k = x_val / 1000.0
    if abs(k - round(k)) < 1e-6:
        return f"{int(round(k))}K"
    return f"{k:.1f}K"


def parse_args():
    p = argparse.ArgumentParser(description="Plot multiplication MI metrics from a CSV.")
    p.add_argument("csv_path", type=Path, help="Path to mi_metrics_mul.csv")
    p.add_argument("--out-z", type=Path, default=None, help="Output PDF path for MI conditioned on z.")
    p.add_argument("--out-c", type=Path, default=None, help="Output PDF path for MI conditioned on carries.")
    p.add_argument("--out-heat", type=Path, default=None, help="Output PDF path for delta heatmap.")
    p.add_argument(
        "--places",
        type=str,
        default=None,
        help="Comma-separated place ids to plot (e.g. '0,1,2,5,10,20,30,40').",
    )
    p.add_argument("--plot-all", action="store_true", help="Plot all available places in line charts.")
    p.add_argument("--max-x", type=int, default=MAX_X_DISPLAY, help="Maximum x-axis display for line plots.")
    return p.parse_args()


def detect_places(df: pd.DataFrame):
    pat = re.compile(r"^mi_mul/p(\d+)-(z|carries)(-base)?$")
    places = set()
    for col in df.columns:
        m = pat.match(col)
        if m:
            places.add(int(m.group(1)))
    return sorted(places)


def choose_places(all_places, places_arg=None, plot_all=False):
    if not all_places:
        return []
    if plot_all:
        return all_places
    if places_arg:
        wanted = []
        for chunk in places_arg.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            wanted.append(int(chunk))
        return [p for p in wanted if p in all_places]

    # default informative subset
    anchors = [0, 1, 2, 5, 10, 20, 30]
    last = all_places[-1]
    chosen = sorted(set([p for p in anchors if p in all_places] + [last]))
    return chosen


def get_col(df: pd.DataFrame, col: str):
    if col not in df.columns:
        return None
    return df[col].to_numpy()


def plot_two_panel(df, x, places, mode, out_path, max_x):
    """mode is 'z' or 'carries'."""
    fig, axs = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    ax_top, ax_bottom = axs

    cmap = plt.get_cmap("tab20")
    base_handles, model_handles, labels_base, labels_model = [], [], [], []

    for i, p in enumerate(places):
        color = cmap(i % cmap.N)
        base_col = f"mi_mul/p{p}-{mode}-base"
        model_col = f"mi_mul/p{p}-{mode}"
        y_base = get_col(df, base_col)
        y_model = get_col(df, model_col)
        if y_base is None or y_model is None:
            continue

        h_base, = ax_top.plot(x, y_base, linestyle="--", alpha=ALPHA_BASE, linewidth=LINEWIDTH, color=color)
        h_model, = ax_top.plot(x, y_model, linewidth=LINEWIDTH, color=color)
        base_handles.append(h_base)
        model_handles.append(h_model)
        labels_base.append(f"p{p} (data)")
        labels_model.append(f"p{p} (model)")

    ax_top.set_ylabel("Mutual information", fontsize=LABEL_FONTSIZE)
    ax_top.grid(**HGRID_KW)
    ax_top.relim()
    ax_top.autoscale_view()
    ymin, ymax = ax_top.get_ylim()
    ax_top.set_ylim(max(0.0, ymin), ymax)

    if base_handles or model_handles:
        ncols = 2 if len(labels_base) <= 8 else 3
        ax_top.legend(
            base_handles + model_handles,
            labels_base + labels_model,
            fontsize=LEGEND_FONTSIZE,
            loc="upper center",
            ncol=ncols,
            bbox_to_anchor=(0.5, 0.98),
            frameon=True,
            framealpha=0.92,
            fancybox=True,
            borderaxespad=0.35,
        )

    y_loss = get_col(df, "train_loss")
    if y_loss is not None:
        ax_bottom.plot(x, y_loss, linewidth=LINEWIDTH)
    ax_bottom.set_ylabel("Train loss", fontsize=LABEL_FONTSIZE)
    ax_bottom.set_xlabel("Training steps", fontsize=LABEL_FONTSIZE)
    ax_bottom.grid(**HGRID_KW)
    x_max_data = float(np.nanmax(x)) if len(x) > 0 else 0.0
    x_upper = max_x if max_x is not None else x_max_data
    if x_max_data > 0:
        x_upper = min(float(x_upper), x_max_data)
    if x_upper <= 0:
        x_upper = max_x if max_x is not None else 1.0
    ax_bottom.set_xlim(0, x_upper)
    ax_bottom.xaxis.set_major_formatter(FuncFormatter(k_formatter))

    ax_top.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)
    ax_bottom.tick_params(axis="both", which="major", labelsize=TICK_FONTSIZE)

    title = "MI conditioned on z" if mode == "z" else "MI conditioned on carries"
    fig.suptitle(title, fontsize=LABEL_FONTSIZE + 1)
    fig.subplots_adjust(bottom=0.09, top=0.91, hspace=0.12)
    fig.savefig(str(out_path), **SAVE_KW)
    plt.close(fig)
    print(f"Saved {mode} figure to {out_path}")


def plot_delta_heatmap(df, x, places, out_path):
    """Plot model-data delta MI heatmaps for z and carries."""
    if not places:
        print("No places detected; skipping heatmap.")
        return

    mat_z = []
    mat_c = []
    usable_places = []
    for p in places:
        z = get_col(df, f"mi_mul/p{p}-z")
        z0 = get_col(df, f"mi_mul/p{p}-z-base")
        c = get_col(df, f"mi_mul/p{p}-carries")
        c0 = get_col(df, f"mi_mul/p{p}-carries-base")
        if z is None or z0 is None or c is None or c0 is None:
            continue
        usable_places.append(p)
        mat_z.append(z - z0)
        mat_c.append(c - c0)

    if not usable_places:
        print("No complete place columns for heatmap; skipping.")
        return

    mat_z = np.vstack(mat_z)
    mat_c = np.vstack(mat_c)

    vmax = float(max(np.nanmax(np.abs(mat_z)), np.nanmax(np.abs(mat_c))))
    vmax = max(vmax, 1e-6)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    im0 = axs[0].imshow(mat_z, aspect="auto", cmap="coolwarm", norm=norm, interpolation="nearest")
    axs[0].set_title("ΔMI = model - data (conditioned on z)")
    axs[0].set_yticks(np.arange(len(usable_places)))
    axs[0].set_yticklabels([f"p{p}" for p in usable_places])

    im1 = axs[1].imshow(mat_c, aspect="auto", cmap="coolwarm", norm=norm, interpolation="nearest")
    axs[1].set_title("ΔMI = model - data (conditioned on carries)")
    axs[1].set_yticks(np.arange(len(usable_places)))
    axs[1].set_yticklabels([f"p{p}" for p in usable_places])

    # mark x ticks by actual step values
    n = len(x)
    tick_ids = np.linspace(0, max(0, n - 1), num=min(8, n), dtype=int)
    axs[1].set_xticks(tick_ids)
    axs[1].set_xticklabels([k_formatter(x[i]) for i in tick_ids])
    axs[1].set_xlabel("Training steps")

    cbar = fig.colorbar(im1, ax=axs, shrink=0.96, pad=0.01)
    cbar.set_label("MI difference")

    fig.tight_layout()
    fig.savefig(str(out_path), **SAVE_KW)
    plt.close(fig)
    print(f"Saved heatmap to {out_path}")


def main():
    args = parse_args()
    csv_path = args.csv_path.expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    out_z = args.out_z.expanduser().resolve() if args.out_z else (csv_path.parent / "mi_mul_conditioned_on_z.pdf")
    out_c = args.out_c.expanduser().resolve() if args.out_c else (csv_path.parent / "mi_mul_conditioned_on_carries.pdf")
    out_h = args.out_heat.expanduser().resolve() if args.out_heat else (csv_path.parent / "mi_mul_delta_heatmap.pdf")

    df = pd.read_csv(csv_path)
    if "iter" not in df.columns:
        raise KeyError("Column 'iter' not found in CSV.")
    x = df["iter"].to_numpy()

    all_places = detect_places(df)
    if not all_places:
        raise ValueError("No multiplication MI columns found (expected columns like 'mi_mul/p0-z').")

    line_places = choose_places(all_places, places_arg=args.places, plot_all=args.plot_all)
    if not line_places:
        raise ValueError("No valid places selected to plot.")

    plot_two_panel(df, x, line_places, mode="z", out_path=out_z, max_x=args.max_x)
    plot_two_panel(df, x, line_places, mode="carries", out_path=out_c, max_x=args.max_x)
    plot_delta_heatmap(df, x, all_places, out_h)


if __name__ == "__main__":
    main()
