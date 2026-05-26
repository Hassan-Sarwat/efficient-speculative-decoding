"""analyze_results.py — consolidate benchmark metrics → CSV and comparison charts.

Usage:
    python analyze_results.py              # reads/writes outputs/
    python analyze_results.py --dir outputs_recent
"""
import argparse
import json
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--dir", default="outputs", help="Folder containing metrics_*.json files")
args = parser.parse_args()

OUTPUTS = Path(args.dir)
SCENARIO_ORDER = ["easy", "medium", "hard"]
TYPE_ORDER = ["base", "cod", "cot"]
TYPE_LABELS = {
    "base": "Base (Untrained)",
    "cod": "Chain of Draft (CoD)",
    "cot": "Chain of Thought (CoT)",
}
SCENARIO_LABELS = {
    "easy": "Easy\n(GSM8K)",
    "medium": "Medium\n(MATH L1-2)",
    "hard": "Hard\n(MATH L3-4)",
}
COLORS = {"base": "#95a5a6", "cod": "#27ae60", "cot": "#2980b9"}
HATCH = "///"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
})


def lookup(df, type_, scenario, speculative, col):
    rows = df[
        (df["type"] == type_)
        & (df["scenario"] == scenario)
        & (df["speculative"] == speculative)
    ]
    return float(rows[col].values[0]) if len(rows) else np.nan


def panel_not_measured(ax, title):
    """Show a placeholder when a metric wasn't collected in this run."""
    ax.set_title(title)
    ax.text(0.5, 0.5, "Not measured\nin this run",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=13, color="#aaaaaa", style="italic")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_facecolor("#f8f9fa")


def is_all_zero(data_list):
    return all(
        all(v == 0 or np.isnan(v) for v in series)
        for series in data_list
    )


def draw_grouped_bars(
    ax, data, x_labels, series_labels, colors, hatches=None,
    bar_width=0.78, annotate=False, fmt=".2f",
):
    """
    data: list of arrays (one per series), each len == len(x_labels).
    Returns flat list of bar containers.
    """
    n_s = len(series_labels)
    bw = bar_width / n_s
    x = np.arange(len(x_labels))
    all_rects = []
    for i, (label, vals) in enumerate(zip(series_labels, data)):
        offsets = x + (i - n_s / 2 + 0.5) * bw
        hatch = hatches[i] if hatches else ""
        rects = ax.bar(
            offsets, vals, width=bw * 0.92,
            color=colors[i], hatch=hatch, alpha=0.88,
            edgecolor="white", linewidth=0.8, label=label,
        )
        all_rects.append(rects)
        if annotate:
            for rect, v in zip(rects, vals):
                if np.isfinite(v) and v != 0:
                    ax.annotate(
                        f"{v:{fmt}}",
                        xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7,
                    )
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.yaxis.grid(True, alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)
    return all_rects


# ── 1. Parse JSON → DataFrame ──────────────────────────────────────────────
PATTERN = re.compile(r"metrics_(base|cod|cot)_(easy|medium|hard)_benchmark\.json$")
rows = []
for fp in sorted(OUTPUTS.glob("metrics_*_benchmark.json")):
    m = PATTERN.match(fp.name)
    if not m:
        continue
    typ, scenario = m.group(1), m.group(2)
    data = json.loads(fp.read_text())
    for key in ("baseline", "speculative"):
        if key not in data:
            continue
        e = data[key]
        rows.append({
            "type": typ,
            "speculative": 1 if key == "speculative" else 0,
            "scenario": scenario,
            **{k: e[k] for k in [
                "avg_tokens_generated", "acceptance_rate", "accuracy",
                "total_samples", "peak_vram_gb", "tokens_per_second",
                "avg_ttft_ms", "avg_itl_ms", "total_duration_sec",
                "system_efficiency",
            ]},
        })

df = pd.DataFrame(rows)
csv_path = OUTPUTS / "all_metrics.csv"
df.to_csv(csv_path, index=False)
print(f"CSV written to {csv_path} ({len(df)} rows)")

# ── Precompute label lists used in multiple charts ─────────────────────────
scen_lbls = [SCENARIO_LABELS[s] for s in SCENARIO_ORDER]
trained = ["cod", "cot"]

# 4 series: cod-baseline, cod-speculative, cot-baseline, cot-speculative
spec_series_labels = []
spec_series_colors = []
spec_series_hatches = []
for t in trained:
    spec_series_labels.append(f"{TYPE_LABELS[t]} Baseline")
    spec_series_colors.append(COLORS[t])
    spec_series_hatches.append("")
    spec_series_labels.append(f"{TYPE_LABELS[t]} + Spec. Decoding")
    spec_series_colors.append(COLORS[t])
    spec_series_hatches.append(HATCH)


def spec_series(col, multiplier=1.0):
    """Build data list for the 4 spec-comparison series."""
    out = []
    for t in trained:
        out.append([lookup(df, t, s, 0, col) * multiplier for s in SCENARIO_ORDER])
        out.append([lookup(df, t, s, 1, col) * multiplier for s in SCENARIO_ORDER])
    return out


# ── 2. Chart 1: 2×3 Comparison Dashboard ──────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
run_label = OUTPUTS.name.replace("_", " ").title()
fig.suptitle(
    f"CoT vs CoD vs Base — Benchmark Comparison Dashboard ({run_label})",
    fontsize=17, fontweight="bold", y=0.995,
)

# [0,0] Accuracy — baseline, all three model types
ax = axes[0, 0]
data_acc = [[lookup(df, t, s, 0, "accuracy") for s in SCENARIO_ORDER] for t in TYPE_ORDER]
clrs_all = [COLORS[t] for t in TYPE_ORDER]
lbls_all = [TYPE_LABELS[t] for t in TYPE_ORDER]
draw_grouped_bars(ax, data_acc, scen_lbls, lbls_all, clrs_all, annotate=True, fmt=".3f")
ax.set_title("Accuracy (Baseline)")
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1.15)
ax.legend(loc="upper right", framealpha=0.95)

# [0,1] Avg tokens / response — baseline, all three types
ax = axes[0, 1]
data_tok = [[lookup(df, t, s, 0, "avg_tokens_generated") for s in SCENARIO_ORDER] for t in TYPE_ORDER]
draw_grouped_bars(ax, data_tok, scen_lbls, lbls_all, clrs_all, annotate=True, fmt=".0f")
ax.set_title("Avg Tokens per Response (Baseline)")
ax.set_ylabel("Tokens")
ax.legend(loc="upper left", framealpha=0.95)

# [0,2] Throughput (tokens/sec) — baseline vs speculative, cod & cot
ax = axes[0, 2]
draw_grouped_bars(
    ax, spec_series("tokens_per_second"), scen_lbls,
    spec_series_labels, spec_series_colors, hatches=spec_series_hatches,
    annotate=True, fmt=".0f",
)
ax.set_title("Throughput: Tokens/sec\n(Baseline vs Speculative Decoding)")
ax.set_ylabel("Tokens per Second")
ax.legend(loc="lower right", framealpha=0.95, ncol=1, fontsize=8)

# [1,0] TTFT — baseline vs speculative, cod & cot (converted to seconds)
ax = axes[1, 0]
_ttft_data = spec_series("avg_ttft_ms", multiplier=1 / 1000)
if is_all_zero(_ttft_data):
    panel_not_measured(ax, "Avg Time-to-First-Token\n(Baseline vs Speculative Decoding)")
else:
    draw_grouped_bars(ax, _ttft_data, scen_lbls, spec_series_labels,
                      spec_series_colors, hatches=spec_series_hatches, annotate=False)
    ax.set_title("Avg Time-to-First-Token\n(Baseline vs Speculative Decoding)")
    ax.set_ylabel("TTFT (seconds)")
    ax.legend(loc="upper left", framealpha=0.95, ncol=1, fontsize=8)

# [1,1] Inter-Token Latency — baseline vs speculative, cod & cot
ax = axes[1, 1]
_itl_data = spec_series("avg_itl_ms")
if is_all_zero(_itl_data):
    panel_not_measured(ax, "Avg Inter-Token Latency (ms)\n(Baseline vs Speculative Decoding)")
else:
    draw_grouped_bars(ax, _itl_data, scen_lbls, spec_series_labels,
                      spec_series_colors, hatches=spec_series_hatches,
                      annotate=True, fmt=".1f")
    ax.set_title("Avg Inter-Token Latency (ms)\n(Baseline vs Speculative Decoding)")
    ax.set_ylabel("ITL (ms)")
    ax.legend(loc="upper right", framealpha=0.95, ncol=1, fontsize=8)

# [1,2] Accuracy stability — does speculative decoding change accuracy?
ax = axes[1, 2]
draw_grouped_bars(
    ax, spec_series("accuracy"), scen_lbls,
    spec_series_labels, spec_series_colors, hatches=spec_series_hatches,
    annotate=True, fmt=".3f", bar_width=0.75,
)
ax.set_title("Accuracy: Baseline vs Speculative\n(Does Spec. Decoding Change Quality?)")
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1.15)
ax.legend(loc="upper right", framealpha=0.95, ncol=1, fontsize=8)

# Add a legend note for hatching
hatch_patch = plt.Rectangle((0, 0), 1, 1, fc="gray", hatch=HATCH, alpha=0.5)
solid_patch = plt.Rectangle((0, 0), 1, 1, fc="gray", alpha=0.88)
fig.legend(
    [solid_patch, hatch_patch], ["Baseline (no spec. decoding)", "With Speculative Decoding"],
    loc="lower center", ncol=2, fontsize=10, framealpha=0.9,
    bbox_to_anchor=(0.5, -0.02),
)

plt.tight_layout(rect=[0, 0.03, 1, 0.99])
chart1_path = OUTPUTS / "comparison_charts.png"
fig.savefig(chart1_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Chart saved to {chart1_path}")


# ── 3. Chart 2: Speculative Decoding Deep-Dive ─────────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(17, 5.5))
fig2.suptitle(
    f"Speculative Decoding Deep-Dive: CoD vs CoT ({run_label})",
    fontsize=15, fontweight="bold", y=1.03,
)

acc_lbls = [TYPE_LABELS[t] for t in trained]
acc_clrs = [COLORS[t] for t in trained]

# [0] Acceptance rate (speculative runs only)
ax = axes2[0]
ar_data = [
    [lookup(df, t, s, 1, "acceptance_rate") for s in SCENARIO_ORDER]
    for t in trained
]
draw_grouped_bars(ax, ar_data, scen_lbls, acc_lbls, acc_clrs, annotate=True, fmt=".3f")
ax.set_title("Draft Token Acceptance Rate")
ax.set_ylabel("Acceptance Rate")
ax.set_ylim(0, 1.05)
ax.legend(framealpha=0.95)

# [1] Throughput speedup: speculative / baseline
ax = axes2[1]
speedup_data = []
for t in trained:
    speedups = []
    for s in SCENARIO_ORDER:
        base_tps = lookup(df, t, s, 0, "tokens_per_second")
        spec_tps = lookup(df, t, s, 1, "tokens_per_second")
        speedups.append(spec_tps / base_tps if np.isfinite(base_tps) and base_tps > 0 else np.nan)
    speedup_data.append(speedups)

draw_grouped_bars(ax, speedup_data, scen_lbls, acc_lbls, acc_clrs, annotate=True, fmt=".2f")
ax.axhline(1.0, color="#e74c3c", linestyle="--", linewidth=1.5, alpha=0.8, label="1× (no speedup)")
ax.set_title("Throughput Speedup\n(Speculative ÷ Baseline tokens/sec)")
ax.set_ylabel("Speedup Ratio (×)")
ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
handles, lbls_ = ax.get_legend_handles_labels()
ax.legend(handles, lbls_, framealpha=0.95)

# [2] System efficiency (speculative runs)
ax = axes2[2]
eff_data = [
    [lookup(df, t, s, 1, "system_efficiency") for s in SCENARIO_ORDER]
    for t in trained
]
draw_grouped_bars(ax, eff_data, scen_lbls, acc_lbls, acc_clrs, annotate=True, fmt=".3f")
ax.set_title("System Efficiency\n(Speculative Runs)")
ax.set_ylabel("System Efficiency")
ax.set_ylim(0, max(max(s) for s in eff_data if s) * 1.2)
ax.legend(framealpha=0.95)

plt.tight_layout()
chart2_path = OUTPUTS / "speculative_analysis.png"
fig2.savefig(chart2_path, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"Chart saved to {chart2_path}")
