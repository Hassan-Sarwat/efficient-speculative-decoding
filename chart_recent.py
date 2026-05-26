"""chart_recent.py — 6-panel focused chart for outputs_recent."""
import json
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUTPUTS = Path("outputs_recent")
SCENARIO_ORDER = ["easy", "medium", "hard"]
SCENARIO_LABELS = {
    "easy": "Easy\n(GSM8K)",
    "medium": "Medium\n(MATH L1-2)",
    "hard": "Hard\n(MATH L3-4)",
}
TRAINED = ["cod", "cot"]
TYPE_LABELS = {
    "cod": "Chain of Draft (CoD)",
    "cot": "Chain of Thought (CoT)",
}
C = {"cod": "#27ae60", "cot": "#2980b9"}
HATCH = "///"
K = 5  # num_speculative_tokens

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
})

# ── Parse all metrics files ───────────────────────────────────────────────────
PATTERN = re.compile(r"metrics_(cod|cot)_(easy|medium|hard)_benchmark\.json$")

# metrics[type][scenario][key] = value
metrics = {}
for fp in sorted(OUTPUTS.glob("metrics_*_benchmark.json")):
    m = PATTERN.match(fp.name)
    if not m:
        continue
    typ, scenario = m.group(1), m.group(2)
    data = json.loads(fp.read_text())
    if typ not in metrics:
        metrics[typ] = {}
    metrics[typ][scenario] = {
        "baseline": data.get("baseline", {}),
        "speculative": data.get("speculative", {}),
    }


def get(typ, scenario, mode, field, default=np.nan):
    try:
        return metrics[typ][scenario][mode][field]
    except (KeyError, TypeError):
        return default


scen_lbls = [SCENARIO_LABELS[s] for s in SCENARIO_ORDER]

# ── Helper: grouped bar plot ──────────────────────────────────────────────────
def grouped_bars(ax, data, x_labels, series_labels, colors, hatches=None,
                 bar_width=0.75, annotate=False, fmt=".2f"):
    n_s = len(series_labels)
    bw = bar_width / n_s
    x = np.arange(len(x_labels))
    for i, (label, vals) in enumerate(zip(series_labels, data)):
        offsets = x + (i - n_s / 2 + 0.5) * bw
        hatch = hatches[i] if hatches else ""
        rects = ax.bar(
            offsets, vals, width=bw * 0.92,
            color=colors[i], hatch=hatch, alpha=0.88,
            edgecolor="white", linewidth=0.8, label=label,
        )
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


# ── Build the figure ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(19, 11))
fig.suptitle(
    "Chain of Draft vs Chain of Thought — Speculative Decoding Analysis (Recent Run)",
    fontsize=15, fontweight="bold", y=0.995,
)

# ── [0,0] Accuracy: Baseline vs Speculative ──────────────────────────────────
ax = axes[0, 0]
series_4_labels = []
series_4_colors = []
series_4_hatches = []
for t in TRAINED:
    series_4_labels += [f"{TYPE_LABELS[t]} Baseline", f"{TYPE_LABELS[t]} + Spec."]
    series_4_colors += [C[t], C[t]]
    series_4_hatches += ["", HATCH]

data_acc = []
for t in TRAINED:
    data_acc.append([get(t, s, "baseline", "accuracy") for s in SCENARIO_ORDER])
    data_acc.append([get(t, s, "speculative", "accuracy") for s in SCENARIO_ORDER])

grouped_bars(ax, data_acc, scen_lbls, series_4_labels, series_4_colors,
             hatches=series_4_hatches, annotate=True, fmt=".3f")
ax.set_title("Accuracy: Baseline vs Speculative")
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1.15)
ax.legend(loc="lower left", framealpha=0.95, ncol=1)

# ── [0,1] Avg Tokens per Response (baseline only) ────────────────────────────
ax = axes[0, 1]
data_tok = [
    [get(t, s, "baseline", "avg_tokens_generated") for s in SCENARIO_ORDER]
    for t in TRAINED
]
grouped_bars(ax, data_tok, scen_lbls,
             [TYPE_LABELS[t] for t in TRAINED], [C[t] for t in TRAINED],
             annotate=True, fmt=".0f")
ax.set_title("Avg Tokens per Response (Baseline)")
ax.set_ylabel("Tokens")
ax.legend(framealpha=0.95)

# ── [0,2] Throughput: Baseline vs Speculative ────────────────────────────────
ax = axes[0, 2]
data_tps = []
for t in TRAINED:
    data_tps.append([get(t, s, "baseline", "tokens_per_second") for s in SCENARIO_ORDER])
    data_tps.append([get(t, s, "speculative", "tokens_per_second") for s in SCENARIO_ORDER])

grouped_bars(ax, data_tps, scen_lbls, series_4_labels, series_4_colors,
             hatches=series_4_hatches, annotate=True, fmt=".0f")
ax.set_title("Throughput: Tokens/sec\n(Baseline vs Speculative)")
ax.set_ylabel("Tokens per Second")
ax.legend(loc="upper left", framealpha=0.95, ncol=1)

# ── [1,0] Throughput Speedup ──────────────────────────────────────────────────
ax = axes[1, 0]
speedup_data = []
for t in TRAINED:
    vals = []
    for s in SCENARIO_ORDER:
        b = get(t, s, "baseline", "tokens_per_second")
        sp = get(t, s, "speculative", "tokens_per_second")
        vals.append(sp / b if np.isfinite(b) and b > 0 else np.nan)
    speedup_data.append(vals)

grouped_bars(ax, speedup_data, scen_lbls,
             [TYPE_LABELS[t] for t in TRAINED], [C[t] for t in TRAINED],
             annotate=True, fmt=".2f")
ax.axhline(1.0, color="#e74c3c", linestyle="--", linewidth=1.5, alpha=0.8,
           zorder=0, label="1× (no speedup)")
ax.set_title("Throughput Speedup\n(Speculative ÷ Baseline)")
ax.set_ylabel("Speedup Ratio (×)")
handles, lbls_ = ax.get_legend_handles_labels()
ax.legend(handles, lbls_, framealpha=0.95)

# ── [1,1] Draft Token Acceptance Rate ────────────────────────────────────────
ax = axes[1, 1]
ar_data = [
    [get(t, s, "speculative", "acceptance_rate") for s in SCENARIO_ORDER]
    for t in TRAINED
]
grouped_bars(ax, ar_data, scen_lbls,
             [TYPE_LABELS[t] for t in TRAINED], [C[t] for t in TRAINED],
             annotate=True, fmt=".3f")
ax.set_title("Draft Token Acceptance Rate\n(Per-Token, Speculative Runs)")
ax.set_ylabel("Acceptance Rate")
ax.set_ylim(0, 1.0)
ax.legend(framealpha=0.95)

# ── [1,2] Mean Tokens Accepted per Draft Round ───────────────────────────────
ax = axes[1, 2]
mean_accepted = []
for t in TRAINED:
    vals = []
    for s in SCENARIO_ORDER:
        n_acc = get(t, s, "speculative", "num_accepted_tokens", 0)
        n_drafts = get(t, s, "speculative", "num_drafts", 0)
        vals.append(n_acc / n_drafts if n_drafts > 0 else np.nan)
    mean_accepted.append(vals)

grouped_bars(ax, mean_accepted, scen_lbls,
             [TYPE_LABELS[t] for t in TRAINED], [C[t] for t in TRAINED],
             annotate=True, fmt=".2f")
ax.axhline(K, color="#e74c3c", linestyle="--", linewidth=1.5, alpha=0.8,
           zorder=0, label=f"k={K} (max possible)")
ax.set_title(f"Mean Tokens Accepted per Draft Round\n(k={K} speculative tokens, max={K})")
ax.set_ylabel("Avg Accepted Tokens / Round")
ax.set_ylim(0, K * 1.15)
handles, lbls_ = ax.get_legend_handles_labels()
ax.legend(handles, lbls_, framealpha=0.95)

# ── Global legend for hatch pattern ──────────────────────────────────────────
solid_patch = mpatches.Patch(fc="#888888", alpha=0.88, label="Baseline (no spec. decoding)")
hatch_patch = mpatches.Patch(fc="#888888", hatch=HATCH, alpha=0.5, label="With Speculative Decoding")
fig.legend(
    handles=[solid_patch, hatch_patch],
    loc="lower center", ncol=2, fontsize=10, framealpha=0.9,
    bbox_to_anchor=(0.5, -0.02),
)

plt.tight_layout(rect=[0, 0.03, 1, 0.99])
out_path = OUTPUTS / "focused_analysis.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Chart saved to {out_path}")
