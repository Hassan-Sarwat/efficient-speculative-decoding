"""chart_speedup_analysis.py — Full speed + quality breakdown: CoD+Spec vs CoT vs Base.

Panels:
  [0,0] Wall time per sample (all 5 configs)
  [0,1] Accuracy (all 5 configs)
  [0,2] Accuracy per second — quality-efficiency tradeoff
  [1,0] Avg tokens per response (verbosity / token efficiency)
  [1,1] Tokens per second (raw + spec-decode accelerated)
  [1,2] Draft token acceptance rate (spec runs only)

Prints a detailed text summary to stdout.
"""
import json
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUTPUTS = Path("outputs")
SCENARIO_ORDER = ["easy", "medium", "hard"]
SCENARIO_LABELS = {
    "easy": "Easy\n(GSM8K)",
    "medium": "Medium\n(MATH L1-2)",
    "hard": "Hard\n(MATH L3-4)",
}

# 5 configs: (type, mode, display_label, color, hatch)
CONFIGS = [
    ("base", "baseline",    "Base (Untrained)",       "#95a5a6", ""),
    ("cod",  "baseline",    "CoD Baseline",            "#a8d5b5", ""),
    ("cot",  "baseline",    "CoT Baseline",            "#a8c8e8", ""),
    ("cod",  "speculative", "CoD + Spec. Decoding",    "#27ae60", "///"),
    ("cot",  "speculative", "CoT + Spec. Decoding",    "#2980b9", "///"),
]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 7.5,
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
})

# ── Parse metrics files ───────────────────────────────────────────────────────
PATTERN = re.compile(r"metrics_(base|cod|cot)_(easy|medium|hard)_benchmark\.json$")
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
        "baseline":    data.get("baseline", {}),
        "speculative": data.get("speculative", {}),
    }


def get(typ, scenario, mode, field, default=np.nan):
    try:
        return metrics[typ][scenario][mode][field]
    except (KeyError, TypeError):
        return default


def sec_per_sample(typ, scenario, mode):
    dur = get(typ, scenario, mode, "total_duration_sec")
    n   = get(typ, scenario, mode, "total_samples")
    return dur / n if np.isfinite(dur) and n > 0 else np.nan


scen_lbls = [SCENARIO_LABELS[s] for s in SCENARIO_ORDER]
cfg_labels  = [c[2] for c in CONFIGS]
cfg_colors  = [c[3] for c in CONFIGS]
cfg_hatches = [c[4] for c in CONFIGS]


def grouped_bars(ax, data, x_labels, series_labels, colors, hatches=None,
                 bar_width=0.85, annotate=False, fmt=".2f"):
    n_s = len(series_labels)
    bw  = bar_width / n_s
    x   = np.arange(len(x_labels))
    for i, (label, vals) in enumerate(zip(series_labels, data)):
        offsets = x + (i - n_s / 2 + 0.5) * bw
        hatch   = hatches[i] if hatches else ""
        rects   = ax.bar(
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
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6.5,
                    )
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.yaxis.grid(True, alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)


# ── Text summary ──────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  SPEEDUP ANALYSIS: CoD+Spec vs CoT vs Base")
print("=" * 72)

scen_names = {"easy": "Easy (GSM8K)", "medium": "Medium (MATH L1-2)", "hard": "Hard (MATH L3-4)"}
for s in SCENARIO_ORDER:
    print(f"\n{'-'*72}")
    print(f"  {scen_names[s]}")
    print(f"{'-'*72}")
    header = f"  {'Configuration':<26} {'TPS':>7} {'Tokens':>7} {'Sec/Smpl':>9} {'Accuracy':>9} {'Acc/Sec':>9}"
    print(header)
    print(f"  {'-'*22:<26} {'-'*7} {'-'*7} {'-'*9} {'-'*9} {'-'*9}")

    cod_spec_sps = sec_per_sample("cod", s, "speculative")
    for typ, mode, label, _, _ in CONFIGS:
        tps    = get(typ, s, mode, "tokens_per_second")
        tokens = get(typ, s, mode, "avg_tokens_generated")
        acc    = get(typ, s, mode, "accuracy")
        sps    = sec_per_sample(typ, s, mode)
        acc_s  = acc / sps if np.isfinite(sps) and sps > 0 else np.nan
        tps_s    = f"{tps:.1f}" if np.isfinite(tps) else "—"
        tokens_s = f"{tokens:.0f}" if np.isfinite(tokens) else "—"
        sps_s    = f"{sps:.3f}" if np.isfinite(sps) else "—"
        acc_s_s  = f"{acc:.3f}" if np.isfinite(acc) else "—"
        accsec_s = f"{acc_s:.3f}" if np.isfinite(acc_s) else "—"
        print(f"  {label:<26} {tps_s:>7} {tokens_s:>7} {sps_s:>9} {acc_s_s:>9} {accsec_s:>9}")

    print(f"\n  Speedup of CoD+Spec vs:")
    for typ, mode, label, _, _ in CONFIGS:
        if typ == "cod" and mode == "speculative":
            continue
        sps = sec_per_sample(typ, s, mode)
        if np.isfinite(sps) and np.isfinite(cod_spec_sps) and cod_spec_sps > 0:
            ratio = sps / cod_spec_sps
            print(f"    {label:<30} {ratio:.2f}×")

    # decomposed speedup for CoD+Spec vs CoT Baseline
    cot_base_tokens = get("cot", s, "baseline", "avg_tokens_generated")
    cod_spec_tokens = get("cod", s, "speculative", "avg_tokens_generated")
    cot_base_tps    = get("cot", s, "baseline",    "tokens_per_second")
    cod_spec_tps    = get("cod", s, "speculative",  "tokens_per_second")
    if all(np.isfinite(v) and v > 0 for v in [cot_base_tokens, cod_spec_tokens, cot_base_tps, cod_spec_tps]):
        token_component = cot_base_tokens / cod_spec_tokens
        tps_component   = cod_spec_tps / cot_base_tps
        print(f"\n  Speedup decomposition (CoD+Spec vs CoT Baseline):")
        print(f"    Token efficiency:   {token_component:.2f}× (CoD uses {cod_spec_tokens:.0f} vs CoT {cot_base_tokens:.0f} tokens/response)")
        print(f"    TPS acceleration:   {tps_component:.2f}× (CoD+Spec {cod_spec_tps:.0f} vs CoT baseline {cot_base_tps:.0f} tok/s)")
        print(f"    Combined (product): {token_component * tps_component:.2f}×")

    ar_cod = get("cod", s, "speculative", "acceptance_rate")
    ar_cot = get("cot", s, "speculative", "acceptance_rate")
    if np.isfinite(ar_cod) and np.isfinite(ar_cot):
        print(f"\n  Acceptance rates (spec decoding):")
        print(f"    CoD: {ar_cod:.3f}   CoT: {ar_cot:.3f}   Δ = {ar_cod - ar_cot:+.3f}")

print(f"\n{'='*72}\n")


# ── Build figure ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(21, 12))
fig.suptitle(
    "CoD + Speculative Decoding vs CoT vs Base — Full Speed & Quality Analysis",
    fontsize=15, fontweight="bold", y=0.998,
)

# [0,0] Wall time per sample
ax = axes[0, 0]
data_sps = [
    [sec_per_sample(typ, s, mode) for s in SCENARIO_ORDER]
    for typ, mode, *_ in CONFIGS
]
grouped_bars(ax, data_sps, scen_lbls, cfg_labels, cfg_colors,
             hatches=cfg_hatches, annotate=True, fmt=".2f")
ax.set_title("Wall Time per Sample\n(lower = faster)")
ax.set_ylabel("Seconds per Sample")
ax.legend(loc="upper left", framealpha=0.95, fontsize=7)

# [0,1] Accuracy
ax = axes[0, 1]
data_acc = [
    [get(typ, s, mode, "accuracy") for s in SCENARIO_ORDER]
    for typ, mode, *_ in CONFIGS
]
grouped_bars(ax, data_acc, scen_lbls, cfg_labels, cfg_colors,
             hatches=cfg_hatches, annotate=True, fmt=".3f")
ax.set_title("Accuracy\n(higher = better)")
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1.15)
ax.legend(loc="upper right", framealpha=0.95, fontsize=7)

# [0,2] Accuracy per second — quality-efficiency tradeoff
ax = axes[0, 2]
data_aps = []
for typ, mode, *_ in CONFIGS:
    vals = []
    for s in SCENARIO_ORDER:
        acc = get(typ, s, mode, "accuracy")
        sps = sec_per_sample(typ, s, mode)
        vals.append(acc / sps if np.isfinite(acc) and np.isfinite(sps) and sps > 0 else np.nan)
    data_aps.append(vals)

grouped_bars(ax, data_aps, scen_lbls, cfg_labels, cfg_colors,
             hatches=cfg_hatches, annotate=True, fmt=".2f")
ax.set_title("Accuracy per Second\n(quality × speed tradeoff — higher = better)")
ax.set_ylabel("Accuracy / Sec-per-Sample")
ax.legend(loc="upper right", framealpha=0.95, fontsize=7)

# [1,0] Avg tokens per response
ax = axes[1, 0]
data_tok = [
    [get(typ, s, mode, "avg_tokens_generated") for s in SCENARIO_ORDER]
    for typ, mode, *_ in CONFIGS
]
grouped_bars(ax, data_tok, scen_lbls, cfg_labels, cfg_colors,
             hatches=cfg_hatches, annotate=True, fmt=".0f")
ax.set_title("Avg Tokens per Response\n(token efficiency — lower = more concise)")
ax.set_ylabel("Tokens")
ax.legend(loc="upper left", framealpha=0.95, fontsize=7)

# [1,1] Tokens per second
ax = axes[1, 1]
data_tps = [
    [get(typ, s, mode, "tokens_per_second") for s in SCENARIO_ORDER]
    for typ, mode, *_ in CONFIGS
]
grouped_bars(ax, data_tps, scen_lbls, cfg_labels, cfg_colors,
             hatches=cfg_hatches, annotate=True, fmt=".0f")
ax.set_title("Throughput: Tokens per Second\n(higher = faster raw generation)")
ax.set_ylabel("Tokens per Second")
ax.legend(loc="lower right", framealpha=0.95, fontsize=7)

# [1,2] Acceptance rate (spec runs only)
ax = axes[1, 2]
spec_cfgs  = [(typ, mode, label, color, hatch) for typ, mode, label, color, hatch in CONFIGS if mode == "speculative"]
ar_data    = [[get(typ, s, "speculative", "acceptance_rate") for s in SCENARIO_ORDER] for typ, mode, *_ in spec_cfgs]
ar_labels  = [c[2] for c in spec_cfgs]
ar_colors  = [c[3] for c in spec_cfgs]
ar_hatches = [c[4] for c in spec_cfgs]

grouped_bars(ax, ar_data, scen_lbls, ar_labels, ar_colors,
             hatches=ar_hatches, annotate=True, fmt=".3f")
ax.set_title("Draft Token Acceptance Rate\n(speculative runs only — higher = better draft alignment)")
ax.set_ylabel("Acceptance Rate")
ax.set_ylim(0, 1.05)
ax.legend(framealpha=0.95, fontsize=7)

# ── Global hatch legend ───────────────────────────────────────────────────────
solid_patch = mpatches.Patch(fc="#888888", alpha=0.88, label="Baseline (no spec. decoding)")
hatch_patch = mpatches.Patch(fc="#888888", hatch="///", alpha=0.5, label="With Speculative Decoding")
fig.legend(
    handles=[solid_patch, hatch_patch],
    loc="lower center", ncol=2, fontsize=10, framealpha=0.9,
    bbox_to_anchor=(0.5, -0.015),
)

plt.tight_layout(rect=[0, 0.03, 1, 0.998])
out_path = OUTPUTS / "speedup_analysis.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Chart saved to {out_path}")
