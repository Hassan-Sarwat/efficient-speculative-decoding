"""
Generate all figures for the write-up.

Sources
-------
* W&B project `peft_cob`  – training loss + GPU metrics parsed from output.log
* outputs/all_metrics.csv – inference throughput / VRAM / accuracy

Output
------
All PNGs are written to images/.

Run
---
    python scripts/generate_plots.py [--entity <wandb-entity>]
    python scripts/generate_plots.py --skip-wandb   # inference plots only

The script reads WANDB_API_KEY from .env automatically.
"""

import argparse
import ast
import os
import re
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import wandb
from dotenv import load_dotenv

load_dotenv()

IMAGES_DIR = Path(__file__).parent.parent / "images"
IMAGES_DIR.mkdir(exist_ok=True)

METRICS_CSV = Path(__file__).parent.parent / "outputs" / "all_metrics.csv"

PROJECT = "peft_cob"

PALETTE = {
    "cot": "#4C72B0",
    "cod": "#DD8452",
    "base": "#8172B2",
}
SPEC_ALPHA = 1.0
BASE_ALPHA = 0.55

SCENARIO_ORDER = ["easy", "medium", "hard"]
SCENARIO_LABELS = {
    "easy": "Easy (GSM8K)",
    "medium": "Medium (MATH L1-2)",
    "hard": "Hard (MATH L3-4)",
}
TYPE_LABELS = {"cot": "CoT", "cod": "CoD", "base": "Base"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "figure.dpi": 150,
})


# ── helpers ──────────────────────────────────────────────────────────────────

def parse_run_name(name: str) -> dict | None:
    """Return {model_type, scenario, reasoning_type} or None if unrecognised.

    Handles both orderings produced by different training runs:
      target_easy_cot_<ts>   (scenario before type)
      target_cot_hard_<ts>   (type before scenario)
    """
    m = re.match(r"^(target|draft)_(easy|medium|hard)_(cot|cod)", name)
    if m:
        return {"model_type": m.group(1), "scenario": m.group(2), "reasoning_type": m.group(3)}
    m = re.match(r"^(target|draft)_(cot|cod)_(easy|medium|hard)", name)
    if m:
        return {"model_type": m.group(1), "scenario": m.group(3), "reasoning_type": m.group(2)}
    return None


def parse_run_logs(run) -> pd.DataFrame:
    """Download output.log and parse step-level metric dicts printed to stdout.

    Training emits one dict per step, e.g.:
      {'loss': 0.99, 'grad_norm': 1.14, 'learning_rate': 0.0,
       'epoch': 0.01, 'gpu_peak_mem_gb': 29.08, 'gpu_current_mem_gb': 27.94, ...}
    """
    try:
        with tempfile.TemporaryDirectory() as tmp:
            run.file("output.log").download(root=tmp, replace=True)
            log_path = Path(tmp) / "output.log"
            if not log_path.exists():
                return pd.DataFrame()

            rows = []
            with open(log_path) as f:
                for line in f:
                    line = line.strip()
                    if not (line.startswith("{'loss':") or line.startswith('{"loss":')):
                        continue
                    try:
                        row = ast.literal_eval(line)
                        if isinstance(row, dict) and "loss" in row:
                            rows.append(row)
                    except (ValueError, SyntaxError):
                        pass

            if rows:
                df = pd.DataFrame(rows)
                df["step"] = range(1, len(df) + 1)
                print(f"    parsed {len(df)} steps from output.log "
                      f"(cols: {[c for c in df.columns if c != 'step']})")
                return df
            return pd.DataFrame()

    except Exception as e:
        print(f"    could not parse logs for {run.name}: {e}")
        return pd.DataFrame()


def fetch_wandb_runs(entity: str | None) -> list[dict]:
    """Return list of run dicts.  Each dict has a 'log_df' key with the parsed
    output.log DataFrame (preferred) and a 'history' key with the W&B metric
    history (fallback / cross-check).
    """
    api = wandb.Api()
    path = f"{entity}/{PROJECT}" if entity else PROJECT
    print(f"Fetching runs from W&B project: {path}")
    runs = api.runs(path)

    records = []
    for run in runs:
        meta = parse_run_name(run.name)
        if meta is None:
            print(f"  skip: {run.name!r} (unrecognised format)")
            continue

        print(f"  pulling: {run.name}  state={run.state}")
        history = run.history(samples=2000, pandas=True)
        log_df = parse_run_logs(run)

        records.append({
            **meta,
            "run_name": run.name,
            "history": history,   # W&B metrics (may be empty for early runs)
            "log_df": log_df,     # parsed output.log (always has data if run completed)
            "summary": dict(run.summary),
        })

    return records


def _best_df(run: dict) -> pd.DataFrame:
    """Return log_df if it has loss data, otherwise fall back to W&B history."""
    if not run["log_df"].empty and "loss" in run["log_df"].columns:
        return run["log_df"]
    return run["history"]


# ── training loss ─────────────────────────────────────────────────────────────

def _plot_loss_for_model(records: list[dict], model_type: str) -> None:
    """3×2 grid: rows = scenarios, cols = reasoning types."""
    rtypes = ["cot", "cod"]
    fig, axes = plt.subplots(len(SCENARIO_ORDER), len(rtypes), figsize=(12, 10))
    fig.suptitle(
        f"Training Loss — {model_type.capitalize()} Model",
        fontsize=14, fontweight="bold",
    )

    for row, scenario in enumerate(SCENARIO_ORDER):
        for col, rtype in enumerate(rtypes):
            ax = axes[row][col]
            ax.set_title(f"{SCENARIO_LABELS[scenario]}  |  {TYPE_LABELS[rtype]}", fontsize=9)
            ax.set_xlabel("Step", fontsize=8)
            ax.set_ylabel("Loss", fontsize=8)
            ax.tick_params(labelsize=7)

            match = [r for r in records
                     if r["scenario"] == scenario
                     and r["reasoning_type"] == rtype
                     and r["model_type"] == model_type]
            if not match:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, color="grey", fontsize=9)
                continue

            run = match[-1]
            df = _best_df(run)

            loss_col = next(
                (c for c in ("loss", "train/loss", "train_loss") if c in df.columns),
                None,
            )
            if loss_col is None:
                ax.text(0.5, 0.5, "no loss data", ha="center", va="center",
                        transform=ax.transAxes, color="grey", fontsize=9)
                continue

            step_col = "step" if "step" in df.columns else "_step"
            steps = df[step_col] if step_col in df.columns else pd.Series(range(len(df)))
            loss = df[loss_col].dropna()
            steps = steps.iloc[: len(loss)]

            ax.plot(steps, loss, color=PALETTE[rtype], linewidth=1.8)

            # annotate final loss
            ax.annotate(
                f"final: {loss.iloc[-1]:.3f}",
                xy=(steps.iloc[-1], loss.iloc[-1]),
                xytext=(-5, 8), textcoords="offset points",
                fontsize=7, color=PALETTE[rtype],
            )

    plt.tight_layout()
    out = IMAGES_DIR / f"training_loss_{model_type}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


def plot_training_loss(records: list[dict]) -> None:
    _plot_loss_for_model(records, "target")
    _plot_loss_for_model(records, "draft")


# ── training GPU memory ───────────────────────────────────────────────────────

def plot_training_gpu(records: list[dict]) -> None:
    """GPU memory per step from output.log (gpu_current_mem_gb + gpu_peak_mem_gb)."""
    for model_type in ["target", "draft"]:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        fig.suptitle(
            f"GPU Memory During Training — {model_type.capitalize()} Model",
            fontsize=13, fontweight="bold",
        )

        for ax, scenario in zip(axes, SCENARIO_ORDER):
            ax.set_title(SCENARIO_LABELS[scenario], fontsize=10)
            ax.set_xlabel("Training Step", fontsize=9)
            if ax is axes[0]:
                ax.set_ylabel("GPU Memory (GB)", fontsize=9)

            any_data = False
            for rtype in ["cot", "cod"]:
                match = [r for r in records
                         if r["scenario"] == scenario
                         and r["reasoning_type"] == rtype
                         and r["model_type"] == model_type]
                if not match:
                    continue

                run = match[-1]
                df = run["log_df"]

                if df.empty or "gpu_current_mem_gb" not in df.columns:
                    print(f"  no gpu_current_mem_gb in log_df for {run['run_name']}")
                    continue

                steps = df["step"]
                current = df["gpu_current_mem_gb"].dropna()
                peak = df["gpu_peak_mem_gb"].dropna() if "gpu_peak_mem_gb" in df.columns else None

                color = PALETTE[rtype]
                ax.plot(steps[: len(current)], current,
                        color=color, linewidth=1.6, label=f"{TYPE_LABELS[rtype]} current")
                if peak is not None:
                    ax.fill_between(
                        steps[: len(peak)], current[: len(peak)], peak,
                        color=color, alpha=0.15, label=f"{TYPE_LABELS[rtype]} peak",
                    )
                any_data = True

            if not any_data:
                ax.text(0.5, 0.5, "no GPU log data", ha="center", va="center",
                        transform=ax.transAxes, color="grey", fontsize=9)

            ax.legend(fontsize=8)

        plt.tight_layout()
        out = IMAGES_DIR / f"training_gpu_{model_type}.png"
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out.name}")


# ── inference VRAM ────────────────────────────────────────────────────────────

def plot_inference_vram(df: pd.DataFrame) -> None:
    types = ["base", "cot", "cod"]
    x = np.arange(len(SCENARIO_ORDER))
    width = 0.13

    fig, ax = plt.subplots(figsize=(11, 5))
    offsets = np.linspace(-(len(types) * width), len(types) * width, len(types) * 2)

    i = 0
    for rtype in types:
        sub = df[df["type"] == rtype]
        bv = np.array([
            sub[(sub["scenario"] == s) & (sub["speculative"] == 0)]["peak_vram_gb"].values
            for s in SCENARIO_ORDER
        ], dtype=object)
        sv = np.array([
            sub[(sub["scenario"] == s) & (sub["speculative"] == 1)]["peak_vram_gb"].values
            for s in SCENARIO_ORDER
        ], dtype=object)
        bv = np.array([v[0] if len(v) else np.nan for v in bv], dtype=float)
        sv = np.array([v[0] if len(v) else np.nan for v in sv], dtype=float)

        color = PALETTE[rtype]
        ax.bar(x + offsets[i], bv, width, color=color, alpha=BASE_ALPHA,
               label=f"{TYPE_LABELS[rtype]} baseline")
        i += 1
        if rtype != "base":
            ax.bar(x + offsets[i], sv, width, color=color, alpha=SPEC_ALPHA,
                   label=f"{TYPE_LABELS[rtype]} speculative")
            i += 1

    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIO_ORDER])
    ax.set_ylabel("Peak VRAM (GB)")
    ax.set_title("Peak GPU Memory During Inference", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    out = IMAGES_DIR / "inference_vram.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


# ── throughput ────────────────────────────────────────────────────────────────

def plot_throughput(df: pd.DataFrame) -> None:
    types = ["cot", "cod"]
    x = np.arange(len(SCENARIO_ORDER))
    width = 0.18
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    entries = []

    for rtype in types:
        sub = df[df["type"] == rtype]
        for spec, label in [(0, "baseline"), (1, "speculative")]:
            vals = np.array([
                sub[(sub["scenario"] == s) & (sub["speculative"] == spec)]["tokens_per_second"].values
                for s in SCENARIO_ORDER
            ], dtype=object)
            vals = np.array([v[0] if len(v) else np.nan for v in vals], dtype=float)
            entries.append((f"{TYPE_LABELS[rtype]} {label}", rtype, bool(spec), vals))

    fig, ax = plt.subplots(figsize=(10, 5))
    for (label, rtype, is_spec, vals), offset in zip(entries, offsets):
        ax.bar(x + offset, vals, width, color=PALETTE[rtype],
               alpha=SPEC_ALPHA if is_spec else BASE_ALPHA, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIO_ORDER])
    ax.set_ylabel("Tokens / second")
    ax.set_title("Inference Throughput: Baseline vs Speculative Decoding",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = IMAGES_DIR / "inference_throughput.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


# ── acceptance rate ───────────────────────────────────────────────────────────

def plot_acceptance_rate(df: pd.DataFrame) -> None:
    spec_df = df[df["speculative"] == 1]
    x = np.arange(len(SCENARIO_ORDER))
    width = 0.3

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, rtype in enumerate(["cot", "cod"]):
        sub = spec_df[spec_df["type"] == rtype]
        vals = np.array([
            sub[sub["scenario"] == s]["acceptance_rate"].values
            for s in SCENARIO_ORDER
        ], dtype=object)
        vals = np.array([v[0] if len(v) else np.nan for v in vals], dtype=float)
        ax.bar(x + (i - 0.5) * width, vals * 100, width,
               color=PALETTE[rtype], label=TYPE_LABELS[rtype])

    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIO_ORDER])
    ax.set_ylabel("Draft Token Acceptance Rate (%)")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax.set_title("Speculative Decoding — Draft Token Acceptance Rate",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    out = IMAGES_DIR / "acceptance_rate.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


# ── accuracy ──────────────────────────────────────────────────────────────────

def plot_accuracy(df: pd.DataFrame) -> None:
    types = ["base", "cot", "cod"]
    x = np.arange(len(SCENARIO_ORDER))
    width = 0.13
    offsets = np.linspace(-2.5 * width, 2.5 * width, len(types) * 2 - 1)

    fig, ax = plt.subplots(figsize=(11, 5))
    i = 0
    for rtype in types:
        sub = df[df["type"] == rtype]
        for spec, alpha in [(0, BASE_ALPHA), (1, SPEC_ALPHA)]:
            if rtype == "base" and spec == 1:
                continue
            vals = np.array([
                sub[(sub["scenario"] == s) & (sub["speculative"] == spec)]["accuracy"].values
                for s in SCENARIO_ORDER
            ], dtype=object)
            vals = np.array([v[0] if len(v) else np.nan for v in vals], dtype=float)
            ax.bar(x + offsets[i], vals * 100, width, color=PALETTE[rtype], alpha=alpha,
                   label=f"{TYPE_LABELS[rtype]} {'spec.' if spec else 'base.'}")
            i += 1

    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIO_ORDER])
    ax.set_ylabel("Accuracy (%)")
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax.set_title("Answer Accuracy by Scenario and Reasoning Type",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    out = IMAGES_DIR / "accuracy.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


# ── speedup ───────────────────────────────────────────────────────────────────

def plot_speedup(df: pd.DataFrame) -> None:
    x = np.arange(len(SCENARIO_ORDER))
    width = 0.3

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, rtype in enumerate(["cot", "cod"]):
        sub = df[df["type"] == rtype]
        speedups = []
        for s in SCENARIO_ORDER:
            base = sub[(sub["scenario"] == s) & (sub["speculative"] == 0)]["tokens_per_second"].values
            spec = sub[(sub["scenario"] == s) & (sub["speculative"] == 1)]["tokens_per_second"].values
            speedups.append(spec[0] / base[0] if len(base) and len(spec) else np.nan)
        ax.bar(x + (i - 0.5) * width, speedups, width,
               color=PALETTE[rtype], label=TYPE_LABELS[rtype])

    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--", alpha=0.5, label="No speedup")
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIO_ORDER])
    ax.set_ylabel("Speedup (× baseline)")
    ax.set_title("Speculative Decoding Speedup over Baseline", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    out = IMAGES_DIR / "speedup.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default=None,
                        help="W&B entity (username or team). Omit to use default.")
    parser.add_argument("--skip-wandb", action="store_true",
                        help="Skip W&B pulls and only regenerate inference plots.")
    args = parser.parse_args()

    print("\n=== Inference plots from all_metrics.csv ===")
    df = pd.read_csv(METRICS_CSV)
    plot_inference_vram(df)
    plot_throughput(df)
    plot_acceptance_rate(df)
    plot_accuracy(df)
    plot_speedup(df)

    if args.skip_wandb:
        print("\nSkipped W&B (--skip-wandb). Done.")
        return

    print("\n=== Training plots from W&B (using output.log) ===")
    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        print("WANDB_API_KEY not set — skipping training plots.")
        return

    wandb.login(key=api_key)
    records = fetch_wandb_runs(args.entity)

    if not records:
        print("No matching W&B runs found.")
        return

    plot_training_loss(records)
    plot_training_gpu(records)

    print(f"\nDone. All images saved to {IMAGES_DIR}/")


if __name__ == "__main__":
    main()
