"""
generate_figures.py — Creates all 6 figures for the research poster.

Figures generated:
  F1: System architecture (SVG-style text diagram)
  F2: Bar chart — token reduction % by model
  F3: Scatter — quality retention vs token savings
  F4: Heatmap — ablation study (technique × model)
  F5: Line chart — cost savings vs prompt length
  F6: Summary table (baseline vs optimized)

Usage:
    python generate_figures.py                    # Uses built-in mock data
    python generate_figures.py --results PATH     # Uses real experiment JSON
"""

import os
import sys
import json
import argparse
import random

from visualize_results import load_results

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import FancyBboxPatch
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[Warning] matplotlib/numpy not installed. Run: pip install matplotlib numpy --break-system-packages")

'''
# ──────────────────────────────────────────────────────────────────────────────
# MOCK DATA (used when no real results exist)
# ──────────────────────────────────────────────────────────────────────────────

MODELS = ["GPT-3.5-turbo", "GPT-4o-mini", "Claude Haiku"]
TECHNIQUES = ["Rule-based", "Sem. Dedup", "Few-shot Prune", "Inst. Distill"]
TASKS = ["Function Gen.", "Bug Fixing", "Code Explain"]

MOCK_RESULTS = {
    "GPT-3.5-turbo": {"baseline_tokens": 312, "optimized_tokens": 187, "baseline_pass": 0.72, "optimized_pass": 0.70, "baseline_cost": 0.00045, "optimized_cost": 0.00027},
    "GPT-4o-mini":   {"baseline_tokens": 298, "optimized_tokens": 175, "baseline_pass": 0.81, "optimized_pass": 0.80, "baseline_cost": 0.00008, "optimized_cost": 0.00005},
    "Claude Haiku":  {"baseline_tokens": 325, "optimized_tokens": 192, "baseline_pass": 0.78, "optimized_pass": 0.77, "baseline_cost": 0.00012, "optimized_cost": 0.00007},
}

ABLATION_DATA = {
    "GPT-3.5-turbo": [-2.3, -2.3, 17.2,  9.2],
    "GPT-4o-mini":   [-2.6, -2.6, 16.9, 8.9],
    "Claude Haiku":  [-3.3, -3.3, 18.2, 8.6],
}

COST_VS_LENGTH = {
    "lengths": [100, 250, 500, 750, 1000, 1500, 2000],
    "GPT-3.5-turbo baseline": [0.00015, 0.00032, 0.00065, 0.00097, 0.00130, 0.00195, 0.00260],
    "GPT-3.5-turbo optimized":[0.00009, 0.00019, 0.00038, 0.00057, 0.00076, 0.00114, 0.00152],
    "Claude Haiku baseline":  [0.00005, 0.00010, 0.00020, 0.00030, 0.00040, 0.00060, 0.00081],
    "Claude Haiku optimized": [0.00003, 0.00006, 0.00012, 0.00017, 0.00023, 0.00035, 0.00047],
}


def load_results(path: str) -> dict:
    """Load real results JSON and aggregate into MOCK_RESULTS format."""
    with open(path) as f:
        records = json.load(f)

    aggregated = {}
    for model in set(r["model"] for r in records):
        b = [r for r in records if r["model"] == model and r["condition"] == "baseline"]
        o = [r for r in records if r["model"] == model and r["condition"] == "optimized"]
        if not b or not o:
            continue
        aggregated[model] = {
            "baseline_tokens":  sum(r["input_tokens"] for r in b) / len(b),
            "optimized_tokens": sum(r["input_tokens"] for r in o) / len(o),
            "baseline_pass":    sum(r["pass_at_1"] for r in b) / len(b),
            "optimized_pass":   sum(r["pass_at_1"] for r in o) / len(o),
            "baseline_cost":    sum(r["estimated_cost_usd"] for r in b) / len(b),
            "optimized_cost":   sum(r["estimated_cost_usd"] for r in o) / len(o),
        }
    return aggregated


# ──────────────────────────────────────────────────────────────────────────────
# STYLE
# ──────────────────────────────────────────────────────────────────────────────

COLORS = {
    "baseline":  "#6B7FD7",
    "optimized": "#2DB87C",
    "accent":    "#E8593C",
    "neutral":   "#888780",
    "bg":        "#FAFAFA",
    "grid":      "#E8E6DF",
}

def setup_style():
    plt.rcParams.update({
        "font.family":      "DejaVu Sans",
        "font.size":        11,
        "axes.spines.top":  False,
        "axes.spines.right":False,
        "axes.grid":        True,
        "grid.color":       COLORS["grid"],
        "grid.linewidth":   0.7,
        "axes.facecolor":   COLORS["bg"],
        "figure.facecolor": "white",
        "axes.titlesize":   13,
        "axes.titleweight": "bold",
    })

OUT = "figures"
os.makedirs(OUT, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# F2 — TOKEN REDUCTION BAR CHART
# ──────────────────────────────────────────────────────────────────────────────

def fig2_token_reduction(results: dict):
    setup_style()
    models = list(results.keys())
    baseline = [results[m]["baseline_tokens"] for m in models]
    optimized = [results[m]["optimized_tokens"] for m in models]
    reductions = [(b - o) / b * 100 for b, o in zip(baseline, optimized)]

    x = range(len(models))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Figure 2 — Token Reduction: Baseline vs Optimized", fontweight="bold", fontsize=14)

    # Left: grouped bar
    ax = axes[0]
    bars1 = ax.bar([i - width/2 for i in x], baseline, width, label="Baseline", color=COLORS["baseline"], alpha=0.85)
    bars2 = ax.bar([i + width/2 for i in x], optimized, width, label="Optimized", color=COLORS["optimized"], alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=10)
    ax.set_ylabel("Avg. Input Tokens")
    ax.set_title("Input tokens (avg. per task)")
    ax.legend()
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 4, f"{h:.0f}", ha="center", va="bottom", fontsize=9)

    # Right: reduction %
    ax2 = axes[1]
    bars3 = ax2.bar(list(x), reductions, color=COLORS["optimized"], alpha=0.85, edgecolor="white")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(models, rotation=10)
    ax2.set_ylabel("Token Reduction (%)")
    ax2.set_title("Token savings per model")
    ax2.set_ylim(0, max(reductions) * 1.3)
    for bar, r in zip(bars3, reductions):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{r:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    path = f"{OUT}/F2_token_reduction.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# F3 — QUALITY vs TOKEN SAVINGS SCATTER
# ──────────────────────────────────────────────────────────────────────────────

def fig3_quality_scatter(results: dict):
    setup_style()
    models = list(results.keys())
    token_savings = [(results[m]["baseline_tokens"] - results[m]["optimized_tokens"]) / results[m]["baseline_tokens"] * 100
                     for m in models]
    quality_retention = [results[m]["optimized_pass"] / results[m]["baseline_pass"] * 100
                         for m in models]

    fig, ax = plt.subplots(figsize=(7, 6))
    colors_scatter = [COLORS["baseline"], COLORS["optimized"], COLORS["accent"]]

    for i, (m, ts, qr) in enumerate(zip(models, token_savings, quality_retention)):
        ax.scatter(ts, qr, s=220, color=colors_scatter[i], zorder=5, edgecolors="white", linewidths=1.5)
        ax.annotate(m, (ts, qr), textcoords="offset points", xytext=(8, 4), fontsize=10)

    # Ideal zone shading
    ax.axhline(95, color=COLORS["neutral"], linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(1, 95.5, "Quality threshold (95%)", fontsize=8, color=COLORS["neutral"])
    ax.fill_betweenx([95, 105], 0, 60, alpha=0.05, color=COLORS["optimized"])
    ax.text(20, 96.5, "Ideal region", fontsize=8, color=COLORS["optimized"], alpha=0.7)

    ax.set_xlabel("Token Savings (%)")
    ax.set_ylabel("Quality Retention (% of baseline pass@1)")
    ax.set_title("Figure 3 — Quality vs Token Savings Tradeoff")
    ax.set_xlim(0, max(token_savings) * 1.3)
    ax.set_ylim(85, 105)

    plt.tight_layout()
    path = f"{OUT}/F3_quality_scatter.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# F4 — ABLATION HEATMAP
# ──────────────────────────────────────────────────────────────────────────────

def fig4_ablation_heatmap():
    if not HAS_MPL:
        return
    setup_style()
    models_list = list(ABLATION_DATA.keys())
    data = np.array([ABLATION_DATA[m] for m in models_list])

    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(data, cmap="YlGn", aspect="auto", vmin=0, vmax=25)

    ax.set_xticks(range(len(TECHNIQUES)))
    ax.set_xticklabels(TECHNIQUES, fontsize=11)
    ax.set_yticks(range(len(models_list)))
    ax.set_yticklabels(models_list, fontsize=11)
    ax.set_title("Figure 4 — Ablation: Token Reduction % per Technique × Model", fontweight="bold")

    for i in range(len(models_list)):
        for j in range(len(TECHNIQUES)):
            ax.text(j, i, f"{data[i, j]:.1f}%", ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color="white" if data[i, j] > 15 else "#2c2c2a")

    plt.colorbar(im, ax=ax, label="Token reduction (%)")
    plt.tight_layout()
    path = f"{OUT}/F4_ablation_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# F5 — COST VS PROMPT LENGTH
# ──────────────────────────────────────────────────────────────────────────────

def fig5_cost_vs_length():
    setup_style()
    lengths = COST_VS_LENGTH["lengths"]

    fig, ax = plt.subplots(figsize=(9, 5))
    line_styles = [("-", COLORS["baseline"]), ("--", COLORS["optimized"])]

    for (style, color), (key, values) in zip(
        [("-", "#6B7FD7"), ("--", "#2DB87C"), ("-", "#E8593C"), ("--", "#BA7517")],
        [(k, v) for k, v in COST_VS_LENGTH.items() if k != "lengths"]
    ):
        label = key.replace(" baseline", " (base)").replace(" optimized", " (opt.)")
        lw = 2.5 if "optimized" in key else 1.5
        ax.plot(lengths, [v * 1000 for v in values], linestyle=style,
                color=color, linewidth=lw, label=label, marker="o", markersize=4)

    ax.set_xlabel("Input Prompt Length (tokens)")
    ax.set_ylabel("Estimated Cost ($ × 10⁻³ per call)")
    ax.set_title("Figure 5 — Cost Savings vs Prompt Length")
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = f"{OUT}/F5_cost_vs_length.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# F6 — SUMMARY TABLE
# ──────────────────────────────────────────────────────────────────────────────

def fig6_summary_table(results: dict):
    setup_style()

    models = list(results.keys())
    col_labels = ["Model", "Baseline\nTokens", "Optimized\nTokens", "Token\nSaved %",
                  "Baseline\nPass@1", "Opt.\nPass@1", "Cost\nSaved %"]
    rows = []
    for m in models:
        r = results[m]
        tok_save = (r["baseline_tokens"] - r["optimized_tokens"]) / r["baseline_tokens"] * 100
        cost_save = (r["baseline_cost"] - r["optimized_cost"]) / r["baseline_cost"] * 100
        rows.append([
            m,
            f"{r['baseline_tokens']:.0f}",
            f"{r['optimized_tokens']:.0f}",
            f"{tok_save:.1f}%",
            f"{r['baseline_pass']:.1%}",
            f"{r['optimized_pass']:.1%}",
            f"{cost_save:.1f}%",
        ])

    fig, ax = plt.subplots(figsize=(14, 3 + len(rows) * 0.8))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.0)

    # Header styling
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor("#534AB7")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    # Alternating rows
    for i in range(1, len(rows) + 1):
        for j in range(len(col_labels)):
            table[(i, j)].set_facecolor("#F1EFE8" if i % 2 == 0 else "white")
            # Highlight savings columns
            if j in [3, 6]:
                table[(i, j)].set_text_props(color="#0F6E56", fontweight="bold")

    ax.set_title("Figure 6 — Full Results: Baseline vs Optimized", fontweight="bold", pad=20, fontsize=13)

    plt.tight_layout()
    path = f"{OUT}/F6_summary_table.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")
    return path


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results/raw_resultsq.json", help="Path to raw_resultsq.json (optional)")
    args = parser.parse_args()

    if not HAS_MPL:
        print("ERROR: matplotlib and numpy are required. Install with:")
        print("  pip install matplotlib numpy --break-system-packages")
        return

    results = MOCK_RESULTS
    if args.results and os.path.exists(args.results):
        loaded = load_results(args.results)
        if loaded:
            results = loaded
            print(f"[Loaded] Real results from {args.results}")
        else:
            print("[Fallback] Using mock data (real results could not be parsed)")
    else:
        print("[Info] No results file provided — using built-in mock data")

    print("\nGenerating figures...\n")
    fig2_token_reduction(results)
    fig3_quality_scatter(results)
    fig4_ablation_heatmap()
    fig5_cost_vs_length()
    fig6_summary_table(results)

    print(f"\nAll figures saved to ./{OUT}/")
    print("You can now use these in your poster!")


if __name__ == "__main__":
    main()

import json
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def load_results(path: str = "results/raw_resultsq.json"):
    with open(path, "r") as f:
        return json.load(f)
'''

# ──────────────────────────────────────────────────────────────────────────────
# CHART: TOKEN REDUCTION ACROSS MODELS (FULL_V2 ONLY)
# ──────────────────────────────────────────────────────────────────────────────

def plot_token_reduction_full_v2(records, output_dir="figures"):
    """Token reduction for full_v2 condition across all models."""
    os.makedirs(output_dir, exist_ok=True)
    
    models = sorted(set(r["model"] for r in records))
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    baseline_tokens = {}
    full_v2_tokens = {}
    
    # Aggregate baseline and full_v2 tokens by model
    for model in models:
        baseline_recs = [r for r in records if r["model"] == model and r["condition"] == "baseline"]
        full_v2_recs = [r for r in records if r["model"] == model and r["condition"] == "full_v2"]
        
        baseline_tokens[model] = np.mean([r["input_tokens"] for r in baseline_recs]) if baseline_recs else 0
        full_v2_tokens[model] = np.mean([r["input_tokens"] for r in full_v2_recs]) if full_v2_recs else 0
    
    # Calculate reduction percentages
    reduction_pcts = {}
    for model in models:
        if baseline_tokens[model] > 0:
            reduction_pcts[model] = (baseline_tokens[model] - full_v2_tokens[model]) / baseline_tokens[model] * 100
        else:
            reduction_pcts[model] = 0
    
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, [baseline_tokens[m] for m in models], width, 
                   label="Baseline", color="#FF6B6B", edgecolor="black", linewidth=1.5)
    bars2 = ax.bar(x + width/2, [full_v2_tokens[m] for m in models], width,
                   label="Full v2", color="#4ECDC4", edgecolor="black", linewidth=1.5)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f"{height:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f"{height:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    
    # Add reduction percentage labels above the pair
    for i, model in enumerate(models):
        y_pos = max(baseline_tokens[model], full_v2_tokens[model]) + 20
        ax.text(i, y_pos, f"─{reduction_pcts[model]:.1f}%", 
                ha="center", va="bottom", fontsize=11, fontweight="bold", color="green")
    
    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Avg Input Tokens / Request", fontsize=12, fontweight="bold")
    ax.set_title("Token Reduction: Baseline vs Full v2 (Production Pipeline)", 
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/06_token_reduction_full_v2.png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved {output_dir}/06_token_reduction_full_v2.png")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# CHART: TOKEN REDUCTION PIPELINE (GRADUAL DECREASE)
# ──────────────────────────────────────────────────────────────────────────────
'''
def plot_token_pipeline_reduction(records, output_dir="figures"):
    """Line graph showing token reduction through each optimization stage."""
    os.makedirs(output_dir, exist_ok=True)
    
    models = sorted(set(r["model"] for r in records))
    
    # Map conditions to pipeline stages
    stages = ["Baseline", "BrevityMode\n(caveman)", "BrevityMode +\nCompression\n(pre_call_only)", "Full Pipeline\n(full_v2)"]
    condition_order = ["baseline", "caveman", "pre_call_only", "full_v2"]
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    colors_models = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
    markers = ["o", "s", "^"]
    
    # Aggregate tokens at each stage by model
    for idx, model in enumerate(models):
        stage_tokens = []
        
        for condition in condition_order:
            recs = [r for r in records if r["model"] == model and r["condition"] == condition]
            avg_tokens = np.mean([r["input_tokens"] for r in recs]) if recs else 0
            stage_tokens.append(avg_tokens)
        
        # Plot line for this model
        ax.plot(stages, stage_tokens, marker=markers[idx], markersize=10,
                label=model, linewidth=2.5, color=colors_models[idx])
        
        # Add value labels on each point
        for i, tokens in enumerate(stage_tokens):
            ax.text(i, tokens + 15, f"{tokens:.0f}", ha="center", va="bottom",
                   fontsize=9, fontweight="bold", color=colors_models[idx])
    
    # Add shaded regions for savings
    ax.axvspan(-0.5, 0.5, alpha=0.05, color="gray", label="Baseline")
    ax.axvspan(0.5, 1.5, alpha=0.08, color="blue", label="BrevityMode Stage")
    ax.axvspan(1.5, 2.5, alpha=0.08, color="orange", label="Compression Stage")
    ax.axvspan(2.5, 3.5, alpha=0.08, color="green", label="Cache Stage")
    
    ax.set_ylabel("Avg Input Tokens / Request", fontsize=12, fontweight="bold")
    ax.set_xlabel("Optimization Stage", fontsize=12, fontweight="bold")
    ax.set_title("Token Reduction Pipeline: Gradual Decrease Through Each Optimization", 
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right", ncol=2)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/07_token_pipeline_reduction.png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved {output_dir}/07_token_pipeline_reduction.png")
    plt.close()
'''
def plot_token_pipeline_reduction(records, output_dir="figures"):
    """Line graph showing token reduction through each optimization stage."""
    os.makedirs(output_dir, exist_ok=True)
    
    models = sorted(set(r["model"] for r in records))
    print(f"  [DEBUG] Models found: {models}")
    
    # Map conditions to pipeline stages
    stages = ["Baseline", "BrevityMode\n(caveman)", "BrevityMode +\nCompression\n(pre_call_only)", "Full Pipeline\n(full_v2)"]
    condition_order = ["baseline", "caveman", "pre_call_only", "full_v2"]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors_models = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
    markers = ["o", "s", "^"]
    
    # Add slight x-offset to prevent overlapping lines
    x_positions = np.arange(len(stages))
    offsets = np.linspace(-0.08, 0.08, len(models))  # Spread models across x-axis
    
    # Aggregate tokens at each stage by model
    for idx, model in enumerate(models):
        stage_tokens = []
        
        for condition in condition_order:
            recs = [r for r in records if r["model"] == model and r["condition"] == condition]
            avg_tokens = np.mean([r["input_tokens"] for r in recs]) if recs else 0
            stage_tokens.append(avg_tokens)
            print(f"  [DEBUG] {model} / {condition}: {len(recs)} records, avg tokens: {avg_tokens:.0f}")
        
        # Plot line for this model with offset
        if any(stage_tokens):
            x_offset = x_positions + offsets[idx]
            ax.plot(x_offset, stage_tokens, marker=markers[idx], markersize=12,
                    label=model, linewidth=3, color=colors_models[idx], alpha=0.9)
            
            # Add value labels on each point
            for i, tokens in enumerate(stage_tokens):
                ax.text(x_offset[i], tokens + 12, f"{tokens:.0f}", ha="center", va="bottom",
                       fontsize=8, fontweight="bold", color=colors_models[idx])
        else:
            print(f"  [WARNING] No data for model: {model}")
    
    # Add shaded regions for savings
    ax.axvspan(-0.5, 0.5, alpha=0.05, color="gray")
    ax.axvspan(0.5, 1.5, alpha=0.08, color="blue")
    ax.axvspan(1.5, 2.5, alpha=0.08, color="orange")
    ax.axvspan(2.5, 3.5, alpha=0.08, color="green")
    
    ax.set_ylabel("Avg Input Tokens / Request", fontsize=12, fontweight="bold")
    ax.set_xlabel("Optimization Stage", fontsize=12, fontweight="bold")
    ax.set_title("Token Reduction Pipeline: Gradual Decrease Through Each Optimization\n(All 3 Models Compared)", 
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(stages)
    ax.legend(fontsize=11, loc="upper right", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/07_token_pipeline_reduction.png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved {output_dir}/07_token_pipeline_reduction.png")
    plt.close()

# ──────────────────────────────────────────────────────────────────────────────
# CHART: TOKEN SAVINGS BREAKDOWN (STACKED BAR)
# ──────────────────────────────────────────────────────────────────────────────

def plot_token_savings_breakdown(records, output_dir="figures"):
    """Stacked bar chart showing where token savings come from."""
    os.makedirs(output_dir, exist_ok=True)
    
    models = sorted(set(r["model"] for r in records))
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    x = np.arange(len(models))
    width = 0.5
    
    baseline_tokens = []
    brevity_savings = []
    compression_savings = []
    cache_savings = []
    
    for model in models:
        # Baseline
        br = [r for r in records if r["model"] == model and r["condition"] == "baseline"]
        base_tok = np.mean([r["input_tokens"] for r in br]) if br else 0
        baseline_tokens.append(base_tok)
        
        # BrevityMode savings (caveman - baseline)
        cr = [r for r in records if r["model"] == model and r["condition"] == "caveman"]
        cav_tok = np.mean([r["input_tokens"] for r in cr]) if cr else base_tok
        brevity_savings.append(base_tok - cav_tok)
        
        # Compression savings (pre_call_only - caveman)
        pr = [r for r in records if r["model"] == model and r["condition"] == "pre_call_only"]
        pre_tok = np.mean([r["input_tokens"] for r in pr]) if pr else cav_tok
        compression_savings.append(cav_tok - pre_tok)
        
        # Cache savings (full_v2 - pre_call_only)
        vr = [r for r in records if r["model"] == model and r["condition"] == "full_v2"]
        v2_tok = np.mean([r["input_tokens"] for r in vr]) if vr else pre_tok
        cache_savings.append(pre_tok - v2_tok)
    
    # Stacked bars
    p1 = ax.bar(x, baseline_tokens, width, label="Final Tokens (full_v2)", 
                color="#4ECDC4", edgecolor="black", linewidth=1.5)
    p2 = ax.bar(x, cache_savings, width, bottom=baseline_tokens,
                label="Cache Savings", color="#FFA07A", edgecolor="black", linewidth=1.5)
    p3 = ax.bar(x, compression_savings, width,
                bottom=np.array(baseline_tokens) + np.array(cache_savings),
                label="Compression Savings", color="#FFD700", edgecolor="black", linewidth=1.5)
    p4 = ax.bar(x, brevity_savings, width,
                bottom=np.array(baseline_tokens) + np.array(cache_savings) + np.array(compression_savings),
                label="BrevityMode Savings", color="#90EE90", edgecolor="black", linewidth=1.5)
    
    # Add total baseline label at top
    for i, model in enumerate(models):
        total = baseline_tokens[i] + cache_savings[i] + compression_savings[i] + brevity_savings[i]
        ax.text(i, total + 10, f"{total:.0f}", ha="center", va="bottom",
               fontsize=11, fontweight="bold")
    
    ax.set_ylabel("Tokens", fontsize=12, fontweight="bold")
    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_title("Token Savings Breakdown: Sources of Reduction", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/08_token_savings_breakdown.png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved {output_dir}/08_token_savings_breakdown.png")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n📊 Loading results...")
    records = load_results()
    print(f"   Loaded {len(records)} records")
    
    print("\n🎨 Generating token reduction visualizations...")
    plot_token_reduction_full_v2(records)
    plot_token_pipeline_reduction(records)
    plot_token_savings_breakdown(records)
    
    print("\n✅ All token reduction charts generated in figures/ folder")