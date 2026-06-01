"""
Generates charts from experiment_runner_v2 results.
Creates PNG visualizations of token savings, cost reduction, and cache hit rates.
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from collections import defaultdict

# ──────────────────────────────────────────────────────────────────────────────
# LOAD RESULTS
# ──────────────────────────────────────────────────────────────────────────────

def load_results(path: str = "results/raw_results.json"):
    with open(path, "r") as f:
        return json.load(f)


def aggregate_by_condition_model(records):
    """Group records by (model, condition)"""
    agg = defaultdict(lambda: {"tokens": [], "costs": [], "hits": 0, "total": 0})
    
    for r in records:
        key = (r["model"], r["condition"])
        agg[key]["tokens"].append(r["input_tokens"])
        agg[key]["costs"].append(r["estimated_cost_usd"])
        if r["cache_hit"]:
            agg[key]["hits"] += 1
        agg[key]["total"] += 1
    
    return agg


def aggregate_by_variant(records):
    """Group by variant for cache analysis"""
    agg = defaultdict(lambda: {"hits": 0, "total": 0, "tokens": []})
    
    for r in records:
        if r["condition"] != "full_v2":
            continue
        key = r["variant"]
        agg[key]["tokens"].append(r["input_tokens"])
        if r["cache_hit"]:
            agg[key]["hits"] += 1
        agg[key]["total"] += 1
    
    return agg


# ──────────────────────────────────────────────────────────────────────────────
# CHART 1: TOKEN REDUCTION BY CONDITION
# ──────────────────────────────────────────────────────────────────────────────

def plot_token_reduction(records, output_dir="figures"):
    os.makedirs(output_dir, exist_ok=True)
    
    agg = aggregate_by_condition_model(records)
    models = sorted(set(r["model"] for r in records))
    conditions = ["baseline", "pre_call_only", "full_v2"]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(models))
    width = 0.25
    
    for i, cond in enumerate(conditions):
        tokens = [
            np.mean(agg[(m, cond)]["tokens"]) if (m, cond) in agg else 0
            for m in models
        ]
        ax.bar(x + i * width, tokens, width, label=cond)
    
    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Avg Input Tokens / Request", fontsize=12, fontweight="bold")
    ax.set_title("Token Reduction: Baseline vs Pre-Call vs Full v2", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/01_token_reduction.png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved {output_dir}/01_token_reduction.png")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# CHART 2: COST REDUCTION BY CONDITION
# ──────────────────────────────────────────────────────────────────────────────

def plot_cost_reduction(records, output_dir="figures"):
    agg = aggregate_by_condition_model(records)
    models = sorted(set(r["model"] for r in records))
    conditions = ["baseline", "pre_call_only", "full_v2"]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(models))
    width = 0.25
    
    for i, cond in enumerate(conditions):
        costs = [
            np.mean(agg[(m, cond)]["costs"]) if (m, cond) in agg else 0
            for m in models
        ]
        ax.bar(x + i * width, costs, width, label=cond)
    
    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Avg Cost / Request ($)", fontsize=12, fontweight="bold")
    ax.set_title("Cost Reduction: Baseline vs Pre-Call vs Full v2", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/02_cost_reduction.png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved {output_dir}/02_cost_reduction.png")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# CHART 3: CACHE HIT RATES BY VARIANT
# ──────────────────────────────────────────────────────────────────────────────

def plot_cache_hit_rates(records, output_dir="figures"):
    agg = aggregate_by_variant(records)
    variants = ["original", "paraphrase_1", "paraphrase_2", "exact_repeat"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    hit_rates = [
        (agg[v]["hits"] / agg[v]["total"] * 100) if agg[v]["total"] > 0 else 0
        for v in variants
    ]
    
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A"]
    bars = ax.bar(variants, hit_rates, color=colors, edgecolor="black", linewidth=1.5)
    
    # Add value labels on bars
    for bar, rate in zip(bars, hit_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f"{rate:.1f}%", ha="center", va="bottom", fontweight="bold")
    
    ax.set_ylabel("Cache Hit Rate (%)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Request Variant", fontsize=12, fontweight="bold")
    ax.set_title("Cache Hit Rates by Request Type (full_v2)", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/03_cache_hit_rates.png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved {output_dir}/03_cache_hit_rates.png")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# CHART 4: COST SAVINGS AT SCALE
# ──────────────────────────────────────────────────────────────────────────────

def plot_cost_at_scale(records, output_dir="figures"):
    agg = aggregate_by_condition_model(records)
    models = sorted(set(r["model"] for r in records))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    scales = [1_000, 10_000, 100_000]
    
    for idx, (ax, scale) in enumerate(zip(axes, scales)):
        baseline_costs = [
            np.mean(agg[(m, "baseline")]["costs"]) * scale if (m, "baseline") in agg else 0
            for m in models
        ]
        v2_costs = [
            np.mean(agg[(m, "full_v2")]["costs"]) * scale if (m, "full_v2") in agg else 0
            for m in models
        ]
        
        x = np.arange(len(models))
        width = 0.35
        
        ax.bar(x - width/2, baseline_costs, width, label="Baseline", color="#FF6B6B")
        ax.bar(x + width/2, v2_costs, width, label="Full v2", color="#4ECDC4")
        
        ax.set_ylabel("Daily Cost ($)", fontsize=10, fontweight="bold")
        ax.set_xlabel("Model", fontsize=10, fontweight="bold")
        ax.set_title(f"{scale:,} requests/day", fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/04_cost_at_scale.png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved {output_dir}/04_cost_at_scale.png")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# CHART 5: COST REDUCTION PERCENTAGE
# ──────────────────────────────────────────────────────────────────────────────

def plot_cost_reduction_pct(records, output_dir="figures"):
    agg = aggregate_by_condition_model(records)
    models = sorted(set(r["model"] for r in records))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    reductions = []
    for m in models:
        baseline = np.mean(agg[(m, "baseline")]["costs"]) if (m, "baseline") in agg else 1
        v2 = np.mean(agg[(m, "full_v2")]["costs"]) if (m, "full_v2") in agg else baseline
        pct = (baseline - v2) / baseline * 100 if baseline > 0 else 0
        reductions.append(pct)
    
    colors = ["#2ECC71" if r > 0 else "#E74C3C" for r in reductions]
    bars = ax.barh(models, reductions, color=colors, edgecolor="black", linewidth=1.5)
    
    # Add value labels
    for bar, val in zip(bars, reductions):
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2.,
                f"{val:.1f}%", ha="left" if val > 0 else "right", va="center",
                fontweight="bold", fontsize=11)
    
    ax.set_xlabel("Cost Reduction (%)", fontsize=12, fontweight="bold")
    ax.set_title("Cost Reduction: Full v2 vs Baseline", fontsize=14, fontweight="bold")
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.grid(axis="x", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/05_cost_reduction_pct.png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved {output_dir}/05_cost_reduction_pct.png")
    plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n📊 Loading results...")
    records = load_results()
    print(f"   Loaded {len(records)} records")
    
    print("\n🎨 Generating visualizations...")
    plot_token_reduction(records)
    plot_cost_reduction(records)
    plot_cache_hit_rates(records)
    plot_cost_at_scale(records)
    plot_cost_reduction_pct(records)
    
    print("\n✅ All charts generated in figures/ folder")