#!/usr/bin/env python3
"""
verify_reproduction.py — compare a local eval result against the paper numbers.

Usage
-----
Standard / PRO log (a .log / .txt file that contains a "FINAL: X/Y = Z.Z%" line):

    python scripts/verify_reproduction.py \
        --config anc_v7kl01_10k --benchmark spatial_std \
        --log /tmp/repro_spatial_std_seed7.log

LIBERO-PRO with a specific perturbation:

    python scripts/verify_reproduction.py \
        --config anc_v7kl01_10k --benchmark libero_pro --perturbation lan \
        --log /tmp/repro_pro_lan_seed7.log

LIBERO-Plus (one or more shard JSONs; script aggregates):

    python scripts/verify_reproduction.py \
        --config anc_v7kl01_10k --benchmark libero_plus \
        --plus-shards /tmp/*plus*_results.json

Available --config values: run with --list-configs.

Exit codes: 0 = pass, 2 = warn, 3 = fail, 1 = usage/parse error.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = REPO_ROOT / "results" / "3seed_registry.json"

# ─────────────────────────────────────────────────────────────────────────────
# Static per-checkpoint paper numbers for the four HF release checkpoints.
# Spatial has multi-seed numbers in the 3seed registry; the other three are
# single-seed (seed=7) reference values from each ckpt's MODEL_CARD.md.
# ─────────────────────────────────────────────────────────────────────────────
STATIC_NUMBERS = {
    "object_v7kl015_2.5k": {
        "label": "Anchor-Align v7+KL=0.15 (LIBERO Object, 2.5k)",
        "libero_pro": {
            "lan":    {"mean_sr_pct": 100.0, "stderr_pp": 0.0, "n_seeds": 1},
            "object": {"mean_sr_pct":  89.6, "stderr_pp": 0.0, "n_seeds": 1},
            "swap":   {"mean_sr_pct":   0.0, "stderr_pp": 0.0, "n_seeds": 1},
        },
        "libero_plus": {"mean_sr_pct": 83.6, "stderr_pp": 0.0, "n_seeds": 1},
    },
    "goal_v7kl01_25k": {
        "label": "Anchor-Align v7+KL=0.1 (LIBERO Goal, 25k)",
        "spatial_std": {"mean_sr_pct": 97.8, "stderr_pp": 0.0, "n_seeds": 1},
        "libero_pro": {
            "lan":    {"mean_sr_pct": 96.4, "stderr_pp": 0.0, "n_seeds": 1},
            "object": {"mean_sr_pct": 76.6, "stderr_pp": 0.0, "n_seeds": 1},
            "swap":   {"mean_sr_pct":  2.2, "stderr_pp": 0.0, "n_seeds": 1},
        },
        "libero_plus": {"mean_sr_pct": 72.8, "stderr_pp": 0.0, "n_seeds": 1},
    },
    "l10_v7kl015_45k": {
        "label": "Anchor-Align v7+KL=0.15 (LIBERO-10, 45k)",
        "spatial_std": {"mean_sr_pct": 90.4, "stderr_pp": 0.0, "n_seeds": 1},
        "libero_pro": {
            "lan":    {"mean_sr_pct": 89.8, "stderr_pp": 0.0, "n_seeds": 1},
            "object": {"mean_sr_pct": 39.6, "stderr_pp": 0.0, "n_seeds": 1},
            "swap":   {"mean_sr_pct":  0.6, "stderr_pp": 0.0, "n_seeds": 1},
        },
        "libero_plus": {"mean_sr_pct": 69.2, "stderr_pp": 0.0, "n_seeds": 1},
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Terminal colors (auto-disabled if not a TTY)
# ─────────────────────────────────────────────────────────────────────────────
def _c(code: str, s: str) -> str:
    if not sys.stdout.isatty():
        return s
    return f"\033[{code}m{s}\033[0m"


GREEN = lambda s: _c("32", s)   # noqa: E731
YELLOW = lambda s: _c("33", s)  # noqa: E731
RED = lambda s: _c("31", s)     # noqa: E731
BOLD = lambda s: _c("1", s)     # noqa: E731
DIM = lambda s: _c("2", s)      # noqa: E731


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────
def load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        return {}
    with REGISTRY_PATH.open() as f:
        return json.load(f)


def lookup_paper_number(config: str, benchmark: str, perturbation: Optional[str]) -> Optional[dict]:
    """Return {'mean_sr_pct': float, 'stderr_pp': float, 'n_seeds': int, 'source': str} or None."""
    registry = load_registry()

    # First, try the multi-seed registry
    entries = registry.get("configs", {})
    if config in entries:
        cfg_entry = entries[config]
        results = cfg_entry.get("results", {})
        if benchmark in results:
            branch = results[benchmark]
            if benchmark == "libero_pro":
                if perturbation and perturbation in branch:
                    agg = branch[perturbation].get("aggregate")
                    if agg:
                        return {**agg, "source": "results/3seed_registry.json",
                                "label": cfg_entry.get("label", config)}
            else:
                agg = branch.get("aggregate")
                if agg:
                    return {**agg, "source": "results/3seed_registry.json",
                            "label": cfg_entry.get("label", config)}

    # Fallback to static numbers (single-seed release checkpoints)
    if config in STATIC_NUMBERS:
        entry = STATIC_NUMBERS[config]
        if benchmark in entry:
            branch = entry[benchmark]
            if benchmark == "libero_pro":
                if perturbation and perturbation in branch:
                    return {**branch[perturbation],
                            "source": f"{config} MODEL_CARD.md (seed=7 reference)",
                            "label": entry.get("label", config)}
            else:
                return {**branch,
                        "source": f"{config} MODEL_CARD.md (seed=7 reference)",
                        "label": entry.get("label", config)}
    return None


def list_configs() -> None:
    registry = load_registry()
    entries = registry.get("configs", {})
    print(BOLD("Multi-seed registry configs (from results/3seed_registry.json):"))
    for k, v in entries.items():
        avail = list(v.get("results", {}).keys())
        print(f"  {k:<30}  {v.get('label', '')}")
        print(f"    benchmarks: {', '.join(avail)}")
    print()
    print(BOLD("Static reference checkpoints (from MODEL_CARD.md, seed=7):"))
    for k, v in STATIC_NUMBERS.items():
        avail = [b for b in ("spatial_std", "libero_pro", "libero_plus") if b in v]
        print(f"  {k:<30}  {v['label']}")
        print(f"    benchmarks: {', '.join(avail)}")


# ─────────────────────────────────────────────────────────────────────────────
# Parsers for local eval output
# ─────────────────────────────────────────────────────────────────────────────
def parse_final_from_log(path: Path) -> Optional[dict]:
    """Parse Standard / PRO eval log. Look for 'FINAL: X/Y' or the aggregate at end."""
    try:
        text = path.read_text(errors="ignore")
    except OSError as e:
        print(RED(f"error: cannot read log {path}: {e}"))
        return None

    # Preferred: 'FINAL: X/Y' (used by run_libero_eval_batched, run_libero_pro_eval)
    m = re.search(r"FINAL:\s*(\d+)/(\d+)", text)
    if m:
        s, n = int(m.group(1)), int(m.group(2))
        return {"successes": s, "episodes": n, "sr_pct": 100.0 * s / n}

    # Fallback: 'Round X/X: A/B (Z.Z%)' — final round line
    round_matches = re.findall(r"Round\s+\d+/\d+:\s*(\d+)/(\d+)", text)
    if round_matches:
        s, n = int(round_matches[-1][0]), int(round_matches[-1][1])
        return {"successes": s, "episodes": n, "sr_pct": 100.0 * s / n}

    # Fallback: 'Total successes: X\nTotal episodes: Y' (older format)
    ts = re.search(r"Total successes:\s*(\d+)", text)
    te = re.search(r"Total episodes:\s*(\d+)", text)
    if ts and te:
        s, n = int(ts.group(1)), int(te.group(1))
        return {"successes": s, "episodes": n, "sr_pct": 100.0 * s / n}

    print(RED(f"error: could not find a result line in {path}"))
    print(DIM("  Expected one of: 'FINAL: X/Y', 'Round N/N: X/Y', or 'Total successes: X'."))
    return None


def parse_plus_shards(paths: list[Path]) -> Optional[dict]:
    """Aggregate LIBERO-Plus shard JSONs by deduping on (category, task_id, task_name)."""
    if not paths:
        print(RED("error: --plus-shards did not match any file"))
        return None
    seen: dict = {}
    for p in paths:
        try:
            d = json.loads(p.read_text())
        except Exception as e:
            print(YELLOW(f"warning: skipping unreadable shard {p}: {e}"))
            continue
        # Per-category if all tasks come from one category (single-cat shards)
        summary_cats = list(d.get("summary", {}).get("per_category", {}).keys())
        default_cat = summary_cats[0] if len(summary_cats) == 1 else None
        for entry in d.get("per_task_results", []):
            cat = entry.get("category") or default_cat or "unknown"
            uid = (cat, entry.get("task_id"), entry.get("task_name"))
            seen[uid] = bool(entry.get("success", False))
    if not seen:
        print(RED("error: no per-task results parsed from any shard"))
        return None
    s = sum(1 for ok in seen.values() if ok)
    n = len(seen)
    return {"successes": s, "episodes": n, "sr_pct": 100.0 * s / n}


# ─────────────────────────────────────────────────────────────────────────────
# Verdict
# ─────────────────────────────────────────────────────────────────────────────
def verdict(delta_pp: float, warn_tol: float, fail_tol: float) -> tuple[int, str, str]:
    a = abs(delta_pp)
    if a <= warn_tol:
        return 0, "PASS", GREEN("✅ PASS")
    if a <= fail_tol:
        return 2, "WARN", YELLOW("⚠️  WARN")
    return 3, "FAIL", RED("❌ FAIL")


def emit_report(config: str, benchmark: str, perturbation: Optional[str],
                paper: dict, local: dict, exit_code: int, verdict_str: str,
                warn_tol: float, fail_tol: float) -> None:
    print()
    print(BOLD("Reproduction check"))
    print("─" * 60)
    print(f"config:      {config}  ({paper.get('label', '?')})")
    bench_display = benchmark + (f" / {perturbation}" if perturbation else "")
    print(f"benchmark:   {bench_display}")

    n_seeds = paper.get("n_seeds", 1)
    seed_note = f"{n_seeds}-seed mean" if n_seeds > 1 else "seed=7 reference"
    stderr_str = f" ± {paper['stderr_pp']:.2f}" if paper.get("stderr_pp", 0) > 0 else ""
    print(f"paper:       {paper['mean_sr_pct']:.2f}%{stderr_str}  ({seed_note})")
    print(f"yours:       {local['sr_pct']:.2f}%  ({local['successes']}/{local['episodes']})")
    delta = local["sr_pct"] - paper["mean_sr_pct"]
    delta_str = f"{delta:+.2f} pp"
    if abs(delta) <= warn_tol:
        delta_str = GREEN(delta_str)
    elif abs(delta) <= fail_tol:
        delta_str = YELLOW(delta_str)
    else:
        delta_str = RED(delta_str)
    print(f"delta:       {delta_str}")
    print(f"tolerance:   warn ±{warn_tol:.1f} pp / fail ±{fail_tol:.1f} pp")
    print(f"source:      {paper.get('source', '?')}")
    print()
    print(verdict_str)
    if exit_code == 0:
        print(DIM("  Your setup reproduces the paper number within statistical variance."))
    elif exit_code == 2:
        print(DIM("  Outside the warning band. Common causes:"))
        print(DIM("    - different GPU family (H100, A100, RTX 40xx vs GH200)"))
        print(DIM("    - different CUDA/cuDNN minor version"))
        print(DIM("  If your delta is < fail-tol you're still within acceptable bounds."))
    else:
        print(DIM("  Outside the failure band. Likely causes:"))
        print(DIM("    - wrong checkpoint directory"))
        print(DIM("    - inference flags changed (see REPRODUCE.md 'Inference flags used everywhere')"))
        print(DIM("    - LIBERO-PRO or LIBERO-Plus repo not on PYTHONPATH"))
        print(DIM("    - VLM backbone weights mismatch"))
        print(DIM(f"  Please open an issue at https://github.com/dwipddalal/Anchor-Align/issues"))
        print(DIM(f"  with your GPU model, CUDA/PyTorch versions, and this log."))
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--list-configs", action="store_true",
                   help="Print all known configs + benchmarks and exit.")
    p.add_argument("--config", type=str,
                   help="Config key. See --list-configs.")
    p.add_argument("--benchmark", type=str,
                   choices=["spatial_std", "libero_pro", "libero_plus"],
                   help="Which benchmark family.")
    p.add_argument("--perturbation", type=str, choices=["lan", "object", "swap"],
                   help="Required when --benchmark libero_pro.")
    p.add_argument("--log", type=str,
                   help="Path to a Standard/PRO eval log (.txt or .log).")
    p.add_argument("--plus-shards", nargs="+", default=None,
                   help="Glob(s) or explicit paths to LIBERO-Plus shard JSONs to aggregate.")
    p.add_argument("--warn-tol", type=float, default=2.0,
                   help="Absolute tolerance in pp for the warning band. Default: 2.0")
    p.add_argument("--fail-tol", type=float, default=3.0,
                   help="Absolute tolerance in pp for the failure band. Default: 3.0")
    args = p.parse_args()

    if args.list_configs:
        list_configs()
        return 0

    if not args.config or not args.benchmark:
        p.print_help()
        return 1

    if args.benchmark == "libero_pro" and not args.perturbation:
        print(RED("error: --benchmark libero_pro requires --perturbation {lan,object,swap}"))
        return 1

    paper = lookup_paper_number(args.config, args.benchmark, args.perturbation)
    if paper is None:
        print(RED(f"error: no paper number for config={args.config} benchmark={args.benchmark}"
                  + (f" perturbation={args.perturbation}" if args.perturbation else "")))
        print(DIM("Run --list-configs to see supported combinations."))
        return 1

    # Parse local result
    if args.benchmark == "libero_plus":
        if not args.plus_shards:
            print(RED("error: --benchmark libero_plus requires --plus-shards <glob-or-path>"))
            return 1
        paths: list[Path] = []
        for spec in args.plus_shards:
            expanded = glob.glob(spec)
            if expanded:
                paths.extend(Path(x) for x in expanded)
            elif Path(spec).exists():
                paths.append(Path(spec))
        local = parse_plus_shards(paths)
    else:
        if not args.log:
            print(RED("error: --benchmark spatial_std / libero_pro requires --log <path>"))
            return 1
        local = parse_final_from_log(Path(args.log))

    if local is None:
        return 1

    delta = local["sr_pct"] - paper["mean_sr_pct"]
    exit_code, _, verdict_str = verdict(delta, args.warn_tol, args.fail_tol)
    emit_report(args.config, args.benchmark, args.perturbation,
                paper, local, exit_code, verdict_str,
                args.warn_tol, args.fail_tol)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
