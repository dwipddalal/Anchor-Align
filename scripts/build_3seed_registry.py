#!/usr/bin/env python3
"""Build a registry of 3-seed multi-seed eval results, mapping each config
to its checkpoint path, seeds, log locations, and final aggregated numbers.

Outputs:
  results/3seed_registry.json   (machine-readable)
  results/3seed_registry.md     (human-readable summary)
"""
import glob
import json
import os
import re
import statistics
from collections import defaultdict

ROOT = os.environ.get("REPO_DIR", ".")
LOGS = os.path.join(ROOT, "experiments/logs/3seed")

# Authoritative checkpoint mapping. Edit if new configs are added.
CONFIGS = {
    "anc_v7kl01_10k": {
        "label": "Anchor-Align (v7+KL=0.1)",
        "description": "v7 alignment (aw=0.02, threshold=0.15) + KL distillation (kw=0.1) on libero_spatial. Headline method.",
        "checkpoint_path": "Dwipz/Anchor-Align/libero-spatial",
        "modal_volume_path": "/checkpoints/anchor_v7kl01_10k",
        "wandb_run_id": "(internal)",
        "training": {
            "dataset": "libero_spatial_no_noops",
            "max_steps": 10000,
            "batch_size": 32,
            "learning_rate": 2e-4,
            "lora_rank": 64,
            "image_aug": True,
            "kl_loss_weight": 0.1,
            "kl_layers": "all",
            "kl_sigma": 1.0,
            "use_alignment": True,
            "align_version": 7,
            "align_weight": 0.02,
            "align_dir_l2_threshold": 0.15,
        },
    },
    "bl_action_10k": {
        "label": "Action-only baseline",
        "description": "Standard BC: L1-regression action head only (no anchoring, no alignment).",
        "checkpoint_path": "(internal, not released) baseline-spatial-10k--20260227_042158--10000_chkpt",
        "modal_volume_path": "/checkpoints/baseline_action_10k",
        "wandb_run_id": "bsb4fnzu",
        "training": {
            "dataset": "libero_spatial_no_noops",
            "max_steps": 10000,
            "batch_size": 32,
            "learning_rate": 2e-4,
            "lora_rank": 64,
            "image_aug": True,
            "kl_loss_weight": 0,
            "use_alignment": False,
        },
    },
}

SEEDS = [7, 21, 42]
EVAL_CONFIG = {
    "spatial_std": {"num_parallel_envs": 50, "num_trials_per_task": 50, "use_seed_in_env": True},
    "libero_pro": {"num_parallel_envs": 50, "use_seed_in_env": True, "perturbations": ["lan", "object", "swap"]},
    "libero_plus": {"batch_size": 48, "num_shards": 4, "shard_ranges": [[0, 600], [600, 1200], [1200, 1800], [1800, 2402]]},
}


def parse_pct_log(path):
    try:
        text = open(path).read()
    except OSError:
        return None
    m = re.search(r"FINAL:\s*(\d+)/(\d+)", text)
    if m:
        s, n = int(m.group(1)), int(m.group(2))
        return {"successes": s, "episodes": n, "sr_pct": s / n * 100 if n else None}
    eps = re.findall(r"Total episodes:\s*(\d+)", text)
    suc = re.findall(r"Total successes:\s*(\d+)", text)
    if eps and suc:
        s, n = int(suc[-1]), int(eps[-1])
        return {"successes": s, "episodes": n, "sr_pct": s / n * 100 if n else None}
    return None


def collect_one(cfg, kind, key=None):
    """For a (cfg, kind) pair, find seed-keyed eval logs and return per-seed results."""
    if kind == "spatial_std":
        pattern = os.path.join(LOGS, "spatial_std", f"{cfg}_seed*--*.log")
    elif kind == "libero_pro":
        pattern = os.path.join(LOGS, "libero_pro", f"{cfg}_{key}_seed*--*.log")
    else:
        return {}

    by_seed = {}
    for f in sorted(glob.glob(pattern)):
        m = re.search(rf"seed(\d+)", os.path.basename(f))
        if not m:
            continue
        seed = int(m.group(1))
        parsed = parse_pct_log(f)
        if parsed and parsed["sr_pct"] is not None:
            # Keep latest valid result for each seed
            by_seed[seed] = {**parsed, "log": os.path.relpath(f, ROOT)}
    return by_seed


def _parse_plus_log_sr(log_path):
    """Parse Plus log for cumulative-progress 'X/Y = Z.Z%' lines. The LAST one in the
    file is the shard's final total (or, if it crashed, partial). Returns (succ, n)
    or None.
    """
    try:
        text = open(log_path).read()
    except OSError:
        return None
    # The eval prints '... | X/Y = Z.Z%' as a running counter and again as a final
    # summary. Rich-formatted multi-line wrapping breaks 'Successes:' across lines,
    # so we just match the bare 'X/Y = Z.Z%' pattern and take the last occurrence.
    matches = re.findall(r"(\d+)/(\d+) = \d+\.\d+%", text)
    if matches:
        s, n = int(matches[-1][0]), int(matches[-1][1])
        return (s, n) if n > 0 else None
    return None


def collect_plus(cfg):
    """Plus is sharded: aggregate the 4 shards per seed.
    Sources (in order of preference):
      1. JSON results in experiments/logs/libero_plus/ (Delta or Modal-with-copy-fix)
      2. JSON results pulled from Modal volume to a sibling dir
      3. Log files on Modal volume parsed for 'Successes: X/Y = Z.Z%'
    """
    pattern = os.path.join(ROOT, "experiments/logs/libero_plus", f"*3seed_{cfg}_seed*_shard*_results.json")
    by_seed = defaultdict(lambda: {"shards": {}, "successes": 0, "episodes": 0, "shard_count": 0})
    seen_keys = set()
    for f in sorted(glob.glob(pattern)):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        m = re.search(rf"3seed_{cfg}_seed(\d+)_shard(\d+)", os.path.basename(f))
        if not m:
            continue
        seed = int(m.group(1))
        shard = int(m.group(2))
        sr = d.get("summary", {}).get("overall_sr")
        n = len(d.get("per_task_results", []))
        if sr is None or n == 0:
            continue
        succ = round(sr / 100 * n)
        seen_keys.add((seed, shard))
        # Avoid double-counting: only keep latest shard if multiple match
        if shard not in by_seed[seed]["shards"]:
            by_seed[seed]["shards"][shard] = {"successes": succ, "episodes": n, "result_json": os.path.relpath(f, ROOT)}
            by_seed[seed]["successes"] += succ
            by_seed[seed]["episodes"] += n
            by_seed[seed]["shard_count"] += 1

    # Fallback: parse Modal /vol log files for cells we don't have JSON for
    # Logs filename pattern: {cfg}_seed{seed}_shard{shard}--YYYYMMDD_HHMMSS.log
    # These exist if `modal volume get /results/libero_plus ...` was run
    log_dirs = [
        os.path.join(ROOT, "experiments/logs/3seed/libero_plus"),
        "/tmp",  # ad-hoc pulls land here
    ]
    log_pattern = rf"{cfg}_seed(\d+)_shard(\d+)--\d+_\d+\.log"
    # Collect ALL candidate logs per (seed, shard) and pick the one with the most
    # episodes — modal-restarted shards leave partial logs; we want the one that
    # actually finished (or at least covered the most tasks).
    candidates = defaultdict(list)  # (seed, shard) -> [(succ, n, path), ...]
    for log_dir in log_dirs:
        if not os.path.isdir(log_dir):
            continue
        for fname in sorted(os.listdir(log_dir)):
            m = re.match(log_pattern, fname)
            if not m:
                continue
            seed = int(m.group(1))
            shard = int(m.group(2))
            if (seed, shard) in seen_keys:
                continue
            f = os.path.join(log_dir, fname)
            parsed = _parse_plus_log_sr(f)
            if parsed:
                candidates[(seed, shard)].append((parsed[0], parsed[1], f))
    for (seed, shard), entries in candidates.items():
        # Pick the entry with the most episodes evaluated
        entries.sort(key=lambda e: (-e[1], e[2]))
        succ, n, f = entries[0]
        if shard not in by_seed[seed]["shards"]:
            by_seed[seed]["shards"][shard] = {
                "successes": succ, "episodes": n, "result_json": None, "from_log": os.path.relpath(f, ROOT) if f.startswith(ROOT) else f,
            }
            by_seed[seed]["successes"] += succ
            by_seed[seed]["episodes"] += n
            by_seed[seed]["shard_count"] += 1
            seen_keys.add((seed, shard))

    out = {}
    for seed, v in by_seed.items():
        if v["episodes"]:
            out[seed] = {
                "successes": v["successes"],
                "episodes": v["episodes"],
                "sr_pct": v["successes"] / v["episodes"] * 100,
                "shards_complete": v["shard_count"],
                "shards_expected": 4,
                "shard_breakdown": v["shards"],
            }
    return out


def aggregate(per_seed):
    """Compute mean ± stderr from a {seed: {sr_pct: ...}} dict."""
    pcts = [v["sr_pct"] for v in per_seed.values() if v.get("sr_pct") is not None]
    if not pcts:
        return None
    n = len(pcts)
    mean = statistics.mean(pcts)
    stderr = statistics.stdev(pcts) / (n ** 0.5) if n > 1 else None
    return {"mean_sr_pct": mean, "stderr_pp": stderr, "n_seeds": n, "seeds": sorted(per_seed.keys())}


def main():
    registry = {
        "_meta": {
            "campaign": "VLA-Adapter 3-seed multi-seed eval",
            "created": "2026-05-08",
            "seeds": SEEDS,
            "eval_config": EVAL_CONFIG,
            "notes": [
                "All evals use --use_seed_in_env=True (cfg.seed mixed into env_seed) per the if-else wrap added 2026-05-07.",
                "For Plus: env_seed = cfg.seed + task_id (eval script's native logic).",
                "For std/PRO: env_seed = cfg.seed * 100000 + task_id * 1000 + ep_idx (when use_seed_in_env=True).",
                "Eval scripts: experiments/robot/libero/run_libero_eval_batched.py, experiments/robot/libero_pro/run_libero_pro_eval.py, experiments/robot/libero_plus/run_libero_plus_eval_batched.py",
                "Modal H100 used for some evals (anchor std s7, anchor s42 sh2).",
                "Delta GH200 used for the rest.",
            ],
        },
        "configs": {},
    }

    for cfg, info in CONFIGS.items():
        std = collect_one(cfg, "spatial_std")
        pro = {pert: collect_one(cfg, "libero_pro", pert) for pert in EVAL_CONFIG["libero_pro"]["perturbations"]}
        plus = collect_plus(cfg)

        registry["configs"][cfg] = {
            **info,
            "results": {
                "spatial_std": {
                    "per_seed": std,
                    "aggregate": aggregate(std),
                },
                "libero_pro": {
                    pert: {"per_seed": pro[pert], "aggregate": aggregate(pro[pert])}
                    for pert in EVAL_CONFIG["libero_pro"]["perturbations"]
                },
                "libero_plus": {
                    "per_seed": plus,
                    "aggregate": aggregate(plus),
                },
            },
        }

    # ---- Write JSON ----
    out_dir = os.path.join(ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "3seed_registry.json")
    with open(json_path, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"JSON written: {json_path}")

    # ---- Write Markdown summary ----
    md_path = os.path.join(out_dir, "3seed_registry.md")
    md = ["# 3-seed eval registry\n\n",
          "*Auto-generated by `scripts/build_3seed_registry.py`. Source of truth for which checkpoint produced which number.*\n\n"]

    md.append("## Configs\n\n")
    md.append("| Tag | Label | Checkpoint path |\n|---|---|---|\n")
    for cfg, info in CONFIGS.items():
        md.append(f"| `{cfg}` | {info['label']} | `{info['checkpoint_path']}` |\n")
    md.append("\n")

    md.append("## Aggregated results (mean ± stderr across seeds)\n\n")
    md.append("| Config | Spatial std | PRO LAN | PRO OBJ | PRO SWAP | Plus |\n")
    md.append("|---|---:|---:|---:|---:|---:|\n")
    for cfg in CONFIGS:
        rec = registry["configs"][cfg]["results"]
        cells = []
        for kind, key in [("spatial_std", None), ("libero_pro", "lan"), ("libero_pro", "object"),
                          ("libero_pro", "swap"), ("libero_plus", None)]:
            agg = rec[kind][key]["aggregate"] if key else rec[kind]["aggregate"]
            if agg is None:
                cells.append("—")
            elif agg.get("stderr_pp") is None:
                cells.append(f"{agg['mean_sr_pct']:.2f} (n={agg['n_seeds']})")
            else:
                cells.append(f"{agg['mean_sr_pct']:.2f} ± {agg['stderr_pp']:.2f} (n={agg['n_seeds']})")
        md.append(f"| **{CONFIGS[cfg]['label']}** | " + " | ".join(cells) + " |\n")
    md.append("\n")

    md.append("## Per-seed breakdown\n\n")
    for cfg, info in CONFIGS.items():
        md.append(f"### {info['label']} (`{cfg}`)\n")
        md.append(f"- **Checkpoint**: `{info['checkpoint_path']}`\n")
        md.append(f"- **Modal volume path**: `{info['modal_volume_path']}`\n")
        md.append(f"- **WandB run**: {info['wandb_run_id']}\n\n")

        rec = registry["configs"][cfg]["results"]

        # Std
        std = rec["spatial_std"]["per_seed"]
        if std:
            md.append("#### LIBERO Spatial std (500 eps)\n")
            md.append("| Seed | s/n | SR | Log |\n|---:|---:|---:|---|\n")
            for s in sorted(std.keys()):
                v = std[s]
                md.append(f"| {s} | {v['successes']}/{v['episodes']} | {v['sr_pct']:.2f}% | `{v['log']}` |\n")
            agg = rec["spatial_std"]["aggregate"]
            if agg and agg.get("stderr_pp") is not None:
                md.append(f"| **Mean ± SE** | | **{agg['mean_sr_pct']:.2f} ± {agg['stderr_pp']:.2f}** (n={agg['n_seeds']}) | |\n\n")
            else:
                md.append("\n")

        # PRO
        for pert in EVAL_CONFIG["libero_pro"]["perturbations"]:
            data = rec["libero_pro"][pert]["per_seed"]
            if data:
                md.append(f"#### LIBERO-PRO {pert.upper()} (500 eps)\n")
                md.append("| Seed | s/n | SR | Log |\n|---:|---:|---:|---|\n")
                for s in sorted(data.keys()):
                    v = data[s]
                    md.append(f"| {s} | {v['successes']}/{v['episodes']} | {v['sr_pct']:.2f}% | `{v['log']}` |\n")
                agg = rec["libero_pro"][pert]["aggregate"]
                if agg and agg.get("stderr_pp") is not None:
                    md.append(f"| **Mean ± SE** | | **{agg['mean_sr_pct']:.2f} ± {agg['stderr_pp']:.2f}** (n={agg['n_seeds']}) | |\n\n")
                else:
                    md.append("\n")

        # Plus
        plus = rec["libero_plus"]["per_seed"]
        if plus:
            md.append("#### LIBERO-Plus (~2402 eps per seed, 4-shard aggregate)\n")
            md.append("| Seed | s/n | SR | Shards complete |\n|---:|---:|---:|---:|\n")
            for s in sorted(plus.keys()):
                v = plus[s]
                md.append(f"| {s} | {v['successes']}/{v['episodes']} | {v['sr_pct']:.2f}% | {v['shards_complete']}/{v['shards_expected']} |\n")
            agg = rec["libero_plus"]["aggregate"]
            if agg and agg.get("stderr_pp") is not None:
                md.append(f"| **Mean ± SE** | | **{agg['mean_sr_pct']:.2f} ± {agg['stderr_pp']:.2f}** (n={agg['n_seeds']}) | |\n\n")
            else:
                md.append("\n")
        md.append("\n---\n\n")

    md.append("## Reproducing\n\n")
    md.append("```bash\n# Run aggregator + registry build\npython scripts/aggregate_3seed_results.py\npython scripts/build_3seed_registry.py\n```\n")

    with open(md_path, "w") as f:
        f.writelines(md)
    print(f"Markdown written: {md_path}")


if __name__ == "__main__":
    main()
