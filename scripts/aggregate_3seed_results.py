#!/usr/bin/env python3
"""Aggregate 3-seed eval results into mean ± stderr table.

Reads logs from experiments/logs/3seed/{spatial_std,libero_pro,libero_plus}
and prints a markdown summary. Run after all 48 jobs complete.
"""
import glob
import json
import os
import re
import statistics
from collections import defaultdict

ROOT = os.environ.get("REPO_DIR", ".")
LOGS = os.path.join(ROOT, "experiments/logs/3seed")


def parse_pct_log(path):
    try:
        with open(path) as f:
            text = f.read()
    except OSError:
        return None
    m = re.search(r"FINAL:\s*(\d+)/(\d+)", text)
    if m:
        s, n = int(m.group(1)), int(m.group(2))
        return (s / n * 100, s, n) if n else None
    eps = re.findall(r"Total episodes:\s*(\d+)", text)
    suc = re.findall(r"Total successes:\s*(\d+)", text)
    if eps and suc:
        return int(suc[-1]) / int(eps[-1]) * 100, int(suc[-1]), int(eps[-1])
    return None


def collect(pattern, name_re):
    """Return dict[(cfg, key)] -> list[(seed, pct, succ, eps)]."""
    out = defaultdict(list)
    for f in sorted(glob.glob(pattern)):
        m = re.search(name_re, os.path.basename(f))
        if not m:
            continue
        d = m.groupdict()
        cfg = d["cfg"]
        seed = int(d["seed"])
        key = d.get("key", "")
        parsed = parse_pct_log(f)
        if parsed is None:
            continue
        pct, s, n = parsed
        out[(cfg, key)].append((seed, pct, s, n, f))
    return out


def collect_plus_shards(pattern):
    """Plus is sharded — aggregate shards per (cfg, seed) by reading json results."""
    by_cfg_seed = defaultdict(lambda: {"succ": 0, "eps": 0, "shards": 0})
    for f in sorted(glob.glob(pattern)):
        # filename: 3seed_{cfg}_seed{seed}_shard{N}_results.json (from run_id_note)
        # or fallback: directly from log
        try:
            d = json.load(open(f))
        except (OSError, json.JSONDecodeError):
            continue
        note = d.get("config", {}).get("run_id_note") or ""
        # Fall back to filename if config.run_id_note is None/missing
        if not note:
            base = os.path.basename(f)
            m_fn = re.search(r"3seed_(\w+_seed\d+_shard\d+)", base)
            if m_fn:
                note = "3seed_" + m_fn.group(1)
        m = re.match(r"3seed_(?P<cfg>[^_]+(?:_[^_]+)*?)_seed(?P<seed>\d+)_shard\d+", note)
        if not m:
            continue
        cfg = m.group("cfg")
        seed = int(m.group("seed"))
        sr = d.get("summary", {}).get("overall_sr")
        n = len(d.get("per_task_results", []))
        if sr is None or n == 0:
            continue
        succ = round(sr / 100 * n)
        by_cfg_seed[(cfg, seed)]["succ"] += succ
        by_cfg_seed[(cfg, seed)]["eps"] += n
        by_cfg_seed[(cfg, seed)]["shards"] += 1
    out = {}
    for (cfg, seed), v in by_cfg_seed.items():
        if v["eps"]:
            out[(cfg, seed)] = (v["succ"] / v["eps"] * 100, v["succ"], v["eps"], v["shards"])
    return out


def fmt_seeds(seeds_pcts):
    """seeds_pcts: list[(seed, pct, ...)]; return mean, stderr, n."""
    pcts = [p for _, p, *_ in seeds_pcts]
    if not pcts:
        return None, None, 0
    if len(pcts) == 1:
        return pcts[0], None, 1
    mean = statistics.mean(pcts)
    stderr = statistics.stdev(pcts) / (len(pcts) ** 0.5)
    return mean, stderr, len(pcts)


def fmt(mean, stderr, n):
    if mean is None:
        return "—"
    if stderr is None:
        return f"{mean:.2f} (n=1)"
    return f"{mean:.2f} ± {stderr:.2f} (n={n})"


def main():
    print("# 3-seed eval results (mean ± stderr across seeds)\n")

    # Spatial std
    sp = collect(
        os.path.join(LOGS, "spatial_std", "*.log"),
        r"(?P<cfg>anc_v7kl01_10k|bl_action_10k)_seed(?P<seed>\d+)",
    )
    print("## LIBERO Spatial std (500 eps)\n")
    print("| Config | Mean ± SE | Seeds |")
    print("|---|---|---|")
    for cfg in ["anc_v7kl01_10k", "bl_action_10k"]:
        runs = sp.get((cfg, ""), [])
        mean, se, n = fmt_seeds(runs)
        seeds = sorted({s for s, *_ in runs})
        print(f"| {cfg} | {fmt(mean, se, n)} | {seeds} |")
    print()

    # PRO
    pro = collect(
        os.path.join(LOGS, "libero_pro", "*.log"),
        r"(?P<cfg>anc_v7kl01_10k|bl_action_10k)_(?P<key>lan|object|swap)_seed(?P<seed>\d+)",
    )
    print("## LIBERO-PRO (500 eps per perturbation)\n")
    print("| Config | LAN | OBJECT | SWAP |")
    print("|---|---|---|---|")
    for cfg in ["anc_v7kl01_10k", "bl_action_10k"]:
        cells = []
        for pert in ["lan", "object", "swap"]:
            runs = pro.get((cfg, pert), [])
            mean, se, n = fmt_seeds(runs)
            cells.append(fmt(mean, se, n))
        print(f"| {cfg} | {cells[0]} | {cells[1]} | {cells[2]} |")
    print()

    # Plus (aggregate by shard then by seed)
    plus_by_cfgseed = collect_plus_shards(
        os.path.join(ROOT, "experiments/logs/libero_plus/*3seed_*_results.json")
    )
    print("## LIBERO-Plus (~2402 eps per seed, sharded across 4 jobs)\n")
    print("| Config | Mean ± SE | Per-seed | Total eps/seed |")
    print("|---|---|---|---|")
    by_cfg = defaultdict(list)
    for (cfg, seed), (pct, s, n, sh) in plus_by_cfgseed.items():
        by_cfg[cfg].append((seed, pct, s, n, sh))
    for cfg in ["anc_v7kl01_10k", "bl_action_10k"]:
        runs = sorted(by_cfg.get(cfg, []))
        mean, se, n = fmt_seeds(runs)
        per_seed = ", ".join(f"s{seed}={pct:.1f}% ({sh}/4 sh)" for seed, pct, *_, sh in runs)
        eps_str = ", ".join(str(eps) for *_, eps, _ in runs)
        print(f"| {cfg} | {fmt(mean, se, n)} | {per_seed} | {eps_str} |")
    print()

    # Footer
    print("## Status check\n")
    print(f"- spatial_std logs: {len(glob.glob(os.path.join(LOGS, 'spatial_std', '*.log')))}/6 expected")
    print(f"- libero_pro logs:  {len(glob.glob(os.path.join(LOGS, 'libero_pro', '*.log')))}/18 expected")
    print(f"- libero_plus json: {len(glob.glob(os.path.join(ROOT, 'experiments/logs/libero_plus/*3seed_*_results.json')))}/24 expected")


if __name__ == "__main__":
    main()
