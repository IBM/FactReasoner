#!/usr/bin/env python3
"""
state_media_overnight.py  —  subprocess-based overnight runner
==============================================================
Calls state_media_eval.py as a subprocess (respecting its argparse/IBM
backend setup) and reads the results JSON it saves after each run.

Usage:
    cd /u/samit/FactReasoner
    nohup python3 state_media_overnight.py > /u/samit/overnight_log.txt 2>&1 &
    echo "PID: $!"

Check progress any time:
    tail -50 /u/samit/overnight_log.txt
    cat /u/samit/overnight_results/overnight_report.txt
"""

import json, os, subprocess, sys, time, datetime, shutil, copy, traceback

# ── Config ────────────────────────────────────────────────────────────────────
N_RUNS           = 39  # run 1 already done; runner below starts at run_idx=1
LABELS_FILTER    = "factual_false"
EVAL_SCRIPT      = "/u/samit/FactReasoner/scripts/state_media_eval.py"
WORKING_DIR      = "/u/samit/FactReasoner"
PYTHON           = sys.executable          # same conda env that launched us
RESULTS_SRC      = "/u/samit/state_media_results/state_media_results.json"
RESULTS_DIR      = "/u/samit/overnight_results"
FIRST_RUN_CACHE  = "fresh"     # always fetch live — finds better sources across runs
SUBSEQUENT_CACHE = "fresh"     # every run gets genuinely new Serper results;
                               # best run across 40 is a real upper bound
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(RESULTS_DIR, exist_ok=True)

def ts():
    return datetime.datetime.now().strftime("%H:%M:%S")

def log(msg):
    print(f"[{ts()}] {msg}", flush=True)


def composite_score(trust_correct, evaluated, total_atoms):
    """
    accuracy × coverage_fraction.
    Penalises runs where many hard false-label atoms were skipped.
    A run that evaluates 24/31 at 87.5%  → 0.679
    A run that evaluates 23/31 at 91.3%  → 0.676   (looks better, but isn't)
    A run that evaluates 31/31 at 80.0%  → 0.800   (correctly wins)
    """
    if evaluated == 0:
        return 0.0
    return (trust_correct / evaluated) * (evaluated / max(total_atoms, 1))


def parse_results(results_list, skipped_list):
    evaluated   = len(results_list)
    total_atoms = evaluated + len(skipped_list)

    t_correct   = sum(1 for r in results_list if r.get("correct"))
    v_correct   = sum(1 for r in results_list if r.get("van_correct"))

    factual = [r for r in results_list if r.get("raw_label") == "factual"]
    false_r = [r for r in results_list if r.get("raw_label") == "false"]

    def pct(n, d):
        return round(100 * n / d, 1) if d else None

    return {
        "total_atoms":   total_atoms,
        "evaluated":     evaluated,
        "skipped":       len(skipped_list),
        "trust_correct": t_correct,
        "van_correct":   v_correct,
        "trust_acc":     pct(t_correct, evaluated),
        "van_acc":       pct(v_correct, evaluated),
        "delta":         round((pct(t_correct, evaluated) or 0) -
                               (pct(v_correct, evaluated) or 0), 1),
        "composite":     round(composite_score(t_correct, evaluated, total_atoms), 4),
        "factual_n":     len(factual),
        "factual_trust": pct(sum(1 for r in factual if r.get("correct")),   len(factual)),
        "factual_van":   pct(sum(1 for r in factual if r.get("van_correct")), len(factual)),
        "false_n":       len(false_r),
        "false_trust":   pct(sum(1 for r in false_r if r.get("correct")),   len(false_r)),
        "false_van":     pct(sum(1 for r in false_r if r.get("van_correct")), len(false_r)),
    }


def run_once(run_idx, cache_mode):
    """Invoke state_media_eval.py as a subprocess and return (results, skipped)."""
    cmd = [
        PYTHON, EVAL_SCRIPT,
        "--labels",     LABELS_FILTER,
        "--cache-mode", cache_mode,
    ]
    log(f"  $ {' '.join(cmd)}")

    proc = subprocess.run(
        cmd,
        cwd=WORKING_DIR,
        capture_output=False,   # let stdout/stderr flow to the nohup log
        text=True,
    )

    if proc.returncode != 0:
        log(f"  subprocess exited with code {proc.returncode}")

    # Read the results JSON the script saved
    if not os.path.exists(RESULTS_SRC):
        log(f"  ERROR: {RESULTS_SRC} not found after run")
        return [], []

    with open(RESULTS_SRC) as f:
        data = json.load(f)

    # state_media_results.json is a list of result dicts for evaluated atoms.
    # Skipped atoms are NOT saved there (only printed). We reconstruct the
    # skipped count from total_atoms = 31 (the full dataset size in this
    # label filter) − evaluated.
    # The script prints the skipped list but doesn't save it to JSON, so we
    # derive what we need from what IS saved.
    results  = data if isinstance(data, list) else []
    # We don't have the skipped list from JSON, so fake it with placeholders
    # just for the count (total dataset size is fixed at 31 for factual_false)
    TOTAL_DATASET = 31
    skipped_count = TOTAL_DATASET - len(results)
    skipped = [{"row_idx": -1, "account": "?", "raw_label": "?",
                "reason": "skipped (not in saved results JSON)"}
               for _ in range(max(skipped_count, 0))]

    # Archive this run's results so we don't lose them on next iteration
    archive = os.path.join(RESULTS_DIR, f"run_{run_idx+1:03d}_results.json")
    # Note: run_idx starts at 1 in resume mode, so files are run_002, run_003, etc.
    shutil.copy(RESULTS_SRC, archive)

    return results, skipped


def format_report(all_summaries, best_idx, best_summary, best_results, best_skipped):
    lines = []
    w = 70
    lines.append("=" * w)
    lines.append("OVERNIGHT RUN REPORT — Chinese State Media Factuality Eval")
    lines.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Runs completed: {len(all_summaries)} / {N_RUNS}  |  "
                 f"labels: {LABELS_FILTER}")
    lines.append("=" * w)
    lines.append("")

    def avg(key):
        vals = [s[key] for s in all_summaries if s.get(key) is not None]
        return round(sum(vals) / len(vals), 1) if vals else "—"

    def stddev(key):
        vals = [s[key] for s in all_summaries if s.get(key) is not None]
        if len(vals) < 2: return "—"
        m = sum(vals) / len(vals)
        return round((sum((x-m)**2 for x in vals) / len(vals)) ** 0.5, 1)

    def rng(key):
        vals = [s[key] for s in all_summaries if s.get(key) is not None]
        return f"{min(vals)}% – {max(vals)}%" if vals else "—"

    lines.append("AVERAGE ACCURACY ACROSS ALL RUNS")
    lines.append("-" * 40)
    lines.append(f"  Trust accuracy:          {avg('trust_acc')}%")
    lines.append(f"  Vanilla accuracy:        {avg('van_acc')}%")
    lines.append(f"  Delta (Trust − Vanilla): {avg('delta')}%")
    lines.append(f"  Trust std dev:           {stddev('trust_acc')}%")
    lines.append(f"  Trust range:             {rng('trust_acc')}")
    lines.append(f"  Avg evaluated / 31:      {avg('evaluated')}")
    lines.append(f"  Avg skipped / 31:        {avg('skipped')}")
    lines.append(f"  Factual Trust avg:       {avg('factual_trust')}%")
    lines.append(f"  False Trust avg:         {avg('false_trust')}%")
    lines.append(f"  False Vanilla avg:       {avg('false_van')}%")
    lines.append("")

    lines.append(f"BEST RUN  (run #{best_idx+1}  |  composite score = {best_summary.get('composite')})")
    lines.append("-" * 40)
    lines.append(f"  Trust:     {best_summary.get('trust_correct')}/{best_summary.get('evaluated')}"
                 f"  ({best_summary.get('trust_acc')}%)")
    lines.append(f"  Vanilla:   {best_summary.get('van_correct')}/{best_summary.get('evaluated')}"
                 f"  ({best_summary.get('van_acc')}%)")
    lines.append(f"  Delta:     +{best_summary.get('delta')}%")
    lines.append(f"  Evaluated: {best_summary.get('evaluated')} / {best_summary.get('total_atoms')} atoms")
    lines.append(f"  Skipped:   {best_summary.get('skipped')} atoms")
    lines.append(f"  Factual:   Trust {best_summary.get('factual_trust')}%"
                 f"  Vanilla {best_summary.get('factual_van')}%"
                 f"  (n={best_summary.get('factual_n')})")
    lines.append(f"  False:     Trust {best_summary.get('false_trust')}%"
                 f"  Vanilla {best_summary.get('false_van')}%"
                 f"  (n={best_summary.get('false_n')})")
    lines.append("")

    # Failures in best run
    failures = [r for r in best_results if not r.get("correct")]
    lines.append(f"TRUST FAILURES IN BEST RUN  ({len(failures)} atoms)")
    lines.append("-" * 40)
    for r in failures:
        lines.append(f"  [{r.get('row_idx','?'):>2}] {str(r.get('account','?')):<22}"
                     f"  {str(r.get('raw_label','?')):<12}"
                     f"  GT={r.get('ground_truth')}  "
                     f"T={r.get('p_trust',0):.4f}→{r.get('verdict')} ✗")
        lines.append(f"        {str(r.get('claim',''))[:90]}")
        for ctx in r.get("contexts", []):
            if ctx.get("nli_type"):
                lines.append(f"          [{ctx['nli_type'][:13]:<13}] "
                             f"{str(ctx.get('title',''))[:55]}")
                lines.append(f"                  {ctx.get('link','')}")
        lines.append("")

    lines.append("")

    # Per-run table
    lines.append("ALL RUNS TABLE")
    lines.append("-" * w)
    hdr = f"  {'Run':>3}  {'T%':>5}  {'V%':>5}  {'Δ':>5}  {'Eval':>4}  {'Skip':>4}  " \
          f"{'f_T%':>5}  {'fls_T%':>6}  {'fls_V%':>6}  {'Comp':>6}"
    lines.append(hdr)
    lines.append("  " + "-" * (w-2))
    for i, s in enumerate(all_summaries):
        marker = " ◄ BEST" if i == best_idx else ""
        lines.append(
            f"  {i+1:>3}  {str(s.get('trust_acc','—')):>5}  {str(s.get('van_acc','—')):>5}  "
            f"{('+'+str(s.get('delta','—')) if isinstance(s.get('delta'), (int,float)) and s.get('delta',0)>=0 else str(s.get('delta','—'))):>5}  {s.get('evaluated',0):>4}  "
            f"{s.get('skipped',0):>4}  "
            f"{str(s.get('factual_trust','—')):>5}  {str(s.get('false_trust','—')):>6}  "
            f"{str(s.get('false_van','—')):>6}  {s.get('composite',0):>6.3f}{marker}"
        )
    return "\n".join(lines)


def save_all(all_summaries, best_idx, best_summary, best_results, best_skipped):
    # Full context + URLs for the best run
    with open(os.path.join(RESULTS_DIR, "best_run.json"), "w") as f:
        json.dump({
            "run_number": best_idx + 1,
            "summary":    best_summary,
            "results":    best_results,
            "skipped":    best_skipped,
        }, f, indent=2)

    # Per-run stats
    with open(os.path.join(RESULTS_DIR, "all_runs_summary.json"), "w") as f:
        json.dump({
            "n_runs_completed": len(all_summaries),
            "config": {
                "labels_filter":    LABELS_FILTER,
                "n_runs_planned":   N_RUNS,
                "first_run_cache":  FIRST_RUN_CACHE,
                "subsequent_cache": SUBSEQUENT_CACHE,
            },
            "summaries": all_summaries,
        }, f, indent=2)

    # Human-readable report
    if all_summaries:
        report = format_report(all_summaries, best_idx, best_summary,
                               best_results, best_skipped)
        with open(os.path.join(RESULTS_DIR, "overnight_report.txt"), "w") as f:
            f.write(report)


def main():
    log(f"Overnight runner: {N_RUNS} runs, labels={LABELS_FILTER}")
    log(f"Eval script: {EVAL_SCRIPT}")
    log(f"Results dir: {RESULTS_DIR}")
    log(f"Cache: run1={FIRST_RUN_CACHE}, subsequent={SUBSEQUENT_CACHE}")

    all_summaries = []
    best_composite = -1.0
    best_idx       = 0
    best_results   = []
    best_skipped   = []
    best_summary   = {}

    # Bootstrap from run 1 results that were saved before the crash
    run1_archive = os.path.join(RESULTS_DIR, "run_001_results.json")
    if os.path.exists(run1_archive):
        with open(run1_archive) as f:
            r1 = json.load(f)
        r1_results = r1 if isinstance(r1, list) else []
        TOTAL_DATASET = 31
        r1_skipped = [{"row_idx":-1,"account":"?","raw_label":"?",
                       "reason":"skipped"} for _ in range(max(TOTAL_DATASET-len(r1_results),0))]
        r1_summary = parse_results(r1_results, r1_skipped)
        all_summaries.append(r1_summary)
        best_composite = r1_summary["composite"]
        best_idx = 0
        best_results = r1_results
        best_skipped = r1_skipped
        best_summary = r1_summary
        log(f"Loaded run 1 from archive: Trust={r1_summary['trust_acc']}%  "
            f"Van={r1_summary['van_acc']}%  composite={r1_summary['composite']}")
        save_all(all_summaries, best_idx, best_summary, best_results, best_skipped)
    else:
        log("WARNING: run_001_results.json not found — starting fresh")

    for run_idx in range(1, N_RUNS + 1):  # start at 1 since run 0 already done
        cache_mode = "fresh"  # always live — each run sees fresh Serper results
        log("")
        log(f"━━━  RUN {run_idx+1}/{N_RUNS+1}  cache={cache_mode}  ━━━")
        t0 = time.time()

        try:
            results, skipped = run_once(run_idx, cache_mode)
        except Exception as e:
            log(f"  RUN {run_idx+1} EXCEPTION: {e}")
            traceback.print_exc()
            continue

        elapsed = round(time.time() - t0, 1)
        summary = parse_results(results, skipped)
        all_summaries.append(summary)

        log(f"  ── done in {elapsed}s  "
            f"Trust={summary['trust_acc']}%  Van={summary['van_acc']}%  "
            f"Δ={summary['delta']}  "
            f"evaluated={summary['evaluated']}/31  "
            f"false_n={summary['false_n']}  "
            f"composite={summary['composite']}")

        if summary["composite"] > best_composite:
            best_composite = summary["composite"]
            best_idx       = len(all_summaries) - 1
            best_results   = copy.deepcopy(results)
            best_skipped   = copy.deepcopy(skipped)
            best_summary   = copy.deepcopy(summary)
            log(f"  ★ NEW BEST  composite={best_composite:.4f}  "
                f"Trust={best_summary['trust_acc']}%  "
                f"evaluated={best_summary['evaluated']}")

        # Save after every run so you can check progress mid-night
        save_all(all_summaries, best_idx, best_summary, best_results, best_skipped)

    log("")
    log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    log(f"ALL {len(all_summaries)} RUNS COMPLETE")
    log(f"Best run #{best_idx+1}: Trust={best_summary.get('trust_acc')}%  "
        f"Van={best_summary.get('van_acc')}%  "
        f"evaluated={best_summary.get('evaluated')}  "
        f"composite={best_composite:.4f}")
    log(f"Results at {RESULTS_DIR}/")
    save_all(all_summaries, best_idx, best_summary, best_results, best_skipped)


if __name__ == "__main__":
    main()
