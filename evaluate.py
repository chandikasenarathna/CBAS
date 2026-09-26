# evaluate.py — Offline benchmark of CBAS against the KeyRecs dataset
#
# Replicates the deployed scenario:
#   TRAIN  on a participant's fixed-text windows  (~ fixed-text enrolment)
#   TEST   on that participant's held-out FREE-text windows (~ free-text monitoring)
#   IMPOSTORS = other participants' free-text windows
#
# Modes:
#   --mode fixed : train on fixed-text only (demonstrates the Gap-2 problem)
#   --mode mixed : train on fixed-text + a few free-text windows (the fix)
#
# Outputs (in evaluation/):
#   keyrecs_results.csv  — per-user metrics
#   summary.json         — aggregate metrics (mean EER, FAR/FRR at lock threshold)
#   far_frr_curve.png    — pooled FAR/FRR vs threshold with EER marker
#   det_curve.png        — DET curve
#   score_dist.png       — genuine vs impostor score distributions
#
# Usage:
#   python evaluate.py [--mode fixed|mixed] [--max-users N] [--impostors K]
#                      [--window 30] [--grid] [--seed 42]

import os, sys, json, argparse, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.keyrecs import load_all
from utils.model import UserModel
from utils.metrics import eer, far_frr, sweep
import config

def run_benchmark(participants, mode, max_users, n_impostors, seed,
                  contamination=None, nu=None,
                  n_train_fixed=15, n_train_free=5, n_test=20,
                  min_train=10, min_test=10):
    rng = np.random.default_rng(seed)

    eligible = [p for p, d in participants.items()
                if len(d["fixed"]) >= min_train and len(d["free"]) >= min_test]
    eligible.sort()
    if max_users:
        eligible = eligible[:max_users]
    print(f"[EVAL] Eligible genuine users: {len(eligible)} (mode={mode})")

    rows = []
    pooled_gen, pooled_imp = [], []

    for ui, user in enumerate(eligible, 1):
        data = participants[user]
        if mode == "mixed":
            train = data["fixed"][:n_train_fixed] + data["free"][:n_train_free]
            test  = data["free"][n_train_free : n_train_free + n_test]
        else:
            train = data["fixed"][:n_train_fixed]
            test  = data["free"][:n_test]
        if len(test) < min_test:
            continue

        # Impostors: other participants' free-text windows
        others = [p for p in eligible if p != user]
        imp_users = rng.choice(others, size=min(n_impostors, len(others)), replace=False)
        impostor = []
        for p in imp_users:
            impostor.extend(participants[p]["free"][:10])

        kwargs = {}
        if contamination is not None: kwargs["contamination"] = contamination
        if nu is not None:            kwargs["nu"] = nu
        model = UserModel(**kwargs)
        model.fit(train)

        gen_scores = model.score_batch(test)
        imp_scores = model.score_batch(impostor)

        e, t_eer = eer(gen_scores, imp_scores)
        lock_t = model.thresholds["lock"]
        far_lock, frr_lock = far_frr(gen_scores, imp_scores, lock_t)

        rows.append({
            "user": user,
            "n_train": len(train),
            "n_genuine": len(gen_scores),
            "n_impostor": len(imp_scores),
            "eer": round(e, 4),
            "eer_threshold": round(t_eer, 4),
            "lock_threshold": round(lock_t, 4),
            "far_at_lock": round(far_lock, 4),
            "frr_at_lock": round(frr_lock, 4),
        })
        pooled_gen.extend(gen_scores.tolist())
        pooled_imp.extend(imp_scores.tolist())
        print(f"[EVAL] {ui:>3}/{len(eligible)} {user}: EER={e:.3f} "
              f"FAR@lock={far_lock:.3f} FRR@lock={frr_lock:.3f}")

    return rows, np.array(pooled_gen), np.array(pooled_imp)


def grid_search(participants, seed, max_users=15, **bench_kw):
    print("[GRID] Hyperparameter grid search (mean EER over subset of users)")
    results = []
    for contamination in (0.01, 0.05, 0.1):
        for nu in (0.01, 0.05, 0.1):
            rows, _, _ = run_benchmark(
                participants, mode="fixed", max_users=max_users,
                n_impostors=5, seed=seed, contamination=contamination, nu=nu,
                **bench_kw)
            mean_eer = float(np.mean([r["eer"] for r in rows]))
            results.append((contamination, nu, mean_eer))
            print(f"[GRID] contamination={contamination:<5} nu={nu:<5} -> mean EER={mean_eer:.4f}")
    results.sort(key=lambda r: r[2])
    best_c, best_nu, best_eer = results[0]
    print(f"[GRID] BEST: contamination={best_c}, nu={best_nu}, mean EER={best_eer:.4f}")
    return best_c, best_nu


def make_plots(gen, imp, out_dir, mode):
    thresholds, fars, frrs = sweep(gen, imp, n_points=300)
    e, t_eer = eer(gen, imp)

    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, fars, label="FAR (impostors accepted)", color="#ef4444")
    plt.plot(thresholds, frrs, label="FRR (genuine rejected)", color="#3b82f6")
    plt.axvline(t_eer, ls="--", color="#888", label=f"EER = {e:.3f} @ {t_eer:.2f}")
    plt.xlabel("Acceptance threshold"); plt.ylabel("Error rate")
    plt.title(f"CBAS pooled FAR/FRR vs threshold — KeyRecs ({mode} training)")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "far_frr_curve.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 6))
    plt.plot(fars * 100, frrs * 100, color="#6366f1")
    plt.plot([0, 100], [0, 100], ls="--", color="#aaa")
    plt.scatter([e * 100], [e * 100], color="#ef4444", zorder=5,
                label=f"EER = {e*100:.2f}%")
    plt.xlabel("False Acceptance Rate (%)"); plt.ylabel("False Rejection Rate (%)")
    plt.title(f"DET curve — KeyRecs ({mode} training)")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "det_curve.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5))
    bins = np.linspace(0, 1, 60)
    plt.hist(gen, bins=bins, alpha=.6, density=True, label=f"Genuine (n={len(gen)})",
             color="#22c55e")
    plt.hist(imp, bins=bins, alpha=.6, density=True, label=f"Impostor (n={len(imp)})",
             color="#ef4444")
    plt.xlabel("Combined anomaly score (1 = legitimate)"); plt.ylabel("Density")
    plt.title(f"Score distributions — KeyRecs ({mode} training)")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "score_dist.png"), dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="CBAS KeyRecs benchmark")
    ap.add_argument("--mode", choices=["fixed", "mixed"], default="mixed")
    ap.add_argument("--max-users", type=int, default=40)
    ap.add_argument("--impostors", type=int, default=10)
    ap.add_argument("--window", type=int, default=config.WINDOW_SIZE)
    ap.add_argument("--train-fixed", type=int, default=15)
    ap.add_argument("--train-free", type=int, default=5)
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--grid", action="store_true", help="run hyperparameter grid search first")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    bench_kw = dict(n_train_fixed=args.train_fixed, n_train_free=args.train_free,
                    n_test=args.n_test)

    out_dir = os.path.join(config.BASE_DIR, "evaluation")
    os.makedirs(out_dir, exist_ok=True)
    data_dir = os.path.join(config.DATA_DIR, "keyrecs")

    t0 = time.time()
    print(f"[EVAL] Loading KeyRecs from {data_dir} (window={args.window}) ...")
    participants = load_all(data_dir, window_size=args.window)
    print(f"[EVAL] Loaded {len(participants)} participants in {time.time()-t0:.1f}s")

    contamination = nu = None
    if args.grid:
        contamination, nu = grid_search(participants, args.seed, **bench_kw)

    rows, gen, imp = run_benchmark(
        participants, args.mode, args.max_users, args.impostors, args.seed,
        contamination=contamination, nu=nu, **bench_kw)

    if not rows:
        print("[EVAL] No eligible users — check dataset path and window size.")
        return

    eers       = [r["eer"] for r in rows]
    fars_lock  = [r["far_at_lock"] for r in rows]
    frrs_lock  = [r["frr_at_lock"] for r in rows]
    pooled_e, pooled_t = eer(gen, imp)

    summary = {
        "dataset": "KeyRecs (DOI: 10.5281/zenodo.7886743)",
        "mode": args.mode,
        "window_size": args.window,
        "n_users": len(rows),
        "mean_eer": round(float(np.mean(eers)), 4),
        "std_eer": round(float(np.std(eers)), 4),
        "pooled_eer": round(pooled_e, 4),
        "pooled_eer_threshold": round(pooled_t, 4),
        "mean_far_at_lock": round(float(np.mean(fars_lock)), 4),
        "mean_frr_at_lock": round(float(np.mean(frrs_lock)), 4),
        "targets": {"FAR": 0.05, "FRR": 0.10, "EER": 0.08},
        "targets_met": {
            "FAR": bool(np.mean(fars_lock) <= 0.05),
            "FRR": bool(np.mean(frrs_lock) <= 0.10),
            "EER": bool(np.mean(eers) <= 0.08),
        },
    }

    import csv
    csv_path = os.path.join(out_dir, "keyrecs_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    make_plots(gen, imp, out_dir, args.mode)

    print("\n" + "=" * 60)
    print(f"  RESULTS ({args.mode} training, {len(rows)} users)")
    print(f"  Mean EER        : {summary['mean_eer']*100:.2f}%  (target <= 8%)")
    print(f"  Mean FAR @ lock : {summary['mean_far_at_lock']*100:.2f}%  (target <= 5%)")
    print(f"  Mean FRR @ lock : {summary['mean_frr_at_lock']*100:.2f}%  (target <= 10%)")
    print(f"  Artefacts saved to evaluation/: keyrecs_results.csv, summary.json,")
    print(f"  far_frr_curve.png, det_curve.png, score_dist.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
