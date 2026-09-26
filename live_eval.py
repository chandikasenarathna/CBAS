# live_eval.py — Live FAR/FRR/EER evaluation from labelled typing samples
#
# Uses the samples collected via the web app's Evaluation page (/eval):
#   label "genuine"  -> your own typing        (used for FRR)
#   all other labels -> impersonated behaviours (used for FAR)
#
# Usage:
#   python live_eval.py <username> [--include-keyrecs N]
#
# Outputs (evaluation/live/):
#   live_results.csv, live_summary.json, live_far_frr_curve.png, live_score_dist.png

import os, sys, json, argparse, csv
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from utils.database import get_eval_samples, get_user
from utils.model import UserModel, load_thresholds
from utils.metrics import eer, far_frr, sweep
from utils.keyrecs import load_free_text


def main():
    ap = argparse.ArgumentParser(description="CBAS live evaluation")
    ap.add_argument("username")
    ap.add_argument("--include-keyrecs", type=int, default=0,
                    help="also add N KeyRecs participants as extra impostors")
    args = ap.parse_args()

    user = get_user(args.username)
    if not user:
        print(f"[LIVE] Unknown user '{args.username}'.")
        return

    samples = get_eval_samples(user["id"])
    if not samples:
        print("[LIVE] No evaluation samples found. Collect them via the /eval page first.")
        return

    genuine_fv  = [s["features"] for s in samples if s["label"] == "genuine"]
    impostor_fv = [s["features"] for s in samples if s["label"] != "genuine"]

    print(f"[LIVE] Genuine windows: {len(genuine_fv)} | Impostor windows: {len(impostor_fv)}")

    if len(genuine_fv) < 5 or len(impostor_fv) < 5:
        print("[LIVE] Need at least 5 genuine and 5 impostor windows. Collect more data.")
        return

    model = UserModel.load(args.username)
    thresholds = load_thresholds(args.username)
    print(f"[LIVE] Per-user thresholds: {thresholds}")

    gen_scores = model.score_batch(genuine_fv)
    imp_scores = model.score_batch(impostor_fv)

    # Optional: add KeyRecs cross-user impostors for a stronger FAR estimate
    if args.include_keyrecs > 0:
        data_dir = os.path.join(config.DATA_DIR, "keyrecs")
        kr = load_free_text(data_dir, window_size=config.WINDOW_SIZE,
                            max_participants=args.include_keyrecs)
        kr_fv = [fv for part in kr.values() for fv in part[:10]]
        if kr_fv:
            kr_scores = model.score_batch(kr_fv)
            imp_scores = np.concatenate([imp_scores, kr_scores])
            print(f"[LIVE] Added {len(kr_scores)} KeyRecs impostor windows "
                  f"from {len(kr)} participants.")

    e, t_eer = eer(gen_scores, imp_scores)
    far_lock, frr_lock = far_frr(gen_scores, imp_scores, thresholds["lock"])
    far_warn, frr_warn = far_frr(gen_scores, imp_scores, thresholds["warn"])

    # Per-label breakdown
    by_label = {}
    for s in samples:
        if s["label"] == "genuine":
            continue
        by_label.setdefault(s["label"], []).append(s["features"])
    label_rows = []
    for label, fvs in sorted(by_label.items()):
        sc = model.score_batch(fvs)
        accepted = float(np.mean(sc >= thresholds["lock"]))
        label_rows.append({"label": label, "n": len(fvs),
                           "mean_score": round(float(np.mean(sc)), 4),
                           "far_at_lock": round(accepted, 4)})

    # ── Save artefacts ─────────────────────────────────────────────────────
    out_dir = os.path.join(config.BASE_DIR, "evaluation", "live")
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "live_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["type", "score"])
        for s in gen_scores: w.writerow(["genuine", round(float(s), 4)])
        for s in imp_scores: w.writerow(["impostor", round(float(s), 4)])

    summary = {
        "user": args.username,
        "n_genuine": len(gen_scores),
        "n_impostor": len(imp_scores),
        "eer": round(e, 4),
        "eer_threshold": round(t_eer, 4),
        "thresholds": thresholds,
        "far_at_lock": round(far_lock, 4),
        "frr_at_lock": round(frr_lock, 4),
        "far_at_warn": round(far_warn, 4),
        "frr_at_warn": round(frr_warn, 4),
        "per_label": label_rows,
        "targets": {"FAR": 0.05, "FRR": 0.10, "EER": 0.08},
        "targets_met": {"FAR": far_lock <= 0.05, "FRR": frr_lock <= 0.10, "EER": e <= 0.08},
    }
    with open(os.path.join(out_dir, "live_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Plots
    th, fars, frrs = sweep(gen_scores, imp_scores, n_points=300)
    plt.figure(figsize=(7, 5))
    plt.plot(th, fars, label="FAR", color="#ef4444")
    plt.plot(th, frrs, label="FRR", color="#3b82f6")
    plt.axvline(t_eer, ls="--", color="#888", label=f"EER = {e:.3f}")
    plt.axvline(thresholds["lock"], ls=":", color="#f59e0b",
                label=f"lock threshold = {thresholds['lock']:.2f}")
    plt.xlabel("Acceptance threshold"); plt.ylabel("Error rate")
    plt.title(f"Live FAR/FRR — user '{args.username}'")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "live_far_frr_curve.png"), dpi=150)
    plt.close()

    bins = np.linspace(0, 1, 60)
    plt.figure(figsize=(7, 5))
    plt.hist(gen_scores, bins=bins, alpha=.6, density=True,
             label=f"Genuine (n={len(gen_scores)})", color="#22c55e")
    plt.hist(imp_scores, bins=bins, alpha=.6, density=True,
             label=f"Impostor (n={len(imp_scores)})", color="#ef4444")
    plt.axvline(thresholds["lock"], ls=":", color="#f59e0b", label="lock threshold")
    plt.xlabel("Combined anomaly score"); plt.ylabel("Density")
    plt.title(f"Live score distributions — user '{args.username}'")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "live_score_dist.png"), dpi=150)
    plt.close()

    # ── Report ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  LIVE EVALUATION — user '{args.username}'")
    print(f"  EER          : {e*100:.2f}% (threshold {t_eer:.2f})  [target <= 8%]")
    print(f"  FAR @ lock   : {far_lock*100:.2f}%  [target <= 5%]")
    print(f"  FRR @ lock   : {frr_lock*100:.2f}%  [target <= 10%]")
    print("  Per-label breakdown:")
    for r in label_rows:
        print(f"    {r['label']:<22} n={r['n']:<4} mean={r['mean_score']:.3f} FAR={r['far_at_lock']*100:.1f}%")
    print(f"  Artefacts: evaluation/live/live_results.csv, live_summary.json,")
    print(f"             live_far_frr_curve.png, live_score_dist.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
