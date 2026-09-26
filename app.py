# app.py — CBAS v2: Continuous Behavioural Authentication System
# Key changes from v1:
#   - Password-based login (real authentication)
#   - Browser-side JS keystroke capture (not pynput)
#   - Proper enrolment with structured passage typing + free-text typing
#   - Enrolment passages are split server-side into fixed-size windows, so
#     training features match runtime monitoring features exactly
#   - Per-user calibrated models and trust thresholds
#   - Real server-side session lock on low trust (unlocked only via MFA)
#   - Evaluation capture mode for the live FAR/FRR protocol

import uuid, time
from flask import (Flask, render_template, request, redirect,
                   url_for, session, jsonify, flash)

import config
from utils.database import (init_db, create_user, get_user, get_user_by_id,
                              verify_password, set_enrolled,
                              save_enrolment_window, get_enrolment_samples,
                              count_enrolment_windows, count_free_enrolment,
                              clear_enrolment,
                              save_eval_samples, count_eval_samples,
                              clear_eval_samples,
                              log_event, get_recent_events,
                              # Admin functions
                              get_all_users, set_admin, get_user_anomaly_summary,
                              get_all_anomaly_events, get_user_trust_history,
                              get_user_all_events, delete_user_full, get_system_stats)
from utils.features import events_to_full_feature_vector, windows_to_full_feature_vectors
from utils.model    import train_models, models_exist, score_features, delete_models, load_thresholds
from utils.trust    import TrustEngine, LEVEL_WARN, LEVEL_MFA
from utils.mfa      import generate_secret, get_qr_base64, verify_code

# ── App setup ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = config.SECRET_KEY

# ── Per-session state ──────────────────────────────────────────────────────────
_trust_engines = {}   # session_id -> TrustEngine


# ══════════════════════════════════════════════════════════════════════════════
# AUTH ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "").strip()

        if not username or not password:
            flash("Username and password are required.", "error")
            return render_template("register.html")
        if len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
            return render_template("register.html")
        if get_user(username):
            flash("Username already taken.", "error")
            return render_template("register.html")

        secret = generate_secret()
        uid = create_user(username, password, secret)
        qr = get_qr_base64(secret, username)
        return render_template("register.html", qr=qr, secret=secret,
                               username=username, show_qr=True)

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "").strip()

        if not verify_password(username, password):
            flash("Invalid username or password.", "error")
            return render_template("login.html")

        user = get_user(username)
        session["user_id"]    = user["id"]
        session["username"]   = username
        session["is_admin"]   = bool(user.get("is_admin", 0))
        session["session_id"] = str(uuid.uuid4())
        session["auth_time"]  = time.time()
        return redirect(url_for("dashboard"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    sid = session.get("session_id")
    if sid and sid in _trust_engines:
        del _trust_engines[sid]
    session.clear()
    return redirect(url_for("login"))


def _require_login():
    """Returns (user_dict, error_response). If error_response is not None, return it."""
    if "user_id" not in session:
        return None, redirect(url_for("login"))
    user = get_user_by_id(session["user_id"])
    if user is None:
        session.clear()
        return None, redirect(url_for("login"))
    return user, None


@app.context_processor
def inject_user():
    """Ensure current_user and latest is_admin flag are available in all templates."""
    current_user = None
    if "user_id" in session:
        current_user = get_user_by_id(session["user_id"])
        if current_user:
            session["is_admin"] = bool(current_user.get("is_admin", 0))
    return {"current_user": current_user}



def _get_engine(user):
    """Get or create the trust engine for this session, configured with the
    per-user thresholds derived at training time."""
    sid = session.get("session_id", "unknown")
    if sid not in _trust_engines:
        _trust_engines[sid] = TrustEngine(thresholds=load_thresholds(user["username"]))
    return _trust_engines[sid]


def _json_body():
    """Return a JSON object or a safe empty object for API endpoints."""
    return request.get_json(silent=True) or {}


# ══════════════════════════════════════════════════════════════════════════════
# ENROLMENT
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/enrol")
def enrol():
    user, err = _require_login()
    if err:
        return err
    n = count_enrolment_windows(user["id"])
    n_free = count_free_enrolment(user["id"])
    target = config.ENROL_MIN_WINDOWS
    return render_template("enrol.html",
                           user=user,
                           enrol_text=config.ENROL_TEXT,
                           n_passages=config.ENROL_PASSAGES,
                           n_windows=n,
                           n_free=n_free,
                           require_free=config.ENROL_REQUIRE_FREE,
                           config=config,
                           free_target=config.ENROL_MIN_FREE_WINDOWS,
                           target=target,
                           progress=min(100, int(n / max(target, 1) * 100)))


@app.route("/enrol/submit", methods=["POST"])
def enrol_submit():
    """
    Receive keystroke events from JS, split into fixed-size windows
    (identical to runtime monitoring windows), extract features, save.
    Free-text submissions are marked with passage_num = -1.
    """
    user, err = _require_login()
    if err:
        return jsonify({"error": "Not logged in"}), 401

    data = _json_body()
    events = data.get("events", [])
    passage_type = data.get("passage_type", "fixed")   # 'fixed' | 'free'
    passage_num = data.get("passage_num", 0)

    if len(events) < config.WINDOW_SIZE:
        return jsonify({"error": f"Too few keystrokes ({len(events)}). "
                                 f"Need at least {config.WINDOW_SIZE}."}), 400

    # Split into windows -> one 12-feature vector per window.
    # This is what makes training features match monitoring features.
    vectors = windows_to_full_feature_vectors(events, config.WINDOW_SIZE)
    if not vectors:
        return jsonify({"error": "Could not extract valid features. Try again."}), 400

    store_num = -1 if passage_type == "free" else passage_num
    for fv in vectors:
        save_enrolment_window(user["id"], fv, store_num)

    n = count_enrolment_windows(user["id"])
    n_free = count_free_enrolment(user["id"])
    target = config.ENROL_MIN_WINDOWS

    return jsonify({
        "status": "saved",
        "windows_added": len(vectors),
        "n": n,
        "n_free": n_free,
        "target": target,
        "pct": min(100, int(n / max(target, 1) * 100))
    })


@app.route("/enrol/train", methods=["POST"])
def enrol_train():
    """Train models on collected enrolment data."""
    user, err = _require_login()
    if err:
        return jsonify({"error": "Not logged in"}), 401

    samples = get_enrolment_samples(user["id"])
    target = config.ENROL_MIN_WINDOWS

    if len(samples) < target:
        return jsonify({"error": f"Need at least {target} windows, got {len(samples)}."}), 400

    free_windows = count_free_enrolment(user["id"])
    if config.ENROL_REQUIRE_FREE and free_windows < config.ENROL_MIN_FREE_WINDOWS:
        return jsonify({"error": f"Add at least {config.ENROL_MIN_FREE_WINDOWS} free-text windows before training "
                                 "so the profile represents natural typing."}), 400

    try:
        result = train_models(user["username"], samples)
        set_enrolled(user["id"], True)
        log_event(user["id"], session.get("session_id", "-"), "model_trained",
                  details=(f"total={result['n_total']} training={result['n_vectors']} "
                           f"validation={result['n_calibration']} thresholds={result['thresholds']}"))
        return jsonify({"status": "trained", "result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/enrol/reset", methods=["POST"])
def enrol_reset():
    user, err = _require_login()
    if err:
        return jsonify({"error": "Not logged in"}), 401
    clear_enrolment(user["id"])
    delete_models(user["username"])
    set_enrolled(user["id"], False)
    return jsonify({"status": "reset"})


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD / MONITORING
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/dashboard")
def dashboard():
    user, err = _require_login()
    if err:
        return err
    events = get_recent_events(user["id"], limit=20)
    return render_template("dashboard.html", user=user, events=events,
                           config=config,
                           profile_ready=models_exist(user["username"]))


@app.route("/monitor/score", methods=["POST"])
def monitor_score():
    """
    Receive a window of keystroke events from JS, score them, update trust.
    This is the core continuous authentication endpoint.
    A locked session stays locked until MFA passes — further typing is ignored.
    """
    user, err = _require_login()
    if err:
        return jsonify({"error": "Not logged in"}), 401

    if not user["enrolled"]:
        return jsonify({"error": "Not enrolled"}), 400
    if not models_exist(user["username"]):
        return jsonify({"error": "Your saved model is incomplete or uses an older calibration. Reset enrolment and train a new profile."}), 409

    sid = session.get("session_id", "unknown")
    engine = _get_engine(user)

    # ── Hard session lock: no further scoring until MFA passes ─────────────
    if engine.locked:
        return jsonify({**engine.state(), "history": engine.recent_history(50)})

    # Extract features from JS events
    data = _json_body()
    events = data.get("events", [])

    if not isinstance(events, list) or len(events) < config.WINDOW_SIZE:
        return jsonify({"error": f"Need one complete {config.WINDOW_SIZE}-keystroke window to score."}), 400

    fv = events_to_full_feature_vector(events)
    if all(v == 0.0 for v in fv):
        return jsonify({"error": "No valid timing features were captured. Type normally and try again."}), 400

    # Score against trained models
    try:
        result = score_features(user["username"], fv)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Update trust score
    state = engine.update(result["combined"])

    # Log events
    log_event(user["id"], sid, "score_update",
              trust_score=state["score"],
              details=f"IF={result['if_score']} SVM={result['svm_score']} raw_if={result['if_raw']} raw_svm={result['svm_raw']}")

    if state["level"] == LEVEL_WARN:
        log_event(user["id"], sid, "warning", trust_score=state["score"])
    elif state["level"] == LEVEL_MFA:
        log_event(user["id"], sid, "mfa_trigger", trust_score=state["score"])
        log_event(user["id"], sid, "session_lock", trust_score=state["score"])

    return jsonify({
        **state,
        "if_score"  : result["if_score"],
        "svm_score" : result["svm_score"],
        "if_raw"    : result["if_raw"],
        "svm_raw"   : result["svm_raw"],
        "history"   : engine.recent_history(50),
    })


# ══════════════════════════════════════════════════════════════════════════════
# MFA
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/mfa/verify", methods=["POST"])
def mfa_verify():
    user, err = _require_login()
    if err:
        return jsonify({"error": "Not logged in"}), 401

    code   = request.json.get("code", "").strip()
    sid    = session.get("session_id", "unknown")
    engine = _trust_engines.get(sid)

    if verify_code(user["mfa_secret"], code):
        if engine:
            engine.mfa_passed()
        log_event(user["id"], sid, "mfa_pass",
                  trust_score=engine.score if engine else 1.0)
        log_event(user["id"], sid, "session_unlock",
                  trust_score=engine.score if engine else 1.0)
        return jsonify({"status": "pass", "score": engine.score if engine else 1.0})
    else:
        if engine:
            engine.mfa_failed()
        log_event(user["id"], sid, "mfa_fail")
        return jsonify({"status": "fail"})


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION CAPTURE (live FAR/FRR protocol)
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/eval")
def eval_page():
    user, err = _require_login()
    if err:
        return err
    if not user["enrolled"]:
        flash("Complete enrolment first.", "error")
        return redirect(url_for("enrol"))
    counts = count_eval_samples(user["id"])
    return render_template("eval.html", user=user, counts=counts,
                           window_size=config.WINDOW_SIZE)


@app.route("/eval/submit", methods=["POST"])
def eval_submit():
    """Store one or more labelled window feature vectors for live evaluation."""
    user, err = _require_login()
    if err:
        return jsonify({"error": "Not logged in"}), 401

    data = _json_body()
    events = data.get("events", [])
    label  = (data.get("label") or "").strip().lower().replace(" ", "_")
    if not label:
        return jsonify({"error": "Missing label"}), 400

    vectors = windows_to_full_feature_vectors(events, config.WINDOW_SIZE)
    if not vectors:
        return jsonify({"error": "No valid windows in submission"}), 400

    save_eval_samples(user["id"], label, vectors)
    return jsonify({"status": "saved", "added": len(vectors),
                    "counts": count_eval_samples(user["id"])})


@app.route("/eval/reset", methods=["POST"])
def eval_reset():
    user, err = _require_login()
    if err:
        return jsonify({"error": "Not logged in"}), 401
    clear_eval_samples(user["id"])
    return jsonify({"status": "reset"})


# ══════════════════════════════════════════════════════════════════════════════
# ADMIN DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

def _require_admin():
    """Returns (user_dict, error_response). Checks login AND admin flag."""
    user, err = _require_login()
    if err:
        return None, err
    if not user.get("is_admin"):
        flash("Access denied. Admin privileges required.", "error")
        return None, redirect(url_for("dashboard"))
    return user, None


@app.route("/admin")
def admin_dashboard():
    user, err = _require_admin()
    if err:
        return err
    stats = get_system_stats()
    users = get_all_users()
    anomalies = get_all_anomaly_events(limit=config.ADMIN_EVENTS_PER_PAGE)
    return render_template("admin_dashboard.html",
                           user=user, stats=stats, users=users,
                           anomalies=anomalies, config=config)


@app.route("/admin/users/<int:uid>/toggle-admin", methods=["POST"])
def admin_toggle_admin(uid):
    user, err = _require_admin()
    if err:
        return err
    target = get_user_by_id(uid)
    if not target:
        flash("User not found.", "error")
        return redirect(url_for("admin_dashboard"))
    if uid == user["id"]:
        flash("You cannot change your own admin status.", "error")
        return redirect(url_for("admin_dashboard"))
    new_val = not bool(target.get("is_admin", 0))
    set_admin(uid, new_val)
    flash(f"{'Promoted' if new_val else 'Demoted'} {target['username']}.", "success")
    return redirect(url_for("admin_dashboard"))


@app.route("/admin/users/<int:uid>/delete", methods=["GET", "POST"])
def admin_delete_user(uid):
    user, err = _require_admin()
    if err:
        return err
    target = get_user_by_id(uid)
    if not target:
        flash("User not found.", "error")
        return redirect(url_for("admin_dashboard"))
    if uid == user["id"]:
        flash("You cannot delete your own currently logged-in account.", "error")
        return redirect(url_for("admin_dashboard"))
    
    uname = target["username"]
    try:
        from utils.model import delete_models
        delete_models(uname)
    except Exception as e:
        print(f"[Admin] Warning: model deletion failed for {uname}: {e}")

    try:
        delete_user_full(uid)
        flash(f"User '{uname}' and all associated data have been permanently deleted.", "success")
    except Exception as e:
        print(f"[Admin] DB deletion error: {e}")
        flash(f"Failed to delete user: {e}", "error")

    return redirect(url_for("admin_dashboard"))


@app.route("/api/admin/users/<int:uid>/delete", methods=["DELETE", "POST"])
def api_admin_delete_user(uid):
    user, err = _require_admin()
    if err:
        return jsonify({"error": "Unauthorized"}), 403
    target = get_user_by_id(uid)
    if not target:
        return jsonify({"error": "User not found"}), 404
    if uid == user["id"]:
        return jsonify({"error": "Cannot delete yourself"}), 400
    uname = target["username"]
    try:
        from utils.model import delete_models
        delete_models(uname)
    except Exception as e:
        pass
    delete_user_full(uid)
    return jsonify({"status": "deleted", "username": uname, "id": uid})



@app.route("/admin/users/<int:uid>/reset-enrolment", methods=["POST"])
def admin_reset_enrolment(uid):
    user, err = _require_admin()
    if err:
        return err
    target = get_user_by_id(uid)
    if not target:
        flash("User not found.", "error")
        return redirect(url_for("admin_dashboard"))
    clear_enrolment(uid)
    delete_models(target["username"])
    set_enrolled(uid, False)
    flash(f"Enrolment reset for '{target['username']}'.", "success")
    return redirect(url_for("admin_user_detail", uid=uid))


@app.route("/admin/users/<int:uid>")
def admin_user_detail(uid):
    user, err = _require_admin()
    if err:
        return err
    target = get_user_by_id(uid)
    if not target:
        flash("User not found.", "error")
        return redirect(url_for("admin_dashboard"))
    anomaly_summary = get_user_anomaly_summary(uid)
    events = get_user_all_events(uid, limit=config.ADMIN_EVENTS_PER_PAGE)
    trust_history = get_user_trust_history(uid, limit=200)
    # Compute pattern analysis stats
    scores = [e["trust_score"] for e in trust_history if e["trust_score"] is not None]
    pattern_stats = {}
    if scores:
        import statistics
        pattern_stats = {
            "avg_score": round(statistics.mean(scores), 4),
            "min_score": round(min(scores), 4),
            "max_score": round(max(scores), 4),
            "std_dev":   round(statistics.stdev(scores), 4) if len(scores) > 1 else 0,
            "total_scores": len(scores),
        }
    return render_template("admin_user_detail.html",
                           user=user, target=target,
                           anomaly_summary=anomaly_summary,
                           events=events, trust_history=trust_history,
                           pattern_stats=pattern_stats, config=config)


# ══════════════════════════════════════════════════════════════════════════════
# ADMIN API
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/admin/stats")
def api_admin_stats():
    user, err = _require_admin()
    if err:
        return jsonify({"error": "Unauthorized"}), 403
    return jsonify(get_system_stats())


@app.route("/api/admin/anomalies")
def api_admin_anomalies():
    user, err = _require_admin()
    if err:
        return jsonify({"error": "Unauthorized"}), 403
    return jsonify(get_all_anomaly_events(limit=config.ADMIN_EVENTS_PER_PAGE))


@app.route("/api/admin/user/<int:uid>/trust-history")
def api_admin_trust_history(uid):
    user, err = _require_admin()
    if err:
        return jsonify({"error": "Unauthorized"}), 403
    return jsonify(get_user_trust_history(uid, limit=200))


# ══════════════════════════════════════════════════════════════════════════════
# API
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/api/events")
def api_events():
    user, err = _require_login()
    if err:
        return jsonify([])
    return jsonify(get_recent_events(user["id"], limit=50))


# ══════════════════════════════════════════════════════════════════════════════
# Run
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# CLI Commands
# ══════════════════════════════════════════════════════════════════════════════

import click

@app.cli.command("create-admin")
@click.option("--username", default=config.ADMIN_DEFAULT_USERNAME, help="Admin username")
@click.option("--password", prompt=True, hide_input=True, confirmation_prompt=True, help="Admin password")
def create_admin_cmd(username, password):
    """Create or promote an admin user."""
    init_db()
    existing = get_user(username)
    if existing:
        set_admin(existing["id"], True)
        print(f"[ADMIN] Promoted existing user '{username}' to admin.")
    else:
        secret = generate_secret()
        uid = create_user(username, password, secret)
        set_admin(uid, True)
        print(f"[ADMIN] Created admin user '{username}' (id={uid}).")
        print(f"[ADMIN] MFA secret: {secret}")
        print(f"[ADMIN] Set up your authenticator app with this secret.")


if __name__ == "__main__":
    init_db()
    print(f"\n{'='*60}")
    print("  CBAS v2 — Continuous Behavioural Authentication System")
    print(f"  Running at http://127.0.0.1:{config.PORT}")
    print(f"{'='*60}\n")
    app.run(host="127.0.0.1", port=config.PORT, debug=config.DEBUG, threaded=True)
