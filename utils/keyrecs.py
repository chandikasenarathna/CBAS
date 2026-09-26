# utils/keyrecs.py — Loader for the KeyRecs Keystroke Dynamics benchmark dataset
#
# Source: Dias, T., Vitorino, J., Maia, E., Sousa, O., Praça, I. (2023).
#         KeyRecs: Keystroke Dynamics Dataset. Zenodo. DOI: 10.5281/zenodo.7886743
#         (cite the Data in Brief article: DOI 10.1016/j.dib.2023.109509)
#
# KeyRecs provides inter-key latencies in a digraph model:
#   DU.key1.key1 = dwell time of key1 (Down->Up)
#   DD.key1.key2 = Down(key1) -> Down(key2) latency
#   UD.key1.key2 = Up(key1)   -> Down(key2) latency  (= flight time)
# Times are in SECONDS. Negative flight values indicate rollover typing.
#
# This loader reconstructs synthetic press/release event streams per participant
# (press_{i+1} = press_i + DD, release_i = press_i + dwell) and feeds them
# through the SAME windowing/feature pipeline as live browser capture
# (windows_to_full_feature_vectors), so benchmark features are perfectly
# comparable with runtime features.

import os, csv
from collections import defaultdict
from utils.features import windows_to_full_feature_vectors

FIXED_CSV = "fixed-text.csv"
FREE_CSV  = "free-text.csv"

# Fixed-text password "v.p.w.j.k.e.u.r.k.b" (10 keys).
# Column layout per key i (0-based, after participant/session/repetition):
#   col(3 + 5*i)     = DU.k_i.k_i  (dwell of key i)
#   col(3 + 5*i + 1) = DD.k_i.k_{i+1}  (only for i < 9)
_FIXED_KEYS = 10


def _synthesize_events(dwells, dds):
    """
    Build a synthetic event list [{press_time, release_time}] (ms) from
    per-key dwell times and down-down latencies (seconds).
    """
    events = []
    press = 0.0
    for i, dwell in enumerate(dwells):
        try:
            d = float(dwell)
        except (TypeError, ValueError):
            # Cannot determine press time of the next key — stop the stream here
            if i > 0:
                break
            press += 0.15
            continue
        release = press + d
        events.append({"press_time": press * 1000.0, "release_time": release * 1000.0})
        if i < len(dds):
            try:
                press += float(dds[i])
            except (TypeError, ValueError):
                break
    return events


def load_free_text(data_dir, window_size=30, max_participants=None):
    """
    Parse free-text.csv (long format, one digraph per row).
    Returns {participant_id: [feature_vector, ...]} built from each
    participant's concatenated session streams.
    """
    path = os.path.join(data_dir, FREE_CSV)
    # (participant, session) -> list of rows in file order
    streams = defaultdict(list)
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        idx = {name: i for i, name in enumerate(header)}
        i_part = idx["participant"]; i_sess = idx["session"]
        i_du = idx["DU.key1.key1"];   i_dd = idx["DD.key1.key2"]
        for row in reader:
            if len(row) <= max(i_part, i_sess, i_du, i_dd):
                continue
            streams[(row[i_part], row[i_sess])].append((row[i_du], row[i_dd]))

    # Merge sessions per participant in session order
    per_participant = defaultdict(list)
    for (part, sess), rows in streams.items():
        per_participant[part].append((sess, rows))

    out = {}
    for part, sess_list in sorted(per_participant.items()):
        if max_participants and len(out) >= max_participants:
            break
        events = []
        for sess, rows in sorted(sess_list):
            dwells = [r[0] for r in rows]
            dds    = [r[1] for r in rows]
            events.extend(_synthesize_events(dwells, dds))
        vectors = windows_to_full_feature_vectors(events, window_size)
        if vectors:
            out[part] = vectors
    return out


def load_fixed_text(data_dir, window_size=30, max_participants=None):
    """
    Parse fixed-text.csv (wide format, one password repetition per row).
    Repetitions are concatenated per participant (in session/repetition order)
    and split into windows. Returns {participant_id: [feature_vector, ...]}.

    Flight times are derived as DD - dwell (the dataset's UD column in the
    fixed-text file uses a different convention; DD and dwell are reliable).
    """
    path = os.path.join(data_dir, FIXED_CSV)
    per_participant = defaultdict(list)  # participant -> list of rows
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            # 3 id cols + 9 digraphs x 5 cols + final dwell + total time = 50
            if len(row) < 3 + 5 * (_FIXED_KEYS - 1) + 2:
                continue
            per_participant[row[0]].append(row)

    out = {}
    for part, rows in sorted(per_participant.items()):
        if max_participants and len(out) >= max_participants:
            break
        # Sort by (session, repetition) numerically when possible
        def sort_key(r):
            try:
                return (int(r[1]), int(r[2]))
            except ValueError:
                return (r[1], r[2])
        events = []
        for row in sorted(rows, key=sort_key):
            dwells = [row[3 + 5 * i] for i in range(_FIXED_KEYS)]
            dds    = [row[3 + 5 * i + 1] for i in range(_FIXED_KEYS - 1)]
            events.extend(_synthesize_events(dwells, dds))
        vectors = windows_to_full_feature_vectors(events, window_size)
        if vectors:
            out[part] = vectors
    return out


def load_all(data_dir, window_size=30, max_participants=None):
    """
    Returns {participant_id: {"fixed": [...], "free": [...]}} for participants
    present in at least one of the two files.
    """
    fixed = load_fixed_text(data_dir, window_size, max_participants)
    free  = load_free_text(data_dir, window_size, max_participants)
    participants = {}
    for part in set(fixed) | set(free):
        participants[part] = {"fixed": fixed.get(part, []), "free": free.get(part, [])}
    return participants
