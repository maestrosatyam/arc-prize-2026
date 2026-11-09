#!/usr/bin/env python3
"""Stratified sample of P0 level attempts and per-case dossiers.

Subcommands
  sample   write cases.csv from the P0 benchmark.json (seeded, reproducible)
  pattern  print the Kaggle --file-pattern regex covering the sampled passes
  build    write one dossier per case from the downloaded events/requests logs

Inputs
  experiments/runs/kaggle-P0-commit-v1/benchmark.json   per-pass records
  <raw>/<game>_p<k>_requests.jsonl                       model requests/responses
  <raw>/artifacts/<game>_p<k>_events.jsonl               per-step boards + transcripts

The dossiers carry no labels.
"""
import argparse
import csv
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(ROOT, "experiments", "runs", "kaggle-P0-commit-v1", "benchmark.json")
PROV = os.path.join(ROOT, "kaggle", "arc3", "p0", "PROVENANCE.json")
OUT = os.path.join(ROOT, "experiments", "arc3-failure-annotation")

ALWAYS_LOW = ["bp35", "g50t", "lf52", "sk48", "wa30"]
SEED = 17
N_LOW_PASSES = 2
MIN_LATE_LEVEL = 6
N_SUCCESS = 8
MAX_DOSSIER_BYTES = 25 * 1024


# ----------------------------------------------------------------------------
# benchmark helpers

def load_runs():
    d = json.load(open(BENCH))
    runs = []
    for g in d["game_runs"]:
        m = re.search(r"_p(\d+)\.html$", g.get("solver_analysis_html") or "")
        g["pass"] = int(m.group(1)) if m else None
        g["game4"] = g["game_id"][:4]
        g["failed_level"] = None if g["state"] == "won" else g["levels_completed"] + 1
        runs.append(g)
    # fall back to order within game if the html name is missing
    by = defaultdict(list)
    for g in runs:
        by[g["game_id"]].append(g)
    for gid, lst in by.items():
        if any(x["pass"] is None for x in lst):
            for i, x in enumerate(lst):
                x["pass"] = i
    return runs


def run_for(runs, game4, p):
    for g in runs:
        if g["game4"] == game4 and g["pass"] == p:
            return g
    raise KeyError((game4, p))


# ----------------------------------------------------------------------------
# sample

def cmd_sample(args):
    runs = load_runs()
    rng = random.Random(SEED)
    cases = []

    # stratum A: always-low games, N_LOW_PASSES passes each (all passes failed)
    for g4 in ALWAYS_LOW:
        passes = sorted(g["pass"] for g in runs if g["game4"] == g4 and g["state"] != "won")
        for p in sorted(rng.sample(passes, N_LOW_PASSES)):
            g = run_for(runs, g4, p)
            cases.append(("A", g, g["failed_level"], "F"))

    # stratum B: every failed pass at level >= MIN_LATE_LEVEL outside stratum A
    for g in sorted(runs, key=lambda x: (x["game4"], x["pass"])):
        if g["game4"] in ALWAYS_LOW or g["state"] == "won":
            continue
        if g["failed_level"] >= MIN_LATE_LEVEL:
            cases.append(("B", g, g["failed_level"], "F"))

    # stratum S: matched successes. keys = distinct (game, level) failed above;
    # candidates = passes of the same game that cleared that level.
    keys = sorted({(g["game4"], lvl) for _, g, lvl, _ in cases})
    cand = {}
    for g4, lvl in keys:
        ps = sorted(g["pass"] for g in runs if g["game4"] == g4 and g["levels_completed"] >= lvl)
        if ps:
            cand[(g4, lvl)] = ps
    picked_games = set()
    order = list(cand.keys())
    rng.shuffle(order)
    for g4, lvl in order:
        if len([c for c in cases if c[0] == "S"]) >= N_SUCCESS:
            break
        if g4 in picked_games:
            continue
        p = rng.choice(cand[(g4, lvl)])
        picked_games.add(g4)
        cases.append(("S", run_for(runs, g4, p), lvl, "S"))

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "cases.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "stratum", "game_id", "pass", "level", "outcome",
                    "number_of_levels", "levels_completed", "pass_state", "final_score",
                    "base_actions_level", "actions_level"])
        for st, g, lvl, oc in cases:
            cid = f"{g['game4']}_p{g['pass']}_L{lvl}_{oc}"
            w.writerow([cid, st, g["game_id"], g["pass"], lvl, oc, g["number_of_levels"],
                        g["levels_completed"], g["state"], f"{g['final_score']:.2f}",
                        g["base_actions_per_level"][lvl - 1], g["actions_per_level"][lvl - 1]])
    print(f"wrote {path}: {len(cases)} cases "
          f"(A={sum(c[0]=='A' for c in cases)}, B={sum(c[0]=='B' for c in cases)}, "
          f"S={sum(c[0]=='S' for c in cases)})")


def read_cases():
    with open(os.path.join(OUT, "cases.csv")) as f:
        return list(csv.DictReader(f))


def cmd_pattern(args):
    passes = sorted({f"{c['game_id']}_p{c['pass']}" for c in read_cases()})
    print("(" + "|".join(passes) + r")_(requests|events)\.jsonl$")


# ----------------------------------------------------------------------------
# log parsing

STEP_RE = re.compile(r"Current state: step (\d+), level (\d+)\.")
HDR_RE = re.compile(r"^\[(SYSTEM PROMPT|USER PROMPT|ANALYZER STATUS|MODEL RESPONSE META|THINKING|CONTENT|"
                    r"TOOL CALL: \w+|TOOL RESULT: \w+|[A-Z][A-Z _]+)\]\s*$", re.M)
ACTION_RE = re.compile(r"\[action\] batch of (\d+) actions?, (\d+) executed: ([^\n]*)")
GUARD_PAT = re.compile(r"(truncat|guard|reject|not allowed|blocked|refus|ignored|invalid|"
                       r"timed out|timeout|Timeout|budget|limit reached|cut off|too long|"
                       r"Yielded control|not executed|skipped|discard|evict|trimmed|dropped)", re.I)


def load_events(path):
    evs = [json.loads(l) for l in open(path)]
    for e in evs:
        for k in ("action_num", "analysis_step", "level", "score", "batch_index", "batch_size"):
            if k in e and isinstance(e[k], str):
                try:
                    e[k] = int(float(e[k]))
                except ValueError:
                    pass
        for k in ("board_changed", "level_completed", "game_over", "done", "run_complete"):
            if k in e and isinstance(e[k], str):
                e[k] = e[k].lower() == "true"
    return evs


def board_diff(a, b):
    """Return (n_changed, bbox) between two board_ascii strings."""
    if not a or not b:
        return None, None
    ra, rb = a.split("\n"), b.split("\n")
    n = 0
    r0 = c0 = 10 ** 6
    r1 = c1 = -1
    for r, (x, y) in enumerate(zip(ra, rb)):
        if x == y:
            continue
        for c, (p, q) in enumerate(zip(x, y)):
            if p != q:
                n += 1
                r0, r1 = min(r0, r), max(r1, r)
                c0, c1 = min(c0, c), max(c1, c)
    return n, (None if n == 0 else (r0, r1, c0, c1))


def parse_transcript(t):
    """Split a transcript into ordered (header, body) blocks."""
    blocks = []
    pos = 0
    hdrs = list(HDR_RE.finditer(t))
    for i, m in enumerate(hdrs):
        end = hdrs[i + 1].start() if i + 1 < len(hdrs) else len(t)
        blocks.append((m.group(1), t[m.end():end].strip("\n")))
    return blocks


def analysis_turns(ev):
    """Turn one analysis event into a dict with user prompt, status, and model requests."""
    blocks = parse_transcript(ev.get("transcript", ""))
    out = {"user": "", "status": "", "requests": [], "meta": []}
    cur = None
    for h, b in blocks:
        if h == "USER PROMPT":
            out["user"] = b
        elif h == "ANALYZER STATUS":
            out["status"] = b
        elif h == "MODEL RESPONSE META":
            cur = {"thinking": "", "content": "", "code": "", "result": "", "meta": b}
            out["requests"].append(cur)
        elif h == "THINKING":
            if cur is None:
                cur = {"thinking": "", "content": "", "code": "", "result": "", "meta": ""}
                out["requests"].append(cur)
            cur["thinking"] = b
        elif h == "CONTENT":
            if cur is None:
                cur = {"thinking": "", "content": "", "code": "", "result": "", "meta": ""}
                out["requests"].append(cur)
            cur["content"] = b
        elif h.startswith("TOOL CALL"):
            if cur is None:
                cur = {"thinking": "", "content": "", "code": "", "result": "", "meta": ""}
                out["requests"].append(cur)
            m = re.search(r"<parameter=code>\n?(.*?)\n?</parameter>", b, re.S)
            cur["code"] = m.group(1) if m else b
        elif h.startswith("TOOL RESULT"):
            if cur is not None:
                cur["result"] = b
    m = STEP_RE.search(out["user"])
    out["step"] = int(m.group(1)) if m else None
    out["level"] = int(m.group(2)) if m else None
    m = re.search(r"history_messages: (\d+)", out["status"])
    out["history_messages"] = int(m.group(1)) if m else None
    m = re.search(r"message: (.*)", out["status"])
    out["status_msg"] = m.group(1).strip() if m else ""
    return out


def load_requests(path):
    if not path or not os.path.exists(path):
        return None
    recs = []
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            msgs = r.get("messages") or []
            last_user = None
            for m in reversed(msgs):
                if m.get("role") == "user":
                    c = m.get("content")
                    if isinstance(c, list):
                        c = " ".join(x.get("text", "") for x in c if x.get("type") == "text")
                    last_user = c
                    break
            m = STEP_RE.search(last_user or "")
            recs.append({
                "event": r.get("event"),
                "analysis_step": int(r.get("analysis_step") or 0),
                "idx": int(r.get("request_index_within_turn") or 0),
                "n_msgs": len(msgs),
                "step": int(m.group(1)) if m else None,
                "level": int(m.group(2)) if m else None,
                "usage": r.get("usage") or {},
                "finish": r.get("finish_reason"),
            })
    return recs


# ----------------------------------------------------------------------------
# dossier text helpers

def clip(s, n, tail=0):
    s = (s or "").strip()
    if len(s) <= n + tail:
        return s
    if tail:
        return s[:n].rstrip() + f" [...{len(s) - n - tail} chars...] " + s[-tail:].lstrip()
    return s[:n].rstrip() + f" [...{len(s) - n} chars]"


def one_line(s, n):
    return clip(re.sub(r"\s+", " ", s or ""), n)


def code_summary(code):
    if not code:
        return "(no code)"
    lines = [l for l in code.split("\n") if l.strip()]
    acts = re.findall(r"action\(([^\n]*)\)", code)
    defs = re.findall(r"^def (\w+)\(", code, re.M)
    s = f"{len(lines)} lines"
    if defs:
        s += "; defines " + ",".join(defs[:4])
    if acts:
        s += "; action(" + one_line(acts[0], 110) + ")" + (f" +{len(acts) - 1} more" if len(acts) > 1 else "")
    else:
        s += "; inspect only"
    return s


def result_summary(res, n=220):
    if not res:
        return "(no result)"
    exe = ACTION_RE.search(res)
    s = ""
    if exe:
        s += f"executed {exe.group(2)}/{exe.group(1)}: {one_line(exe.group(3), 120)}; "
    body = ACTION_RE.sub("", res).strip()
    if "Traceback" in body or "Error" in body.split("\n")[-1:][0]:
        s += "ERROR: " + one_line(body[-160:], 160)
    elif body:
        s += one_line(body, n)
    return s.strip()


def harness_notes(user):
    """Compact list of the harness remarks that matter (animation, level, guards)."""
    notes = []
    for line in (user or "").split("\n"):
        l = line.strip()
        if not l:
            continue
        if l.startswith(("Executed actions:", "The last executed action", "You are still", "Level ",
                         "The code executed", "Your last python snippet", "Game over", "GAME OVER",
                         "The game", "A level", "Your retained functions", "Note:", "Warning", "WARNING")):
            notes.append(one_line(l, 230))
        elif GUARD_PAT.search(l) and not l.startswith(("Only tool", "Keep tool output", "Use Python",
                                                       "When ready", "When calling", "If you use",
                                                       "Only letter", "For the most recent", "Ground",
                                                       "Focus on", "You may call")):
            notes.append("! " + one_line(l, 230))
    return notes


def guard_lines(text):
    out = []
    for line in (text or "").split("\n"):
        if GUARD_PAT.search(line) and not line.lstrip().startswith(("#", "print")):
            out.append(one_line(line, 200))
    return out


# ----------------------------------------------------------------------------
# build

def build_case(case, run, prov, raw_dir):
    game_id, p, lvl = case["game_id"], int(case["pass"]), int(case["level"])
    ev_path = os.path.join(raw_dir, "artifacts", f"{game_id}_p{p}_events.jsonl")
    rq_path = os.path.join(raw_dir, f"{game_id}_p{p}_requests.jsonl")
    evs = load_events(ev_path)
    reqs = load_requests(rq_path)

    acts = [e for e in evs if e["type"] == "action"]
    ana = [e for e in evs if e["type"] == "analysis"]
    init = [e for e in evs if e["type"] == "initial"]

    # level_before for each action = level of previous action event
    prev_level = 1
    prev_board = init[0].get("board_ascii") if init else None
    level_actions = []
    all_actions = []
    for e in acts:
        e["level_before"] = prev_level
        n, bbox = board_diff(prev_board, e.get("board_ascii"))
        e["n_changed"], e["bbox"] = n, bbox
        all_actions.append(e)
        if prev_level == lvl:
            level_actions.append(e)
        prev_level = e["level"]
        prev_board = e.get("board_ascii") or prev_board

    turns = [analysis_turns(e) for e in ana]
    for t, e in zip(turns, ana):
        t["analysis_step"] = e.get("analysis_step")
        t["action_num"] = e.get("action_num")
        t["time"] = (re.search(r"\| (\d\d:\d\d:\d\d) \|", e.get("transcript", "")[:200]) or [None, None])[1]
    level_turns = [t for t in turns if t["level"] == lvl]
    # context trims: history_messages drops between consecutive turns in the whole game
    trims = []
    prev_hm = None
    for t in turns:
        hm = t["history_messages"]
        if hm is not None and prev_hm is not None and hm < prev_hm - 2:
            trims.append((t["analysis_step"], t["level"], prev_hm, hm))
        if hm is not None:
            prev_hm = hm
    level_trims = [x for x in trims if x[1] == lvl]

    # tokens and wall time from benchmark history
    hist = run["history"]
    apl = run["actions_per_level"]
    start = sum(apl[:lvl - 1])
    end = start + apl[lvl - 1]
    seg = hist[start:end]
    gen = sum(h.get("generated_tokens", 0) for h in seg)
    unc = sum(h.get("uncached_input_tokens", 0) for h in seg)
    t0 = hist[start - 1]["wallclock_seconds"] if start > 0 else 0.0
    t1 = seg[-1]["wallclock_seconds"] if seg else t0
    is_last_level = (end == len(hist))
    if is_last_level:
        gen += run.get("final_generated_tokens", 0) or 0
        unc += run.get("final_uncached_input_tokens", 0) or 0
        t1 = run.get("final_wallclock_seconds") or t1
    wall = t1 - t0

    # request-level usage on this level
    req_stats = None
    if reqs:
        resp = [r for r in reqs if r["event"] == "response" and r["level"] == lvl]
        rq = [r for r in reqs if r["event"] == "request" and r["level"] == lvl]
        pt = [r["usage"].get("prompt_tokens", 0) for r in resp]
        ct = [r["usage"].get("completion_tokens", 0) for r in resp]
        it = [(r["usage"].get("prompt_tokens_details") or {}).get("image_tokens", 0) for r in resp]
        fin = Counter(r["finish"] for r in resp)
        req_stats = {
            "n_requests": len(rq), "prompt_sum": sum(pt), "prompt_max": max(pt) if pt else 0,
            "completion_sum": sum(ct), "image_sum": sum(it), "finish": dict(fin),
            "msgs": [r["n_msgs"] for r in rq],
        }

    # ------------------------------------------------------------------ text
    L = []
    cid = case["case_id"]
    L.append(f"# Dossier {cid}")
    L.append("")
    L.append("## Provenance")
    L.append(f"- game: {game_id}; pass: p{p}; level: {lvl} of {run['number_of_levels']}; "
             f"outcome of this level: {'cleared' if case['outcome'] == 'S' else 'not cleared (pass ended here)'}")
    L.append(f"- pass state: {run['state']}; levels completed: {run['levels_completed']}; final score {run['final_score']:.2f}")
    L.append(f"- run: experiments/runs/kaggle-P0-commit-v1 (label {prov.get('run_label', 'P0')}); "
             f"source kernel {prov.get('source_kernel')}; notebook sha256 {prov.get('pulled_file_sha256')}; "
             f"modification: {prov.get('modification')}")
    L.append(f"- logs: {game_id}_p{p}_requests.jsonl, artifacts/{game_id}_p{p}_events.jsonl "
             f"(Kaggle output of team-account-1/arc-agi-3-p0-unmodified-dfranzen)")
    L.append("")
    L.append("## Budget on this level")
    base = run["base_actions_per_level"][lvl - 1]
    L.append(f"- baseline actions for level: {base}; actions spent: {apl[lvl - 1]} "
             f"(ratio {apl[lvl - 1] / base:.2f}); game-wide actions before this level: {start}")
    L.append(f"- generated tokens: {gen}; uncached input tokens: {unc}; elapsed wall time: {wall:.0f} s "
             f"({wall / 60:.1f} min; shared-GPU job, elapsed not exclusive)")
    if req_stats:
        L.append(f"- model requests on level: {req_stats['n_requests']}; prompt tokens sum {req_stats['prompt_sum']} "
                 f"(max per request {req_stats['prompt_max']}); completion tokens {req_stats['completion_sum']}; "
                 f"image tokens {req_stats['image_sum']}; finish reasons {req_stats['finish']}")
    L.append(f"- analyzer turns on level: {len(level_turns)}; context trims on level: {len(level_trims)} "
             f"(game total {len(trims)})")
    if level_trims:
        L.append("  trims (analysis_step, history_messages before->after): " +
                 ", ".join(f"s{a}: {b}->{c}" for a, _, b, c in level_trims))
    resets = [e for e in level_actions if (e.get("action_name") == "RESET" or e.get("game_over"))]
    L.append(f"- resets/game-overs on level: {len(resets)}" +
             (": " + ", ".join(f"act{e['action_num']} {e.get('action_display')}{' GAME_OVER' if e.get('game_over') else ''}" for e in resets[:12]) if resets else ""))
    cnt = Counter(e.get("action_display") for e in level_actions)
    L.append(f"- action mix: " + ", ".join(f"{k}x{v}" for k, v in cnt.most_common(12)))
    nochange = [e["action_num"] for e in level_actions if not e.get("board_changed")]
    L.append(f"- actions with no board change: {len(nochange)}" + (f" (e.g. {nochange[:15]})" if nochange else ""))
    L.append("")

    # retained functions (last seen on this level)
    retained = ""
    for t in level_turns:
        m = re.search(r"Your retained functions from earlier Python calls in this game: ([^\n]*)", t["user"])
        if m:
            retained = m.group(1)
    if retained:
        L.append(f"- retained functions (last seen): {one_line(retained, 400)}")
        L.append("")
    if level_turns:
        L.append(f"- first/last analyzer turn on level: {level_turns[0]['time']} / {level_turns[-1]['time']} UTC "
                 f"(games run interleaved; gaps are scheduler time, not model time)")
        L.append("")

    # action table (rows with flags always kept; long tables thinned in the middle)
    rows = []
    for k, e in enumerate(level_actions):
        flags = []
        nxt = all_actions[all_actions.index(e) + 1] if all_actions.index(e) + 1 < len(all_actions) else None
        bi, bs = e.get("batch_index"), e.get("batch_size")
        if isinstance(bi, int) and isinstance(bs, int) and bi < bs and (nxt is None or nxt.get("batch_index") == 1):
            flags.append(f"BATCH_CUT({bi}/{bs})")
        if e.get("level_completed"):
            flags.append("LEVEL_COMPLETED")
        if e.get("game_over"):
            flags.append("GAME_OVER")
        if e.get("action_name") == "RESET":
            flags.append("RESET")
        if not e.get("board_changed"):
            flags.append("no-change")
        bb = e["bbox"]
        rows.append((bool(flags), f"{e['action_num']} | {e.get('batch_index')}/{e.get('batch_size')} | {e.get('action_display')} | "
                     f"{e['n_changed']} | {'' if not bb else f'{bb[0]}-{bb[1]},{bb[2]}-{bb[3]}'} | {' '.join(flags)}"))

    def action_table(max_rows):
        out = ["## Actions on this level (events)",
               "act | step-in-batch | action | cells changed vs previous board | bbox r0-r1,c0-c1 | flags"]
        if len(rows) <= max_rows:
            out += [r for _, r in rows]
        else:
            head_n, tail_n = max_rows // 3, max_rows - max_rows // 3
            keep = set(range(head_n)) | set(range(len(rows) - tail_n, len(rows))) | {i for i, (f, _) in enumerate(rows) if f}
            skipped = 0
            for i, (_, r) in enumerate(rows):
                if i in keep:
                    if skipped:
                        out.append(f"(... {skipped} rows omitted ...)")
                        skipped = 0
                    out.append(r)
                else:
                    skipped += 1
            if skipped:
                out.append(f"(... {skipped} rows omitted ...)")
        return "\n".join(out) + "\n"

    def digest_block(i, t, belief_n, end_n, result_n):
        b = [f"### Turn {i + 1} (analysis_step {t['analysis_step']}, step {t['step']}, {t['time'] or ''})"]
        for n in [x for x in harness_notes(t["user"]) if not x.startswith("Your retained functions")][:5]:
            b.append(f"- harness: {n}")
        for j, r in enumerate(t["requests"]):
            think = r["thinking"] or r["content"]
            b.append(f"- req {j + 1} belief: \"{one_line(think, belief_n)}\"" if think else f"- req {j + 1}: (no text)")
            if end_n and r["thinking"] and len(r["thinking"]) > belief_n + end_n:
                b.append(f"  end: \"{one_line(r['thinking'][-end_n:], end_n)}\"")
            b.append(f"  code: {code_summary(r['code'])}")
            b.append(f"  result: {result_summary(r['result'], result_n)}")
            g = guard_lines(r["result"])
            if g:
                b.append("  guard/notice: " + " || ".join(g[:3]))
        if t["status_msg"]:
            b.append(f"- analyzer: {one_line(t['status_msg'], 160)}; history_messages={t['history_messages']}")
        return "\n".join(b)

    def digest(belief_n, end_n, result_n, keep_head, keep_tail):
        blocks = []
        n = len(level_turns)
        for i, t in enumerate(level_turns):
            if n > keep_head + keep_tail and keep_head <= i < n - keep_tail:
                if not blocks or not blocks[-1].startswith("(..."):
                    blocks.append(f"(... {n - keep_head - keep_tail} turn(s) omitted for size; the actions table above lists what was executed ...)")
                continue
            blocks.append(digest_block(i, t, belief_n, end_n, result_n))
        return "## Chronological digest of analyzer turns on this level\n" \
               "(turn = one analyzer call, possibly several model requests; quotes are the model's own reasoning)\n\n" \
               + "\n\n".join(blocks) + "\n"

    def full_quotes(n_full, think_n, code_n, res_n):
        out = []
        sel = level_turns[-n_full:] if n_full else []
        for i, t in enumerate(sel):
            k = len(level_turns) - len(sel) + i + 1
            f = [f"### Turn {k} (analysis_step {t['analysis_step']}, step {t['step']}) in full"]
            f.append("harness prompt (trimmed):")
            f.append("    " + "\n    ".join(x for x in harness_notes(t["user"])[:8] if not x.startswith("Your retained functions")))
            for j, r in enumerate(t["requests"]):
                f.append(f"req {j + 1} reasoning:")
                f.append("    " + clip(r["thinking"] or r["content"], think_n, think_n // 3).replace("\n", "\n    "))
                f.append(f"req {j + 1} code:")
                f.append("    " + clip(r["code"], code_n, code_n // 3).replace("\n", "\n    "))
                f.append(f"req {j + 1} result:")
                f.append("    " + clip(r["result"], res_n, res_n // 3).replace("\n", "\n    "))
            if t["status_msg"]:
                f.append(f"analyzer: {t['status_msg']}")
            out.append("\n".join(f))
        if not out:
            return ""
        return "## Last turns on this level, quoted more fully\n\n" + "\n\n".join(out) + "\n"

    head = "\n".join(L) + "\n"
    ladder = [
        # (max action rows, belief, end, result, keep_head, keep_tail, n_full, think, code, res)
        (400, 260, 220, 220, 999, 999, 4, 1400, 900, 700),
        (160, 240, 200, 200, 999, 999, 3, 1200, 800, 600),
        (120, 220, 160, 180, 999, 999, 3, 1000, 700, 500),
        (100, 200, 0, 160, 999, 999, 2, 900, 600, 450),
        (90, 180, 0, 140, 999, 999, 2, 800, 500, 400),
        (80, 160, 0, 120, 999, 999, 2, 700, 450, 350),
        (80, 160, 0, 120, 8, 12, 2, 700, 450, 350),
        (70, 160, 0, 120, 6, 10, 2, 700, 450, 350),
        (60, 140, 0, 100, 5, 8, 2, 600, 400, 300),
        (60, 140, 0, 100, 4, 6, 1, 600, 400, 300),
        (50, 120, 0, 80, 3, 5, 1, 600, 400, 300),
        (40, 120, 0, 80, 2, 3, 1, 500, 300, 250),
    ]
    text = None
    for (mr, bn, en, rn, kh, kt, nf, tn, cn, rsn) in ladder:
        text = head + action_table(mr) + "\n" + digest(bn, en, rn, kh, kt) + "\n" + full_quotes(nf, tn, cn, rsn)
        if len(text.encode()) <= MAX_DOSSIER_BYTES:
            break
    if len(text.encode()) > MAX_DOSSIER_BYTES:
        text = text.encode()[:MAX_DOSSIER_BYTES - 40].decode(errors="ignore") + "\n(... truncated for size ...)\n"
    return text, {
        "case_id": cid, "n_turns": len(level_turns), "n_requests": (req_stats or {}).get("n_requests"),
        "actions": apl[lvl - 1], "base": base, "gen_tokens": gen, "wall_s": round(wall),
        "trims": len(level_trims), "resets": len(resets), "no_change_actions": len(nochange),
    }


def cmd_build(args):
    runs = load_runs()
    prov = json.load(open(PROV)) if os.path.exists(PROV) else {}
    prov["run_label"] = "P0"
    cases = read_cases()
    if args.only:
        cases = [c for c in cases if c["case_id"] in set(args.only.split(","))]
    ddir = os.path.join(OUT, "dossiers")
    os.makedirs(ddir, exist_ok=True)
    index = []
    for c in cases:
        run = run_for(runs, c["game_id"][:4], int(c["pass"]))
        ev = os.path.join(args.raw, "artifacts", f"{c['game_id']}_p{c['pass']}_events.jsonl")
        if not os.path.exists(ev):
            print(f"skip {c['case_id']}: missing {ev}", file=sys.stderr)
            continue
        text, meta = build_case(c, run, prov, args.raw)
        path = os.path.join(ddir, f"{c['case_id']}.md")
        with open(path, "w") as f:
            f.write(text)
        meta["bytes"] = len(text.encode())
        index.append(meta)
        print(f"{c['case_id']}: {meta['bytes']} B, turns={meta['n_turns']}, actions={meta['actions']}/{meta['base']}, "
              f"trims={meta['trims']}, resets={meta['resets']}")
    with open(os.path.join(OUT, "dossier_index.json"), "w") as f:
        json.dump(index, f, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sample").set_defaults(fn=cmd_sample)
    sub.add_parser("pattern").set_defaults(fn=cmd_pattern)
    b = sub.add_parser("build")
    b.add_argument("--raw", required=True, help="directory holding <game>_p<k>_requests.jsonl and artifacts/")
    b.add_argument("--only", default="", help="comma-separated case ids")
    b.set_defaults(fn=cmd_build)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
