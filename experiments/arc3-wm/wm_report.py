"""WM1 report: P0 against P0 + world-model tools on the 25 public games, and how the model used the tools.

Reads a run's archive (benchmark.json, artifacts/*_events.jsonl, transcripts/*.txt) and the baseline's
benchmark.json, and writes into --out:

  comparison.md          scores, the game-paired permutation test, matched-token scores, levels per pass, the rule
  per_game.csv           per game: mean score, levels, tokens, wall time, both arms
  tool_calls.jsonl       one row per python tool call that touched a world-model tool (code, parsed results)
  levels.csv             one row per (game, pass, level) played: cleared, turns, tool calls by kind, verify outcome,
                         plan and execute outcomes, errors
  tool_usage.md          adoption, verify/plan/execute outcomes, errors, and their relation to clears
  cases/*.md             the turns of selected levels (code and tool results), for reading what the model believed

Selection of cases (up to --cases per kind): verified but not cleared; plan found but not cleared; three or more
verify calls never ok; cleared with a verified model; tool errors or timeouts.

  python3 -I experiments/arc3-wm/wm_report.py --run experiments/runs/kaggle-WM1-public-v1 \
      --baseline experiments/runs/kaggle-P0-commit-v1 --out experiments/arc3-wm/report
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import re
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("analysis_arc3_kaggle", ROOT / "scripts" / "analysis_arc3_kaggle.py")
AK = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(AK)

TOOLS = ("verify_model", "plan_actions", "execute_plan", "rollout", "action_counts", "candidate_clicks",
         "tile_centers", "transitions_of", "objects_of", "diff_rows", "crop")
BUDGETS = (100_000, 150_000, 200_000)
HEADER = re.compile(r"^--- analysis_step=(\d+) \| action=(\d+) \| (\d\d:\d\d:\d\d) \| tool-agent ---$")
MARKERS = ("[SYSTEM PROMPT]", "[USER PROMPT]", "[ANALYZER STATUS]", "[MODEL RESPONSE META]", "[THINKING]",
           "[TOOL CALL: python]", "[TOOL RESULT: python]")
ERRORS = {"traceback": re.compile(r"Traceback \(most recent call last\)"), "name_error": re.compile(r"NameError"),
          "timeout": re.compile(r"timed out|TimeoutError|time limit", re.I),
          "sandbox_start": re.compile(r"could not start|Argument list too long|E2BIG"),
          "key_error": re.compile(r"KeyError")}


# ------------------------------------------------------------------------------------------------ benchmark rows
def load_rows(run_dir: Path, tag: str) -> list[dict]:
    """AK.load for an archive path (the same score and token fields)."""
    b = json.loads((run_dir / "benchmark.json").read_text())
    rows, seen = [], {}
    for r in b["game_runs"]:
        g = r["game_id"].split("-")[0]
        p = seen.get(g, 0)
        seen[g] = p + 1
        h = r["history"]
        cum, t = [], 0
        for x in h:
            t += int(x.get("generated_tokens") or 0)
            cum.append(t)
        apl, base, lc = r["actions_per_level"], r["base_actions_per_level"], r["levels_completed"]
        clear_idx, acc = [], 0
        for a in apl[:lc]:
            acc += a
            clear_idx.append(acc)
        clear_tok = [cum[min(len(cum), c) - 1] if cum else 0 for c in clear_idx]
        rows.append({"run": tag, "game": g, "game_id": r["game_id"], "pass": p, "levels": len(base), "cleared": lc,
                     "won": r["state"] == "won", "score": r["final_score"], "actions": len(h), "tokens": t,
                     "wall_s": r.get("final_wallclock_seconds"), "apl": apl, "base": base, "clear_tok": clear_tok})
    return rows


def permutation(a: list[dict], b: list[dict], draws: int = 20_000, seed: int = 0) -> tuple[float, float]:
    """Mean of b minus mean of a over game-passes; two-sided p with pass labels shuffled within each game."""
    by = defaultdict(lambda: ([], []))
    for r in a:
        by[r["game"]][0].append(r["score"])
    for r in b:
        by[r["game"]][1].append(r["score"])
    games = sorted(g for g in by if by[g][0] and by[g][1])

    def stat(groups):
        return st.mean([x for _, bb in groups for x in bb]) - st.mean([x for aa, _ in groups for x in aa])

    obs = stat([by[g] for g in games])
    rng = random.Random(seed)
    hits = 0
    for _ in range(draws):
        groups = []
        for g in games:
            pool = by[g][0] + by[g][1]
            rng.shuffle(pool)
            groups.append((pool[:len(by[g][0])], pool[len(by[g][0]):]))
        if abs(stat(groups)) >= abs(obs) - 1e-12:
            hits += 1
    return obs, (hits + 1) / (draws + 1)


# ------------------------------------------------------------------------------------------------ transcripts
def parse_transcript(path: Path) -> list[dict]:
    turns, cur, sec, buf = [], None, None, []

    def flush():
        if cur is not None and sec is not None:
            cur["sections"].append((sec, "\n".join(buf)))

    for line in path.read_text(errors="replace").splitlines():
        m = HEADER.match(line)
        if m:
            flush()
            cur = {"step": int(m.group(1)), "action": int(m.group(2)), "time": m.group(3), "sections": []}
            turns.append(cur)
            sec, buf = None, []
            continue
        if line.strip() in MARKERS:
            flush()
            sec, buf = line.strip(), []
            continue
        buf.append(line)
    flush()
    for t in turns:
        t["code"] = [_code_of(s) for k, s in t["sections"] if k == "[TOOL CALL: python]"]
        t["results"] = [s for k, s in t["sections"] if k == "[TOOL RESULT: python]"]
        t["thinking"] = "\n".join(s for k, s in t["sections"] if k == "[THINKING]")
    return turns


def _code_of(text: str) -> str:
    m = re.search(r"<parameter=code>\n?(.*?)</parameter>", text, re.S)
    return m.group(1) if m else text


def own_definitions(code: str) -> set:
    """Tool names the snippet defines or assigns itself (its calls are then the model's own helper, not the tool)."""
    return {n for n in TOOLS if re.search(rf"^\s*def\s+{n}\s*\(|^\s*{n}\s*=", code, re.M)}


def tool_calls_in(code: str) -> Counter:
    c = Counter()
    mine = own_definitions(code)
    for name in TOOLS:
        if name in mine:
            c["shadowed_" + name] += 1
            continue
        c[name] += len(re.findall(rf"(?<![\w.]){name}\s*\(", code))
    c["last_execute_plan"] = len(re.findall(r"\blast_execute_plan\b", code))
    c["defines_step"] = len(re.findall(r"^\s*def\s+step\s*\(", code, re.M))
    return +c


def parse_results(text: str) -> dict:
    out = {}
    oks = re.findall(r"'ok':\s*(True|False)", text)
    if oks:
        out["verify_ok"] = [o == "True" for o in oks]
        out["checked"] = [int(x) for x in re.findall(r"'checked':\s*(\d+)", text)]
        out["matched"] = [int(x) for x in re.findall(r"'matched':\s*(\d+)", text)]
    founds = re.findall(r"'found':\s*(True|False)", text)
    if founds:
        out["plan_found"] = [f == "True" for f in founds]
        out["plan_reason"] = re.findall(r"'reason':\s*'(\w+)'", text)
    if "'executed':" in text:
        out["executed"] = [int(x) for x in re.findall(r"'executed':\s*(\d+)", text)]
        out["exec_stopped"] = re.findall(r"'stopped':\s*'?([\w ]+?)'?[,}]", text)
    out["errors"] = sorted(k for k, rx in ERRORS.items() if rx.search(text))
    return out


def level_of_turn(events: list[dict], turn: dict) -> int | None:
    """The level the turn played on: its first action's level, else the level after the last earlier action."""
    same = [e for e in events if e.get("analysis_step") == turn["step"] and e.get("type") == "action"]
    if same:
        return same[0].get("level")
    prior = [e for e in events if e.get("type") == "action" and (e.get("action_num") or 0) <= turn["action"]]
    return prior[-1].get("level") if prior else 1


# ------------------------------------------------------------------------------------------------ report
def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", type=int, default=5)
    a = ap.parse_args(argv)
    run, base, out = Path(a.run), Path(a.baseline), Path(a.out)
    (out / "cases").mkdir(parents=True, exist_ok=True)
    R, B = load_rows(run, "WM1"), load_rows(base, "P0")

    # scores
    d, p = permutation(B, R)
    mt = {bud: (st.mean(AK.score_at_budget(r, bud) for r in B), st.mean(AK.score_at_budget(r, bud) for r in R))
          for bud in BUDGETS}
    lv = (st.mean(r["cleared"] for r in B), st.mean(r["cleared"] for r in R))
    games = sorted({r["game"] for r in B} | {r["game"] for r in R})
    with open(out / "per_game.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["game", "p0_score", "wm1_score", "delta", "p0_levels", "wm1_levels", "p0_tokens", "wm1_tokens",
                    "p0_wall_min", "wm1_wall_min"])
        for g in games:
            b_, r_ = [x for x in B if x["game"] == g], [x for x in R if x["game"] == g]
            if not b_ or not r_:
                continue
            m = lambda rows, k: st.mean(x[k] or 0 for x in rows)  # noqa: E731
            w.writerow([g, round(m(b_, "score"), 2), round(m(r_, "score"), 2), round(m(r_, "score") - m(b_, "score"), 2),
                        round(m(b_, "cleared"), 2), round(m(r_, "cleared"), 2), int(m(b_, "tokens")), int(m(r_, "tokens")),
                        round(m(b_, "wall_s") / 60, 1), round(m(r_, "wall_s") / 60, 1)])
    better = d >= 5 and p < 0.05 and mt[150_000][1] >= mt[150_000][0]
    worse = d <= -5 and p < 0.05 and mt[150_000][1] < mt[150_000][0]
    verdict = "better" if better else "worse" if worse else "inconclusive"

    # transcripts -> tool calls and levels
    calls, levels = [], {}
    tdir = run / "transcripts"
    for tp in sorted(tdir.glob("*.txt")) if tdir.is_dir() else []:
        m = re.match(r"(.+)_p(\d+)\.txt$", tp.name)
        if not m:
            continue
        gid, ps = m.group(1), int(m.group(2))
        ev_path = run / "artifacts" / f"{gid}_p{ps}_events.jsonl"
        events = [json.loads(x) for x in ev_path.read_text().splitlines() if x.strip()] if ev_path.is_file() else []
        for t in parse_transcript(tp):
            lvl = level_of_turn(events, t)
            key = (gid.split("-")[0], ps, lvl)
            L = levels.setdefault(key, Counter())
            L["turns"] += 1
            for code, res in zip(t["code"], t["results"] + [""] * len(t["code"])):
                tc = tool_calls_in(code)
                pr = parse_results(res)
                for k, v in tc.items():
                    L[k] += v
                if pr.get("verify_ok"):
                    L["verify_ok_any"] = max(L["verify_ok_any"], int(any(pr["verify_ok"])))
                    L["verify_reports"] += len(pr["verify_ok"])
                    L["max_checked"] = max([L["max_checked"]] + pr.get("checked", []))
                if pr.get("plan_found"):
                    L["plan_found_any"] = max(L["plan_found_any"], int(any(pr["plan_found"])))
                    for r_ in pr.get("plan_reason", []):
                        L["plan_reason_" + r_] += 1
                for s in pr.get("exec_stopped", []):
                    L["exec_stop_" + s.strip().replace(" ", "_")] += 1
                for e in pr["errors"]:
                    L["err_" + e] += 1
                if tc:
                    calls.append({"game": key[0], "pass": ps, "level": lvl, "step": t["step"], "action": t["action"],
                                  "time": t["time"], "tools": dict(tc), "results": {k: v for k, v in pr.items() if v},
                                  "code": code[:6000], "result_head": res[:3000]})
    cleared = {(r["game"], r["pass"]): r["cleared"] for r in R}
    with open(out / "tool_calls.jsonl", "w") as f:
        for c in calls:
            f.write(json.dumps(c) + "\n")
    cols = ["turns", "verify_model", "verify_reports", "verify_ok_any", "max_checked", "plan_actions", "plan_found_any",
            "execute_plan", "rollout", "defines_step", "err_traceback", "err_timeout", "err_name_error", "err_sandbox_start"]
    lrows = []
    for (g, ps, lvl), L in sorted(levels.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] or 0)):
        lrows.append({"game": g, "pass": ps, "level": lvl, "cleared": int((lvl or 0) <= cleared.get((g, ps), 0)),
                      **{c: L.get(c, 0) for c in cols},
                      "plan_reasons": ";".join(f"{k[12:]}={v}" for k, v in L.items() if k.startswith("plan_reason_")),
                      "exec_stops": ";".join(f"{k[10:]}={v}" for k, v in L.items() if k.startswith("exec_stop_"))})
    with open(out / "levels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(lrows[0]) if lrows else ["game"])
        w.writeheader()
        w.writerows(lrows)

    # usage summary
    used = [r for r in lrows if r["verify_model"] or r["plan_actions"] or r["execute_plan"]]
    gp_used = {(r["game"], r["pass"]) for r in used}

    def rate(rows, key):
        return f"{sum(1 for r in rows if r[key]) }/{len(rows)}"
    U = ["# WM1: how the model used the world-model tools", "",
         f"Levels played: {len(lrows)}; levels with any world-model call: {len(used)}; game-passes with any: "
         f"{len(gp_used)}/{len(R)}.", "",
         "| Group | Levels | Cleared |", "|---|---|---|"]
    groups = {
        "no world-model call": [r for r in lrows if r not in used],
        "verify_model called, never ok": [r for r in used if r["verify_model"] and not r["verify_ok_any"]],
        "verify_model ok at least once": [r for r in used if r["verify_ok_any"]],
        "plan found at least once": [r for r in used if r["plan_found_any"]],
        "execute_plan called": [r for r in used if r["execute_plan"]],
        "a step defined but verify never called": [r for r in lrows if r["defines_step"] and not r["verify_model"]],
    }
    for name, rows in groups.items():
        U.append(f"| {name} | {len(rows)} | {rate(rows, 'cleared') if rows else '-'} |")
    tot = Counter()
    for r in lrows:
        for c in cols[1:]:
            tot[c] += r[c] if isinstance(r[c], int) else 0
    def tally(field):
        c = Counter()
        for r in lrows:
            for x in filter(None, r[field].split(";")):
                k, v = x.split("=")
                c[k] += int(v)
        return ", ".join(f"{k} {v}" for k, v in c.most_common()) or "none"

    shadow = Counter()
    for L in levels.values():
        for k, v in L.items():
            if k.startswith("shadowed_"):
                shadow[k[9:]] += v
    U += ["", "Snippets that define their own helper under a tool's name (the tool is then shadowed in that snippet): "
          + (", ".join(f"{k} {v}" for k, v in shadow.most_common()) or "none")]
    U += ["", "Totals over all levels: " + ", ".join(f"{k} {v}" for k, v in tot.items()), "",
          "Plan stop reasons: " + tally("plan_reasons"), "", "Execute stops: " + tally("exec_stops")]
    (out / "tool_usage.md").write_text("\n".join(U) + "\n")

    # cases
    kinds = {
        "verified_not_cleared": [r for r in used if r["verify_ok_any"] and not r["cleared"]],
        "plan_found_not_cleared": [r for r in used if r["plan_found_any"] and not r["cleared"]],
        "verify_never_ok": sorted([r for r in used if r["verify_model"] >= 3 and not r["verify_ok_any"]],
                                  key=lambda r: -r["verify_model"]),
        "cleared_with_verified_model": [r for r in used if r["verify_ok_any"] and r["cleared"]],
        "tool_errors": [r for r in used if r["err_timeout"] or r["err_sandbox_start"] or r["err_name_error"]],
    }
    index = ["# Cases", ""]
    for kind, rows in kinds.items():
        index.append(f"## {kind} ({len(rows)})")
        for r in rows[:a.cases]:
            name = f"{kind}__{r['game']}_p{r['pass']}_L{r['level']}.md"
            gid = next(x["game_id"] for x in R if x["game"] == r["game"])
            turns = parse_transcript(run / "transcripts" / f"{gid}_p{r['pass']}.txt")
            ev_path = run / "artifacts" / f"{gid}_p{r['pass']}_events.jsonl"
            events = [json.loads(x) for x in ev_path.read_text().splitlines() if x.strip()] if ev_path.is_file() else []
            body = [f"# {r['game']} pass {r['pass']} level {r['level']} ({kind}; cleared {r['cleared']})", ""]
            for t in turns:
                if level_of_turn(events, t) != r["level"]:
                    continue
                body.append(f"## step {t['step']} (action {t['action']}, {t['time']})")
                if t["thinking"].strip():
                    body += ["Thinking (head):", "", "> " + t["thinking"].strip()[:1500].replace("\n", "\n> "), ""]
                for code, res in zip(t["code"], t["results"] + [""] * len(t["code"])):
                    body += ["```python", code.strip()[:5000], "```", "Result (head):", "```", res.strip()[:2500], "```", ""]
            (out / "cases" / name).write_text("\n".join(body) + "\n")
            index.append(f"- [{name}](cases/{name}): verify calls {r['verify_model']}, ok {r['verify_ok_any']}, "
                         f"plan calls {r['plan_actions']}, found {r['plan_found_any']}, execute {r['execute_plan']}")
        index.append("")
    (out / "cases.md").write_text("\n".join(index) + "\n")

    # comparison
    C = ["# WM1 against P0 on the 25 public games", "",
         f"P0: {len(B)} game-passes, mean {st.mean(r['score'] for r in B):.2f}. "
         f"WM1: {len(R)} game-passes, mean {st.mean(r['score'] for r in R):.2f}.", "",
         f"Delta (WM1 - P0) {d:+.2f}, game-paired permutation p {p:.3f} (20,000 draws).", "",
         "| Measure | P0 | WM1 |", "|---|---|---|",
         *[f"| score at {bud // 1000}k generated tokens per game-pass | {x:.2f} | {y:.2f} |" for bud, (x, y) in mt.items()],
         f"| levels cleared per pass (sum over games, mean of passes) | {lv[0] * 25:.1f} | {lv[1] * 25:.1f} |",
         f"| generated tokens per game-pass (mean) | {st.mean(r['tokens'] for r in B):,.0f} | {st.mean(r['tokens'] for r in R):,.0f} |",
         f"| actions per game-pass (mean) | {st.mean(r['actions'] for r in B):.0f} | {st.mean(r['actions'] for r in R):.0f} |", "",
         f"Pre-registered rule (experiments/arc3-wm/DESIGN.md): **{verdict}**.", "",
         "Per game: per_game.csv. Tool use: tool_usage.md, levels.csv, tool_calls.jsonl, cases.md."]
    (out / "comparison.md").write_text("\n".join(C) + "\n")
    print("\n".join(C))


if __name__ == "__main__":
    main()
