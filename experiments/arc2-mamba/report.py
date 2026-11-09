"""Markdown tables for RESULTS.md from results.json (printed to stdout).

    ../../envs/mamba/bin/python report.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORDER = ["copy_input:baseline", "mamba:icl", "transformer:icl", "gru:icl", "hybrid:icl", "mamba:ttt", "transformer:ttt", "gru:ttt",
         "hybrid:ttt", "pretrained130m:fewshot"]


def frac(a, b):
    return f"{a}/{b}" if b else "-"


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "-"


def main():
    r = json.loads((HERE / "results.json").read_text())
    print("### Training (7200 process-CPU seconds per arm)\n")
    print("| Arm | Params | Steps | Sequences | Tokens | CPU s | Wall s | Paused s | gen tf-exact | gen tok-acc | "
          "meta-val tf-exact | meta-val tok-acc |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for arm, t in r["training"].items():
        s, v = t["state"], t["final_val"]
        print(f"| {arm} | {t['params']:,} | {s['step']} | {s['seqs']:,} | {s['tokens']:,} | {s['cpu_used']:.0f} | "
              f"{s['wall_used']:.0f} | {s['paused_s']:.0f} | {frac(round(v['gen']['tf_exact'] * v['gen']['n']), v['gen']['n'])} | "
              f"{v['gen']['tok_acc']:.3f} | {frac(round(v['meta_val']['tf_exact'] * v['meta_val']['n']), v['meta_val']['n'])} | "
              f"{v['meta_val']['tok_acc']:.3f} |")
    print("\n### Held-out training tasks, last train pair as query (hpair)\n")
    print("| Run | exact@2 | exact@1 | shape@1 | cell acc (attempt 1) | attempted | <=10 exact@2 | 11-15 exact@2 | "
          "16-30 exact@2 | new exact@2 | inherited exact@2 | CPU s |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k in ORDER:
        x = r["runs"].get(k)
        if not x or not x["hpair"]["outputs"]:
            continue
        h = x["hpair"]
        b = [x.get(f"hpair{s}", {}) for s in ("<=10", "11-15", "16-30")]
        g = [x.get(f"hpair_{s}", {}) for s in ("new", "inherited")]
        print(f"| {k} | {frac(h['exact2'], h['outputs'])} | {frac(h['exact1'], h['outputs'])} | "
              f"{frac(h['shape1'], h['outputs'])} | {pct(h['cells_ok'], h['cells'])} | {frac(h['attempted'], h['outputs'])} | "
              + " | ".join(frac(y.get("exact2", 0), y.get("outputs", 0)) for y in b + g) + f" | {h['cpu_s']:.0f} |")
    print("\n### hpair, tasks with all grids <= 15x15 (121 queries; TTT attempts only these)\n")
    print("| Run | exact@2 | exact@1 | shape@1 | cell acc | changed cells correct (same-shape) | unchanged cells correct | CPU s |")
    print("|---|---|---|---|---|---|---|---|")
    for k in ORDER:
        x = r["runs"].get(k)
        if not x or "hpair_le15" not in x or not x["hpair_le15"]["outputs"]:
            continue
        h, c = x["hpair_le15"], x["hpair_changed_le15"]
        chg = f"{frac(c['changed_ok'], c['changed'])} ({pct(c['changed_ok'], c['changed'])})" if c["changed"] else "0 by definition"
        kept = f"{frac(c['kept_ok'], c['kept'])} ({pct(c['kept_ok'], c['kept'])})" if c["kept"] else "-"
        print(f"| {k} | {frac(h['exact2'], h['outputs'])} | {frac(h['exact1'], h['outputs'])} | {frac(h['shape1'], h['outputs'])} | "
              f"{pct(h['cells_ok'], h['cells'])} ({frac(h['cells_ok'], h['cells'])}) | {chg} | {kept} | {h['cpu_s']:.0f} |")
    print("\n### Held-out training tasks, real test outputs, host evaluator (htest)\n")
    print("| Run | exact@2 | exact@1 | attempted | <=10 | 11-15 | 16-30 | new | inherited | CPU s |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for k in ORDER:
        x = r["runs"].get(k)
        if not x or "htest_host" not in x:
            continue
        h = x["htest_host"]
        a = h["all"]
        print(f"| {k} | {frac(a['exact2'], a['outputs'])} | {frac(a['exact1'], a['outputs'])} | "
              f"{frac(a['attempted'], a['outputs'])} | "
              + " | ".join(frac(h[s]["exact2"], h[s]["outputs"]) for s in ("<=10", "11-15", "16-30", "new", "inherited"))
              + f" | {x['htest_cpu_s']:.0f} |")
    print("\n### Synthetic suite (9 tasks, 1 test output each)\n")
    print("| Run | exact@2 | shape@1 | cell acc | solved | CPU s |")
    print("|---|---|---|---|---|---|")
    for k in ORDER:
        x = r["runs"].get(k)
        if not x or not x["synth"]["outputs"]:
            continue
        s = x["synth"]
        solved = ", ".join(t for t, v in x["synth_per_task"].items() if v) or "none"
        print(f"| {k} | {frac(s['exact2'], s['outputs'])} | {frac(s['shape1'], s['outputs'])} | {pct(s['cells_ok'], s['cells'])} | "
              f"{solved} | {s['cpu_s']:.0f} |")
    print("\n### Solved held-out ids\n")
    for k in ORDER:
        x = r["runs"].get(k)
        if not x:
            continue
        print(f"- {k}: hpair {x.get('hpair_solved_ids') or 'none'}; htest {x.get('htest_host', {}).get('solved_ids') or 'none'}"
              f"; htest solves not among the run's own candidates: {x.get('htest_host', {}).get('solved_not_from_model', '-')}")
    for k in ORDER:
        x = r["runs"].get(k)
        if x and "ttt_steps_hist" in x:
            print(f"- {k} chosen TTT steps: {x['ttt_steps_hist']}")


if __name__ == "__main__":
    main()
