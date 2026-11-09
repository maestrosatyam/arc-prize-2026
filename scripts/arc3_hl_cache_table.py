"""ARC-3 hint arms: prefix-cache, throughput and equal-action table from the public-run archives.

Per run (FPG, OIF, HLB, HLA, HLS, HLC): prefix-cache hit (arc3_hl_analyze.cache_hit), actions (benchmark.json), the
median decode-batch generation speed from serve.log, HLB against FPG at 100 and 150 actions per game-pass
(arc3_hl_analyze.score_at, game-paired permutation), and HLC's hint mix from hl_lookups.jsonl when present.
usage: envs/arc3/bin/python -I scripts/arc3_hl_cache_table.py [--out PATH]
"""
import argparse
import json
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import arc3_hl_analyze as H  # noqa: E402

A = H.A
ROOT = Path(__file__).resolve().parent.parent
RUNS = {r: ROOT / f"experiments/runs/kaggle-A3-{r}-public-v1" for r in ("FPG", "OIF", "HLB", "HLA", "HLS", "HLC")}
DECODE = re.compile(r"Decode batch.*gen throughput \(token/s\): ([\d.]+)")


def decode_median(run: Path) -> float | None:
    vals = [float(m.group(1)) for line in (run / "serve.log").open(encoding="utf-8", errors="replace")
            if (m := DECODE.search(line))]
    return round(st.median(vals), 1) if vals else None


def at_actions(ref: list[dict], arm: list[dict], n: int) -> dict:
    a = [{"game": r["game"], "score": H.score_at(r, n)} for r in ref]
    b = [{"game": r["game"], "score": H.score_at(r, n)} for r in arm]
    d, p = A.WM.permutation(a, b)
    return {"ref": round(st.mean(x["score"] for x in a), 2), "arm": round(st.mean(x["score"] for x in b), 2),
            "D": round(d, 2), "p": round(p, 4)}


def hint_mix(run: Path) -> dict | None:
    p = run / "hl_lookups.jsonl"
    if not p.is_file():
        return None
    n, src = 0, {}
    for line in p.open(encoding="utf-8"):
        r = json.loads(line)
        if r.get("kind") == "lookup":
            n += 1
            for k, v in (r.get("sources") or {}).items():
                src[k] = src.get(k, 0) + v
    total = sum(src.values())
    return {"lookups": n, "hints": total, "sources": src,
            "team_share": round(src.get("team", 0) / total, 4) if total else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    rows = {}
    for name, run in RUNS.items():
        bench = A.load_runs(run / "benchmark.json")
        rows[name] = {"archive": str(run.relative_to(ROOT)), **H.cache_hit(run),
                      "actions": sum(int(r.get("actions") or 0) for r in bench),
                      "decode_tok_s_median": decode_median(run)}
    fpg = H._scored(A.load_runs(RUNS["FPG"] / "benchmark.json"))
    hlb = H._scored(A.load_runs(RUNS["HLB"] / "benchmark.json"))
    out = {"runs": rows,
           "hlb_vs_fpg_at_actions": {n: at_actions(fpg, hlb, n) for n in (100, 150)},
           "hlc_hint_mix": hint_mix(RUNS["HLC"])}
    for name, r in rows.items():
        print(f"ARC-3 {name}: hit {r['hit']:.1%} over {r['prefill_lines']} prefill lines, new {r['new_tokens'] / 1e6:.1f}M,"
              f" actions {r['actions']} ({r['actions'] / rows['FPG']['actions']:.3f} of FPG),"
              f" decode median {r['decode_tok_s_median']} tok/s")
    for n, x in out["hlb_vs_fpg_at_actions"].items():
        print(f"ARC-3 HLB vs FPG at {n} actions: {x['arm']} vs {x['ref']}, D {x['D']:+.2f}, p {x['p']}")
    print(f"ARC-3 HLC hint mix: {out['hlc_hint_mix']}")
    if a.out:
        a.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
