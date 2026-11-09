"""Replay v7-vr's and v7-vr2's ranking (harness code embedded in each built notebook) on the VR-bank's two replicates.
usage: python -I experiments/arc2-bank/v7vr_repeat_replay.py <repo root>"""
import ast, importlib, json, os, sys, tempfile, collections
ROOT = sys.argv[1]
NB = {"v7vr": f"{ROOT}/kaggle/arc2/hybrid_vr/v7vr/arc-agi-2-hybrid-v7vr.ipynb",
      "v7vr2": f"{ROOT}/kaggle/arc2/hybrid_vr/v7vr2/arc-agi-2-hybrid-v7vr2.ipynb"}
P2 = f"{ROOT}/experiments/runs/kaggle-A2-VRbank-v1/hybrid/p2"
tmp = tempfile.mkdtemp(); sys.path.insert(0, tmp)
def load(nb_path, pkg):
    nb = json.load(open(nb_path))
    src = next(("".join(c["source"]) if isinstance(c["source"], list) else c["source"]) for c in nb["cells"]
               if "VR_FILES = {" in ("".join(c["source"]) if isinstance(c["source"], list) else c["source"]))
    a = src.index("VR_FILES = {")
    files = ast.literal_eval(src[a + len("VR_FILES = "): src.index("\n}\n", a) + 2])
    os.makedirs(os.path.join(tmp, pkg, "arc_harness"))
    open(os.path.join(tmp, pkg, "__init__.py"), "w").close()
    for rel, txt in files.items():
        if rel.startswith("arc_harness/") and rel.endswith(".py"):
            open(os.path.join(tmp, pkg, rel), "w").write(txt)
    return importlib.import_module(f"{pkg}.arc_harness.selection")
sel = {k: load(p, k) for k, p in NB.items()}
ch = json.load(open(f"{P2}/_kaggle_challenges_vrbank_tasks.json"))
sol = json.load(open(f"{P2}/_kaggle_solutions_vrbank_tasks.json"))
b0sub = json.load(open(f"{ROOT}/experiments/runs/kaggle-A2-VRbank-v1/hybrid/submission_b0.json"))
res = {}
for rep in ("A", "B"):
    by = collections.defaultdict(list)
    for line in open(f"{P2}/rep_{rep}/vr/samples.jsonl"):
        s = json.loads(line); by[s["task_id"]].append(s)
    right = collections.defaultdict(set); moved = collections.defaultdict(int)
    for t in sorted(sol):
        task = {"id": t, "train": ch[t]["train"], "test": [{"input": x["input"]} for x in ch[t]["test"]]}
        b0 = [[e["attempt_1"], e["attempt_2"]] for e in b0sub[t]]
        truth = sol[t]
        for j, tr in enumerate(truth):
            if tr in b0[j]: right["B0"].add((t, j))
        for k, mod in sel.items():
            att = mod.two_attempts(task, by.get(t, []), "vr", b0, "verified_first", use_validator=True)
            for j, e in enumerate(att):
                if truth[j] in (e["attempt_1"], e["attempt_2"]): right[k].add((t, j))
                if [e["attempt_1"], e["attempt_2"]] != b0[j]: moved[k] += 1
    res[rep] = right
    n = sum(len(v) for v in sol.values())
    print(f"rep {rep}: outputs {n}, tasks {len(sol)}, samples {sum(len(v) for v in by.values())}")
    for k in ("B0", "v7vr", "v7vr2"):
        print(f"  {k:6s} right {len(right[k]):2d}" + (f"  (outputs changed from B0: {moved[k]})" if k in moved else ""))
    print(f"  v7vr2 not v7vr: {sorted(right['v7vr2'] - right['v7vr'])}")
    print(f"  v7vr not v7vr2: {sorted(right['v7vr'] - right['v7vr2'])}")
for k in ("v7vr", "v7vr2"):
    a, b = res["A"][k], res["B"][k]
    print(f"{k}: A {len(a)} B {len(b)}; right in A only {sorted(a - b)}; in B only {sorted(b - a)}")
