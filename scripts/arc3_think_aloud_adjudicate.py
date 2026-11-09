"""ARC-AGI-3 think-aloud: adjudication of the two blind coders into key_map.csv (RULE.md section 2).

key = the mechanism's rule states the trick or a step the insight makes explicit (basis T); part = an ingredient the
insight relies on (basic controls, goal, hazards, budget; basis I). Agreed labels stand; part/none disputes become
part. Writes key_map.csv, adjudication.csv and key_map.sha256. Done before any recognition outcome was computed.

Usage: python -I scripts/arc3_think_aloud_adjudicate.py <coder dir outside the repo> experiments/arc3-think-aloud
"""
import csv, json, sys, hashlib
from pathlib import Path
W = Path(sys.argv[1]); OUT = Path(sys.argv[2])
sheet = json.loads((W/'coding_sheet.json').read_text())
lab = {n: {(r['game'], int(r['level']), r['mechanism_id']): r['label'] for r in csv.DictReader(open(W/f'coder_{n}.csv'))} for n in 'AB'}
K, P = 'key', 'part'
# adjudicated (game, level, mechanism suffix) -> label; basis: T = states the trick or an explicit step, I = ingredient
ADJ = {
 ('ar25',3,'m09'):(P,'I'), ('ar25',4,'m09'):(P,'I'), ('ar25',5,'m04'):(K,'T'), ('ar25',7,'m04'):(K,'T'),
 ('bp35',1,'m03'):(K,'T'), ('bp35',2,'m07'):(P,'I'),
 ('cd82',1,'m03'):(P,'I'), ('cd82',2,'m07'):(P,'I'), ('cd82',3,'m12'):(P,'I'), ('cd82',4,'m12'):(P,'I'),
 ('cd82',5,'m12'):(P,'I'), ('cd82',6,'m12'):(P,'I'),
 ('cn04',1,'m06'):(P,'I'),
 ('dc22',1,'m05'):(K,'T'), ('dc22',5,'m14'):(P,'I'), ('dc22',5,'m15'):(P,'I'), ('dc22',6,'m21'):(P,'I'), ('dc22',6,'m22'):(P,'I'),
 ('ft09',3,'m01'):(K,'T'),
 ('g50t',1,'m10'):(P,'I'), ('g50t',5,'m14'):(P,'I'), ('g50t',6,'m14'):(P,'I'), ('g50t',6,'m16'):(P,'I'),
 ('ka59',2,'m05'):(K,'T'), ('ka59',4,'m08'):(K,'T'), ('ka59',5,'m15'):(K,'T'), ('ka59',6,'m14'):(K,'T'), ('ka59',7,'m14'):(K,'T'),
 ('lf52',1,'m02'):(K,'T'),
 ('lp85',6,'m06'):(K,'T'), ('lp85',6,'m07'):(K,'T'), ('lp85',7,'m04'):(K,'T'), ('lp85',7,'m06'):(K,'T'), ('lp85',7,'m07'):(P,'I'),
 ('ls20',3,'m05'):(P,'I'),
 ('m0r0',2,'m03'):(K,'T'), ('m0r0',3,'m10'):(P,'I'), ('m0r0',3,'m11'):(K,'T'), ('m0r0',4,'m11'):(K,'T'),
 ('m0r0',6,'m03'):(K,'T'), ('m0r0',6,'m13'):(K,'T'),
 ('re86',3,'m04'):(P,'I'), ('re86',7,'m07'):(P,'I'),
 ('s5i5',7,'m03'):(K,'T'), ('s5i5',7,'m08'):(P,'I'), ('s5i5',7,'m12'):(K,'T'), ('s5i5',8,'m12'):(K,'T'),
 ('sc25',4,'m10'):(K,'T'), ('sc25',6,'m13'):(K,'T'),
 ('sk48',1,'m05'):(K,'T'),
 ('sp80',1,'m06'):(P,'I'), ('sp80',3,'m08'):(P,'I'),
 ('su15',3,'m05'):(K,'T'), ('su15',6,'m09'):(K,'T'), ('su15',8,'m10'):(K,'T'), ('su15',9,'m10'):(K,'T'),
 ('tn36',1,'m05'):(K,'T'), ('tn36',5,'m13'):(K,'T'),
 ('tu93',1,'m02'):(K,'T'), ('tu93',5,'m08'):(K,'T'), ('tu93',6,'m08'):(K,'T'),
 ('vc33',3,'m02'):(K,'T'), ('vc33',3,'m07'):(P,'I'),
 ('wa30',2,'m08'):(P,'I'), ('wa30',2,'m10'):(K,'T'), ('wa30',2,'m12'):(K,'T'),
 ('wa30',5,'m08'):(P,'I'), ('wa30',5,'m10'):(K,'T'), ('wa30',5,'m12'):(K,'T'),
 ('wa30',6,'m18'):(K,'T'),
}
rows, adj_rows, used = [], [], set()
for g in sheet['games']:
    for lv in g['levels']:
        L = lv['level']
        for m in g['mechanisms']:
            mid = m['mechanism_id']; k = (g['game'], L, mid); suf = mid.split('.')[-1]
            a, b = lab['A'].get(k, 'none'), lab['B'].get(k, 'none')
            key = (g['game'], L, suf)
            if key in ADJ:
                final, basis = ADJ[key]; used.add(key)
                adj_rows.append(dict(track='ARC-3', game=g['game'], level=L, mechanism_id=mid, coder_A=a, coder_B=b, final=final, basis=basis))
            elif a == b:
                final = a
            elif 'key' in (a, b):
                raise SystemExit(f'unadjudicated key dispute {k} {a} {b}')
            else:
                final = 'part'  # part/none dispute: inclusive
            if final != 'none':
                rows.append(dict(game=g['game'], level=L, mechanism_id=mid, label=final))
assert used == set(ADJ), set(ADJ) - used
OUT.mkdir(exist_ok=True)
with open(OUT/'key_map.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=['game','level','mechanism_id','label']); w.writeheader(); w.writerows(rows)
with open(OUT/'adjudication.csv', 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=list(adj_rows[0])); w.writeheader(); w.writerows(adj_rows)
sha = hashlib.sha256((OUT/'key_map.csv').read_bytes()).hexdigest()
(OUT/'key_map.sha256').write_text(sha + '  key_map.csv\n')
from collections import defaultdict
lev = defaultdict(set)
for r in rows: lev[(r['game'], r['level'])].add(r['label'])
print(dict(rows=len(rows), key=sum(r['label']=='key' for r in rows), part=sum(r['label']=='part' for r in rows),
           levels_key=sum('key' in v for v in lev.values()), part_only=sorted(k for k,v in lev.items() if 'key' not in v),
           adjudicated=len(adj_rows), sha=sha))
