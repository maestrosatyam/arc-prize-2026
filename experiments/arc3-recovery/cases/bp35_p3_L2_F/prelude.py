try:
    import json
except Exception:
    pass
try:
    import itertools
except Exception:
    pass
try:
    from collections import Counter
except Exception:
    pass
try:
    import inspect
except Exception:
    pass

def compact(d):
    out={}
    for k,v in d.items():
        if isinstance(v,list):
            out[k]=len(v)
        else:
            out[k]=v
    return out

def cells(frame, ch):
    lines = frame.ascii.split('\n')
    s=set()
    for r,l in enumerate(lines):
        for c,x in enumerate(l):
            if x==ch: s.add((r,c))
    return s

def group(cs):
    rows={}
    for r,c in cs: rows.setdefault(r,[]).append(c)
    return {r:(min(v),max(v),len(v)) for r,v in sorted(rows.items())}

def comps(fr):
    out=[]
    for n in fr.segmentation['nodes']:
        if n['pixels']>=20:
            out.append((n['color'], n['pixels'], tuple(n['boundary'][0]), tuple(n['boundary'][2]) if len(n['boundary'])>2 else None))
    return out

def sets(fr):
    S=set(); N=set()
    for r,l in enumerate(fr.ascii.split('\n')):
        for c,ch in enumerate(l):
            if ch=='S': S.add((r,c))
            elif ch=='N': N.add((r,c))
    return S,N

def score(shift):
    dr,dc = shift
    def sh(s): return {((r+dr)%64,(c+dc)%64) for r,c in s}
    iS = len(sh(Sa)&Sb); iN = len(sh(Na)&Nb)
    return iS, iN

def best_shift(A, B, chars='SN', drange=range(-40,41), dcrange=range(-3,4)):
    out=[]
    for ch in chars:
        pass
    def sets2(fr):
        m={}
        for r,l in enumerate(fr.ascii.split('\n')):
            for c,x in enumerate(l):
                m.setdefault(x,set()).add((r,c))
        return m
    ma, mb = sets2(A), sets2(B)
    best=None
    for dr in drange:
        for dc in dcrange:
            tot=0; det={}
            for ch in chars:
                sa = {( (r+dr)%64, (c+dc)%64) for r,c in ma.get(ch,())}
                inter = len(sa & mb.get(ch,set()))
                det[ch]=inter
                tot += inter
            if best is None or tot>best[0]:
                best=(tot,dr,dc,det)
    return best

def rowprofile(fr, off=0):
    prof = {}
    for r,l in enumerate(fr.ascii.split('\n')):
        wr = (r+off)%64
        s = sum(1 for c,ch in enumerate(l) if ch=='S')
        n = sum(1 for c,ch in enumerate(l) if ch=='N')
        prof[wr] = (s,n)
    return prof

def cmap(fr):
    return {(r,c):ch for r,l in enumerate(fr.ascii.split('\n')) for c,ch in enumerate(l)}

def runs(cells_list):
    out=[]
    for r,c in sorted(cells_list):
        if out and out[-1][0]==r and out[-1][2]==c-1:
            out[-1][2]=c
        else:
            out.append([r,c,c])
    return [tuple(x) for x in out]

def bbox(cells_list):
    rs=[p[0] for p in cells_list]; cs=[p[1] for p in cells_list]
    return (min(rs),min(cs),max(rs),max(cs))

def pebs(fr):
    P = [(p['boundary'][0][0], p['boundary'][0][1]) for p in fr.segmentation['nodes'] if p['color']=='G']
    return P

def plant_nodes(fr):
    out=[]
    for n in fr.segmentation['nodes']:
        if n['color']=='N':
            b=n['boundary']; rs=[p[0] for p in b]; cs=[p[1] for p in b]
            out.append((min(rs),max(rs),min(cs),max(cs)))
    return out

def click_plant(tr,tc,tol=4):
    best=None
    for (r0,r1,c0,c1) in plant_nodes(current_frame):
        rr=min(max(tr,r0),r1); cc=min(max(tc,c0),c1)
        d=abs(rr-tr)+abs(cc-tc)
        if best is None or d<best[0]: best=(d,(rr,cc),(r0,r1,c0,c1))
    if best is None or best[0]>tol: 
        print('no plant near',tr,tc,'best',best); return None
    (rr,cc),bb = best[1],best[2]
    res = action([{'action':'MOUSE','row':rr,'col':cc}])
    print('click plant at',bb,'->',res.get('no_op'),res.get('level_completed'),(res.get('animation') or {}).get('frames'))
    return res

def ppos(fr):
    out={}
    for n in fr.segmentation['nodes']:
        if n['color'] in 'bY':
            b=n['boundary']; rs=[p[0] for p in b]; cs=[p[1] for p in b]
            out[n['color']]=(min(rs),max(rs),min(cs),max(cs))
    return out

def summary(fr):
    m = cmap(fr)
    s = sum(1 for ch in m.values() if ch=='S')
    pk = sorted([(r,c) for (r,c),ch in m.items() if ch in 'PM'])
    prows = sorted(set(r for r,c in pk))
    return s, (prows[0] if prows else None), (pk[0][1] if pk else None), ppos(fr)

def cnt(m,ch): return sum(1 for v in m.values() if v==ch)

def state(fr):
    m = cmap(fr)
    s = cnt(m,'S')
    pr = sorted(set(r for (r,c),ch in m.items() if ch=='p'))
    pk = sorted([(r,c) for (r,c),ch in m.items() if ch in 'P'])
    top0 = sum(1 for c in range(64) if m[(0,c)]=='S')
    return s, (pr[0] if pr else None), (pk[0] if pk else None), top0, ppos(fr).get('b')

def bar(fr):
    m = cmap(fr)
    return sum(1 for c in range(64) if m[(63,c)]=='p'), sum(1 for c in range(64) if m[(0,c)]=='p')

def tstate(fr):
    m = cmap(fr)
    s = cnt(m,'S')
    pr = sorted(set(r for (r,c),ch in m.items() if ch=='p'))
    tide = [r for r in pr if r>36]
    pk = sorted([(r,c) for (r,c),ch in m.items() if ch=='P'])
    return s, (min(tide) if tide else None), (max(pr) if pr else None), (pk[0] if pk else None), ppos(fr).get('b'), bar(fr)[0]
