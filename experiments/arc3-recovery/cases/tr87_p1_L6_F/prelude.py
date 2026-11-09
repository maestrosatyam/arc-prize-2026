try:
    import json
except Exception:
    pass
try:
    import hashlib
except Exception:
    pass

def compact(d):
    out={}
    for k,v in d.items():
        if isinstance(v,list):
            out[k]=len(v)
            for e in v[:12]:
                out.setdefault(k,[]).append(e)
        else:
            out[k]=v
    return out

def whites(fr):
    pts=[]
    for n in fr.segmentation['nodes']:
        if n['color']=='W':
            pts.append((n['pixels'], tuple(sorted((r,c) for r,c in [b for b in n['boundary']]))[:1], n['boundary'][0]))
    return pts

def wcells(fr):
    import re
    rows = fr.ascii.split('\n')
    out=[]
    for r,line in enumerate(rows):
        for c,ch in enumerate(line):
            if ch=='W': out.append((r,c))
    return out

def glyphs_at(fr, r0,r1,c0,c1):
    out=[]
    for n in fr.segmentation['nodes']:
        if n['color']!='B': continue
        b=n['boundary']
        rs=[p[0] for p in b]; cs=[p[1] for p in b]
        if min(rs)>=r0 and max(rs)<=r1 and min(cs)>=c0 and max(cs)<=c1:
            out.append((n['pixels'], n['hash'][:8], n['rotation'], (min(rs),min(cs),max(rs),max(cs))))
    return out

def bar(fr):
    return {'blue': glyphs_at(fr,40,47,14,48), 'pink': glyphs_at(fr,51,58,14,48)}

def hud(fr):
    rows=fr.ascii.split('\n')
    r=rows[63]
    return ''.join(r).strip()

def cursor_slot(fr):
    cols=sorted(c for r,c in wcells(fr))
    return (min(cols)-15)//7

def pinkbar(fr):
    out=[]
    for i in range(5):
        g=glyphs_at(fr,51,58,15+7*i,19+7*i)
        out.append(g[0][1] if g else None)
    return out

def bluebar(fr):
    out=[]
    for i in range(5):
        g=glyphs_at(fr,40,47,15+7*i,19+7*i)
        out.append(g[0][1] if g else None)
    return out

def brackets(fr):
    ws=wcells(fr)
    top=[c for r,c in ws if r==48]; bot=[c for r,c in ws if r==60]
    return (min(top),max(top)),(min(bot),max(bot))

def grid(fr): return [list(l) for l in fr.ascii.split('\n')]

def cellsig(g, r0,r1,c0,c1, bg):
    pts=[(r-r0,c-c0) for r in range(r0,r1+1) for c in range(c0,c1+1) if g[r][c]!=bg]
    return pts

def canon(pts):
    best=None
    cur=pts
    for k in range(4):
        mr=min(r for r,c in cur); mc=min(c for r,c in cur)
        norm=tuple(sorted((r-mr,c-mc) for r,c in cur))
        if best is None or norm<best: best=norm
        cur=[(c,-r) for r,c in cur]
    return best

def sig(g,r0,r1,c0,c1,bg): return canon(cellsig(g,r0,r1,c0,c1,bg))

def h(s): return hashlib.md5(str(s).encode()).hexdigest()[:6]

def gs(g,r0,r1,c0,c1,bg): return canon(cellsig(g,r0,r1,c0,c1,bg))

def hid(s, ids={}):
    k=str(s)
    if k not in ids: ids[k]='x%d'%len(ids)
    return ids[k]

def model(fr, pt=PINK_TILES, yt=YEL_TILES):
    g=grid(fr); m={}
    for p,y in zip(pt,yt):
        ps=hid(gs(g,p[0],p[1],p[2],p[3],'P'))
        n=(y[3]-y[2]+1)//7
        m[ps]=[hid(gs(g,y[0],y[1],y[2]+7*k,y[2]+7*k+6,'Y')) for k in range(n)]
    pb=[hid(gs(g,40,46,18+7*k,24+7*k,'P')) for k in range(4)]
    yb=[hid(gs(g,51,57,8+7*k,14+7*k,'Y')) for k in range(7)]
    tgt=[]
    for p in pb: tgt+=m[p]
    return pb,yb,tgt

def cur_slot(fr):
    t=[c for r,c in wcells(fr) if r==48]
    return (min(t)-8)//7

def show(fr,r0,r1,c0,c1):
    g=grid(fr)
    return [''.join(g[r][c0:c1+1]) for r in range(r0,r1+1)]

def seqsig(fr, r0,r1,c0, n, bg):
    return [hid(gs(g,r0,r1,c0+7*k,c0+7*k+6,bg)) for k in range(n)] if False else None

def seqs(fr,r0,r1,c0,n,bg):
    g=grid(fr)
    return [hid(gs(g,r0,r1,c0+7*k,c0+7*k+6,bg)) for k in range(n)]

def l3model(fr):
    P=[(('Y',4,10,5,1),('S',4,10,15,1)),(('Y',4,10,28,2),('S',4,10,45,2)),(('Y',13,19,5,1),('S',13,19,15,2)),
       (('Y',13,19,35,2),('S',13,19,52,1)),(('Y',22,28,5,3),('S',22,28,29,1)),(('Y',22,28,42,1),('S',22,28,52,1))]
    rules={}
    for y,s in P:
        ys=seqs(fr,y[1],y[2],y[3],y[4],'Y'); ss=seqs(fr,s[1],s[2],s[3],s[4],'S')
        rules[tuple(ys)]=ss
    key=seqs(fr,40,46,4,8,'Y'); cur=seqs(fr,51,57,7,7,'S')
    tgt=[]; i=0; ok=True
    while i<len(key):
        for L in (3,2,1):
            k=tuple(key[i:i+L])
            if k in rules:
                tgt.extend(rules[k]); i+=L; break
        else:
            ok=False; break
    return key,cur,tgt,ok

def cols_with(fr,r0,r1,c0,c1,bg):
    g=grid(fr); return ''.join(str(min(sum(1 for r in range(r0,r1+1) if g[r][c]!=bg),9)) for c in range(c0,c1+1))

def l4model(fr):
    P=[('P',3,9,12,'Y',3,9,22),('S',3,9,35,'P',3,9,45),
       ('S',11,17,12,'P',11,17,22),('P',11,17,35,'Y',11,17,45),
       ('P',19,25,12,'Y',19,25,22),('S',19,25,35,'P',19,25,45),
       ('S',27,33,12,'P',27,33,22),('P',27,33,35,'Y',27,33,45)]
    sp={}; py={}
    for c1,r1a,r1b,c1c,c2,r2a,r2b,c2c in P:
        a=seqs(fr,r1a,r1b,c1c,1,c1)[0]; b=seqs(fr,r2a,r2b,c2c,1,c2)[0]
        if c1=='S': sp[a]=b
        else: py[a]=b
    key=seqs(fr,40,46,7,7,'S'); cur=seqs(fr,51,57,7,7,'Y')
    tgt=[py.get(sp.get(k)) for k in key]
    return key,cur,tgt,sp,py

def wbb(fr):
    ws=wcells(fr); rs=sorted(set(r for r,c in ws))
    return (rs[0],rs[-1],min(c for r,c in ws),max(c for r,c in ws))

def l5(fr):
    # legend tiles: (bg, r0,r1,c0, nglyph)
    L=[(('S',10,16,8,1),('P',10,16,18,1)),
       (('S',10,16,31,1),('P',10,16,41,2)),
       (('S',22,28,8,2),('P',22,28,25,1)),
       (('S',22,28,38,1),('P',22,28,48,1))]
    pairs=[]
    for a,b in L:
        pairs.append((seqs(fr,a[1],a[2],a[3],a[4],'S'), seqs(fr,b[1],b[2],b[3],b[4],'P')))
    sbar=seqs(fr,43,49,14,5,'S'); pbar=seqs(fr,52,58,14,5,'P')
    return pairs,sbar,pbar

def tile_idx(fr):
    ws=wcells(fr); top=min(r for r,c in ws); left=min(c for r,c in ws if r==top)
    best=None
    for i,(r,c) in enumerate(TILE_BB):
        d=abs(top-r)+abs(left-c)
        if best is None or d<best[1]: best=(i,d)
    return best

def tiles(fr):
    T=[('S',10,16,8,1),('P',10,16,18,1),('S',10,16,31,1),('P',10,16,41,2),
       ('S',22,28,8,2),('P',22,28,25,1),('S',22,28,38,1),('P',22,28,48,1)]
    out=[]
    for bg,r0,r1,c0,n in T:
        out.append(seqs(fr,r0,r1,c0,n,bg))
    return out

def tidx(fr):
    ws=wcells(fr); top=min(r for r,c in ws); left=min(c for r,c in ws if r==top)
    return min(range(8), key=lambda i: abs(top-TBB[i][0])+abs(left-TBB[i][1]))

def targets(fr):
    sb=seqs(fr,43,49,14,5,'S'); pb=seqs(fr,52,58,14,5,'P')
    return {0:sb[0],1:pb[0],2:sb[1],3:pb[1],4:sb[2],5:pb[3],6:sb[4],7:pb[4]}, sb, pb

def read6(fr):
    out=[]
    for bg,r0,r1,c0,n in T6: out.append(seqs(fr,r0,r1,c0,n,bg))
    return out

def bars6(fr):
    return seqs(fr,44,50,21,3,'S'), seqs(fr,53,59,11,6,'Y')

def readtile(fr,idx):
    T6=[('S',5,11,9,1),('P',5,11,19,2),('P',5,11,38,1),('Y',5,11,48,1),
        ('S',17,23,9,1),('P',17,23,19,2),('P',17,23,38,1),('Y',17,23,48,1),
        ('S',29,35,9,1),('P',29,35,19,2),('P',29,35,38,1),('Y',29,35,48,1)]
    bg,r0,r1,c0,n=T6[idx]
    return seqs(fr,r0,r1,c0,n,bg)

def cidx(fr):
    ws=wcells(fr); top=min(r for r,c in ws); left=min(c for r,c in ws if r==top)
    return min(range(12), key=lambda i: abs(top-BB[i][0])+abs(left-BB[i][1]))
