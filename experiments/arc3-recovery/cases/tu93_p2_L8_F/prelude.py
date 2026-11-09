try:
    from collections import deque
except Exception:
    pass
try:
    import json
except Exception:
    pass
try:
    import itertools,json
except Exception:
    pass
try:
    import copy
except Exception:
    pass

passable=lambda ch: ch in 'WgNbpOR'

def bfs(start,tmax=60):
    q=deque([start]); prev={start:None}; found=None
    while q and not found:
        p,t=q.popleft()
        if t>tmax: continue
        o=occ(t)
        for nm in 'UDLR':
            land,path=slide(p,nm)
            if land is None or any(x in o for x in path): continue
            if hit(t,land): continue
            if land==(6,6): found=(p,t,nm); break
            st=(land,t+1)
            if st not in prev: prev[st]=(p,t,nm); q.append(st)
    if not found: return None
    path=[]; cur=(found[0],found[1])
    while prev[cur] is not None:
        p,t,nm=prev[cur]; path.append((nm,str(p),str(cur[0]),cur[1])); cur=(p,t)
    path=path[::-1]
    return ''.join(e[0] for e in path)+found[2], path, found

def solve(grid, start, goal):
    from collections import deque
    R=len(grid); C=len(grid[0])
    def isnode(ch): return ch!='B' and ch!='g'
    def slide(p, d):
        r,c=p; dr,dc=d
        while True:
            nr,nc=r+dr,c+dc
            if not(0<=nr<R and 0<=nc+C>0 and 0<=nc<C): return None
            ch=grid[nr][nc]
            if ch=='B': return None
            if ch=='g': r,c=nr,nc; continue
            return (nr,nc)
    dirs=[(-1,0,'U'),(1,0,'D'),(0,-1,'L'),(0,1,'R')]
    q=deque([start]); prev={start:None}
    while q:
        u=q.popleft()
        if u==goal: break
        for dr,dc,nm in dirs:
            v=slide(u,(dr,dc))
            if v and v not in prev:
                prev[v]=(u,nm); q.append(v)
    if goal not in prev: return None
    path=[]; cur=goal
    while prev[cur] is not None:
        u,nm=prev[cur]; path.append(nm); cur=u
    return path[::-1]

def parse():
    a = current_frame.ascii.split('\n')
    rows=list(range(15,48,3)); cols=list(range(15,48,3))
    g=[[a[r][c] for c in cols] for r in rows]
    b=next((i,j) for i in range(11) for j in range(11) if g[i][j]=='b')
    return g,b

def expected(g, p, nm):
    d={'U':(-1,0),'D':(1,0),'L':(0,-1),'R':(0,1)}[nm]
    r,c=p
    while True:
        nr,nc=r+d[0],c+d[1]
        if not(0<=nr<11 and 0<=nc<11) or g[nr][nc]=='B': return None
        if g[nr][nc]=='g': r,c=nr,nc; continue
        return (nr,nc)

def cells(fr):
    a=fr.ascii.split('\n')
    rows=list(range(21,36,3)); cols=list(range(12,51,3))
    g=[[a[r][c] for c in cols] for r in rows]
    pos={}
    for i in range(5):
        for j in range(13):
            if g[i][j] not in 'Bg': pos.setdefault(g[i][j],[]).append((i,j))
    return g,pos

def state():
    g,pos=cells(current_frame)
    return pos.get('b',[None])[0], pos.get('R',[None])[0], current_frame.ascii.split('\n')[63].count('M')

def grid_from(fr):
    a=fr.ascii.split('\n')
    R=len(a); C=len(a[0])
    rs=[r for r in range(R) if any(a[r][c]!='B' for c in range(C))]
    cs=[c for c in range(C) if any(a[r][c]!='B' for r in range(R))]
    # exclude HUD row(s): find rows that are mostly one color at edges
    r0,r1=min(rs),max(rs); c0,c1=min(cs),max(cs)
    return a,(r0,c0,r1,c1)

def gparse(fr, r0, c0, nr, nc, step=3):
    a=fr.ascii.split('\n')
    return [[a[r0+i*step][c0+j*step] for j in range(nc)] for i in range(nr)]

def land(g,p,nm):
    d={'U':(-1,0),'D':(1,0),'L':(0,-1),'R':(0,1)}[nm]
    r,c=p
    while True:
        r,c=r+d[0],c+d[1]
        if not(0<=r<NR and 0<=c<NC): return None
        ch=g[r][c]
        if ch=='B': return None
        if ch=='g': continue
        return (r,c)

def findb(g):
    return next((i,j) for i in range(NR) for j in range(NC) if g[i][j]=='b')

def cur():
    g=gparse(current_frame,18,12,9,13)
    b=next(((i,j) for i in range(NR) for j in range(NC) if g[i][j]=='b'), None)
    reds=[(i,j) for i in range(NR) for j in range(NC) if g[i][j]=='R']
    return g,b,reds

def run(plan, r0=16, c0=21, nr=9, nc=9, stop_after=None):
    out=[]
    def cur():
        g=gparse(current_frame,r0,c0,nr,nc)
        b=next(((i,j) for i in range(nr) for j in range(nc) if g[i][j]=='b'), None)
        haz=[(g[i][j],i,j) for i in range(nr) for j in range(nc) if g[i][j] in 'RO']
        return g,b,haz
    def land(g,p,nm):
        d={'U':(-1,0),'D':(1,0),'L':(0,-1),'R':(0,1)}[nm]
        r,c=p
        while True:
            r,c=r+d[0],c+d[1]
            if not(0<=r<nr and 0<=c<nc): return None
            ch=g[r][c]
            if ch=='B': return None
            if ch=='g': continue
            return (r,c)
    g,b,haz=cur(); out.append(('init',b,haz))
    for k,nm in enumerate(plan):
        exp=land(g,b,nm)
        res=action([{'action':{'U':'UP','D':'DOWN','L':'LEFT','R':'RIGHT'}[nm]}])
        g,b,haz=cur()
        out.append((nm,exp,b,haz,res.get('game_over'),res.get('level_completed')))
        if res.get('game_over') or res.get('level_completed') or b!=exp: break
    for o in out: print(o)

def oz():
    g=gparse(current_frame,16,21,9,9)
    o=[(i,j) for i in range(9) for j in range(9) if g[i][j]=='O']
    b=[(i,j) for i in range(9) for j in range(9) if g[i][j]=='b']
    return o,b

def slide(p,d):
    r,c=p; dr,dc=d
    while True:
        r+=dr; c+=dc
        if not (0<=r<14 and 0<=c<15) or G[r][c] not in 'Wg': return None
        if G[r][c]=='W' or G[r][c]=='N': return (r,c)
        if G[r][c]=='b': return (r,c)
        if G[r][c] in 'OR': return (r,c)

def live():
    g=gparse(current_frame,16,21,9,9)
    b=next(((i,j) for i in range(9) for j in range(9) if g[i][j]=='b'),None)
    o=[(i,j) for i in range(9) for j in range(9) if g[i][j]=='O']
    r=[(i,j) for i in range(9) for j in range(9) if g[i][j]=='R']
    return g,b,o,r

def show(fr,label):
    g=gparse(fr,16,21,9,9)
    pos={}
    for i in range(9):
        for j in range(9):
            if g[i][j] in 'bONR': pos.setdefault(g[i][j],[]).append((i,j))
    print(label,pos)

def oranges(fr):
    seg=fr.segmentation
    ps=[n['boundary'][0] for n in seg['nodes'] if n['color']=='p' and n['pixels']==1]
    out=[]
    for n in seg['nodes']:
        if n['color'] in 'ONb':
            br,bc=n['boundary'][0]; d=''
            if n['color']=='O':
                d='?'
                for (pr,pc) in ps:
                    if br<=pr<=br+2 and bc<=pc<=bc+2:
                        d='U' if pr==br else ('D' if pr==br+2 else ('L' if pc==bc else 'R'))
            out.append((n['color'],(br-14)//3,(bc-9)//3,d))
    return sorted(out)

def og(fr): return oranges(fr)

def idx(name,t):
    line,i0,v=LINES[name]; L=len(line); P=2*(L-1); m=(i0+v*t)%P
    return m if m<L else P-m

def ocell(n,t): return LINES[n][0][idx(n,t)]

def occ(t): return {LN[n][_i(n,t)] for n in 'ABCD'}

def oslide(n,t):
    a=_i(n,t); b=_i(n,t+1); return [LN[n][i] for i in range(min(a,b),max(a,b)+1)]

def _idx(n,t):
    line,i0,v=LINES[n]; L=len(line); P=2*(L-1); m=(i0+v*t)%P
    return m if m<L else P-m

def _i(n,t):
    L=len(LN[n]); P=2*(L-1); i0,v=IV[n]; m=(i0+v*t)%P
    return m if m<L else P-m

def hit(t,cell): return any(cell in oslide(n,t) for n in 'ABCD')

def pat(fr):
    seg=fr.segmentation
    ps=[n['boundary'][0] for n in seg['nodes'] if n['color']=='p' and n['pixels']==1]
    out=[]
    for n in seg['nodes']:
        if n['color'] in 'ORNb':
            br,bc=n['boundary'][0]; d=''
            if n['color'] in 'OR':
                d='?'
                for (pr,pc) in ps:
                    if br<=pr<=br+2 and bc<=pc<=bc+2:
                        d='U' if pr==br else ('D' if pr==br+2 else ('L' if pc==bc else 'R'))
            out.append((n['color'],(br-12)//3,(bc-9)//3,d))
    return sorted(out,key=lambda z:(z[1],z[2]))

def crop(pr,pc,label):
    print(label)
    for r in range(pr,pr+3): print('  ',a[r][pc:pc+3])

def cellc(fr,r,c):
    a=fr.ascii.split('\n'); return a[12+3*r][9+3*c]

def slide2(g,p,d):
    r,c=p; dr,dc=d
    while True:
        r+=dr; c+=dc
        if not(0<=r<14 and 0<=c<15): return None
        ch=g[r][c]
        if ch=='B': return None
        if ch in 'WNgpORb': return (r,c)

def slide3(g,p,d):
    r,c=p; dr,dc=d
    while True:
        r+=dr; c+=dc
        if not(0<=r<14 and 0<=c<15): return None
        ch=g[r][c]
        if ch=='B': return None
        if ch in 'WN': return (r,c)

def first_stop(g,p,d,occ):
    r,c=p; dr,dc=d
    while True:
        r+=dr; c+=dc
        if not(0<=r<14 and 0<=c<15): return None
        ch=g[r][c]
        if ch=='B': return None
        if ch in 'WN': return (r,c)

def reds_fire(g,reds,p):
    occ={x[0] for x in reds}
    for (r,c,d) in reds:
        s=first_stop(g,(r,c),DIRS[d],occ)
        if s==p: return True,(r,c)
    return False,None

def pat_move(g,pat,occ):
    (r,c),d=pat
    s=first_stop(g,(r,c),DIRS[d],occ)
    if s is None:
        d2={'U':'D','D':'U','L':'R','R':'L'}[d]
        s=first_stop(g,(r,c),DIRS[d2],occ)
        if s is None: return ((r,c),d)
        return (s,d2)
    return (s,d)

def build_G(fr):
    return [['W' if ch in 'pORb' else ch for ch in [cellc(fr,r,c) for c in range(15)]] for r in range(14)]

def bfs_l6(G,p0,reds0,c7,r10,goal=(2,1),maxd=40):
    D={'U':(-1,0),'D':(1,0),'L':(0,-1),'R':(0,1)}
    MV={'UP':'U','DOWN':'D','LEFT':'L','RIGHT':'R'}
    FL={'U':'D','D':'U','L':'R','R':'L'}
    def stop(p,d):
        r,c=p; dr,dc=D[d]
        while True:
            r+=dr; c+=dc
            if not(0<=r<14 and 0<=c<15): return None
            ch=G[r][c]
            if ch=='B': return None
            if ch in 'WN': return (r,c)
    def fires(p,reds):
        for (rr,cc,d) in reds:
            if stop((rr,cc),d)==p: return True
        return False
    def pmove(pat):
        (r,c),d=pat
        s=stop((r,c),d)
        if s is None:
            s=stop((r,c),FL[d])
            if s is None: return ((r,c),d)
            return (s,FL[d])
        return (s,d)
    start=(p0,reds0,c7,r10)
    seen={start:None}; q=[(start,0)]
    while q:
        st,d=q.pop(0)
        p,reds,a,b=st
        if p==goal:
            path=[]; cur=st
            while seen[cur] is not None:
                prev,mv=seen[cur]; path.append(mv); cur=prev
            return list(reversed(path))
        if d>=maxd: continue
        for mv in ['UP','DOWN','LEFT','RIGHT']:
            dest=stop(p,MV[mv])
            if dest is None: continue
            if dest in {a[0],b[0]}: continue
            newreds=frozenset(x for x in reds if x[0]!=dest)
            na=pmove(a); nb=pmove(b)
            if na[0]==dest or nb[0]==dest: continue
            if fires(dest,newreds): continue
            ns=(dest,newreds,na,nb)
            if ns not in seen:
                seen[ns]=(st,mv); q.append((ns,d+1))
    return None

def explore_l6(G,p0,reds0,c7,r10,maxd=24):
    D={'U':(-1,0),'D':(1,0),'L':(0,-1),'R':(0,1)}
    MV={'UP':'U','DOWN':'D','LEFT':'L','RIGHT':'R'}
    FL={'U':'D','D':'U','L':'R','R':'L'}
    def stop(p,d):
        r,c=p; dr,dc=D[d]
        while True:
            r+=dr; c+=dc
            if not(0<=r<14 and 0<=c<15): return None
            ch=G[r][c]
            if ch=='B': return None
            if ch in 'WN': return (r,c)
    def fires(p,reds):
        return any(stop((rr,cc),d)==p for (rr,cc,d) in reds)
    def pmove(pat):
        (r,c),d=pat
        s=stop((r,c),d)
        if s is None:
            s=stop((r,c),FL[d])
            if s is None: return ((r,c),d)
            return (s,FL[d])
        return (s,d)
    start=(p0,reds0,c7,r10); seen={start}; frontier=[start]
    best={}
    for depth in range(maxd):
        nxt=[]
        for (p,reds,a,b) in frontier:
            for mv in MV:
                dest=stop(p,MV[mv])
                if dest is None or dest in {a[0],b[0]}: continue
                nr=frozenset(x for x in reds if x[0]!=dest)
                na=pmove(a); nb=pmove(b)
                if na[0]==dest or nb[0]==dest: continue
                if fires(dest,nr): continue
                ns=(dest,nr,na,nb)
                if ns in seen: continue
                seen.add(ns); nxt.append(ns)
        frontier=nxt
        if not frontier: break
        for st in frontier:
            k=len(st[1])
            if k not in best or depth<best[k][0]: best[k]=(depth,sorted(st[0]),sorted(st[1]))
    print('states',len(seen),'frontier at end',len(frontier))
    for k in sorted(best): print('alive reds',k,'first at depth',best[k][0],'player',best[k][1],'reds',best[k][2])
    cells=sorted({st[0] for st in seen})
    print('cells ever occupied:',cells)
    return seen

def bfs_l6b(G,p0,reds0,c7,r10,goal=(2,1),maxd=40):
    D={'U':(-1,0),'D':(1,0),'L':(0,-1),'R':(0,1)}
    MV={'UP':'U','DOWN':'D','LEFT':'L','RIGHT':'R'}
    FL={'U':'D','D':'U','L':'R','R':'L'}
    def stop(p,d):
        r,c=p; dr,dc=D[d]
        while True:
            r+=dr; c+=dc
            if not(0<=r<14 and 0<=c<15): return None
            ch=G[r][c]
            if ch=='B': return None
            if ch in 'WN': return (r,c)
    def fires(p,reds): return any(stop((rr,cc),d)==p for (rr,cc,d) in reds)
    def pmove(pat):
        (r,c),d=pat
        s=stop((r,c),d)
        if s is None:
            s=stop((r,c),FL[d])
            if s is None: return ((r,c),d)
            return (s,FL[d])
        return (s,d)
    start=(p0,reds0,c7,r10); seen={start:None}; q=[(start,0)]
    while q:
        st,d=q.pop(0); p,reds,a,b=st
        if p==goal:
            path=[]; cur=st
            while seen[cur] is not None:
                prev,mv=seen[cur]; path.append(mv); cur=prev
            return list(reversed(path))
        if d>=maxd: continue
        for mv in MV:
            dest=stop(p,MV[mv])
            if dest is None or dest in {a[0],b[0]}: continue
            nr=frozenset(x for x in reds if (x[0],x[1])!=dest)
            na=pmove(a); nb=pmove(b)
            if na[0]==dest or nb[0]==dest: continue
            if fires(dest,nr): continue
            ns=(dest,nr,na,nb)
            if ns not in seen:
                seen[ns]=(st,mv); q.append((ns,d+1))
    return None

def trace_l6(G,p0,reds0,c7,r10,plan):
    D={'U':(-1,0),'D':(1,0),'L':(0,-1),'R':(0,1)}
    MV={'UP':'U','DOWN':'D','LEFT':'L','RIGHT':'R'}
    FL={'U':'D','D':'U','L':'R','R':'L'}
    def stop(p,d):
        r,c=p; dr,dc=D[d]
        while True:
            r+=dr; c+=dc
            if not(0<=r<14 and 0<=c<15): return None
            ch=G[r][c]
            if ch=='B': return None
            if ch in 'WN': return (r,c)
    def landing(reds):
        return {(rr,cc,d):stop((rr,cc),d) for (rr,cc,d) in reds}
    def pmove(pat):
        (r,c),d=pat
        s=stop((r,c),d)
        if s is None:
            s=stop((r,c),FL[d])
            if s is None: return ((r,c),d)
            return (s,FL[d])
        return (s,d)
    p,reds,a,b=p0,reds0,c7,r10
    for i,mv in enumerate(plan):
        dest=stop(p,MV[mv])
        cap = any((x[0],x[1])==dest for x in reds)
        nr=frozenset(x for x in reds if (x[0],x[1])!=dest)
        na=pmove(a); nb=pmove(b)
        land=landing(nr)
        danger=[k for k,v in land.items() if v==dest]
        hitO = na[0]==dest or nb[0]==dest
        p,reds,a,b=dest,nr,na,nb
        print(i+1,mv,'->',p,'CAP' if cap else '   ','O',a[0],b[0],'redfires' if danger else ('ORANGE!' if hitO else ''),'dangers',sorted({v for v in land.values() if v}))
    print('final',p)

def stop(p,d):
    r,c=p; dr,dc=D[d]
    while True:
        r+=dr; c+=dc
        if not(0<=r<14 and 0<=c<15): return None
        ch=G[r][c]
        if ch=='B': return None
        if ch in 'WN': return (r,c)

def pmove(pat):
    (r,c),d=pat
    s=stop((r,c),d)
    if s is None:
        s=stop((r,c),FL[d])
        if s is None: return ((r,c),d)
        return (s,FL[d])
    return (s,d)

def stop2(p,d,g=G):
    r,c=p; dr,dc=D[d]
    while True:
        r+=dr; c+=dc
        if not(0<=r<14 and 0<=c<15): return None
        ch=g[r][c]
        if ch=='B': return None
        if ch in 'WN': return (r,c)

def pmove2(pat):
    (r,c),d=pat
    s=stop2((r,c),d)
    if s is None:
        s=stop2((r,c),FL[d])
        if s is None: return ((r,c),d)
        return (s,FL[d])
    return (s,d)

def blk(fr,r,c,R0=18,C0=12):
    s=fr.ascii.split('\n')
    return [s[R0+3*r+dr][C0+3*c:C0+3*c+3] for dr in range(3)]

def l7(fr):
    a=fr.ascii.split('\n'); R0,C0=18,12
    G=[[a[R0+3*r+1][C0+3*c+1] for c in range(13)] for r in range(9)]
    pats=[]
    for r in range(9):
        for c in range(13):
            ch=G[r][c]
            if ch in 'brR':
                sq=[a[R0+3*r+dr][C0+3*c:C0+3*c+3] for dr in range(3)]
                mk=''
                if 'p' in sq[2]: mk='D'
                elif 'p' in sq[0]: mk='U'
                elif 'p' in sq[1][0]: mk='L'
                elif 'p' in sq[1][2]: mk='R'
                pats.append((ch,r,c,mk))
    return G,pats

def auto(fr):
    a=fr.ascii.split('\n')
    rows=[];cols=[]
    for r in range(60):
        for c in range(64):
            if a[r][c]!='B': rows.append(r); cols.append(c)
    best=None
    for r0 in range(min(rows),min(rows)+3):
        for c0 in range(min(cols),min(cols)+3):
            nr=(max(rows)-r0+1)//3; nc=(max(cols)-c0+1)//3
            bad=0
            for r in range(nr):
                for c in range(nc):
                    if len({a[r0+3*r+dr][c0+3*c+dc] for dr in range(3) for dc in range(3)})>1: bad+=1
            if best is None or bad<best[4]: best=(r0,c0,nr,nc,bad)
    r0,c0,nr,nc,bad=best
    G=[[a[r0+3*r+1][c0+3*c+1] for c in range(nc)] for r in range(nr)]
    pats=[]
    for r in range(nr):
        for c in range(nc):
            if G[r][c] in 'brRO':
                sq=[a[r0+3*r+dr][c0+3*c:C0+3*c+3] if False else a[r0+3*r+dr][c0+3*c:c0+3*c+3] for dr in range(3)]
                mk=''
                if 'p' in sq[2]: mk='D'
                elif 'p' in sq[0]: mk='U'
                elif 'p' in sq[1][0]: mk='L'
                elif 'p' in sq[1][2]: mk='R'
                pats.append((G[r][c],r,c,mk))
    return (r0,c0,nr,nc,bad),G,pats
