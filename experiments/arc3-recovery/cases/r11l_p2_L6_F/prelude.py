try:
    import collections
except Exception:
    pass
try:
    import json
except Exception:
    pass
try:
    import itertools, json
except Exception:
    pass
try:
    import time
except Exception:
    pass
try:
    import copy
except Exception:
    pass

def crop(r0,r1,c0,c1):
    for r in range(r0,r1+1): print(r, a[r][c0:c1+1])

def center(n):
    b=n['boundary']; 
    return (sum(p[0] for p in b)/len(b), sum(p[1] for p in b)/len(b))

def get_grid():
    return [list(r) for r in current_frame.ascii.split('\n')]

def masks(g):
    black=set(); water=set()
    for r in range(64):
        for c in range(64):
            if g[r][c]=='B': black.add((r,c))
            elif g[r][c]=='S': water.add((r,c))
    return black, water

def comps(cells):
    cells=set(cells); out=[]
    while cells:
        s=cells.pop(); stack=[s]; comp=[s]
        while stack:
            r,c=stack.pop()
            for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
                p=(r+dr,c+dc)
                if p in cells: cells.remove(p); stack.append(p); comp.append(p)
        out.append(comp)
    return out

def interior_mask(g):
    # flood fill from an interior black cell over cells != 'g'
    start=(40,40)
    seen={start}; stack=[start]
    while stack:
        r,c=stack.pop()
        for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
            p=(r+dr,c+dc)
            if p in seen: continue
            rr,cc=p
            if not(0<=rr<64 and 0<=cc<64): continue
            if g[rr][cc]=='g': continue
            seen.add(p); stack.append(p)
    return seen

def cells_ok(shape, ctr, need_interior=True):
    r0,c0=ctr
    for dr,dc in shape:
        p=(r0+dr,c0+dc)
        if p in water: return False
        if need_interior and p not in inter: return False
    return True

def clearance(p,water):
    r,c=p; best=99
    for (wr,wc) in water:
        d=max(abs(wr-r),abs(wc-c))
        if d<best: best=d
    return best

def build():
    g=get_grid(); black,water=masks(g); inter=interior_mask(g)
    return g,water,inter

def ok_shape(shape,ctr,water,inter):
    r0,c0=ctr
    for dr,dc in shape:
        p=(r0+dr,c0+dc)
        if p in water or p not in inter: return False
    return True

def fm(pts):
    n=len(pts); return (sum(p[0] for p in pts)//n, sum(p[1] for p in pts)//n)

def bres(a,b):
    r0,c0=a; r1,c1=b; out=[]; dr=abs(r1-r0); dc=-abs(c1-c0); sr=1 if r0<r1 else -1; sc=1 if c0<c1 else -1; err=dr+dc
    while True:
        out.append((r0,c0))
        if (r0,c0)==(r1,c1): break
        e2=2*err
        if e2>=dc: err+=dc; r0+=sr
        if e2<=dr: err+=dr; c0+=sc
    return out

def line_ok(a,b,water):
    return all(p not in water for p in bres(a,b))

def sim_check(seq, water, inter, start, SH):
    BLOB,DIA=SH
    keys=['W','A','B','C','D']; blobs={'O':['W','A','B'],'P':['C','D']}
    pos=dict(start); trace=[]
    for click in seq:
        occ={k:pos[k] for k in keys}
        hit=[k for k,v in occ.items() if v==click and k!='W']
        if hit:
            k=hit[0]
            wpos=pos['W']; pos['W']=pos[k]; pos[k]=wpos
        else:
            if click in occ.values(): return None
            pos['W']=click
        bo=fm([pos[k] for k in blobs['O']]); bp=fm([pos[k] for k in blobs['P']])
        if not ok_shape(BLOB,bo,water,inter) or not ok_shape(BLOB,bp,water,inter): return None
        for k in keys:
            if not ok_shape(DIA,pos[k],water,inter): return None
        vals=list(pos.values())
        if len(set(vals))!=len(vals): return None
        for k in blobs['O']:
            if not line_ok(pos[k],bo,water): return None
        for k in blobs['P']:
            if not line_ok(pos[k],bp,water): return None
        trace.append((click,bo,bp))
    return trace

def shapes():
    BLOB=[(-2,-1),(-2,0),(-2,1),(-1,-2),(-1,-1),(-1,0),(-1,1),(-1,2),(0,-2),(0,-1),(0,0),(0,1),(0,2),(1,-2),(1,-1),(1,0),(1,1),(1,2),(2,-1),(2,0),(2,1)]
    DIA=[(-2,0),(-1,-1),(-1,0),(-1,1),(0,-2),(0,-1),(0,0),(0,1),(0,2),(1,-1),(1,0),(1,1),(2,0)]
    return BLOB,DIA

def swap_test():
    pos={'W':(1,1),'A':(2,2)}
    pos['W'],pos['A']=pos['A'],pos['W']
    return pos

def minclr(pts, water):
    return min(clearance(p,water) for p in pts)

def search2(water, diaOK, blobOK, A0,B0,C0,D0,W0, orangeT, purpleT, orange_gray, purple_gray, exact=True, maxnodes=400000):
    tgtO=(51,40); tgtP=(18,57)
    start={'W':W0,'A':A0,'B':B0,'C':C0,'D':D0}
    keys=['W','A','B','C','D']
    groups={'A':'O','B':'O','C':'P','D':'P'}
    sols=[]; nodes=[0]
    def valid(pos):
        bo=fm([pos['W'],pos['A'],pos['B']]); bp=fm([pos['C'],pos['D']])
        if bo not in blobOK or bp not in blobOK: return None
        vals=[pos[k] for k in keys]
        if any(v not in diaOK for v in vals): return None
        if len(set(vals))!=5: return None
        return (bo,bp)
    def dfs(pos, remaining, clicks, clrs):
        nodes[0]+=1
        if nodes[0]>maxnodes: return
        if not remaining:
            # final white placement
            for ctr in orangeT:
                w=(3*ctr[0]-pos['A'][0]-pos['B'][0], 3*ctr[1]-pos['A'][1]-pos['B'][1])
                if w not in diaOK: continue
                p2=dict(pos); p2['W']=w
                v=valid(p2)
                if not v: continue
                bp=v[1]
                good = (bp==tgtP) if exact else (abs(bp[0]-tgtP[0])<=1 and abs(bp[1]-tgtP[1])<=1)
                if not good: continue
                if exact and v[0]!=tgtO: continue
                sols.append((clicks+[w], min(clrs+[clearance(w,water),clearance(v[0],water),clearance(v[1],water)])))
                return
            return
        for i,X in enumerate(remaining):
            tl = orange_gray if groups[X]=='O' else purple_gray
            cur=pos[X]
            for T in tl:
                p1=dict(pos); p1['W']=T
                v1=valid(p1)
                if not v1: continue
                p2=dict(p1); p2['W']=cur; p2[X]=T
                v2=valid(p2)
                if not v2: continue
                c=min(clearance(v1[0],water),clearance(v2[0],water),clearance(v1[1],water),clearance(v2[1],water),clearance(T,water))
                dfs(p2, remaining[:i]+remaining[i+1:], clicks+[T,cur], clrs+[c])
                if sols: return
    dfs(start, ['A','B','C','D'], [], [])
    return sols, nodes[0]

def valid_T(T,Xcur,othercur):
    if T not in diaOK: return False
    if T in (Xcur,othercur): return False
    return fm([Xcur,T,othercur]) in blobOK

def eval_sol(T1,T2,W, water, diaOK, blobOK):
    A0=(9,49); B0=(21,8)
    steps=[]
    # click T1: W=T1,A=A0,B=B0
    for (w,a,b) in [(T1,A0,B0),(A0,T1,B0),(T2,T1,B0),(B0,T1,T2),(W,T1,T2)]:
        bo=fm([w,a,b])
        if bo not in blobOK: return None
        for p in (w,a,b):
            if p not in diaOK: return None
        pts=[w,a,b]
        if len(set(pts))!=3: return None
        lines=[line_ok(p,bo,water) for p in pts]
        steps.append((bo, min(clearance(p,water) for p in pts+[bo]), all(lines)))
    return steps

def base():
    g=[list(r) for r in current_frame.ascii.split('\n')]
    water=set(); 
    for r in range(64):
        for c in range(64):
            if g[r][c]=='S': water.add((r,c))
    seen={(40,40)}; stack=[(40,40)]
    while stack:
        r,c=stack.pop()
        for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
            p=(r+dr,c+dc)
            if p in seen or not(0<=p[0]<64 and 0<=p[1]<64) or g[p[0]][p[1]]=='g': continue
            seen.add(p); stack.append(p)
    BLOB,DIA=shapes()
    diaOK=set(); blobOK=set()
    for (r,c) in seen:
        if all(((r+dr,c+dc) in seen and (r+dr,c+dc) not in water) for dr,dc in DIA): diaOK.add((r,c))
        if all(((r+dr,c+dc) in seen and (r+dr,c+dc) not in water) for dr,dc in BLOB): blobOK.add((r,c))
    # chebyshev clearance map
    INF=99; dist={p:0 for p in water}
    frontier=set(water)
    d=0
    while frontier:
        nxt=set()
        for (r,c) in frontier:
            for dr in (-1,0,1):
                for dc in (-1,0,1):
                    p=(r+dr,c+dc)
                    if p not in dist and 0<=r+dr<64 and 0<=c+dc<64:
                        dist[p]=d+1; nxt.add(p)
        frontier=nxt; d+=1
        if d>12: break
    clr=lambda p: dist.get(p,13)
    return g,water,seen,diaOK,blobOK,clr

def solve_chain(pos, mem, ch, target, diaOK, blobOK, pool, maxnodes=200000):
    keys=[k for k in pos if mem[k]==ch]
    grays=[k for k in keys if k!='W']
    n=len(keys)
    tgt_sum=(n*target[0], n*target[1])
    sols=[]; cnt=[0]
    def cents(ps): return fm(ps)
    def dfs(p, rem, clicks):
        cnt[0]+=1
        if cnt[0]>maxnodes: return
        if not rem:
            ps=[p[k] for k in keys]
            for dr in range(n):
                for dc in range(n):
                    Wp=(tgt_sum[0]+dr-sum(x[0] for x in ps if x!=p['W'])-0, 0)
                break
            # compute required white pos: sum of others + W = tgt_sum (with floor tolerance)
            oth=[p[k] for k in keys if k!='W']
            so=(sum(x[0] for x in oth), sum(x[1] for x in oth))
            for dr in range(n):
                for dc in range(n):
                    Wr=tgt_sum[0]+dr-so[0]; Wc=tgt_sum[1]+dc-so[1]
                    Wp=(Wr,Wc)
                    if Wp in p.values(): continue
                    if Wp not in diaOK: continue
                    if fm(list(oth)+[Wp])!=target: continue
                    sols.append(clicks+[Wp]); return
            return
        for i,X in enumerate(rem):
            cur=p[X]
            for T in pool:
                if T in p.values() and T!=cur: continue
                p2=dict(p); p2['W']=T; p2[X]=cur  # after full macro: white ends at cur, X at T
                ps=[p2[k] for k in keys]
                if fm(ps) not in blobOK: continue
                dfs(p2, rem[:i]+rem[i+1:], clicks+[T,cur])
                if sols: return
    dfs(dict(pos), grays, [])
    return sols[:1], cnt[0]

def mkpool(t,rad,mc,D,C):
    P=[p for p in D if max(abs(p[0]-t[0]),abs(p[1]-t[1]))<=rad and C(p)>=mc]
    P.sort(key=lambda p:(abs(p[0]-t[0])+abs(p[1]-t[1]))); return P

def base2(minwater=20):
    g=[list(r) for r in current_frame.ascii.split('\n')]
    S={(r,c) for r in range(64) for c in range(64) if g[r][c]=='S'}
    seen0=set(); comps=[]
    for p in S:
        if p in seen0: continue
        st=[p]; comp=set()
        while st:
            q=st.pop()
            if q in comp or q not in S: continue
            comp.add(q); seen0.add(q)
            for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
                nq=(q[0]+dr,q[1]+dc)
                if nq in S and nq not in comp: st.append(nq)
        comps.append(comp)
    water=set().union(*[c for c in comps if len(c)>=minwater]) if comps else set()
    seen={(40,40)}; stack=[(40,40)]
    while stack:
        r,c=stack.pop()
        for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
            p=(r+dr,c+dc)
            if p in seen or not(0<=p[0]<64 and 0<=p[1]<64) or g[p[0]][p[1]]=='g': continue
            seen.add(p); stack.append(p)
    BLOB,DIA=shapes()
    diaOK={p for p in seen if all(((p[0]+dr,p[1]+dc) in seen and (p[0]+dr,p[1]+dc) not in water) for dr,dc in DIA)}
    blobOK={p for p in seen if all(((p[0]+dr,p[1]+dc) in seen and (p[0]+dr,p[1]+dc) not in water) for dr,dc in BLOB)}
    INF=10**9; dist={}; frontier=set(water); d=0
    for p in water: dist[p]=0
    while frontier and d<12:
        nxt=set()
        for (r,c) in frontier:
            for dr in (-1,0,1):
                for dc in (-1,0,1):
                    q=(r+dr,c+dc)
                    if q not in dist and 0<=r+dr<64 and 0<=c+dc<64: dist[q]=d+1; nxt.add(q)
        frontier=nxt; d+=1
    clr=lambda p: dist.get(p,13)
    return g,water,seen,diaOK,blobOK,clr,[len(c) for c in comps]

def plan_chain(pos, mem, ch, target, pool, diaOK, blobOK, maxnodes=300000):
    keys=[k for k in pos if mem[k]==ch]; n=len(keys)
    grays=[k for k in keys if k!='W']
    sols=[]; cnt=[0]
    def finals(p):
        oth=[p[k] for k in keys if k!='W']
        so=(sum(x[0] for x in oth), sum(x[1] for x in oth)); out=[]
        for dr in range(n):
            for dc in range(n):
                Wp=(n*target[0]+dr-so[0], n*target[1]+dc-so[1])
                if Wp in p.values() or Wp not in diaOK: continue
                if fm(list(oth)+[Wp])!=target: continue
                out.append(Wp)
        return out
    def dfs(p, rem, clicks):
        cnt[0]+=1
        if cnt[0]>maxnodes or sols: return
        if not rem:
            for Wp in finals(p): sols.append(clicks+[Wp]); return
            return
        for i,X in enumerate(rem):
            cur=p[X]
            for T in pool:
                if T!=cur and T in p.values(): continue
                p2=dict(p); p2['W']=cur; p2[X]=T
                if fm([p2[k] for k in keys]) not in blobOK: continue
                dfs(p2, rem[:i]+rem[i+1:], clicks+[T,cur])
                if sols: return
    dfs(dict(pos), grays, [])
    return sols[:1], cnt[0]

def apply_clicks(clicks):
    global pos
    for click in clicks:
        hit=[k for k,v in pos.items() if v==click and k!='W']
        if hit:
            k=hit[0]; w=pos['W']; pos['W']=pos[k]; pos[k]=w
            if mem[k]!=mem['W']: mem[k],mem['W']=mem['W'],mem[k]
        else: pos['W']=click
        full.append(click)

def indiamond(cell, ctr, exclude=()):
    if ctr in exclude: return False
    return max(abs(cell[0]-ctr[0]),abs(cell[1]-ctr[1]))<=2 and abs(cell[0]-ctr[0])+abs(cell[1]-ctr[1])<=3

def sim(cs_,pos,mem):
    for c in cs_:
        cands=sorted([(abs(c[0]-v[0])+abs(c[1]-v[1]),k) for k,v in pos.items() if k!='W'])
        near=[(d,k) for d,k in cands if d<=4]
        if near:
            d,k=near[0]
            if d!=0: return 'OFFCENTER '+str(c)+str(k)+str(d)
            w=pos['W']; pos['W']=pos[k]; pos[k]=w
            if mem[k]!=mem['W']: mem[k],mem['W']=mem['W'],mem[k]
        else:
            if c not in diaOK: return 'NOTDIA '+str(c)
            pos['W']=c
        for chh in ('A','B'):
            cen=fm([pos[k] for k in pos if mem[k]==chh])
            if cen not in blobOK: return chh+'BLOB '+str(cen)
    return 'ok'

def reach(pos, mem, ch, target, diaOK, blobOK, pool, maxmove=2, maxnodes=200000):
    keys=[k for k in pos if mem[k]==ch]; n=len(keys)
    grays=[k for k in keys if k!='W']
    sols=[]; cnt=[0]
    def finals(p):
        oth=[p[k] for k in keys if k!='W']; so=(sum(x[0] for x in oth),sum(x[1] for x in oth)); out=[]
        for dr in range(n):
            for dc in range(n):
                Wp=(n*target[0]+dr-so[0], n*target[1]+dc-so[1])
                if Wp in p.values() or Wp not in diaOK: continue
                if fm(list(oth)+[Wp])!=target: continue
                out.append(Wp)
        return out
    def dfs(p, rem, used, clicks):
        cnt[0]+=1
        if cnt[0]>maxnodes or sols: return
        for Wp in finals(p): sols.append(clicks+[Wp]); return
        if used>=maxmove or not rem: return
        for i,X in enumerate(rem):
            cur=p[X]
            for T in pool:
                if T==cur or T in p.values(): continue
                p2=dict(p); p2['W']=cur; p2[X]=T
                if fm([p2[k] for k in keys]) not in blobOK: continue
                dfs(p2, rem[:i]+rem[i+1:], used+1, clicks+[T,cur])
                if sols: return
    dfs(dict(pos), grays, 0, [])
    return sols[:1], cnt[0]

def apply(cs_):
    for c in cs_:
        hit=[k for k,v in pos.items() if v==c and k!='W']
        if hit:
            k=hit[0]; w=pos['W']; pos['W']=pos[k]; pos[k]=w
            if mem[k]!=mem['W']: mem[k],mem['W']=mem['W'],mem[k]
        else: pos['W']=c
        plan.append(c)

def reach2(pos, mem, ch, target, diaOK, blobOK, pool, maxmove=2, mind=4, maxnodes=200000):
    keys=[k for k in pos if mem[k]==ch]; n=len(keys)
    grays=[k for k in keys if k!='W']
    others=[p for k,p in pos.items() if mem[k]!=ch]
    sols=[]; cnt=[0]
    def spaced(ps):
        for i in range(len(ps)):
            for j in range(i+1,len(ps)):
                if abs(ps[i][0]-ps[j][0])+abs(ps[i][1]-ps[j][1])<mind: return False
        return True
    def finals(p):
        oth=[p[k] for k in keys if k!='W']; so=(sum(x[0] for x in oth),sum(x[1] for x in oth)); out=[]
        for dr in range(n):
            for dc in range(n):
                Wp=(n*target[0]+dr-so[0], n*target[1]+dc-so[1])
                if Wp in p.values() or Wp not in diaOK: continue
                if fm(list(oth)+[Wp])!=target: continue
                if not spaced(list(oth)+[Wp]+others): continue
                out.append(Wp)
        return out
    def dfs(p, rem, used, clicks):
        cnt[0]+=1
        if cnt[0]>maxnodes or sols: return
        for Wp in finals(p): sols.append(clicks+[Wp]); return
        if used>=maxmove or not rem: return
        for i,X in enumerate(rem):
            cur=p[X]
            for T in pool:
                if T==cur or T in p.values(): continue
                p2=dict(p); p2['W']=cur; p2[X]=T
                ps=[p2[k] for k in keys]
                if fm(ps) not in blobOK: continue
                if not spaced(ps+[p for k,p in p2.items() if mem[k]!=ch]): continue
                dfs(p2, rem[:i]+rem[i+1:], used+1, clicks+[T,cur])
                if sols: return
    dfs(dict(pos), grays, 0, [])
    return sols[:1], cnt[0]

def apply2(cs_,pos,mem,plan):
    for c in cs_:
        hit=[k for k,v in pos.items() if v==c and k!='W']
        if hit:
            k=hit[0]; w=pos['W']; pos['W']=pos[k]; pos[k]=w
            if mem[k]!=mem['W']: mem[k],mem['W']=mem['W'],mem[k]
        else: pos['W']=c
        plan.append(c)

def find_c():
    out=[]
    for r,row in enumerate(a):
        for c,ch in enumerate(row):
            if ch=='c': out.append((r,c))
    return out

def centers(frame):
    out=[]
    for r,row in enumerate(frame.ascii.split('\n')):
        for c,ch in enumerate(row):
            if ch=='c': out.append((r,c))
    return out

def blob_center(f):
    seg=f.segmentation
    out=[]
    for n in seg['nodes']:
        if n['pixels']>=20 and n['color']=='W':
            b0=n['boundary'][0]
            rs=[p for p in range(64)]
            out.append((n['pixels'],b0))
    return out

def reach3(pos, mem, ch, target, diaOK, blobOK, pool, maxmove=2, mind=4, safe=6, maxnodes=300000):
    keys=[k for k in pos if mem[k]==ch]; n=len(keys)
    grays=[k for k in keys if k!='W']
    sols=[]; cnt=[0]
    def spaced(ps):
        for i in range(len(ps)):
            for j in range(i+1,len(ps)):
                if abs(ps[i][0]-ps[j][0])+abs(ps[i][1]-ps[j][1])<mind: return False
        return True
    def safecell(c, p, X):
        for k,v in p.items():
            if k==X or k=='W': continue
            if abs(c[0]-v[0])+abs(c[1]-v[1])<safe: return False
        return True
    def finals(p):
        oth=[p[k] for k in keys if k!='W']; so=(sum(x[0] for x in oth),sum(x[1] for x in oth)); out=[]
        for dr in range(n):
            for dc in range(n):
                Wp=(n*target[0]+dr-so[0], n*target[1]+dc-so[1])
                if Wp in p.values() or Wp not in diaOK: continue
                if fm(list(oth)+[Wp])!=target: continue
                if not spaced(list(oth)+[Wp]): continue
                if not safecell(Wp,p,None): continue
                out.append(Wp)
        return out
    def dfs(p, rem, used, clicks):
        cnt[0]+=1
        if cnt[0]>maxnodes or sols: return
        for Wp in finals(p): sols.append(clicks+[Wp]); return
        if used>=maxmove or not rem: return
        for i,X in enumerate(rem):
            cur=p[X]
            for T in pool:
                if T==cur or T in p.values(): continue
                if not safecell(T,p,X): continue
                p2=dict(p); p2['W']=cur; p2[X]=T
                ps=[p2[k] for k in keys]
                if fm(ps) not in blobOK: continue
                if not spaced(ps): continue
                dfs(p2, rem[:i]+rem[i+1:], used+1, clicks+[T,cur])
                if sols: return
    dfs(dict(pos), grays, 0, [])
    return sols[:1], cnt[0]

def W(r,c): return 0<=r<64 and 0<=c<64 and G[r][c] in 'Ww'

def cell(r,c): return G[r][c] if 0<=r<64 and 0<=c<64 else 'X'

def offs(r0,c0):
    d={}
    for dr,dc in shape:
        d[(dr,dc)]=G[r0+dr][c0+dc]
    return d

def build6(frame):
    A=[list(r) for r in frame.ascii.split('\n')]; R=len(A); C=len(A[0])
    seen=[[False]*C for _ in range(R)]; dq=collections.deque()
    for r in range(R):
        for c in (0,C-1):
            if A[r][c]=='W' and not seen[r][c]: seen[r][c]=True; dq.append((r,c))
    for c in range(C):
        for r in (0,R-1):
            if A[r][c]=='W' and not seen[r][c]: seen[r][c]=True; dq.append((r,c))
    while dq:
        r,c=dq.popleft()
        for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
            r1,c1=r+dr,c+dc
            if 0<=r1<R and 0<=c1<C and not seen[r1][c1] and A[r1][c1]=='W': seen[r1][c1]=True; dq.append((r1,c1))
    free=[[A[r][c]!='g' and not seen[r][c] for c in range(C)] for r in range(R)]
    DIA=[(dr,dc) for dr in range(-2,3) for dc in range(-2,3) if abs(dr)+abs(dc)<=2]
    BL=[(-2,-1),(-2,0),(-2,1),(-1,-2),(-1,-1),(-1,0),(-1,1),(-1,2),(0,-2),(0,-1),(0,0),(0,1),(0,2),(1,-2),(1,-1),(1,0),(1,1),(1,2),(2,-1),(2,0),(2,1)]
    diaOK=set(); blobOK=set()
    for r in range(2,R-2):
        for c in range(2,C-2):
            if all(free[r+dr][c+dc] for dr,dc in DIA): diaOK.add((r,c))
            if all(free[r+dr][c+dc] for dr,dc in BL): blobOK.add((r,c))
    return free,diaOK,blobOK

def reach1(pos, mem, ch, target, diaOK, blobOK, safe=6, mind=4, maxnodes=400000):
    keys=[k for k in pos if mem[k]==ch]; n=len(keys)
    grays=[k for k in keys if k!='W']
    others=[p for k,p in pos.items() if mem[k]!=ch]
    def spaced(ps):
        for i in range(len(ps)):
            for j in range(i+1,len(ps)):
                if abs(ps[i][0]-ps[j][0])+abs(ps[i][1]-ps[j][1])<mind: return False
        return True
    def safecell(c,p,skip):
        for k,v in p.items():
            if k in ('W',skip): continue
            if abs(c[0]-v[0])+abs(c[1]-v[1])<safe: return False
        return True
    def finalW(p):
        oth=[p[k] for k in keys if k!='W']; so=(sum(x[0] for x in oth),sum(x[1] for x in oth)); out=[]
        for dr in range(n):
            for dc in range(n):
                Wp=(n*target[0]+dr-so[0], n*target[1]+dc-so[1])
                if Wp in p.values() or Wp not in diaOK: continue
                if fm(list(oth)+[Wp])!=target: continue
                if not spaced(list(oth)+[Wp]+others): continue
                if not safecell(Wp,p,None): continue
                out.append(Wp)
        return out
    pool=sorted(diaOK)
    # k=0
    r=finalW(dict(pos))
    if r: return [r[0]]
    cnt=0
    for X in grays:
        cur=pos[X]
        for T in pool:
            cnt+=1
            if cnt>maxnodes: break
            if T==cur or T in pos.values(): continue
            if not safecell(T,pos,X): continue
            p2=dict(pos); p2['W']=cur; p2[X]=T
            ps=[p2[k] for k in keys]
            if fm(ps) not in blobOK: continue
            if not spaced(ps+[p for k,p in p2.items() if mem[k]!=ch]): continue
            for Wp in finalW(p2):
                return [T,cur,Wp]
    return None

def reach4(pos,mem,ch,target,diaOK,blobOK,bad_cent=(),safe=6,mind=4):
    keys=[k for k in pos if mem[k]==ch]; n=len(keys); grays=[k for k in keys if k!='W']
    others=[p for k,p in pos.items() if mem[k]!=ch]
    def safecell(c,skip=None):
        for k,v in pos.items():
            if k=='W' or k==skip: continue
            if abs(c[0]-v[0])+abs(c[1]-v[1])<safe: return False
        return True
    def finalWs(p):
        oth=[p[k] for k in keys if k!='W']; so=(sum(x[0] for x in oth),sum(x[1] for x in oth)); out=[]
        for dr in range(n):
            for dc in range(n):
                Wp=(n*target[0]+dr-so[0], n*target[1]+dc-so[1])
                if Wp in p.values() or Wp not in diaOK: continue
                if fm(list(oth)+[Wp])!=target: continue
                if not spaced(list(oth)+[Wp]+others,mind): continue
                if not safecell(Wp): continue
                out.append(Wp)
        return out
    best=[]
    for Wp in finalWs(dict(pos)): best.append((0,[Wp]))
    for X in grays:
        cur=pos[X]
        for T in diaOK:
            if T==cur or T in pos.values() or not safecell(T,X): continue
            p2=dict(pos); p2['W']=cur; p2[X]=T
            ps=[p2[k] for k in keys]; cen=fm(ps)
            if cen not in blobOK or not spaced(ps+[p for k,p in p2.items() if mem[k]!=ch],mind): continue
            pen=10000 if cen in bad_cent else 0
            for Wp in finalWs(p2):
                best.append((abs(T[0]-cur[0])+abs(T[1]-cur[1])+abs(Wp[0]-cur[0])+abs(Wp[1]-cur[1])+pen,[T,cur,Wp]))
    best.sort(key=lambda x:x[0])
    return best[0][1] if best else None

def reach5(pos, mem, ch, target, diaOK, blobOK, bad_cent=(), safe=6, mind=4, box=13, maxsol=1):
    keys=[k for k in pos if mem[k]==ch]; n=len(keys)
    grays=[k for k in keys if k!='W']
    others=[p for k,p in pos.items() if mem[k]!=ch]
    def spaced(ps):
        for i in range(len(ps)):
            for j in range(i+1,len(ps)):
                if abs(ps[i][0]-ps[j][0])+abs(ps[i][1]-ps[j][1])<mind: return False
        return True
    def safecell(c, skip=()):
        for k,v in pos.items():
            if k in skip or k=='W': continue
            if abs(c[0]-v[0])+abs(c[1]-v[1])<safe: return False
        return True
    sols=[]
    X1,X2=grays[0],grays[1]
    c1,c2=pos[X1],pos[X2]
    W0=pos['W']
    T=(n*target[0], n*target[1])
    cells=[(r,c) for r in range(target[0]-box,target[0]+box+1) for c in range(target[1]-box,target[1]+box+1) if (r,c) in diaOK]
    for T1 in cells:
        if T1 in pos.values() or not safecell(T1,(X1,)): continue
        cen1=fm([W0,T1,c2])
        if cen1 not in blobOK or cen1 in bad_cent: continue
        if not spaced([W0,T1,c2]+others): continue
        for T2 in cells:
            if T2==T1 or T2 in pos.values() or T2==c1: continue
            if not safecell(T2,(X1,X2)): continue
            Wp=(T[0]-T1[0]-T2[0], T[1]-T1[1]-T2[1])
            if Wp in (T1,T2,c1,c2,W0) or Wp not in diaOK: continue
            if not spaced([Wp,T1,T2]+others): continue
            if not safecell(Wp): continue
            cen2=fm([c1,T1,T2])
            if cen2 not in blobOK or cen2 in bad_cent: continue
            cost=abs(T1[0]-c1[0])+abs(T1[1]-c1[1])+abs(T2[0]-c2[0])+abs(T2[1]-c2[1])+abs(Wp[0]-c1[0])+abs(Wp[1]-c1[1])
            sols.append((cost,[T1,c1,T2,c2,Wp]))
            if len(sols)>=400: break
        if len(sols)>=400: break
    sols.sort(key=lambda x:x[0])
    return [s[1] for s in sols[:maxsol]]

def centers64(frame):
    out=[]
    for r,row in enumerate(frame.ascii.split('\n')):
        for c,ch in enumerate(row):
            if ch=='c': out.append((r,c))
    return out

def spaced(ps,mind=4):
    for i in range(len(ps)):
        for j in range(i+1,len(ps)):
            if abs(ps[i][0]-ps[j][0])+abs(ps[i][1]-ps[j][1])<mind: return False
    return True

def blobpos(frame):
    A=[list(r) for r in frame.ascii.split('\n')]
    def g(r,c): return A[r][c] if 0<=r<64 and 0<=c<64 else 'X'
    out=[]
    for r in range(2,62):
        for c in range(2,62):
            cells=[g(r+dr,c+dc) for dr,dc in SH]
            if all(x not in 'Bgcw' for x in cells) and any(x not in 'Ww' for x in cells):
                out.append(((r,c),''.join(sorted(set(x for x in cells if x!='W')))))
    # dedupe overlapping: keep those where the center cell is filled
    return out

def border(frame):
    A=frame.ascii.split('\n'); s=set()
    for r in range(64):
        for c in range(64):
            if r<3 or c<3 or r>60 or c>60: s.add((r,c,A[r][c]))
    return s
