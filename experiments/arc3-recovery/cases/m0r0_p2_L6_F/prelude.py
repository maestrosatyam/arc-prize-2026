try:
    import pickle
except Exception:
    pass
try:
    import json
except Exception:
    pass
try:
    from collections import deque
except Exception:
    pass
try:
    from collections import Counter
except Exception:
    pass
try:
    import itertools
except Exception:
    pass
try:
    import collections
except Exception:
    pass
try:
    import time
except Exception:
    pass
try:
    import sys
except Exception:
    pass
try:
    import builtins
except Exception:
    pass
try:
    import traceback
except Exception:
    pass
try:
    import inspect, re
except Exception:
    pass

def brief(d):
    out=[]
    for m in d.get('moved',[]): out.append(f"{m['color']} {m['from']}->{m['to']}")
    for m in d.get('resized',[]): out.append(f"resize {m['color']} {m['before_pixels']}->{m['after_pixels']}")
    for m in d.get('appeared',[]): out.append(f"appear {m.get('color')} {m.get('bbox')}")
    for m in d.get('disappeared',[]): out.append(f"disappear {m.get('color')} {m.get('bbox')}")
    for m in d.get('changed_color',[]): out.append(f"color {m.get('at')} {m.get('before_color')}->{m.get('after_color')}")
    return f"cells={d['changed_cell_count']} | " + '; '.join(out)

def ok(r,c,nr,nc,op):
    for x in range(min(r,nr),max(r,nr)+4):
        for y in range(min(c,nc),max(c,nc)+4):
            if x<0 or x>=R or y<0 or y>=C or not passable[x][y]: return False
            if band[x][y] and band[x][y] not in op: return False
    return True

def fits(f,r,c): return 0<=r and r+4<64 and 0<=c and c+4<64 and all(f[r+dr][c+dc] for dr in range(5) for dc in range(5))

def slide(s,act,mir):
    dr,dc=MOV[act]
    if mir and act=='L': dc=1
    if mir and act=='R': dc=-1
    r,c=s; n=0
    while n<4 and fp(r+dr,c+dc): r+=dr;c+=dc;n+=1
    return (r,c)

def crop(a,r0,r1,c0,c1):
    return '\n'.join(f"{r:2d} "+a[r][c0:c1] for r in range(r0,r1))

def reachable(f,start):
    from collections import deque
    seen={start}; q=deque([start])
    for d in [(-1,0),(1,0),(0,-1),(0,1)]:
        pass
    while q:
        r,c=q.popleft()
        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            p=slide_to(f,r,c,dr,dc)
            if p not in seen: seen.add(p); q.append(p)
    return seen

def slide_to(f,r,c,dr,dc):
    while fits(f,r+dr,c+dc): r+=dr; c+=dc
    return (r,c)

def slide5(f,r,c,dr,dc,cap=5):
    n=0
    while n<cap and fits(f,r+dr,c+dc): r+=dr;c+=dc;n+=1
    return (r,c,n)

def mask(a): return [[(a[r][c] in 'BS') for c in range(64)] for r in range(64)]

def step5(f,r,c,dr,dc,cap=5):
    n=0
    while n<cap and fits(f,r+dr,c+dc): r+=dr;c+=dc;n+=1
    return (r,c)

def Lmove(s,a):
    dr,dc=MOV[a]; return step5(free0,s[0],s[1],dr,dc)

def Rmove(s,a):
    dr,dc=MOV[a]
    if a=='L': dc=1
    elif a=='R': dc=-1
    return step5(free0,s[0],s[1],dr,dc)

def overlap(l,r):
    return not (l[0]+4<r[0] or r[0]+4<l[0] or l[1]+4<r[1] or r[1]+4<l[1])

def touching(l,r):
    # edge adjacent, not overlapping
    dr_ok = (l[0]+5==r[0] or r[0]+5==l[0]) and l[1]<=r[1]+4 and r[1]<=l[1]+4
    dc_ok = (l[1]+5==r[1] or r[1]+5==l[1]) and l[0]<=r[0]+4 and r[0]<=l[0]+4
    return dr_ok or dc_ok

def reach_single(st,mir):
    seen={st}; q=deque([st])
    while q:
        s=q.popleft()
        for a in MOV:
            n = Rmove(s,a) if mir else Lmove(s,a)
            if n not in seen: seen.add(n); q.append(n)
    return seen

def Lm(s,act):
    dr,dc=MOV[act]; return sl(free,s[0],s[1],dr,dc)

def Rm(s,act):
    dr,dc=MOV[act]
    if act=='L': dc=1
    elif act=='R': dc=-1
    return sl(free,s[0],s[1],dr,dc)

def path(src,dst):
    prev={src:None}; q=collections.deque([src])
    while q:
        u=q.popleft()
        if u==dst: break
        r,c=u
        for dr,dc in ((4,0),(-4,0),(0,4),(0,-4)):
            v=(r+dr,c+dc)
            if v in S and v not in prev: prev[v]=u; q.append(v)
    if dst not in prev: return None
    p=[];x=dst
    while x is not None: p.append(x); x=prev[x]
    return list(reversed(p))

def bfs(start,goal):
    prev={start:None}; q=deque([start])
    while q:
        s=q.popleft()
        if goal(*s):
            p=[]; x=s
            while prev[x] is not None: p.append(prev[x][1]); x=prev[x][0]
            return list(reversed(p)), s, len(prev)
        for act in MOV:
            ns=(slide(s[0],act,False),slide(s[1],act,True))
            if ns not in prev: prev[ns]=(s,act); q.append(ns)
    return None,None,len(prev)

def spos():
    out=[]
    for n in current_frame.segmentation['nodes']:
        if n['color']=='S':
            b=n['boundary']
            r0=min(p[0] for p in b); c0=min(p[1] for p in b)
            out.append((r0,c0,n['pixels']))
    return sorted(out,key=lambda t:(t[1],t[0]))

def fits4(f,r,c): return 0<=r and r+4<64 and 0<=c and c+4<64 and all(f[r+dr][c+dc] for dr in range(4) for dc in range(4))

def hazhit(r,c): return any(haz[r+dr][c+dc] for dr in range(4) for dc in range(4))

def st():
    o={}
    for n in current_frame.segmentation['nodes']:
        if n['pixels']>=9:
            b=n['boundary']; o[n['color']]=o.get(n['color'],[])+[(min(p[0] for p in b),min(p[1] for p in b),n['pixels'])]
    return o

def sl(f,r,c,dr,dc,cap=4):
    n=0
    while n<cap and fits4(f,r+dr,c+dc): r+=dr;c+=dc;n+=1
    return (r,c)

def reach(st,mir):
    seen={st}; q=deque([st])
    while q:
        s=q.popleft()
        for act in MOV:
            n=slide_ao(s,act,mir)
            if n not in seen: seen.add(n); q.append(n)
    return seen

def redset(f,r0,r1,c0,c1):
    a=f.ascii.split('\n')
    return {(r,c) for r in range(r0,r1) for c in range(c0,c1) if a[r][c]=='R'}

def redcells(f):
    a=f.ascii.split('\n')
    return {(r,c) for r in range(1,63) for c in range(64) if a[r][c]=='R'}

def fp(r,c): return 0<=r and r+4<64 and 0<=c and c+4<64 and all(passable[r+dr][c+dc] for dr in range(4) for dc in range(4))

def safe(r,c): return not any(haz[r+dr][c+dc] for dr in range(4) for dc in range(4))

def tokhit(r,c): return sum(tok[r+dr][c+dc] for dr in range(4) for dc in range(4))

def tmask(r,c):
    m=0
    for i,(tr,tc) in enumerate([(15,31),(19,11),(31,39)]):
        if r<=tr<=r+3 and c<=tc<=c+3: m|=1<<i
    return m

def solve(start, goalfn, maxd=40):
    st=(start[0],start[1],tmask(*start[0])|tmask(*start[1]))
    prev={st:None}; q=deque([st])
    while q:
        s=q.popleft()
        if goalfn(s):
            p=[]; x=s
            while prev[x] is not None: p.append(prev[x][1]); x=prev[x][0]
            return list(reversed(p)), s
        for act in MOV:
            nl=slide(s[0],act,False); nr=slide(s[1],act,True)
            ns=(nl,nr,s[2]|tmask(*nl)|tmask(*nr))
            if ns not in prev: prev[ns]=(s,act); q.append(ns)
    return None,None

def match(got,pred):
    # got: list of (r,c,px); pred: (lp,rp)
    if len(got)==1: return 'merged', got
    best=None
    import itertools
    for perm in itertools.permutations(got):
        d=abs(perm[0][0]-pred[0][0])+abs(perm[0][1]-pred[0][1])+abs(perm[1][0]-pred[1][0])+abs(perm[1][1]-pred[1][1])
        if best is None or d<best[0]: best=(d,perm)
    return ('ok' if best[0]==0 else 'off%d'%best[0]), best[1]

def swept(pos,dr,dc,size):
    r,c=pos; out=set()
    for k in range(1,5):
        rr,cc=r+dr*k,c+dc*k
        for i in range(size):
            for j in range(size): out.add((rr+i,cc+j))
    return out

def slide_ao(s,act,mir):
    dr,dc=MOV[act]
    if mir and act=='L': dc=1
    if mir and act=='R': dc=-1
    return (s[0]+4*dr,s[1]+4*dc) if swept(s[0],s[1],dr,dc) else s

def dist(pos,tok):
    r,c=pos; tr,tc=tok
    dr=max(0, max(r-(tr+1), tr-(r+3)))
    dc=max(0, max(c-(tc+1), tc-(c+3)))
    return dr+dc

def ch(r,c):
    v=a[r][c]
    return {'B':'#','b':'o','S':'X','p':'p','R':'r','W':'.'}.get(v,v)

def nodes():
    out=[]
    for n in current_frame.segmentation['nodes']:
        if n['pixels']<300:
            b=n['boundary']; out.append((n['color'],(min(p[0] for p in b),min(p[1] for p in b)),n['pixels']))
    return out

def occ(pos,size=4):
    r,c=pos; return {(r+i,c+j) for i in range(size) for j in range(size)}

def tokcells(p):
    r,c=p; return {(r+i,c+j) for i in range(2) for j in range(2)}

def free_cell(r,c,blocked): return base[r][c] and (r,c) not in blocked

def can_move(base,pos,dr,dc,size,blocked):
    r,c=pos
    for k in range(1,5):
        rr,cc=r+dr*k,c+dc*k
        for i in range(size):
            for j in range(size):
                if not (base[rr+i][cc+j] and (rr+i,cc+j) not in blocked): return False
    return True

def block_bfs(base,L0,R0,toks,mir=True):
    MOV={'U':(-1,0),'D':(1,0),'L':(0,-1),'R':(0,1)}
    blocked=set()
    for t in toks: blocked|=tokcells(t)
    def step(s,act,m):
        dr,dc=MOV[act]
        if m and act=='L': dc=1
        if m and act=='R': dc=-1
        return (s[0]+4*dr,s[1]+4*dc) if can_move(base,s,dr,dc,4,blocked) else s
    from collections import deque
    prev={(L0,R0):None}; q=deque([(L0,R0)])
    while q:
        s=q.popleft()
        if s[0]==s[1]:
            p=[];x=s
            while prev[x] is not None: p.append(prev[x][1]); x=prev[x][0]
            return list(reversed(p)),s
        for act in MOV:
            ns=(step(s[0],act,False),step(s[1],act,mir))
            if ns not in prev: prev[ns]=(s,act); q.append(ns)
    return None,None

def base_map(a):
    b=[[ (a[r][c] in 'BSbwY') for c in range(64)] for r in range(64)]
    for r in (0,63):
        for c in range(64): b[r][c]=False
    return b

def blkcells(p):
    r,c=p; return {(r+i,c+j) for i in range(4) for j in range(4)}

def conn(base, blocked, s1, s2):
    from collections import deque
    def cells(pos,size): 
        r,c=pos; return [(r+i,c+j) for i in range(size) for j in range(size)]
    start=cells(s1,4)[0]; target=cells(s2,4)[0]
    seen={start}; q=deque([start])
    while q:
        r,c=q.popleft()
        for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
            nr,nc=r+dr,c+dc
            if 0<=nr<64 and 0<=nc<64 and (nr,nc) not in seen and base[nr][nc] and (nr,nc) not in blocked:
                seen.add((nr,nc)); q.append((nr,nc))
    return target in seen

def tok_reach(base, i, toks, blocks, depth=3):
    MOV={'U':(-1,0),'D':(1,0),'L':(0,-1),'R':(0,1)}
    others=set()
    for j,t in enumerate(toks):
        if j!=i: others|=tokcells(t)
    for b in blocks: others|=blkcells(b)
    from collections import deque
    prev={toks[i]:0}; q=deque([toks[i]])
    while q:
        p=q.popleft()
        if prev[p]>=depth: continue
        for act,(dr,dc) in MOV.items():
            if can_move(base,p,dr,dc,2,others):
                np_=(p[0]+4*dr,p[1]+4*dc)
                if np_ not in prev: prev[np_]=prev[p]+1; q.append(np_)
    return prev

def fits_any(r,c): return base[r][c] and base[r+3][c] and base[r][c+3] and base[r+3][c+3] and all(base[r+i][c+j] for i in range(4) for j in range(4))

def comp(s):
    seen={s}; q=collections.deque([s])
    while q:
        u=q.popleft()
        for v in adj[u]:
            if v not in seen: seen.add(v); q.append(v)
    return seen

def tc(p): r,c=p; return frozenset({(r+i,c+j) for i in range(2) for j in range(2)})

def bc(p): r,c=p; return frozenset({(r+i,c+j) for i in range(4) for j in range(4)})

def sw(p,dr,dc,s):
    r,c=p; out=set()
    for k in range(1,5):
        for i in range(s):
            for j in range(s): out.add((r+dr*k+i,c+dc*k+j))
    return frozenset(out)

def build(base):
    D=[('U',-1,0),('D',1,0),('L',0,-1),('R',0,1)]
    B4=[(r,c) for r in range(2,60,4) for c in range(2,60,4) if all(base[r+i][c+j] for i in range(4) for j in range(4))]
    BW={}
    for p in B4:
        for d,dr,dc in D:
            s=frozenset(((p[0]+dr*k+i)*64+(p[1]+dc*k+j)) for k in range(1,5) for i in range(4) for j in range(4))
            if all(base[x//64][x%64] for x in s): BW[(p,d)]=((p[0]+4*dr,p[1]+4*dc), s)
    B2=[(r,c) for r in range(3,60,4) for c in range(3,60,4) if all(base[r+i][c+j] for i in range(2) for j in range(2))]
    TW={}
    for p in B2:
        for d,dr,dc in D:
            s=frozenset(((p[0]+dr*k+i)*64+(p[1]+dc*k+j)) for k in range(1,5) for i in range(2) for j in range(2))
            if all(base[x//64][x%64] for x in s): TW[(p,d)]=((p[0]+4*dr,p[1]+4*dc), s)
    return B4,B2,BW,TW

def cells4(p): return frozenset((p[0]+i)*64+(p[1]+j) for i in range(4) for j in range(4))

def cells2(p): return frozenset((p[0]+i)*64+(p[1]+j) for i in range(2) for j in range(2))

def joint_bfs(base, L0,R0,T0s, cap=400000):
    from collections import deque
    B4,B2,BW,TW=build(base)
    BF={p:cells4(p) for p in B4}; TF={p:cells2(p) for p in B2}
    ACT=['U','D','L','R']; MIR={'U':'U','D':'D','L':'R','R':'L'}
    start=(0,L0,R0)+tuple(T0s)
    prev={start:None}; q=deque([start]); n=0
    while q:
        s=q.popleft(); n+=1
        if n>cap: return ('CAP',None,n)
        mode,L,R=s[0],s[1],s[2]; T=s[3:6]
        if L==R:
            p=[];x=s
            while prev[x] is not None: p.append(prev[x][1]); x=prev[x][0]
            return (list(reversed(p)), s, n)
        tf=[TF[t] for t in T]
        if mode==0:
            lf=BF[L]; rf=BF[R]
            for a in ACT:
                eL=BW.get((L,a)); eR=BW.get((R,MIR[a]))
                if not eL or not eR: continue
                nl,e=eL; nr,e2=eR
                if any(e & f for f in tf) or (e & rf): continue
                if any(e2 & f for f in tf) or (e2 & lf): continue
                ns=(0,nl,nr)+T
                if ns not in prev: prev[ns]=(s,('M',a)); q.append(ns)
            for i in range(3):
                ns=(i+1,L,R)+T
                if ns not in prev: prev[ns]=(s,('C',i)); q.append(ns)
        else:
            i=mode-1; t=T[i]; lf=BF[L]; rf=BF[R]
            others=lf|rf
            for j in range(3):
                if j!=i: others|=tf[j]
            for a in ACT:
                e=TW.get((t,a))
                if e and not (e[1] & others):
                    nt=list(T); nt[i]=e[0]
                    ns=(mode,L,R)+tuple(nt)
                    if ns not in prev: prev[ns]=(s,('T',a)); q.append(ns)
            ns=(0,L,R)+T
            if ns not in prev: prev[ns]=(s,('CB',)); q.append(ns)
    return ('NONE',None,n)

def blocked_by(tp):
    out=set(); tr,tc=tp
    for r in (tr-1,tr-5):
        for c in (tc-1,tc-5):
            if (r,c) in S4: out.add((r,c))
    return frozenset(out)

def dfs(u):
    children=0; disc[u]=low[u]=t[0]; t[0]+=1
    for v in adj[u]:
        if v not in disc:
            par[v]=u; children+=1; dfs(v); low[u]=min(low[u],low[v])
            if par.get(u) is None and children>1: aps.add(u)
            if par.get(u) is not None and low[v]>=disc[u]: aps.add(u)
        elif v!=par.get(u): low[u]=min(low[u],disc[v])

def sim_plan(base,L,R,T,steps):
    D={'U':(-1,0),'D':(1,0),'L':(0,-1),'R':(0,1)}
    MIR={'U':'U','D':'D','L':'R','R':'L'}
    B4,B2,BW,TW=build(base)
    S4=set(B4); S2=set(B2)
    log=[]
    for st in steps:
        if st[0]=='click': continue
        a=st[0]
        if st[1]=='B':
            eL=BW.get((L,a)); eR=BW.get((R,MIR[a]))
            occ=set().union(*[cells2(t) for t in T])
            okL=eL and not (set(eL[1]) & occ) and not (set(eL[1]) & cells4(R))
            okR=eR and not (set(eR[1]) & occ) and not (set(eR[1]) & cells4(L))
            if okL: L=eL[0]
            if okR: R=eR[0]
            log.append((a,'B',okL,okR,L,R))
        else:
            i=st[1]; e=TW.get((T[i],a))
            occ=set(cells4(L))|set(cells4(R))
            for j in range(3):
                if j!=i: occ|=set(cells2(T[j]))
            if e and not (set(e[1]) & occ):
                T=list(T); T[i]=e[0]; log.append((a,i,True,T[i]))
            else: log.append((a,i,False,T[i]))
    return L,R,tuple(T),log

def block_bfs2(base,L0,R0,toks):
    B4,B2,BW,TW=build(base)
    occ=set().union(*[cells2(t) for t in toks]) if toks else set()
    ACT=['U','D','L','R']; MIR={'U':'U','D':'D','L':'R','R':'L'}
    from collections import deque
    start=(L0,R0); prev={start:None}; q=deque([start])
    while q:
        s=q.popleft()
        L,R=s
        if L==R:
            p=[];x=s
            while prev[x] is not None: p.append(prev[x][1]); x=prev[x][0]
            return list(reversed(p)),s
        for a in ACT:
            eL=BW.get((L,a)); eR=BW.get((R,MIR[a]))
            if not eL or not eR: continue
            if set(eL[1]) & (occ|cells4(R)): continue
            if set(eR[1]) & (occ|cells4(L)): continue
            ns=(eL[0],eR[0])
            if ns not in prev: prev[ns]=(s,a); q.append(ns)
    return None,None

def state3():
    bl=[];tk=[]
    for n in current_frame.segmentation['nodes']:
        if n['pixels']==16: bl.append((min(p[0] for p in n['boundary']),min(p[1] for p in n['boundary'])))
        elif n['pixels']==4: tk.append((n['color'],min(p[0] for p in n['boundary']),min(p[1] for p in n['boundary'])))
    return tuple(sorted(bl)),tuple(sorted(tk))

def click(r,c): return {'action':'MOUSE','row':r,'col':c}

def try_click(r,c):
    try:
        res=action([click(r,c)])
        d=brief(frame_diff())
        return (res.get('no_op'),d)
    except Exception as e:
        return ('err',str(e)[:60])

def tryc(r,c):
    try:
        res=action([click(r,c)])
        return (res.get('no_op'),brief(frame_diff()))
    except Exception as e:
        return ('err',str(e)[:70])

def solve5(f, maxd=40):
    import collections
    a=f.ascii.split('\n'); R=len(a); C=len(a[0])
    wall=[[a[r][c] in 'MPW' for c in range(C)] for r in range(R)]
    act=[[a[r][c] if a[r][c] in 'pNOY' else None for c in range(C)] for r in range(R)]
    S=sorted([(min(p[0] for p in n['boundary']),min(p[1] for p in n['boundary'])) for n in f.segmentation['nodes'] if n['color']=='S' and n['pixels']==16])
    L0,R0=S[0],S[1]
    acts=[(-1,0,'UP'),(1,0,'DOWN'),(0,-1,'LEFT'),(0,1,'RIGHT')]
    start=(L0[0],L0[1],R0[0],R0[1],frozenset())
    prev={start:None}; q=collections.deque([start]); found=None
    while q:
        stt=q.popleft()
        lr,lc,rr,rc,op=stt
        Lc=set((x,y) for x in range(lr,lr+4) for y in range(lc,lc+4))
        Rc=set((x,y) for x in range(rr,rr+4) for y in range(rc,rc+4))
        for dr,dc,nm in acts:
            res=[]
            ok=True; win=False
            for pos,occ in ((( lr,lc),Rc),((rr,rc),Lc)):
                r,c=pos; nr,nc=r+dr*4,c+dc*4
                ar0,ar1=min(r,nr),max(r,nr)+4; ac0,ac1=min(c,nc),max(c,nc)+4
                bad=False
                for x in range(ar0,ar1):
                    for y in range(ac0,ac1):
                        if x<0 or x>=R or y<0 or y>=C or wall[x][y]: bad=True
                if bad: res.append((pos,set())); continue
                tgt=set((x,y) for x in range(nr,nr+4) for y in range(nc,nc+4))
                if tgt & occ: win=True
                res.append(((nr,nc),tgt))
            if win:
                found=(stt,nm); break
            (lpos,ltgt),(rpos,rtgt)=res
            if lpos==(lr,lc) and rpos==(rr,rc): continue
            nops=set(op)
            for tg in (ltgt,rtgt):
                for (x,y) in tg:
                    if act[x][y] and act[x][y] not in nops: nops.add(act[x][y])
            ns=(lpos[0],lpos[1],rpos[0],rpos[1],frozenset(nops))
            if ns not in prev:
                prev[ns]=(stt,nm); q.append(ns)
        if found: break
    if not found: return 'none'
    stt,nm=found; path=[nm]; s=stt
    while prev[s] is not None:
        s,pn=prev[s]; path.append(pn)
    return list(reversed(path)), tuple(stt[:4]), tuple(sorted(stt[4]))

def solve5b(f, maxd=40):
    import collections
    a=f.ascii.split('\n'); R=len(a); C=len(a[0])
    wall=[[a[r][c] in 'MPW' for c in range(C)] for r in range(R)]
    act=[[a[r][c] if a[r][c] in 'pNOY' else None for c in range(C)] for r in range(R)]
    S=sorted([(min(p[0] for p in n['boundary']),min(p[1] for p in n['boundary'])) for n in f.segmentation['nodes'] if n['color']=='S' and n['pixels']==16])
    start=(S[0][0],S[0][1],S[1][0],S[1][1],frozenset())
    acts=[(-1,0,'UP'),(1,0,'DOWN'),(0,-1,'LEFT'),(0,1,'RIGHT')]
    def cells(r,c): return set((x,y) for x in range(r,r+4) for y in range(c,c+4))
    prev={start:None}; q=collections.deque([start]); found=None
    while q:
        lr,lc,rr,rc,op=q.popleft()
        Lc=cells(lr,lc); Rc=cells(rr,rc)
        for dr,dc,nm in acts:
            out=[]; win=False
            for (r,c),occ in (((lr,lc),Rc),((rr,rc),Lc)):
                nr,nc=r+dr*4,c+dc*4
                ar0,ar1=min(r,nr),max(r,nr)+4; ac0,ac1=min(c,nc),max(c,nc)+4
                bad=any(x<0 or x>=R or y<0 or y>=C or wall[x][y] for x in range(ar0,ar1) for y in range(ac0,ac1))
                if bad: out.append(((r,c),None)); continue
                if cells(nr,nc) & occ: win=True
                if any((x,y) in occ for x in range(ar0,ar1) for y in range(ac0,ac1)): out.append(((r,c),None)); continue
                out.append(((nr,nc),cells(nr,nc)))
            if win: found=((lr,lc,rr,rc,op),nm); break
            (lp,lt),(rp,rt)=out
            if lp==(lr,lc) and rp==(rr,rc): continue
            nops=set(op)
            for tg in (lt,rt):
                if tg:
                    for (x,y) in tg:
                        if act[x][y]: nops.add(act[x][y])
            ns=(lp[0],lp[1],rp[0],rp[1],frozenset(nops))
            if ns not in prev: prev[ns]=((lr,lc,rr,rc,op),nm); q.append(ns)
        if found: break
    if not found: return 'none'
    stt,nm=found; path=[nm]; s=stt
    while prev[s] is not None:
        s,pn=prev[s]; path.append(pn)
    return list(reversed(path)), stt

def solve5c(f, maxd=40):
    import collections
    a=f.ascii.split('\n'); R=len(a); C=len(a[0])
    wall=[[a[r][c] in 'MPW' for c in range(C)] for r in range(R)]
    act=[[a[r][c] if a[r][c] in 'pNOY' else None for c in range(C)] for r in range(R)]
    S=sorted([(min(p[0] for p in n['boundary']),min(p[1] for p in n['boundary'])) for n in f.segmentation['nodes'] if n['color']=='S' and n['pixels']==16])
    start=(S[0][0],S[0][1],S[1][0],S[1][1],frozenset())
    acts=[(-1,0,'UP'),(1,0,'DOWN'),(0,-1,'LEFT'),(0,1,'RIGHT')]
    def cells(r,c): return set((x,y) for x in range(r,r+4) for y in range(c,c+4))
    prev={start:None}; q=collections.deque([start]); found=None
    while q:
        lr,lc,rr,rc,op=q.popleft()
        Lc=cells(lr,lc); Rc=cells(rr,rc)
        for dr,dc,nm in acts:
            out=[]
            for (r,c),occ in (((lr,lc),Rc),((rr,rc),Lc)):
                nr,nc=r+dr*4,c+dc*4
                ar0,ar1=min(r,nr),max(r,nr)+4; ac0,ac1=min(c,nc),max(c,nc)+4
                bad=any(x<0 or x>=R or y<0 or y>=C or wall[x][y] for x in range(ar0,ar1) for y in range(ac0,ac1))
                if bad: out.append(((r,c),occ)); continue
                if any((x,y) in occ for x in range(ar0,ar1) for y in range(ac0,ac1)): out.append(((r,c),occ)); continue
                out.append(((nr,nc),cells(nr,nc)))
            (lp,lt),(rp,rt)=out
            if lt & rt: found=((lr,lc,rr,rc,op),nm); break
            if lp==(lr,lc) and rp==(rr,rc): continue
            nops=set(op)
            for tg in (lt,rt):
                for (x,y) in tg:
                    if act[x][y]: nops.add(act[x][y])
            ns=(lp[0],lp[1],rp[0],rp[1],frozenset(nops))
            if ns not in prev: prev[ns]=((lr,lc,rr,rc,op),nm); q.append(ns)
        if found: break
    if not found: return 'none'
    stt,nm=found; path=[nm]; s=stt
    while prev[s] is not None:
        s,pn=prev[s]; path.append(pn)
    return list(reversed(path)), stt

def solve5d(f, maxd=40):
    import collections
    a=f.ascii.split('\n'); R=len(a); C=len(a[0])
    wall=[[a[r][c] in 'MPW' for c in range(C)] for r in range(R)]
    sq=[[None]*C for _ in range(R)]   # square activators (passable)
    band=[[None]*C for _ in range(R)] # closed gates (walls until opened)
    for n in f.segmentation['nodes']:
        if n['color'] in 'pNOY' and n['pixels']>=16:
            tgt = sq if n['pixels']==16 else band
            b=n['boundary']; r0,r1=min(p[0] for p in b),max(p[0] for p in b); c0,c1=min(p[1] for p in b),max(p[1] for p in b)
            for x in range(r0,r1+1):
                for y in range(c0,c1+1): tgt[x][y]=n['color']
    S=sorted([(min(p[0] for p in n['boundary']),min(p[1] for p in n['boundary'])) for n in f.segmentation['nodes'] if n['color']=='S' and n['pixels']==16])
    start=(S[0][0],S[0][1],S[1][0],S[1][1],frozenset())
    acts=[(-1,0,'UP'),(1,0,'DOWN'),(0,-1,'LEFT'),(0,1,'RIGHT')]
    def cells(r,c): return set((x,y) for x in range(r,r+4) for y in range(c,c+4))
    prev={start:None}; q=collections.deque([start]); found=None; cnt=0
    while q:
        lr,lc,rr,rc,op=q.popleft(); cnt+=1
        Lc=cells(lr,lc); Rc=cells(rr,rc)
        for dr,dc,nm in acts:
            out=[]
            for (r,c),occ in (((lr,lc),Rc),((rr,rc),Lc)):
                nr,nc=r+dr*4,c+dc*4
                ar0,ar1=min(r,nr),max(r,nr)+4; ac0,ac1=min(c,nc),max(c,nc)+4
                bad=False
                for x in range(ar0,ar1):
                    for y in range(ac0,ac1):
                        if x<0 or x>=R or y<0 or y>=C or wall[x][y] or (band[x][y] and band[x][y] not in op): bad=True;break
                    if bad: break
                if bad: out.append(((r,c),occ)); continue
                if any((x,y) in occ for x in range(ar0,ar1) for y in range(ac0,ac1)): out.append(((r,c),occ)); continue
                out.append(((nr,nc),cells(nr,nc)))
            (lp,lt),(rp,rt)=out
            if lt & rt: found=((lr,lc,rr,rc,op),nm); break
            if lp==(lr,lc) and rp==(rr,rc): continue
            nops=set(op)
            for tg in (lt,rt):
                for (x,y) in tg:
                    if sq[x][y]: nops.add(sq[x][y])
            ns=(lp[0],lp[1],rp[0],rp[1],frozenset(nops))
            if ns not in prev: prev[ns]=((lr,lc,rr,rc,op),nm); q.append(ns)
        if found: break
    if not found: return ('none',cnt)
    stt,nm=found; path=[nm]; s=stt
    while prev[s] is not None:
        s,pn=prev[s]; path.append(pn)
    return list(reversed(path)), stt, cnt

def reach5(f, idx, opened=('N','O','p')):
    a=f.ascii.split('\n'); R=len(a); C=len(a[0])
    wall=[[a[r][c] in 'MPW' for c in range(C)] for r in range(R)]
    band=[[None]*C for _ in range(R)]
    for n in f.segmentation['nodes']:
        if n['color'] in 'pNOY' and n['pixels']>16:
            b=n['boundary']
            for x in range(min(p[0] for p in b),max(p[0] for p in b)+1):
                for y in range(min(p[1] for p in b),max(p[1] for p in b)+1): band[x][y]=n['color']
    S=sorted([(min(p[0] for p in n['boundary']),min(p[1] for p in n['boundary'])) for n in f.segmentation['nodes'] if n['color']=='S' and n['pixels']==16])
    st=S[idx]; seen={st}; q=[st]
    while q:
        r,c=q.pop()
        for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nr,nc=r+dr*4,c+dc*4
            ar0,ar1=min(r,nr),max(r,nr)+4; ac0,ac1=min(c,nc),max(c,nc)+4
            ok=True
            for x in range(ar0,ar1):
                for y in range(ac0,ac1):
                    if x<0 or x>=R or y<0 or y>=C or wall[x][y] or (band[x][y] and band[x][y] not in opened): ok=False
            if ok and (nr,nc) not in seen: seen.add((nr,nc)); q.append((nr,nc))
    return S,seen

def solve5e(f, maxd=40, pass_through=True):
    import collections
    a=f.ascii.split('\n'); R=len(a); C=len(a[0])
    wall=[[a[r][c] in 'MPW' for c in range(C)] for r in range(R)]
    sq=[[None]*C for _ in range(R)]; band=[[None]*C for _ in range(R)]
    for n in f.segmentation['nodes']:
        if n['color'] in 'pNOY' and n['pixels']>=16:
            tgt = sq if n['pixels']==16 else band
            b=n['boundary']
            for x in range(min(p[0] for p in b),max(p[0] for p in b)+1):
                for y in range(min(p[1] for p in b),max(p[1] for p in b)+1): tgt[x][y]=n['color']
    S=sorted([(min(p[0] for p in n['boundary']),min(p[1] for p in n['boundary'])) for n in f.segmentation['nodes'] if n['color']=='S' and n['pixels']==16])
    start=(S[0][0],S[0][1],S[1][0],S[1][1],frozenset())
    acts=[(-1,0,'UP'),(1,0,'DOWN'),(0,-1,'LEFT'),(0,1,'RIGHT')]
    def cells(r,c): return set((x,y) for x in range(r,r+4) for y in range(c,c+4))
    prev={start:None}; q=collections.deque([start]); found=None; cnt=0; best=(999,None)
    while q:
        lr,lc,rr,rc,op=q.popleft(); cnt+=1
        d=abs(lr-rr)+abs(lc-rc)
        if d<best[0]: best=(d,(lr,lc,rr,rc))
        for dr,dc,nm in acts:
            out=[]
            for (r,c),occ in (((lr,lc),Rc if False else cells(rr,rc)),((rr,rc),cells(lr,lc))):
                nr,nc=r+dr*4,c+dc*4
                ar0,ar1=min(r,nr),max(r,nr)+4; ac0,ac1=min(c,nc),max(c,nc)+4
                bad=any(x<0 or x>=R or y<0 or y>=C or wall[x][y] or (band[x][y] and band[x][y] not in op) for x in range(ar0,ar1) for y in range(ac0,ac1))
                if bad: out.append(((r,c),occ)); continue
                if (not pass_through) and any((x,y) in occ for x in range(ar0,ar1) for y in range(ac0,ac1)): out.append(((r,c),occ)); continue
                out.append(((nr,nc),cells(nr,nc)))
            (lp,lt),(rp,rt)=out
            if lt & rt: found=((lr,lc,rr,rc,op),nm); break
            if lp==(lr,lc) and rp==(rr,rc): continue
            nops=set(op)
            for tg in (lt,rt):
                for (x,y) in tg:
                    if sq[x][y]: nops.add(sq[x][y])
            ns=(lp[0],lp[1],rp[0],rp[1],frozenset(nops))
            if ns not in prev: prev[ns]=((lr,lc,rr,rc,op),nm); q.append(ns)
        if found: break
    if not found: return ('none',cnt,best)
    stt,nm=found; path=[nm]; s=stt
    while prev[s] is not None:
        s,pn=prev[s]; path.append(pn)
    return list(reversed(path)), stt

def sim5(f, plan):
    a=f.ascii.split('\n'); R=len(a); C=len(a[0])
    wall=[[a[r][c] in 'MPW' for c in range(C)] for r in range(R)]
    sq=[[None]*C for _ in range(R)]; band=[[None]*C for _ in range(R)]
    for n in f.segmentation['nodes']:
        if n['color'] in 'pNOY' and n['pixels']>=16:
            tgt=sq if n['pixels']==16 else band
            b=n['boundary']
            for x in range(min(p[0] for p in b),max(p[0] for p in b)+1):
                for y in range(min(p[1] for p in b),max(p[1] for p in b)+1): tgt[x][y]=n['color']
    S=sorted([(min(p[0] for p in n['boundary']),min(p[1] for p in n['boundary'])) for n in f.segmentation['nodes'] if n['color']=='S' and n['pixels']==16])
    L,Rp=S[0],S[1]; op=set(); out=[]
    D={'UP':(-1,0),'DOWN':(1,0),'LEFT':(0,-1),'RIGHT':(0,1)}
    def cells(r,c): return set((x,y) for x in range(r,r+4) for y in range(c,c+4))
    for nm in plan:
        dr,dc=D[nm]; res=[]
        for (r,c),occ in (((*L,),cells(*Rp)),((*Rp,),cells(*L))):
            nr,nc=r+dr*4,c+dc*4
            ar0,ar1=min(r,nr),max(r,nr)+4; ac0,ac1=min(c,nc),max(c,nc)+4
            bad=any(x<0 or x>=R or y<0 or y>=C or wall[x][y] or (band[x][y] and band[x][y] not in op) for x in range(ar0,ar1) for y in range(ac0,ac1))
            thru=any((x,y) in occ for x in range(ar0,ar1) for y in range(ac0,ac1))
            if bad: res.append(((r,c),occ,False))
            else: res.append(((nr,nc),cells(nr,nc),thru))
        (lp,lt,t1),(rp,rt,t2)=res
        for tg in (lt,rt):
            for (x,y) in tg:
                if sq[x][y]: op.add(sq[x][y])
        L,Rp=lp,rp
        out.append((nm,L,Rp,sorted(op),'THRU' if (t1 or t2) else '', 'WIN' if lt&rt else ''))
        if lt&rt: break
    return out

def solve5f(f, maxd=40):
    import collections
    a=f.ascii.split('\n'); R=len(a); C=len(a[0])
    wall=[[a[r][c] in 'MPW' for c in range(C)] for r in range(R)]
    sq=[[None]*C for _ in range(R)]; band=[[None]*C for _ in range(R)]
    for n in f.segmentation['nodes']:
        if n['color'] in 'pNOY' and n['pixels']>=16:
            tgt=sq if n['pixels']==16 else band
            b=n['boundary']
            for x in range(min(p[0] for p in b),max(p[0] for p in b)+1):
                for y in range(min(p[1] for p in b),max(p[1] for p in b)+1): tgt[x][y]=n['color']
    S=sorted([(min(p[0] for p in n['boundary']),min(p[1] for p in n['boundary'])) for n in f.segmentation['nodes'] if n['color']=='S' and n['pixels']==16])
    start=(S[0][0],S[0][1],S[1][0],S[1][1],frozenset())
    acts=[(-1,0,'UP'),(1,0,'DOWN'),(0,-1,'LEFT'),(0,1,'RIGHT')]
    def cells(r,c): return set((x,y) for x in range(r,r+4) for y in range(c,c+4))
    def swept(r,c,nr,nc): return (min(r,nr),max(r,nr)+4,min(c,nc),max(c,nc)+4)
    prev={start:None}; q=collections.deque([start]); found=None; cnt=0
    while q:
        lr,lc,rr,rc,op=q.popleft(); cnt+=1
        for dr,dc,nm in acts:
            cand=[]
            for (r,c) in ((lr,lc),(rr,rc)):
                nr,nc=r+dr*4,c+dc*4
                ar0,ar1,ac0,ac1=swept(r,c,nr,nc)
                bad=any(x<0 or x>=R or y<0 or y>=C or wall[x][y] or (band[x][y] and band[x][y] not in op) for x in range(ar0,ar1) for y in range(ac0,ac1))
                cand.append(None if bad else (nr,nc))
            cands=[x if x else y for x,y in zip(cand,[(lr,lc),(rr,rc)])]
            (al,ar_),(bl,br_)=cand
            sl=swept(lr,lc,al[0],al[1]); sr=swept(rr,rc,bl[0],bl[1])
            def inb(box,cs): return any(box[0]<=x<box[1] and box[2]<=y<box[3] for (x,y) in cs)
            CL=cells(al[0],al[1]); CR=cells(bl[0],bl[1])
            if CL & CR: found=((lr,lc,rr,rc,op),nm); break
            if inb(sl,CR) or inb(sr,CL): continue
            if (al,ar_)==(lr,lc) and (bl,br_)==(rr,rc): continue
            nops=set(op)
            for tg in (CL,CR):
                for (x,y) in tg:
                    if sq[x][y]: nops.add(sq[x][y])
            ns=(al[0],al[1],bl[0],bl[1],frozenset(nops))
            if ns not in prev: prev[ns]=((lr,lc,rr,rc,op),nm); q.append(ns)
        if found: break
    if not found: return ('none',cnt)
    stt,nm=found; path=[nm]; s=stt
    while prev[s] is not None:
        s,pn=prev[s]; path.append(pn)
    return list(reversed(path)), stt, cnt

def solve5g(f, maxd=40):
    import collections
    a=f.ascii.split('\n'); R=len(a); C=len(a[0])
    wall=[[a[r][c] in 'MPW' for c in range(C)] for r in range(R)]
    sq=[[None]*C for _ in range(R)]; band=[[None]*C for _ in range(R)]
    for n in f.segmentation['nodes']:
        if n['color'] in 'pNOY' and n['pixels']>=16:
            tgt=sq if n['pixels']==16 else band
            b=n['boundary']
            for x in range(min(p[0] for p in b),max(p[0] for p in b)+1):
                for y in range(min(p[1] for p in b),max(p[1] for p in b)+1): tgt[x][y]=n['color']
    S=sorted([(min(p[0] for p in n['boundary']),min(p[1] for p in n['boundary'])) for n in f.segmentation['nodes'] if n['color']=='S' and n['pixels']==16])
    start=(S[0][0],S[0][1],S[1][0],S[1][1],frozenset())
    acts=[(-1,0,'UP'),(1,0,'DOWN'),(0,-1,'LEFT'),(0,1,'RIGHT')]
    def cells(r,c): return set((x,y) for x in range(r,r+4) for y in range(c,c+4))
    def swp(r,c,nr,nc): return (min(r,nr),max(r,nr)+4,min(c,nc),max(c,nc)+4)
    prev={start:None}; q=collections.deque([start]); found=None; cnt=0
    while q:
        lr,lc,rr,rc,op=q.popleft(); cnt+=1
        for dr,dc,nm in acts:
            raw=[]
            for (r,c) in ((lr,lc),(rr,rc)):
                nr,nc=r+dr*4,c+dc*4
                ar0,ar1,ac0,ac1=swp(r,c,nr,nc)
                bad=any(x<0 or x>=R or y<0 or y>=C or wall[x][y] or (band[x][y] and band[x][y] not in op) for x in range(ar0,ar1) for y in range(ac0,ac1))
                raw.append((nr,nc) if not bad else None)
            fin=[x if x is not None else y for x,y in zip(raw,[(lr,lc),(rr,rc)])]
            al,bl=fin
            sl=swp(lr,lc,al[0],al[1]); sr=swp(rr,rc,bl[0],bl[1])
            CL=cells(al[0],al[1]); CR=cells(bl[0],bl[1])
            if CL & CR: found=((lr,lc,rr,rc,op),nm); break
            inb=lambda box,cs: any(box[0]<=x<box[1] and box[2]<=y<box[3] for (x,y) in cs)
            if inb(sl,CR) or inb(sr,CL): continue
            if al==(lr,lc) and bl==(rr,rc): continue
            nops=set(op)
            for tg in (CL,CR):
                for (x,y) in tg:
                    if sq[x][y]: nops.add(sq[x][y])
            ns=(al[0],al[1],bl[0],bl[1],frozenset(nops))
            if ns not in prev: prev[ns]=((lr,lc,rr,rc,op),nm); q.append(ns)
        if found: break
    if not found: return ('none',cnt)
    stt,nm=found; path=[nm]; s=stt
    while prev[s] is not None:
        s,pn=prev[s]; path.append(pn)
    return list(reversed(path)), stt, cnt

def solve5h(f, maxd=40, both_move=True):
    import collections
    a=f.ascii.split('\n'); R=len(a); C=len(a[0])
    wall=[[a[r][c] in 'MPW' for c in range(C)] for r in range(R)]
    sq=[[None]*C for _ in range(R)]; band=[[None]*C for _ in range(R)]
    for n in f.segmentation['nodes']:
        if n['color'] in 'pNOY' and n['pixels']>=16:
            tgt=sq if n['pixels']==16 else band
            b=n['boundary']
            for x in range(min(p[0] for p in b),max(p[0] for p in b)+1):
                for y in range(min(p[1] for p in b),max(p[1] for p in b)+1): tgt[x][y]=n['color']
    S=sorted([(min(p[0] for p in n['boundary']),min(p[1] for p in n['boundary'])) for n in f.segmentation['nodes'] if n['color']=='S' and n['pixels']==16])
    start=(S[0][0],S[0][1],S[1][0],S[1][1],frozenset())
    acts=[(-1,0,'UP'),(1,0,'DOWN'),(0,-1,'LEFT'),(0,1,'RIGHT')]
    def cells(r,c): return set((x,y) for x in range(r,r+4) for y in range(c,c+4))
    def swp(r,c,nr,nc): return (min(r,nr),max(r,nr)+4,min(c,nc),max(c,nc)+4)
    prev={start:None}; q=collections.deque([start]); found=None; cnt=0
    while q:
        lr,lc,rr,rc,op=q.popleft(); cnt+=1
        for dr,dc,nm in acts:
            raw=[]
            for (r,c) in ((lr,lc),(rr,rc)):
                nr,nc=r+dr*4,c+dc*4
                ar0,ar1,ac0,ac1=swp(r,c,nr,nc)
                bad=any(x<0 or x>=R or y<0 or y>=C or wall[x][y] or (band[x][y] and band[x][y] not in op) for x in range(ar0,ar1) for y in range(ac0,ac1))
                raw.append((nr,nc) if not bad else None)
            fin=[x if x is not None else y for x,y in zip(raw,[(lr,lc),(rr,rc)])]
            al,bl=fin; CL=cells(al[0],al[1]); CR=cells(bl[0],bl[1])
            if CL & CR and (raw[0] is not None and raw[1] is not None or not both_move):
                found=((lr,lc,rr,rc,op),nm); break
            if raw[0] is None and raw[1] is None: continue
            sl=swp(lr,lc,al[0],al[1]); sr=swp(rr,rc,bl[0],bl[1])
            inb=lambda box,cs: any(box[0]<=x<box[1] and box[2]<=y<box[3] for (x,y) in cs)
            if inb(sl,CR) or inb(sr,CL): continue
            nops=set(op)
            for tg in (CL,CR):
                for (x,y) in tg:
                    if sq[x][y]: nops.add(sq[x][y])
            ns=(al[0],al[1],bl[0],bl[1],frozenset(nops))
            if ns not in prev: prev[ns]=((lr,lc,rr,rc,op),nm); q.append(ns)
        if found: break
    if not found: return ('none',cnt)
    stt,nm=found; path=[nm]; s=stt
    while prev[s] is not None:
        s,pn=prev[s]; path.append(pn)
    return list(reversed(path)), stt, cnt

def solve6(f, maxd=60):
    import collections
    a=f.ascii.split('\n'); R=len(a); C=len(a[0])
    wall=[[a[r][c] in 'MPW' for c in range(C)] for r in range(R)]
    sq=[[None]*C for _ in range(R)]; band=[[None]*C for _ in range(R)]
    sqs={}
    for n in f.segmentation['nodes']:
        if n['color'] in 'pNOY' and n['pixels']>=16:
            b=n['boundary']; r0,r1=min(p[0] for p in b),max(p[0] for p in b); c0,c1=min(p[1] for p in b),max(p[1] for p in b)
            tgt=sq if n['pixels']==16 else band
            for x in range(r0,r1+1):
                for y in range(c0,c1+1): tgt[x][y]=n['color']
            if n['pixels']==16: sqs[n['color']]=sqs.get(n['color'],[])+[(r0,c0)]
    S=sorted([(min(p[0] for p in n['boundary']),min(p[1] for p in n['boundary'])) for n in f.segmentation['nodes'] if n['color']=='S' and n['pixels']==16])
    start=(S[0],S[1])
    acts=[(-1,0,'UP'),(1,0,'DOWN'),(0,-1,'LEFT'),(0,1,'RIGHT')]
    def cells(p): return set((x,y) for x in range(p[0],p[0]+4) for y in range(p[1],p[1]+4))
    def opens(L,Rp):
        s=set()
        for p in (L,Rp):
            c=sq[p[0]][p[1]]
            if c and all(sq[x][y]==c for x in range(p[0],p[0]+4) for y in range(p[1],p[1]+4)): s.add(c)
        return s
    def ok(r,c,nr,nc,op):
        ar0,ar1=min(r,nr),max(r,nr)+4; ac0,ac1=min(c,nc),max(c,nc)+4
        for x in range(ar0,ar1):
            for y in range(ac0,ac1):
                if x<0 or x>=R or y<0 or y>=C or wall[x][y]: return False
                if band[x][y] and band[x][y] not in op: return False
        return True
    def swp(r,c,nr,nc): return (min(r,nr),max(r,nr)+4,min(c,nc),max(c,nc)+4)
    prev={start:None}; q=collections.deque([start]); found=None; cnt=0
    while q:
        L,Rp=q.popleft(); cnt+=1
        op=opens(L,Rp)
        for dr,dc,nm in acts:
            raw=[]
            for (r,c) in (L,Rp):
                nr,nc=r+dr*4,c+dc*4
                raw.append((nr,nc) if ok(r,c,nr,nc,op) else None)
            fin=[x if x is not None else y for x,y in zip(raw,[L,Rp])]
            nL,nR=fin
            if cells(nL) & cells(nR): found=(L,Rp,nm); break
            nop=opens(nL,nR)
            bad=False
            for p in (nL,nR):
                for x in range(p[0],p[0]+4):
                    for y in range(p[1],p[1]+4):
                        if band[x][y] and band[x][y] not in nop: bad=True
            if bad: continue
            sl=swp(L[0],L[1],nL[0],nL[1]); sr=swp(Rp[0],Rp[1],nR[0],nR[1])
            CL=cells(nL); CR=cells(nR)
            inb=lambda box,cs: any(box[0]<=x<box[1] and box[2]<=y<box[3] for (x,y) in cs)
            if inb(sl,CR) or inb(sr,CL): continue
            if nL==L and nR==Rp: continue
            ns=(nL,nR)
            if ns not in prev: prev[ns]=(L,Rp,nm); q.append(ns)
        if found: break
    if not found: return ('none',cnt)
    Lb,Rb,nm=found; path=[nm]; s=(Lb,Rb)
    while prev[s] is not None:
        s,pn=prev[s]; path.append(pn)
    return list(reversed(path))

def solve6b(f, maxd=60):
    import collections
    a=f.ascii.split('\n'); R=len(a); C=len(a[0])
    wall=[[a[r][c] in 'MPW' for c in range(C)] for r in range(R)]
    sq=[[None]*C for _ in range(R)]; band=[[None]*C for _ in range(R)]
    for n in f.segmentation['nodes']:
        if n['color'] in 'pNOY' and n['pixels']>=16:
            b=n['boundary']; r0,r1=min(p[0] for p in b),max(p[0] for p in b); c0,c1=min(p[1] for p in b),max(p[1] for p in b)
            tgt=sq if n['pixels']==16 else band
            for x in range(r0,r1+1):
                for y in range(c0,c1+1): tgt[x][y]=n['color']
    SS=sorted([(min(p[0] for p in n['boundary']),min(p[1] for p in n['boundary'])) for n in f.segmentation['nodes'] if n['color']=='S' and n['pixels']==16])
    start=(SS[0],SS[1])
    acts=[(-1,0,'UP'),(1,0,'DOWN'),(0,-1,'LEFT'),(0,1,'RIGHT')]
    def cells(p0): return set((x,y) for x in range(p0[0],p0[0]+4) for y in range(p0[1],p0[1]+4))
    def opens(Lp,Rp):
        s=set()
        for p0 in (Lp,Rp):
            c=sq[p0[0]][p0[1]]
            if c and all(sq[x][y]==c for x in range(p0[0],p0[0]+4) for y in range(p0[1],p0[1]+4)): s.add(c)
        return s
    def ok(r,c,nr,nc,op):
        for x in range(min(r,nr),max(r,nr)+4):
            for y in range(min(c,nc),max(c,nc)+4):
                if x<0 or x>=R or y<0 or y>=C or wall[x][y]: return False
                if band[x][y] and band[x][y] not in op: return False
        return True
    def swp(r,c,nr,nc): return (min(r,nr),max(r,nr)+4,min(c,nc),max(c,nc)+4)
    prev={start:None}; q=collections.deque([start]); found=None; cnt=0
    while q:
        Lp,Rp=q.popleft(); cnt+=1
        op=opens(Lp,Rp)
        for dr,dc,nm in acts:
            raw=[]
            for (r,c) in (Lp,Rp):
                nr,nc=r+dr*4,c+dc*4
                raw.append((nr,nc) if ok(r,c,nr,nc,op) else None)
            fin=[x if x is not None else y for x,y in zip(raw,[Lp,Rp])]
            nL,nR=fin
            if cells(nL) & cells(nR): found=(Lp,Rp,nm); break
            nop=opens(nL,nR)
            bad=False
            for p0 in (nL,nR):
                for x in range(p0[0],p0[0]+4):
                    for y in range(p0[1],p0[1]+4):
                        if band[x][y] and band[x][y] not in nop: bad=True
            if bad: continue
            sl=swp(Lp[0],Lp[1],nL[0],nL[1]); sr=swp(Rp[0],Rp[1],nR[0],nR[1])
            CL=cells(nL); CR=cells(nR)
            inb=lambda box,cs: any(box[0]<=x<box[1] and box[2]<=y<box[3] for (x,y) in cs)
            if inb(sl,CR) or inb(sr,CL): continue
            if nL==Lp and nR==Rp: continue
            ns=(nL,nR)
            if ns not in prev: prev[ns]=(Lp,Rp,nm); q.append(ns)
        if found: break
    if not found: return ('none',cnt)
    s0,s1,nm=found; path=[nm]; s=(s0,s1)
    while prev[s] is not None:
        s,pn=prev[s]; path.append(pn)
    return list(reversed(path))

def solve6c(f, maxd=60):
    import collections
    a=f.ascii.split('\n'); R=len(a); C=len(a[0])
    wall=[[a[r][c] in 'MPW' for c in range(C)] for r in range(R)]
    sq=[[None]*C for _ in range(R)]; band=[[None]*C for _ in range(R)]
    for n in f.segmentation['nodes']:
        if n['color'] in 'pNOY' and n['pixels']>=16:
            b=n['boundary']; r0,r1=min(p[0] for p in b),max(p[0] for p in b); c0,c1=min(p[1] for p in b),max(p[1] for p in b)
            tgt=sq if n['pixels']==16 else band
            for x in range(r0,r1+1):
                for y in range(c0,c1+1): tgt[x][y]=n['color']
    SS=sorted([(min(p[0] for p in n['boundary']),min(p[1] for p in n['boundary'])) for n in f.segmentation['nodes'] if n['color']=='S' and n['pixels']==16])
    start=(SS[0],SS[1])
    acts=[(-1,0,'UP'),(1,0,'DOWN'),(0,-1,'LEFT'),(0,1,'RIGHT')]
    def cells(p0): return set((x,y) for x in range(p0[0],p0[0]+4) for y in range(p0[1],p0[1]+4))
    def opens(Lp,Rp):
        s=set()
        for p0 in (Lp,Rp):
            c=sq[p0[0]][p0[1]]
            if c and all(sq[x][y]==c for x in range(p0[0],p0[0]+4) for y in range(p0[1],p0[1]+4)): s.add(c)
        return s
    def ok(r,c,nr,nc,op):
        for x in range(min(r,nr),max(r,nr)+4):
            for y in range(min(c,nc),max(c,nc)+4):
                if x<0 or x>=R or y<0 or y>=C or wall[x][y]: return False
                if band[x][y] and band[x][y] not in op: return False
        return True
    def swp(r,c,nr,nc): return (min(r,nr),max(r,nr)+4,min(c,nc),max(c,nc)+4)
    prev={start:None}; q=collections.deque([start]); found=None; cnt=0
    while q:
        Lp,Rp=q.popleft(); cnt+=1
        op=opens(Lp,Rp)
        for dr,dc,nm in acts:
            raw=[]
            for (r,c) in (Lp,Rp):
                nr,nc=r+dr*4,c+dc*4
                raw.append((nr,nc) if ok(r,c,nr,nc,op) else None)
            fin=[x if x is not None else y for x,y in zip(raw,[Lp,Rp])]
            nL,nR=fin
            if cells(nL) & cells(nR): found=(Lp,Rp,nm); break
            nop=opens(nL,nR)
            bad=False
            for p0 in (nL,nR):
                for x in range(p0[0],p0[0]+4):
                    for y in range(p0[1],p0[1]+4):
                        if band[x][y] and band[x][y] not in nop: bad=True
            if bad: continue
            sl=swp(Lp[0],Lp[1],nL[0],nL[1]); sr=swp(Rp[0],Rp[1],nR[0],nR[1])
            CL=cells(nL); CR=cells(nR)
            inb=lambda box,cs: any(box[0]<=x<box[1] and box[2]<=y<box[3] for (x,y) in cs)
            if inb(sl,CR) or inb(sr,CL): continue
            if nL==Lp and nR==Rp: continue
            ns=(nL,nR)
            if ns not in prev: prev[ns]=((Lp,Rp),nm); q.append(ns)
        if found: break
    if not found: return ('none',cnt)
    s0,s1,nm=found; path=[nm]; s=(s0,s1)
    while prev[s] is not None:
        s,pn=prev[s]; path.append(pn)
    return list(reversed(path))

def sim6(f,plan):
    a=f.ascii.split('\n'); R=len(a); C=len(a[0])
    wall=[[a[r][c] in 'MPW' for c in range(C)] for r in range(R)]
    sq=[[None]*C for _ in range(R)]; band=[[None]*C for _ in range(R)]
    for n in f.segmentation['nodes']:
        if n['color'] in 'pNOY' and n['pixels']>=16:
            b=n['boundary']; r0,r1=min(p[0] for p in b),max(p[0] for p in b); c0,c1=min(p[1] for p in b),max(p[1] for p in b)
            tgt=sq if n['pixels']==16 else band
            for x in range(r0,r1+1):
                for y in range(c0,c1+1): tgt[x][y]=n['color']
    SS=sorted([(min(p[0] for p in n['boundary']),min(p[1] for p in n['boundary'])) for n in f.segmentation['nodes'] if n['color']=='S' and n['pixels']==16])
    Lp,Rp=SS
    D={'UP':(-1,0),'DOWN':(1,0),'LEFT':(0,-1),'RIGHT':(0,1)}
    def cells(p0): return set((x,y) for x in range(p0[0],p0[0]+4) for y in range(p0[1],p0[1]+4))
    def opens(Lp,Rp):
        s=set()
        for p0 in (Lp,Rp):
            c=sq[p0[0]][p0[1]]
            if c and all(sq[x][y]==c for x in range(p0[0],p0[0]+4) for y in range(p0[1],p0[1]+4)): s.add(c)
        return s
    def ok(r,c,nr,nc,op):
        for x in range(min(r,nr),max(r,nr)+4):
            for y in range(min(c,nc),max(c,nc)+4):
                if x<0 or x>=R or y<0 or y>=C or wall[x][y]: return False
                if band[x][y] and band[x][y] not in op: return False
        return True
    out=[]
    for i,nm in enumerate(plan):
        dr,dc=D[nm]; op=opens(Lp,Rp); raw=[]
        for (r,c) in (Lp,Rp):
            nr,nc=r+dr*4,c+dc*4
            raw.append((nr,nc) if ok(r,c,nr,nc,op) else None)
        fin=[x if x is not None else y for x,y in zip(raw,[Lp,Rp])]
        Lp,Rp=fin
        win=bool(cells(Lp)&cells(Rp))
        out.append((i,nm,Lp,Rp,sorted(opens(Lp,Rp)),'WIN' if win else ''))
        if win: break
    return out

def solve7(f):
    import collections
    a=f.ascii.split('\n'); R=len(a); C=len(a[0])
    wall=[[a[r][c] in 'MPW' for c in range(C)] for r in range(R)]
    sq=[[None]*C for _ in range(R)]; band=[[None]*C for _ in range(R)]
    for n in f.segmentation['nodes']:
        if n['color'] in 'pNOY' and n['pixels']>=16:
            b=n['boundary']; r0,r1=min(p[0] for p in b),max(p[0] for p in b); c0,c1=min(p[1] for p in b),max(p[1] for p in b)
            tgt=sq if n['pixels']==16 else band
            for x in range(r0,r1+1):
                for y in range(c0,c1+1): tgt[x][y]=n['color']
    SS=sorted([(min(p[0] for p in n['boundary']),min(p[1] for p in n['boundary'])) for n in f.segmentation['nodes'] if n['color']=='S' and n['pixels']==16])
    start=(SS[0],SS[1])
    # (name, (drL,dcL),(drR,dcR))
    acts=[('UP',(-1,0),(-1,0)),('DOWN',(1,0),(1,0)),('LEFT',(0,1),(0,-1)),('RIGHT',(0,-1),(0,1))]
    def cells(p0): return set((x,y) for x in range(p0[0],p0[0]+4) for y in range(p0[1],p0[1]+4))
    def opens(Lp,Rp):
        s=set()
        for p0 in (Lp,Rp):
            c=sq[p0[0]][p0[1]]
            if c and all(sq[x][y]==c for x in range(p0[0],p0[0]+4) for y in range(p0[1],p0[1]+4)): s.add(c)
        return s
    def ok(r,c,nr,nc,op):
        for x in range(min(r,nr),max(r,nr)+4):
            for y in range(min(c,nc),max(c,nc)+4):
                if x<0 or x>=R or y<0 or y>=C or wall[x][y]: return False
                if band[x][y] and band[x][y] not in op: return False
        return True
    def swp(r,c,nr,nc): return (min(r,nr),max(r,nr)+4,min(c,nc),max(c,nc)+4)
    prev={start:None}; q=collections.deque([start]); found=None; cnt=0
    while q:
        Lp,Rp=q.popleft(); cnt+=1
        op=opens(Lp,Rp)
        for nm,(drL,dcL),(drR,dcR) in acts:
            raw=[]
            for (r,c),(dr,dc) in ((Lp,(drL,dcL)),(Rp,(drR,dcR))):
                nr,nc=r+dr*4,c+dc*4
                raw.append((nr,nc) if ok(r,c,nr,nc,op) else None)
            fin=[x if x is not None else y for x,y in zip(raw,[Lp,Rp])]
            nL,nR=fin
            if cells(nL)&cells(nR): found=(Lp,Rp,nm); break
            nop=opens(nL,nR); bad=False
            for p0 in (nL,nR):
                for x in range(p0[0],p0[0]+4):
                    for y in range(p0[1],p0[1]+4):
                        if band[x][y] and band[x][y] not in nop: bad=True
            if bad: continue
            sl=swp(Lp[0],Lp[1],nL[0],nL[1]); sr=swp(Rp[0],Rp[1],nR[0],nR[1])
            CL=cells(nL); CR=cells(nR)
            inb=lambda box,cs: any(box[0]<=x<box[1] and box[2]<=y<box[3] for (x,y) in cs)
            if inb(sl,CR) or inb(sr,CL): continue
            if nL==Lp and nR==Rp: continue
            ns=(nL,nR)
            if ns not in prev: prev[ns]=((Lp,Rp),nm); q.append(ns)
        if found: break
    if not found: return ('none',cnt)
    s0,s1,nm=found; path=[nm]; s=(s0,s1)
    while prev[s] is not None:
        s,pn=prev[s]; path.append(pn)
    return list(reversed(path))

def sim7(f,plan):
    a=f.ascii.split('\n'); R=len(a); C=len(a[0])
    wall=[[a[r][c] in 'MPW' for c in range(C)] for r in range(R)]
    sq=[[None]*C for _ in range(R)]; band=[[None]*C for _ in range(R)]
    for n in f.segmentation['nodes']:
        if n['color'] in 'pNOY' and n['pixels']>=16:
            b=n['boundary']; r0,r1=min(p[0] for p in b),max(p[0] for p in b); c0,c1=min(p[1] for p in b),max(p[1] for p in b)
            tgt=sq if n['pixels']==16 else band
            for x in range(r0,r1+1):
                for y in range(c0,c1+1): tgt[x][y]=n['color']
    SS=sorted([(min(p[0] for p in n['boundary']),min(p[1] for p in n['boundary'])) for n in f.segmentation['nodes'] if n['color']=='S' and n['pixels']==16])
    Lp,Rp=SS
    A={'UP':((-1,0),(-1,0)),'DOWN':((1,0),(1,0)),'LEFT':((0,1),(0,-1)),'RIGHT':((0,-1),(0,1))}
    def cells(p0): return set((x,y) for x in range(p0[0],p0[0]+4) for y in range(p0[1],p0[1]+4))
    def opens(Lp,Rp):
        s=set()
        for p0 in (Lp,Rp):
            c=sq[p0[0]][p0[1]]
            if c and all(sq[x][y]==c for x in range(p0[0],p0[0]+4) for y in range(p0[1],p0[1]+4)): s.add(c)
        return s
    def ok(r,c,nr,nc,op):
        for x in range(min(r,nr),max(r,nr)+4):
            for y in range(min(c,nc),max(c,nc)+4):
                if x<0 or x>=R or y<0 or y>=C or wall[x][y]: return False
                if band[x][y] and band[x][y] not in op: return False
        return True
    out=[]
    for i,nm in enumerate(plan):
        (drL,dcL),(drR,dcR)=A[nm]; op=opens(Lp,Rp)
        raw=[]
        for (r,c),(dr,dc) in ((Lp,(drL,dcL)),(Rp,(drR,dcR))):
            nr,nc=r+dr*4,c+dc*4
            raw.append((nr,nc) if ok(r,c,nr,nc,op) else None)
        fin=[x if x is not None else y for x,y in zip(raw,[Lp,Rp])]
        Lp,Rp=fin; win=bool(cells(Lp)&cells(Rp))
        out.append((i,nm,Lp,Rp,sorted(opens(Lp,Rp)),'WIN' if win else ''))
        if win: break
    return out

def solve8(f):
    import collections
    a=f.ascii.split('\n'); R=len(a); C=len(a[0])
    wall=[[a[r][c] in 'MPW' for c in range(C)] for r in range(R)]
    sq=[[None]*C for _ in range(R)]; band=[[None]*C for _ in range(R)]
    for n in f.segmentation['nodes']:
        if n['color'] in 'pNOY' and n['pixels']>=16:
            b=n['boundary']; r0,r1=min(p[0] for p in b),max(p[0] for p in b); c0,c1=min(p[1] for p in b),max(p[1] for p in b)
            tgt=sq if n['pixels']==16 else band
            for x in range(r0,r1+1):
                for y in range(c0,c1+1): tgt[x][y]=n['color']
    allS=[(min(p[0] for p in n['boundary']),min(p[1] for p in n['boundary'])) for n in f.segmentation['nodes'] if n['color']=='S' and n['pixels']==16]
    L0=[p for p in allS if p[1]<32][0]; R0=[p for p in allS if p[1]>=32][0]
    start=(L0,R0)
    acts=[('UP',(-1,0),(-1,0)),('DOWN',(1,0),(1,0)),('LEFT',(0,1),(0,-1)),('RIGHT',(0,-1),(0,1))]
    def cells(p0): return set((x,y) for x in range(p0[0],p0[0]+4) for y in range(p0[1],p0[1]+4))
    def opens(Lp,Rp):
        s=set()
        for p0 in (Lp,Rp):
            c=sq[p0[0]][p0[1]]
            if c and all(sq[x][y]==c for x in range(p0[0],p0[0]+4) for y in range(p0[1],p0[1]+4)): s.add(c)
        return s
    def ok(r,c,nr,nc,op):
        for x in range(min(r,nr),max(r,nr)+4):
            for y in range(min(c,nc),max(c,nc)+4):
                if x<0 or x>=R or y<0 or y>=C or wall[x][y]: return False
                if band[x][y] and band[x][y] not in op: return False
        return True
    def swp(r,c,nr,nc): return (min(r,nr),max(r,nr)+4,min(c,nc),max(c,nc)+4)
    prev={start:None}; q=collections.deque([start]); found=None; cnt=0
    while q:
        Lp,Rp=q.popleft(); cnt+=1
        op=opens(Lp,Rp)
        for nm,(drL,dcL),(drR,dcR) in acts:
            raw=[]
            for (r,c),(dr,dc) in ((Lp,(drL,dcL)),(Rp,(drR,dcR))):
                nr,nc=r+dr*4,c+dc*4
                raw.append((nr,nc) if ok(r,c,nr,nc,op) else None)
            fin=[x if x is not None else y for x,y in zip(raw,[Lp,Rp])]
            nL,nR=fin
            if cells(nL)&cells(nR): found=(Lp,Rp,nm); break
            nop=opens(nL,nR); bad=False
            for p0 in (nL,nR):
                for x in range(p0[0],p0[0]+4):
                    for y in range(p0[1],p0[1]+4):
                        if band[x][y] and band[x][y] not in nop: bad=True
            if bad: continue
            sl=swp(Lp[0],Lp[1],nL[0],nL[1]); sr=swp(Rp[0],Rp[1],nR[0],nR[1])
            CL=cells(nL); CR=cells(nR)
            inb=lambda box,cs: any(box[0]<=x<box[1] and box[2]<=y<box[3] for (x,y) in cs)
            if inb(sl,CR) or inb(sr,CL): continue
            if nL==Lp and nR==Rp: continue
            ns=(nL,nR)
            if ns not in prev: prev[ns]=((Lp,Rp),nm); q.append(ns)
        if found: break
    if not found: return ('none',cnt)
    s0,s1,nm=found; path=[nm]; s=(s0,s1)
    while prev[s] is not None:
        s,pn=prev[s]; path.append(pn)
    return list(reversed(path))

def sim8(f,plan):
    a=f.ascii.split('\n'); R=len(a); C=len(a[0])
    wall=[[a[r][c] in 'MPW' for c in range(C)] for r in range(R)]
    sq=[[None]*C for _ in range(R)]; band=[[None]*C for _ in range(R)]
    for n in f.segmentation['nodes']:
        if n['color'] in 'pNOY' and n['pixels']>=16:
            b=n['boundary']; r0,r1=min(p[0] for p in b),max(p[0] for p in b); c0,c1=min(p[1] for p in b),max(p[1] for p in b)
            tgt=sq if n['pixels']==16 else band
            for x in range(r0,r1+1):
                for y in range(c0,c1+1): tgt[x][y]=n['color']
    allS=[(min(p[0] for p in n['boundary']),min(p[1] for p in n['boundary'])) for n in f.segmentation['nodes'] if n['color']=='S' and n['pixels']==16]
    Lp=[p for p in allS if p[1]<32][0]; Rp=[p for p in allS if p[1]>=32][0]
    A={'UP':((-1,0),(-1,0)),'DOWN':((1,0),(1,0)),'LEFT':((0,1),(0,-1)),'RIGHT':((0,-1),(0,1))}
    def cells(p0): return set((x,y) for x in range(p0[0],p0[0]+4) for y in range(p0[1],p0[1]+4))
    def opens(Lp,Rp):
        s=set()
        for p0 in (Lp,Rp):
            c=sq[p0[0]][p0[1]]
            if c and all(sq[x][y]==c for x in range(p0[0],p0[0]+4) for y in range(p0[1],p0[1]+4)): s.add(c)
        return s
    def ok(r,c,nr,nc,op):
        for x in range(min(r,nr),max(r,nr)+4):
            for y in range(min(c,nc),max(c,nc)+4):
                if x<0 or x>=R or y<0 or y>=C or wall[x][y]: return False
                if band[x][y] and band[x][y] not in op: return False
        return True
    out=[]
    for i,nm in enumerate(plan):
        (drL,dcL),(drR,dcR)=A[nm]; op=opens(Lp,Rp); raw=[]
        for (r,c),(dr,dc) in ((Lp,(drL,dcL)),(Rp,(drR,dcR))):
            nr,nc=r+dr*4,c+dc*4
            raw.append((nr,nc) if ok(r,c,nr,nc,op) else None)
        fin=[x if x is not None else y for x,y in zip(raw,[Lp,Rp])]
        Lp,Rp=fin; win=bool(cells(Lp)&cells(Rp))
        out.append((i,nm,'L',Lp,'R',Rp,sorted(opens(Lp,Rp)),'WIN' if win else ''))
        if win: break
    return out

def geo(f):
    a=f.ascii.split('\n'); R=len(a); C=len(a[0])
    wall=[[a[r][c] in 'MPW' for c in range(C)] for r in range(R)]
    for (r,c) in [(46,14)]:
        for x in range(r,r+4):
            for y in range(c,c+4): wall[x][y]=True
    sq=[[None]*C for _ in range(R)]; band=[[None]*C for _ in range(R)]
    for col,ps in SQ.items():
        for (r,c) in ps:
            for x in range(r,r+4):
                for y in range(c,c+4): sq[x][y]=col
    for col,ps in BANDS.items():
        for (r,c) in ps:
            for x in range(r,r+4):
                for y in range(c,c+4): band[x][y]=col
    return a,R,C,wall,sq,band

def solve9(f):
    import collections
    a,R,C,wall,sq,band=geo(f)
    allS=[(min(p[0] for p in n['boundary']),min(p[1] for p in n['boundary'])) for n in f.segmentation['nodes'] if n['color']=='S' and n['pixels']==16]
    L0=[p for p in allS if p[1]<32][0]; R0=[p for p in allS if p[1]>=32][0]
    start=(L0,R0)
    acts=[('UP',(-1,0),(-1,0)),('DOWN',(1,0),(1,0)),('LEFT',(0,1),(0,-1)),('RIGHT',(0,-1),(0,1))]
    def cells(p0): return set((x,y) for x in range(p0[0],p0[0]+4) for y in range(p0[1],p0[1]+4))
    def opens(Lp,Rp):
        s=set()
        for p0 in (Lp,Rp):
            c=sq[p0[0]][p0[1]]
            if c: s.add(c)
        return s
    def ok(r,c,nr,nc,op):
        for x in range(min(r,nr),max(r,nr)+4):
            for y in range(min(c,nc),max(c,nc)+4):
                if x<0 or x>=R or y<0 or y>=C or wall[x][y]: return False
                if band[x][y] and band[x][y] not in op: return False
        return True
    def swp(r,c,nr,nc): return (min(r,nr),max(r,nr)+4,min(c,nc),max(c,nc)+4)
    prev={start:None}; q=collections.deque([start]); found=None; cnt=0
    while q:
        Lp,Rp=q.popleft(); cnt+=1
        op=opens(Lp,Rp)
        for nm,(drL,dcL),(drR,dcR) in acts:
            raw=[]
            for (r,c),(dr,dc) in ((Lp,(drL,dcL)),(Rp,(drR,dcR))):
                nr,nc=r+dr*4,c+dc*4
                raw.append((nr,nc) if ok(r,c,nr,nc,op) else None)
            fin=[x if x is not None else y for x,y in zip(raw,[Lp,Rp])]
            nL,nR=fin
            if cells(nL)&cells(nR): found=(Lp,Rp,nm); break
            nop=opens(nL,nR); bad=False
            for p0 in (nL,nR):
                for x in range(p0[0],p0[0]+4):
                    for y in range(p0[1],p0[1]+4):
                        if band[x][y] and band[x][y] not in nop: bad=True
            if bad: continue
            sl=swp(Lp[0],Lp[1],nL[0],nL[1]); sr=swp(Rp[0],Rp[1],nR[0],nR[1])
            CL=cells(nL); CR=cells(nR)
            inb=lambda box,cs: any(box[0]<=x<box[1] and box[2]<=y<box[3] for (x,y) in cs)
            if inb(sl,CR) or inb(sr,CL): continue
            if nL==Lp and nR==Rp: continue
            ns=(nL,nR)
            if ns not in prev: prev[ns]=((Lp,Rp),nm); q.append(ns)
        if found: break
    if not found: return ('none',cnt)
    s0,s1,nm=found; path=[nm]; s=(s0,s1)
    while prev[s] is not None:
        s,pn=prev[s]; path.append(pn)
    return list(reversed(path))

def sim9(f,plan):
    SQ={'p':[(6,14),(50,14)],'N':[(26,34)],'O':[(26,58)]}
    BANDS={'N':[(22,14)],'p':[(22,42),(38,42)],'O':[(38,10)]}
    a=f.ascii.split('\n'); R=len(a); C=len(a[0])
    wall=[[a[r][c] in 'MPW' for c in range(C)] for r in range(R)]
    for x in range(46,50):
        for y in range(14,18): wall[x][y]=True
    sq=[[None]*C for _ in range(R)]; band=[[None]*C for _ in range(R)]
    for col,ps in SQ.items():
        for (r,c) in ps:
            for x in range(r,r+4):
                for y in range(c,c+4): sq[x][y]=col
    for col,ps in BANDS.items():
        for (r,c) in ps:
            for x in range(r,r+4):
                for y in range(c,c+4): band[x][y]=col
    allS=[(min(p[0] for p in n['boundary']),min(p[1] for p in n['boundary'])) for n in f.segmentation['nodes'] if n['color']=='S' and n['pixels']==16]
    Lp=[p for p in allS if p[1]<32][0]; Rp=[p for p in allS if p[1]>=32][0]
    A={'UP':((-1,0),(-1,0)),'DOWN':((1,0),(1,0)),'LEFT':((0,1),(0,-1)),'RIGHT':((0,-1),(0,1))}
    def cells(p0): return set((x,y) for x in range(p0[0],p0[0]+4) for y in range(p0[1],p0[1]+4))
    def opens(Lp,Rp): return set(c for p0 in (Lp,Rp) for c in [sq[p0[0]][p0[1]]] if c)
    def ok(r,c,nr,nc,op):
        for x in range(min(r,nr),max(r,nr)+4):
            for y in range(min(c,nc),max(c,nc)+4):
                if x<0 or x>=R or y<0 or y>=C or wall[x][y]: return False
                if band[x][y] and band[x][y] not in op: return False
        return True
    out=[]
    for i,nm in enumerate(plan):
        (drL,dcL),(drR,dcR)=A[nm]; op=opens(Lp,Rp); raw=[]
        for (r,c),(dr,dc) in ((Lp,(drL,dcL)),(Rp,(drR,dcR))):
            nr,nc=r+dr*4,c+dc*4
            raw.append((nr,nc) if ok(r,c,nr,nc,op) else None)
        fin=[x if x is not None else y for x,y in zip(raw,[Lp,Rp])]
        blocked=['L' if raw[0] is None else '','R' if raw[1] is None else '']
        Lp,Rp=fin; win=bool(cells(Lp)&cells(Rp))
        nop=opens(Lp,Rp); trapped=[p0 for p0 in (Lp,Rp) if any(band[x][y] and band[x][y] not in nop for x in range(p0[0],p0[0]+4) for y in range(p0[1],p0[1]+4))]
        out.append((i,nm,'L',Lp,'R',Rp,'open='+','.join(sorted(nop)),'blk='+''.join(x for x in blocked if x),'TRAP' if trapped else '','WIN' if win else ''))
        if win: break
    return out

def solveA(f):
    import collections
    SQ=SQ2; BANDS=BAND2
    a=f.ascii.split('\n'); R=len(a); C=len(a[0])
    wall=[[a[r][c] in 'MPW' for c in range(C)] for r in range(R)]
    for x in range(46,50):
        for y in range(14,18): wall[x][y]=True
    sq=[[None]*C for _ in range(R)]; band=[[None]*C for _ in range(R)]
    for (col,r0,c0,r1,c1) in SQ:
        for x in range(r0,r1+1):
            for y in range(c0,c1+1): sq[x][y]=col
    for (col,r0,c0,r1,c1) in BANDS:
        for x in range(r0,r1+1):
            for y in range(c0,c1+1): band[x][y]=col
    allS=[(min(p[0] for p in n['boundary']),min(p[1] for p in n['boundary'])) for n in f.segmentation['nodes'] if n['color']=='S' and n['pixels']==16]
    L0=[p for p in allS if p[1]<32][0]; R0=[p for p in allS if p[1]>=32][0]
    start=(L0,R0)
    acts=[('UP',(-1,0),(-1,0)),('DOWN',(1,0),(1,0)),('LEFT',(0,1),(0,-1)),('RIGHT',(0,-1),(0,1))]
    def cells(p0): return set((x,y) for x in range(p0[0],p0[0]+4) for y in range(p0[1],p0[1]+4))
    def opens(Lp,Rp): return set(c for p0 in (Lp,Rp) for c in [sq[p0[0]][p0[1]]] if c)
    def ok(r,c,nr,nc,op):
        for x in range(min(r,nr),max(r,nr)+4):
            for y in range(min(c,nc),max(c,nc)+4):
                if x<0 or x>=R or y<0 or y>=C or wall[x][y]: return False
                if band[x][y] and band[x][y] not in op: return False
        return True
    def swp(r,c,nr,nc): return (min(r,nr),max(r,nr)+4,min(c,nc),max(c,nc)+4)
    prev={start:None}; q=collections.deque([start]); found=None; cnt=0
    while q:
        Lp,Rp=q.popleft(); cnt+=1
        op=opens(Lp,Rp)
        for nm,(drL,dcL),(drR,dcR) in acts:
            raw=[]
            for (r,c),(dr,dc) in ((Lp,(drL,dcL)),(Rp,(drR,dcR))):
                nr,nc=r+dr*4,c+dc*4
                raw.append((nr,nc) if ok(r,c,nr,nc,op) else None)
            fin=[x if x is not None else y for x,y in zip(raw,[Lp,Rp])]
            nL,nR=fin
            if cells(nL)&cells(nR): found=(Lp,Rp,nm); break
            nop=opens(nL,nR); bad=False
            for p0 in (nL,nR):
                for x in range(p0[0],p0[0]+4):
                    for y in range(p0[1],p0[1]+4):
                        if band[x][y] and band[x][y] not in nop: bad=True
            if bad: continue
            sl=swp(Lp[0],Lp[1],nL[0],nL[1]); sr=swp(Rp[0],Rp[1],nR[0],nR[1])
            CL=cells(nL); CR=cells(nR)
            inb=lambda box,cs: any(box[0]<=x<box[1] and box[2]<=y<box[3] for (x,y) in cs)
            if inb(sl,CR) or inb(sr,CL): continue
            if nL==Lp and nR==Rp: continue
            ns=(nL,nR)
            if ns not in prev: prev[ns]=((Lp,Rp),nm); q.append(ns)
        if found: break
    if not found: return ('none',cnt)
    s0,s1,nm=found; path=[nm]; s=(s0,s1)
    while prev[s] is not None:
        s,pn=prev[s]; path.append(pn)
    return list(reversed(path))

def simA(f,plan):
    SQ=[('p',6,14,9,17),('p',50,14,53,17),('N',26,34,29,37),('O',26,58,29,61)]
    BANDS=[('N',22,14,25,25),('p',22,42,25,53),('O',38,10,41,21),('p',38,42,41,53)]
    a=f.ascii.split('\n'); R=len(a); C=len(a[0])
    wall=[[a[r][c] in 'MPW' for c in range(C)] for r in range(R)]
    for x in range(46,50):
        for y in range(14,18): wall[x][y]=True
    sq=[[None]*C for _ in range(R)]; band=[[None]*C for _ in range(R)]
    for (col,r0,c0,r1,c1) in SQ:
        for x in range(r0,r1+1):
            for y in range(c0,c1+1): sq[x][y]=col
    for (col,r0,c0,r1,c1) in BANDS:
        for x in range(r0,r1+1):
            for y in range(c0,c1+1): band[x][y]=col
    allS=[(min(p[0] for p in n['boundary']),min(p[1] for p in n['boundary'])) for n in f.segmentation['nodes'] if n['color']=='S' and n['pixels']==16]
    Lp=[p for p in allS if p[1]<32][0]; Rp=[p for p in allS if p[1]>=32][0]
    A={'UP':((-1,0),(-1,0)),'DOWN':((1,0),(1,0)),'LEFT':((0,1),(0,-1)),'RIGHT':((0,-1),(0,1))}
    def cells(p0): return set((x,y) for x in range(p0[0],p0[0]+4) for y in range(p0[1],p0[1]+4))
    def opens(Lp,Rp): return set(c for p0 in (Lp,Rp) for c in [sq[p0[0]][p0[1]]] if c)
    def ok(r,c,nr,nc,op):
        for x in range(min(r,nr),max(r,nr)+4):
            for y in range(min(c,nc),max(c,nc)+4):
                if x<0 or x>=R or y<0 or y>=C or wall[x][y]: return False
                if band[x][y] and band[x][y] not in op: return False
        return True
    out=[]
    for i,nm in enumerate(plan):
        (drL,dcL),(drR,dcR)=A[nm]; op=opens(Lp,Rp); raw=[]
        for (r,c),(dr,dc) in ((Lp,(drL,dcL)),(Rp,(drR,dcR))):
            nr,nc=r+dr*4,c+dc*4
            raw.append((nr,nc) if ok(r,c,nr,nc,op) else None)
        fin=[x if x is not None else y for x,y in zip(raw,[Lp,Rp])]
        Lp,Rp=fin; win=bool(cells(Lp)&cells(Rp)); nop=opens(Lp,Rp)
        trap=[q0 for q0 in (Lp,Rp) if any(band[x][y] and band[x][y] not in nop for x in range(q0[0],q0[0]+4) for y in range(q0[1],q0[1]+4))]
        out.append((i,nm,Lp,Rp,''.join(sorted(nop)),'blk:'+(''.join('LR'[k] for k in range(2) if raw[k] is None)),'TRAP' if trap else '','WIN' if win else ''))
        if win: break
    return out

def isopen(r,c):
    if a[r][c]!='B': return False
    n=sum(1 for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)) if 0<=r+dr<64 and 0<=c+dc<64 and a[r+dr][c+dc]=='B')
    return n>=3

def pos(f):
    out=[]
    for n in f.segmentation['nodes']:
        if n['color'] in 'Sb' :
            b=n['boundary']; out.append((n['color'],n['pixels'],min(p[0] for p in b),min(p[1] for p in b)))
    return sorted(out)

def build6(f, blue_pass=True):
    a=f.ascii.split('\n'); R=len(a); C=len(a[0])
    passable=[[a[r][c] in 'BSNOb' for c in range(C)] for r in range(R)]
    if not blue_pass:
        for x in range(43,45):
            for y in range(31,33): passable[x][y]=False
    sq=[[None]*C for _ in range(R)]; band=[[None]*C for _ in range(R)]
    for (col,r0,c0,r1,c1) in [('O',14,18,17,21),('N',14,42,17,45),('N',42,18,45,21)]:
        for x in range(r0,r1+1):
            for y in range(c0,c1+1): sq[x][y]=col
    for (col,r0,c0,r1,c1) in [('N',30,14,33,25),('O',30,38,33,49)]:
        for x in range(r0,r1+1):
            for y in range(c0,c1+1): band[x][y]=col
    return R,C,passable,sq,band

def solveB(f, blue_pass=True, maxd=40):
    import collections
    R,C,passable,sq,band=build6(f,blue_pass)
    SS=[(min(p[0] for p in n['boundary']),min(p[1] for p in n['boundary'])) for n in f.segmentation['nodes'] if n['color']=='S' and n['pixels']==16]
    SS.sort(key=lambda p:p[1])
    start=(SS[0],SS[1])
    # block0 = normal mapping (LEFT->col-4), block1 = mirrored (LEFT->col+4)
    acts=[('UP',(-1,0),(-1,0)),('DOWN',(1,0),(1,0)),('LEFT',(0,-1),(0,1)),('RIGHT',(0,1),(0,-1))]
    def cells(p0): return set((x,y) for x in range(p0[0],p0[0]+4) for y in range(p0[1],p0[1]+4))
    def opens(Lp,Rp): return set(c for p0 in (Lp,Rp) for c in [sq[p0[0]][p0[1]]] if c)
    def ok(r,c,nr,nc,op):
        for x in range(min(r,nr),max(r,nr)+4):
            for y in range(min(c,nc),max(c,nc)+4):
                if x<0 or x>=R or y<0 or y>=C or not passable[x][y]: return False
                if band[x][y] and band[x][y] not in op: return False
        return True
    def swp(r,c,nr,nc): return (min(r,nr),max(r,nr)+4,min(c,nc),max(c,nc)+4)
    prev={start:None}; q=collections.deque([start]); found=None
    while q:
        Lp,Rp=q.popleft(); op=opens(Lp,Rp)
        for nm,(drA,dcA),(drB,dcB) in acts:
            raw=[]
            for (r,c),(dr,dc) in ((Lp,(drA,dcA)),(Rp,(drB,dcB))):
                nr,nc=r+dr*4,c+dc*4
                raw.append((nr,nc) if ok(r,c,nr,nc,op) else None)
            fin=[x if x is not None else y for x,y in zip(raw,[Lp,Rp])]
            nL,nR=fin
            if cells(nL)&cells(nR): found=(Lp,Rp,nm); break
            nop=opens(nL,nR); bad=False
            for p0 in (nL,nR):
                for x in range(p0[0],p0[0]+4):
                    for y in range(p0[1],p0[1]+4):
                        if band[x][y] and band[x][y] not in nop: bad=True
            if bad: continue
            sl=swp(Lp[0],Lp[1],nL[0],nL[1]); sr=swp(Rp[0],Rp[1],nR[0],nR[1])
            CL=cells(nL); CR=cells(nR)
            inb=lambda box,cs: any(box[0]<=x<box[1] and box[2]<=y<box[3] for (x,y) in cs)
            if inb(sl,CR) or inb(sr,CL): continue
            if nL==Lp and nR==Rp: continue
            ns=(nL,nR)
            if ns not in prev: prev[ns]=((Lp,Rp),nm); q.append(ns)
        if found: break
    if not found: return ('none',)
    s0,s1,nm=found; path=[nm]; s=(s0,s1)
    while prev[s] is not None:
        s,pn=prev[s]; path.append(pn)
    return list(reversed(path))

def solveC(f, blue_pass=True, allow_trap=True, allow_swap=True):
    import collections
    R,C,passable,sq,band=build6(f,blue_pass)
    SS=[(min(p[0] for p in n['boundary']),min(p[1] for p in n['boundary'])) for n in f.segmentation['nodes'] if n['color']=='S' and n['pixels']==16]
    SS.sort(key=lambda p:p[1]); start=(SS[0],SS[1])
    acts=[('UP',(-1,0),(-1,0)),('DOWN',(1,0),(1,0)),('LEFT',(0,-1),(0,1)),('RIGHT',(0,1),(0,-1))]
    def cells(p0): return set((x,y) for x in range(p0[0],p0[0]+4) for y in range(p0[1],p0[1]+4))
    def opens(Lp,Rp): return set(c for p0 in (Lp,Rp) for c in [sq[p0[0]][p0[1]]] if c)
    def ok(r,c,nr,nc,op):
        for x in range(min(r,nr),max(r,nr)+4):
            for y in range(min(c,nc),max(c,nc)+4):
                if x<0 or x>=R or y<0 or y>=C or not passable[x][y]: return False
                if band[x][y] and band[x][y] not in op: return False
        return True
    prev={start:None}; q=collections.deque([start]); found=None
    while q:
        Lp,Rp=q.popleft(); op=opens(Lp,Rp)
        for nm,(drA,dcA),(drB,dcB) in acts:
            raw=[]
            for (r,c),(dr,dc) in ((Lp,(drA,dcA)),(Rp,(drB,dcB))):
                nr,nc=r+dr*4,c+dc*4
                raw.append((nr,nc) if ok(r,c,nr,nc,op) else None)
            fin=[x if x is not None else y for x,y in zip(raw,[Lp,Rp])]
            nL,nR=fin
            if cells(nL)&cells(nR): found=(Lp,Rp,nm); break
            if not allow_trap:
                nop=opens(nL,nR)
                if any(band[x][y] and band[x][y] not in nop for p0 in (nL,nR) for x in range(p0[0],p0[0]+4) for y in range(p0[1],p0[1]+4)): continue
            if not allow_swap:
                pass
            if nL==Lp and nR==Rp: continue
            ns=(nL,nR)
            if ns not in prev: prev[ns]=((Lp,Rp),nm); q.append(ns)
        if found: break
    if not found: return ('none',)
    s0,s1,nm=found; path=[nm]; s=(s0,s1)
    while prev[s] is not None:
        s,pn=prev[s]; path.append(pn)
    return list(reversed(path))

def reach6(f, start, blue_pass=True, opened=('N','O')):
    R,C,passable,sq,band=build6(f,blue_pass)
    def ok(r,c,nr,nc):
        for x in range(min(r,nr),max(r,nr)+4):
            for y in range(min(c,nc),max(c,nc)+4):
                if x<0 or x>=R or y<0 or y>=C or not passable[x][y]: return False
                if band[x][y] and band[x][y] not in opened: return False
        return True
    seen={start}; q=[start]
    while q:
        p0=q.pop()
        for dr,dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nr,nc=p0[0]+dr*4,p0[1]+dc*4
            if (nr,nc) not in seen and ok(p0[0],p0[1],nr,nc): seen.add((nr,nc)); q.append((nr,nc))
    return seen

def opens(Lp,Rp): return frozenset(c for p0 in (Lp,Rp) for c in [sq[p0[0]][p0[1]]] if c)

def zone(p0):
    r,c=p0
    if r<=29: return 'T'
    return 'B'

def cell(r,c): return current_frame.ascii.split('\n')[r][c]
