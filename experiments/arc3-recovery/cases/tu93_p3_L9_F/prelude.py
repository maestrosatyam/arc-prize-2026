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
    from collections import defaultdict
except Exception:
    pass
try:
    import collections
except Exception:
    pass
try:
    import json
except Exception:
    pass
try:
    from itertools import permutations
except Exception:
    pass

def ch(r,c): return a[r][c]

def passable(i,j): return 0<=i<n and 0<=j<n and RM[i][j]!='B'

def slide(i,j,di,dj):
    while passable(i+di,j+dj): i+=di; j+=dj
    return (i,j)

def mk(RM):
    def passable(i,j): return 0<=i<n and 0<=j<n and RM[i][j]!='B'
    def slide(i,j,di,dj):
        while passable(i+di,j+dj): i+=di; j+=dj
        return (i,j)
    return passable, slide

def free(i,j): return 0<=i<n and 0<=j<n and RM[i][j]!='B'

def moves(p):
    out={}
    for k,(di,dj) in DIRS.items():
        d=(p[0]+2*di,p[1]+2*dj)
        if d in M and M[d]!='B': out[k]=d
    return out

def tilemap(frame):
    a = frame.ascii.split('\n')
    n = 12
    return [''.join(a[15+3*i][15+3*j] for j in range(n)) for i in range(n)]

def ptile(frame):
    a = frame.ascii.split('\n')
    rs = [(r,c) for r in range(64) for c in range(64) if a[r][c]=='b']
    r0, c0 = min(rs)
    return ((r0-15)//3, (c0-15)//3)

def jump_bfs(RM, start, goal):
    n = 12
    def free(i,j): return 0<=i<n and 0<=j<n and RM[i][j]!='B'
    D={'UP':(-1,0),'DOWN':(1,0),'LEFT':(0,-1),'RIGHT':(0,1)}
    prev={start:None}; 
    from collections import deque
    q=deque([start])
    while q:
        cur=q.popleft()
        if cur==goal: break
        for k,(di,dj) in D.items():
            if free(cur[0]+di,cur[1]+dj) and free(cur[0]+2*di,cur[1]+2*dj):
                nx=(cur[0]+2*di,cur[1]+2*dj)
                if nx not in prev: prev[nx]=(cur,k); q.append(nx)
    if goal not in prev: return None
    path=[]; cur=goal
    while prev[cur] is not None:
        p,k=prev[cur]; path.append(k); cur=p
    path.reverse(); return path

def tmap(frame):
    a = frame.ascii.split('\n')
    R0,C0,nr,nc = 21,12,5,13
    return [''.join(a[R0+3*i][C0+3*j] for j in range(nc)) for i in range(nr)]

def find(frame, ch):
    a = frame.ascii.split('\n')
    return [(r,c) for r in range(64) for c in range(64) if a[r][c]==ch]

def t(p): return (((p[0]-21)//3,(p[1]-12)//3) if p else None)

def snap(f):
    red=find(f,'R'); blu=find(f,'b')
    return (t(min(red)) if red else None, t(min(blu)) if blu else None)

def pmoves(P):
    out={}
    for k,(di,dj) in DIRS.items():
        a=(P[0]+di,P[1]+dj); b=(P[0]+2*di,P[1]+2*dj)
        if a in free and b in free: out[k]=b
    return out

def path_cells(P,Q):
    cells=[]; cur=P
    while cur!=Q:
        cur=(cur[0]+ (Q[0]-cur[0])//abs(Q[0]-cur[0]) if Q[0]!=cur[0] else cur[0],
             cur[1]+ (Q[1]-cur[1])//abs(Q[1]-cur[1]) if Q[1]!=cur[1] else cur[1])
        cells.append(cur)
    return cells

def red_responses(R, Pp, conservative=True):
    dr=Pp[0]-R[0]; dc=Pp[1]-R[1]
    resp=set()
    def step(axis, sign):
        di,dj = (sign,0) if axis=='r' else (0,sign)
        a=(R[0]+di,R[1]+dj); b=(R[0]+2*di,R[1]+2*dj)
        if a in free and b in free: resp.add(b)
    if dr==0 and 0<abs(dc)<=4: step('c', 1 if dc>0 else -1)
    elif dc==0 and 0<abs(dr)<=4: step('r', 1 if dr>0 else -1)
    elif conservative and 0<abs(dr)+abs(dc)<=4:
        if dr: step('r', 1 if dr>0 else -1)
        if dc: step('c', 1 if dc>0 else -1)
    resp.add(R)
    return resp

def solve(free, start, goal, red0, conservative=True):
    DIRS={'UP':(-1,0),'DOWN':(1,0),'LEFT':(0,-1),'RIGHT':(0,1)}
    def pmoves(P):
        out={}
        for k,(di,dj) in DIRS.items():
            a=(P[0]+di,P[1]+dj); b=(P[0]+2*di,P[1]+2*dj)
            if a in free and b in free: out[k]=b
        return out
    def seg(P,Q):
        cells=[P]; cur=P
        while cur!=Q:
            cur=(cur[0]+(1 if Q[0]>cur[0] else -1) if Q[0]!=cur[0] else cur[0],
                 cur[1]+(1 if Q[1]>cur[1] else -1) if Q[1]!=cur[1] else cur[1])
            cells.append(cur)
        return set(cells)
    def resp(R,Pp):
        dr=Pp[0]-R[0]; dc=Pp[1]-R[1]; out={R}
        def step(axis,sign):
            di,dj=(sign,0) if axis=='r' else (0,sign)
            a=(R[0]+di,R[1]+dj); b=(R[0]+2*di,R[1]+2*dj)
            if a in free and b in free: out.add(b)
        if dr==0 and 0<abs(dc)<=4: step('c',1 if dc>0 else -1)
        elif dc==0 and 0<abs(dr)<=4: step('r',1 if dr>0 else -1)
        elif conservative and 0<abs(dr)+abs(dc)<=4:
            if dr: step('r',1 if dr>0 else -1)
            if dc: step('c',1 if dc>0 else -1)
        return out
    tiles=sorted(free); reds=sorted(free-{goal})
    succ={}
    for P in tiles:
        if P==goal: continue
        for R in reds:
            if R==P: continue
            for m,Pp in pmoves(P).items():
                if Pp==R: continue
                pp=seg(P,Pp); rs=resp(R,Pp); ok=True; nxt=[]
                for Rp in rs:
                    if Rp==Pp or (seg(R,Rp) & pp): ok=False; break
                    nxt.append((Pp,Rp))
                if ok: succ.setdefault((P,R),[]).append((m,nxt))
    W=set((goal,R) for R in reds)
    for _ in range(200):
        new=set(W)
        for s,ms in succ.items():
            if s in W: continue
            for m,nxt in ms:
                if all(n in W for n in nxt): new.add(s); break
        if new==W: break
        W=new
    # extract shortest strategy
    import collections
    dist={s:0 for s in W if s[0]==goal}
    frontier=[s for s in W if s[0]==goal]
    while frontier:
        cur=frontier.pop(0)
        for s,ms in succ.items():
            pass
    return W, succ

def bar(f):
    a=f.ascii.split('\n')
    row=a[63]
    return row.count('M'), row.strip('M')[:0], len(row)

def bigmap(f, r0=0, c0=0, nr=20, nc=14):
    a=f.ascii.split('\n')
    out=[]
    for i in range(nr):
        row=''
        for j in range(nc):
            r=r0+3*i; c=c0+3*j
            if r>=len(a) or c+2>=len(a[0]): row+='?'; continue
            cells=[a[r][c],a[r][c+1],a[r][c+2],a[r+1][c],a[r+1][c+1],a[r+1][c+2],a[r+2][c],a[r+2][c+1],a[r+2][c+2]]
            # majority non-black if any
            nb=[x for x in cells if x!='B']
            if not nb: row+='B'
            else:
                from collections import Counter
                cnt=Counter(nb)
                row+=cnt.most_common(1)[0][0]
        out.append(row)
    return out

def ctr(i,j,r0=18,c0=12):
    return a[r0+3*i+1][c0+3*j+1]

def build():
    M={}
    rows={0:{4:'W',5:'g',6:'W',7:'g',8:'W'},
          1:{4:'g',6:'g',8:'g'},
          2:{4:'R',5:'g',6:'R',7:'g',8:'W',9:'g',10:'W',11:'g',12:'W'},
          3:{6:'g',12:'g'},
          4:{c:('W' if c%2==0 else 'g') for c in range(13)},
          5:{0:'g',2:'g',5:'g',7:'g',9:'g',11:'g'},
          6:{0:'R',1:'g',2:'W',6:'W',7:'g',8:'W',9:'g',10:'W'},
          7:{2:'g',10:'g'},
          8:{2:'W',3:'g',4:'N',10:'W'}}
    for r,cols in rows.items():
        for c,v in cols.items(): M[(r,c)]=v
    return M

def bfs(start,goal,banned):
    q=collections.deque([start]); par={start:None}
    while q:
        c=q.popleft()
        if c==goal: break
        for k,n in moves(c).items():
            if n in par or n in banned: continue
            par[n]=(c,k); q.append(n)
    if goal not in par: return None
    path=[]; c=goal
    while par[c]: 
        p,k=par[c]; path.append(k); c=p
    return path[::-1]

def plan():
    M=build()
    DIRS={'UP':(-1,0),'DOWN':(1,0),'LEFT':(0,-1),'RIGHT':(0,1)}
    EN={ (2,4):'RIGHT',(2,6):'DOWN',(6,0):'RIGHT'}
    def dang(live):
        s=set()
        for e,d in EN.items():
            if e in live:
                di,dj=DIRS[d]; s.add((e[0]+2*di,e[1]+2*dj))
        return s
    def mv(p):
        out=[]
        for k,(di,dj) in DIRS.items():
            d=(p[0]+2*di,p[1]+2*dj)
            if d in M: out.append((k,d))
        return out
    start=((4,10),frozenset(EN))
    goal=(8,4)
    import collections
    q=collections.deque([start]); par={start:None}
    found=None
    while q:
        p,live=q.popleft()
        for k,d in mv(p):
            if d not in M: continue
            st=(d,live)
            if d==goal: found=(p,live,k,d,live); par[st]=((p,live),k); q.append(st); break
            if d in dang(live) and d not in live: continue   # fatal lunge
            nl=live
            if d in live: nl=frozenset(x for x in live if x!=d)
            st=(d,nl)
            if st in par: continue
            # check landing on enemy tile that is another enemy's danger tile
            if d in live and d in dang(nl): continue
            par[st]=((p,live),k); q.append(st)
        if found: break
    if found is None and (goal,frozenset(EN)) not in par: 
        # search any state with pos==goal
        g=[s for s in par if s[0]==goal]
        if not g: return None
        cur=g[0]
    else:
        g=[s for s in par if s[0]==goal]
        cur=sorted(g,key=lambda s: len(s[1]))[0]
    seq=[]
    while par[cur]:
        p,k=par[cur]; seq.append((k,cur[0],sorted(p[1]-cur[1]))); cur=p
    return seq[::-1]

def ptile2(f):
    a=f.ascii.split('\n')
    for i in range(9):
        for j in range(13):
            if a[18+3*i+1][12+3*j+1]=='b': return (i,j)

def cmap(f, nr=12, nc=14):
    a=f.ascii.split('\n')
    rs=[r for r in range(64) for c in range(64) if a[r][c]!='B' and r<60]
    cs=[c for r in range(64) for c in range(64) if a[r][c]!='B' and r<60]
    r0,c0=min(rs),min(cs)
    M={}
    for i in range(nr):
        for j in range(nc):
            R,C=r0+3*i+1,c0+3*j+1
            if R<60 and C<64: M[(i,j)]=a[R][C]
    return M,(r0,c0)

def orange(f):
    for n in f.segmentation['nodes']:
        if n['color'] in ('O','R'):
            yield n['color'], n['boundary'][0]

def level4_plan():
    a=current_frame.ascii.split('\n')
    r0,c0=16,21
    M={}
    for i in range(9):
        for j in range(9):
            M[(i,j)]=a[r0+3*i+1][c0+3*j+1]
    DIRS={'UP':(-1,0),'DOWN':(1,0),'LEFT':(0,-1),'RIGHT':(0,1)}
    def step(p,k):
        di,dj=DIRS[k]; mid=(p[0]+di,p[1]+dj); q=(p[0]+2*di,p[1]+2*dj)
        if M.get(mid)=='g' and M.get(q,'B') not in ('B','g'): return q
        return None
    O={0:2,1:4,2:6,3:8,4:6,5:4,6:2,7:0}
    def opath(t):
        a1=O[t%8]; a2=O[(t+1)%8]
        return {(6,a1),(6,(a1+a2)//2),(6,a2)}
    start=((8,0),0,False)  # t offset 0 means current t=8 -> use mod index 0 for O(8)=2
    import collections
    q=collections.deque([start]); par={start:None}
    goal=None
    while q:
        (p,tm,red)=q.popleft()
        for k in DIRS:
            n=step(p,k)
            if n is None: continue
            nt=(tm+1)%8
            ocur=opath(tm)  # orange path during this transition (from O(tm) to O(tm+1))
            oend=(6,O[nt])
            nred = red or (n==(0,2))
            if n==(2,2) and not nred: continue   # red lunge
            if n==oend: continue
            if n in ocur and n!=(6,O[tm]): continue
            # crossing: player path tiles
            di,dj=DIRS[k]; pp={(p[0]+di,p[1]+dj),n,p}
            if p[1]==6 and (pp & ocur): continue
            st=(n,nt,nred)
            if st in par: continue
            par[st]=((p,tm,red),k)
            if n==(2,0): goal=st; q.clear(); break
            q.append(st)
        if goal: break
    if goal is None:
        return None,M
    seq=[]; cur=goal
    while par[cur]:
        p,k=par[cur]; seq.append((k,cur[0],cur[1],cur[2])); cur=p
    return seq[::-1],M

def level4_plan2():
    a=current_frame.ascii.split('\n')
    r0,c0=16,21
    M={}
    for i in range(9):
        for j in range(9):
            M[(i,j)]=a[r0+3*i+1][c0+3*j+1]
    DIRS={'UP':(-1,0),'DOWN':(1,0),'LEFT':(0,-1),'RIGHT':(0,1)}
    def step(p,k):
        di,dj=DIRS[k]; mid=(p[0]+di,p[1]+dj); q=(p[0]+2*di,p[1]+2*dj)
        if M.get(mid)=='g' and M.get(q,'B') not in ('B','g'): return q
        return None
    O={0:2,1:4,2:6,3:8,4:6,5:4,6:2,7:0}   # t mod 8 -> orange ROW (col 6)
    def opath(tm):
        a1=O[tm%8]; a2=O[(tm+1)%8]
        return {(a1,6),((a1+a2)//2,6),(a2,6)}
    start=((8,0),0,False)
    import collections
    q=collections.deque([start]); par={start:None}; goal=None
    while q:
        (p,tm,red)=q.popleft()
        for k in DIRS:
            n=step(p,k)
            if n is None: continue
            nt=(tm+1)%8
            ocur=opath(tm); oend=(O[nt],6)
            nred = red or (n==(0,2))
            if n==(2,2) and not nred: continue
            if n==oend: continue
            if n in ocur and n!=(O[tm%8],6): continue
            di,dj=DIRS[k]; pp={p,(p[0]+di,p[1]+dj),n}
            if p[1]==6 and (pp & ocur): continue
            st=(n,nt,nred)
            if st in par: continue
            par[st]=((p,tm,red),k)
            if n==(2,0): goal=st; q.clear(); break
            q.append(st)
        if goal: break
    if goal is None: return None,M
    seq=[]; cur=goal
    while par[cur]:
        p,k=par[cur]; seq.append((k,cur[0],cur[1],cur[2])); cur=p
    return seq[::-1],M

def ents(f):
    d={}
    for n in f.segmentation['nodes']:
        if n['color'] in ('O',):
            r,c=n['boundary'][0]; d['O%d_%d'%((r-14)//3,(c-9)//3)]=None
    return sorted(d)

def sim(route,c0,order,adj=ADJ):
    c=c0; hist=[route[0]]
    for i in range(1,len(route)):
        p=route[i]; d=dist(p)
        nb=[(k,v) for k,v in adj[c].items()]
        best=min(d[v] for k,v in nb)
        cands=sorted([kv for kv in nb if d[kv[1]]==best],key=lambda kv: order.index(kv[0]))
        cn=cands[0][1]
        if cn==p or (p==c and cn==hist[-1]): return 'die@%d'%i
        c=cn; hist.append(p)
        if p==(0,4): return None
    return 'nogoal'

def plan5():
    a=current_frame.ascii.split('\n')
    r0,c0=14,9
    M={}
    for i in range(11):
        for j in range(15):
            M[(i,j)]=a[r0+3*i+1][c0+3*j+1]
    DIRS={'UP':(-1,0),'DOWN':(1,0),'LEFT':(0,-1),'RIGHT':(0,1)}
    def mv(p,k):
        di,dj=DIRS[k]; mid=(p[0]+di,p[1]+dj); q=(p[0]+2*di,p[1]+2*dj)
        if M.get(mid)=='g' and M.get(q,'B') not in ('B','g'): return q,mid
        return None
    def tri(t,lo,hi,start,dirn):
        # position at time t moving 2 per step bouncing in [lo,hi] (even ints), start pos & dir(+1/-1)
        span=(hi-lo)//2; pos=start; d=dirn; 
        for _ in range(t):
            nxt=pos+2*d
            if nxt<lo or nxt>hi: d=-d; nxt=pos+2*d
            pos=nxt
        return pos
    def occ(t):
        s=set()
        A=tri(t,0,10,4,1); Dp=tri(t,0,10,8,-1); B=tri(t,0,14,0,1); C=tri(t,4,8,6,1)
        pts={'A':(A,6),'D':(Dp,6),'B':(6,B),'C':(6,C)}
        return pts
    # verify against observations
    obs={1:{(6,2),(6,6),(8,4)},2:{(4,6),(6,4),(8,6)},3:{(2,6),(4,4),(6,6),(10,6)},4:{(0,6),(6,4),(6,8),(8,6)},5:{(2,6),(6,6),(6,10),(8,4)}}
    ok=True
    for t in range(1,6):
        got=set(occ(t).values())
        if got!=obs[t]: ok=False; print('MISMATCH t',t,'got',sorted(got),'exp',sorted(obs[t]))
    print('model ok',ok)
    def occpaths(t):
        # occupancy and path tiles at transition t -> t+1
        cur=occ(t); nxt=occ(t+1); out={}
        for k in cur:
            p,q=cur[k],nxt[k]
            if p[0]==q[0]: mid=(p[0],(p[1]+q[1])//2)
            else: mid=((p[0]+q[0])//2,p[1])
            out[k]=(p,q,mid)
        return out
    start=((0,8),5)
    import collections
    q=collections.deque([start]); par={start:None}; goal=None
    while q:
        (p,t)=q.popleft()
        if t>25: continue
        op=occpaths(t)
        ocells={x[1] for x in op.values()}
        for k in DIRS:
            r=mv(p,k)
            if not r: continue
            n,mid=r
            nt=t+1
            npath={p,mid,n}
            if n in ocells: continue
            # crossing with any orange path
            bad=False
            for kk,(a1,b1,m1) in op.items():
                seg={a1,m1,b1}
                if npath & seg and not (n==a1 and n not in ocells):
                    bad=True; break
            if bad: continue
            st=(n,nt)
            if st in par: continue
            par[st]=((p,t),k)
            if n==(6,6): goal=st; q.clear(); break
            q.append(st)
        if goal: break
    if goal is None: return None
    seq=[]; cur=goal
    while par[cur]: p,k=par[cur]; seq.append((k,cur[0],cur[1])); cur=p
    return seq[::-1]

def plan5b():
    a=current_frame.ascii.split('\n')
    r0,c0=14,9
    M={}
    for i in range(11):
        for j in range(15):
            M[(i,j)]=a[r0+3*i+1][c0+3*j+1]
    DIRS={'UP':(-1,0),'DOWN':(1,0),'LEFT':(0,-1),'RIGHT':(0,1)}
    def mv(p,k):
        di,dj=DIRS[k]; mid=(p[0]+di,p[1]+dj); q=(p[0]+2*di,p[1]+2*dj)
        if M.get(mid)=='g' and M.get(q,'B') not in ('B','g'): return q,mid
        return None
    def tri(t,lo,hi,start,dirn):
        pos=start; d=dirn
        for _ in range(t):
            nxt=pos+2*d
            if nxt<lo or nxt>hi: d=-d; nxt=pos+2*d
            pos=nxt
        return pos
    def occ(t):
        A=tri(t,0,10,4,1); Dp=tri(t,0,10,8,-1); B=tri(t,0,14,0,1); C=tri(t,4,8,6,1)
        return {'A':(A,6),'D':(Dp,6),'B':(6,B),'C':(C,4)}
    obs={1:{(6,2),(6,6),(8,4)},2:{(4,6),(6,4),(8,6)},3:{(2,6),(4,4),(6,6),(10,6)},4:{(0,6),(6,4),(6,8),(8,6)},5:{(2,6),(6,6),(6,10),(8,4)}}
    ok=True
    for t in range(1,6):
        got=set(occ(t).values())
        if got!=obs[t]: ok=False; print('MISMATCH t',t,sorted(got),sorted(obs[t]))
    print('model ok',ok)
    def occpaths(t):
        cur=occ(t); nxt=occ(t+1); out={}
        for k in cur:
            p,q=cur[k],nxt[k]
            mid=(p[0],(p[1]+q[1])//2) if p[0]==q[0] else ((p[0]+q[0])//2,p[1])
            out[k]=(p,q,mid)
        return out
    start=((0,8),5)
    import collections
    q=collections.deque([start]); par={start:None}; goal=None
    while q:
        (p,t)=q.popleft()
        if t>26: continue
        op=occpaths(t); ocells={x[1] for x in op.values()}
        for k in DIRS:
            r=mv(p,k)
            if not r: continue
            n,mid=r; nt=t+1; npath={p,mid,n}
            if n in ocells: continue
            bad=False
            for kk,(a1,b1,m1) in op.items():
                seg={a1,m1,b1}
                if npath & seg and not (n==a1): bad=True; break
            if bad: continue
            st=(n,nt)
            if st in par: continue
            par[st]=((p,t),k)
            if n==(6,6): goal=st; q.clear(); break
            q.append(st)
        if goal: break
    if goal is None: return None
    seq=[]; cur=goal
    while par[cur]: p,k=par[cur]; seq.append((k,cur[0],cur[1])); cur=p
    return seq[::-1],occ

def plan5c(horizon=40, strict=False):
    a=current_frame.ascii.split('\n')
    r0,c0=14,9
    M={}
    for i in range(11):
        for j in range(15):
            M[(i,j)]=a[r0+3*i+1][c0+3*j+1]
    DIRS={'UP':(-1,0),'DOWN':(1,0),'LEFT':(0,-1),'RIGHT':(0,1)}
    def mv(p,k):
        di,dj=DIRS[k]; mid=(p[0]+di,p[1]+dj); q=(p[0]+2*di,p[1]+2*dj)
        if M.get(mid)=='g' and M.get(q,'B') not in ('B','g'): return q,mid
        return None
    def tri(t,lo,hi,start,dirn):
        pos=start; d=dirn
        for _ in range(t):
            nxt=pos+2*d
            if nxt<lo or nxt>hi: d=-d; nxt=pos+2*d
            pos=nxt
        return pos
    def occ(t):
        return {'A':(tri(t,0,10,4,1),6),'D':(tri(t,0,10,8,-1),6),'B':(6,tri(t,0,14,0,1)),'C':(tri(t,4,8,6,1),4)}
    def occpaths(t):
        cur=occ(t); nxt=occ(t+1); out={}
        for k in cur:
            p,q=cur[k],nxt[k]
            mid=(p[0],(p[1]+q[1])//2) if p[0]==q[0] else ((p[0]+q[0])//2,p[1])
            out[k]=(p,q,mid)
        return out
    start=((0,8),5)
    import collections
    q=collections.deque([start]); par={start:None}; goal=None
    while q:
        (p,t)=q.popleft()
        if t>horizon: continue
        op=occpaths(t); oend={x[1] for x in op.values()}; ostart={x[0] for x in op.values()}
        omid={x[2] for x in op.values()}
        for k in DIRS:
            r=mv(p,k)
            if not r: continue
            n,mid=r; nt=t+1
            if n in oend: continue
            if n in omid: continue
            if mid in oend or mid in omid: continue
            if strict and (n in ostart): continue
            st=(n,nt)
            if st in par: continue
            par[st]=((p,t),k)
            if n==(6,6): goal=st; q.clear(); break
            q.append(st)
        if goal: break
    if goal is None: return None
    seq=[]; cur=goal
    while par[cur]: p,k=par[cur]; seq.append((k,cur[0],cur[1])); cur=p
    return seq[::-1],occ

def occ5(t):
    def tri(t,lo,hi,start,dirn):
        pos=start; d=dirn
        for _ in range(t):
            nxt=pos+2*d
            if nxt<lo or nxt>hi: d=-d; nxt=pos+2*d
            pos=nxt
        return pos
    return sorted([(tri(t,0,10,4,1),6),(tri(t,0,10,8,-1),6),(6,tri(t,0,14,0,1)),(tri(t,4,8,6,1),4)])

def tiles(f,ch):
    return sorted(set(((n['boundary'][0][0]-14)//3,(n['boundary'][0][1]-9)//3) for n in f.segmentation['nodes'] if n['color']==ch))

def snap6(f):
    o=[]
    for n in f.segmentation['nodes']:
        if n['color'] in ('O','R','b'):
            r,c=n['boundary'][0]; o.append((n['color'],(r-12)//3,(c-12)//3))
    eyes=[]
    for n in f.segmentation['nodes']:
        if n['color']=='p':
            r,c=n['boundary'][0]
            # tile of eye
            tr,tc=(r-12)/3,(c-12)/3
            eyes.append((round(tr,2),round(tc,2)))
    return sorted(o),sorted(eyes)

def row6(f):
    a=f.ascii.split('\n'); r0,c0=12,12
    return ''.join(a[r0+3*6+1][c0+3*j+1] for j in range(13))

def rowmid(f,i):
    a=f.ascii.split('\n'); r0,c0=12,12
    return ''.join(a[r0+3*i+1][c0+3*j+1] for j in range(13))

def plan6():
    a=current_frame.ascii.split('\n'); r0,c0=12,12
    M={}
    for i in range(13):
        for j in range(13):
            ch=a[r0+3*i+1][c0+3*j+1]
            M[(i,j)]=ch
    # base walkable map (ignore current occupants): treat O,R,b,N as walkable tiles
    base={}
    for i in range(13):
        for j in range(13):
            ch=M[(i,j)]
            base[(i,j)]= 'W' if ch in 'WONRb' else ch
    DIRS={'UP':(-1,0),'DOWN':(1,0),'LEFT':(0,-1),'RIGHT':(0,1)}
    def mv(p,k):
        di,dj=DIRS[k]; mid=(p[0]+di,p[1]+dj); q=(p[0]+2*di,p[1]+2*dj)
        if base.get(mid)=='g' and base.get(q,'B')!='B': return q,mid
        return None
    REDS={'R1':((6,10),(1,0)),'R2':((10,4),(-1,0)),'R3':((10,6),(0,-1)),'R4':((12,4),(-1,0)),'R5':((12,6),(-1,0))}
    # front2 = tile + 2*dir
    front={k:(p[0]+2*d[0],p[1]+2*d[1]) for k,(p,d) in REDS.items()}
    redtile={k:p for k,(p,d) in REDS.items()}
    def o1(t):  # relative t, position at time t (t=0 -> now at (0,6) about to move down)
        cyc=[0,2,4,2]
        return (cyc[t%4],6)
    def o2(t):
        # at t=0 at col0 moving right; period 12
        pos=0; d=1
        for _ in range(t):
            nxt=pos+2*d
            if nxt>12 or nxt<0: d=-d; nxt=pos+2*d
            pos=nxt
        return (10,pos)
    def opaths(t):
        out=[]
        for f in (o1,o2):
            p=f(t); q=f(t+1)
            mid=(p[0],(p[1]+q[1])//2) if p[0]==q[0] else ((p[0]+q[0])//2,p[1])
            out.append((p,q,mid))
        return out
    import collections
    start=((6,4),0,frozenset(REDS))
    q=collections.deque([start]); par={start:None}; goal=None
    while q:
        (p,t,alive)=q.popleft()
        if t>40: continue
        op=opaths(t); oend={x[1] for x in op}; omid={x[2] for x in op}; ostart={x[0] for x in op}
        for k in DIRS:
            r=mv(p,k)
            if not r: continue
            n,mid=r; nt=t+1
            if n in oend or n in omid or mid in oend or mid in omid: continue
            npass={p,mid,n}
            if any(npass & {a1,m1,b1} for a1,b1,m1 in op if not (n==a1)): continue
            newalive=set(alive)
            cap=None
            for rk in alive:
                if redtile[rk]==n: cap=rk
            if cap: newalive.discard(cap)
            # lunge check by remaining alive reds
            if any(front[rk]==n for rk in newalive): continue
            st=(n,nt,frozenset(newalive))
            if st in par: continue
            par[st]=((p,t,alive),k,cap)
            if n==(2,0): goal=st; q.clear(); break
            q.append(st)
        if goal: break
    if goal is None: return None
    seq=[]; cur=goal
    while par[cur]:
        p,k,cap=par[cur]; seq.append((k,cur[0],cur[1],sorted(par[cur][0][2]) if cap else [],'CAP'+ (cap or ''))); cur=par[cur][0]
    return seq[::-1]

def snap7(f):
    o=[]
    for n in f.segmentation['nodes']:
        if n['color'] in ('b','R','r','N'):
            r,c=n['boundary'][0]; o.append((n['color'],(r-18)//3,(c-12)//3))
    eyes=[((n['boundary'][0][0]-18)/3,(n['boundary'][0][1]-12)/3) for n in f.segmentation['nodes'] if n['color']=='p']
    return sorted(o),[ (round(x,2),round(y,2)) for x,y in sorted(eyes)]

def eyes7(f):
    out=[]
    for n in f.segmentation['nodes']:
        if n['color'] in ('p','Y','O','W') and n['pixels']==1:
            r,c=n['boundary'][0]
            if n['color'] in ('p','Y'): out.append((n['color'],round((r-18)/3,2),round((c-12)/3,2)))
    return sorted(out)

def tgrid(f,nr=11,nc=11,r0=14,c0=14):
    a=f.ascii.split('\n')
    return [[a[r0+3*i+1][c0+3*j+1] for j in range(nc)] for i in range(nr)]

def snap8(f):
    o=[]
    for n in f.segmentation['nodes']:
        if n['color'] in ('b','R','r','N'):
            rr,cc=n['boundary'][0]; o.append((n['color'],(rr-14)//3,(cc-14)//3))
    eyes=[(n['color'],round((n['boundary'][0][0]-14)/3,2),round((n['boundary'][0][1]-14)/3,2)) for n in f.segmentation['nodes'] if n['color'] in ('p','Y')]
    return sorted(o),sorted(eyes)

def cell(i,j): return G[i][j] if 0<=i<nr and 0<=j<nc else 'B'

def dist(src):
    d={src:0}; q=deque([src])
    while q:
        u=q.popleft()
        for v in ADJ[u].values():
            if v not in d: d[v]=d[u]+1; q.append(v)
    return d

def chase_options(c,pnew,ADJ=ADJ):
    d=dist(pnew); best=min(d[v] for v in ADJ[c].values())
    return sorted(v for v in ADJ[c].values() if d[v]==best)

def copt(c,pnew):
    d=dist(pnew); best=min(d[v] for v in ADJ[c].values())
    return set(v for v in ADJ[c].values() if d[v]==best)

def step(c,pnew,order):
    d=DIST[pnew]; best=min(d[v] for v in ADJ[c].values())
    cands=sorted([(k,v) for k,v in ADJ[c].items() if d[v]==best],key=lambda kv: order.index(kv[0]))
    return cands[0][1]

def win_path(order,P0,C0):
    st=(P0,C0); par={st:None}; q=deque([st])
    while q:
        p,c=q.popleft()
        for k,n in ADJ[p].items():
            if n==c: continue
            cn=cstep(c,n,list(order))
            if cn==n or (n==c and cn==p): continue
            s2=(n,cn)
            if s2 in par: continue
            par[s2]=((p,c),k)
            if n==(0,4):
                seq=[]; cur=s2
                while par[cur]:
                    pr,kk=par[cur]; seq.append(kk); cur=pr
                return seq[::-1]
            q.append(s2)
    return None

def dmap(src):
    d={src:0}; q=deque([src])
    while q:
        u=q.popleft()
        for v in ADJ[u].values():
            if v not in d: d[v]=d[u]+1; q.append(v)
    return d

def cstep(c,pnew,order):
    d=DIST[pnew]; best=min(d[v] for v in ADJ[c].values())
    cands=sorted([(k,v) for k,v in ADJ[c].items() if d[v]==best],key=lambda kv: order.index(kv[0]))
    return cands[0][1]

def cands(c,pnew):
    d=DIST[pnew]; best=min(d[v] for v in ADJ[c].values())
    return [(k,v) for k,v in ADJ[c].items() if d[v]==best]

def run(rule,P0,C0):
    p,c=P0,C0; last=None
    for i,mv in enumerate(seq):
        p2=ADJ[p][mv]
        cs=cands(c,p2)
        pick=rule(cs,last,c,p2)
        if pick==p2 or (p2==c and pick==p): return 'die@%d(P%s C%s->%s)'%(i,p2,c,pick)
        last=[k for k,v in ADJ[c].items() if v==pick][0]
        c=pick; p=p2
    return 'WIN' if p==(0,4) else 'nogoal'

def dmap8(src):
    d={src:0}; q=deque([src])
    while q:
        u=q.popleft()
        for v in ADJ8[u].values():
            if v not in d: d[v]=d[u]+1; q.append(v)
    return d

def cands8(c,pnew):
    d=DIST8[pnew]; best=min(d[v] for v in ADJ8[c].values())
    return [(k,v) for k,v in ADJ8[c].items() if d[v]==best]

def make_rules():
    R=[]
    for o in permutations(['UP','DOWN','LEFT','RIGHT']):
        oo=list(o)
        R.append(('ord:'+''.join(x[0] for x in oo), (lambda cs,last,c,p2,oo=oo: sorted(cs,key=lambda kv: oo.index(kv[0]))[0][1])))
    R.append(('rowfirst',lambda cs,last,c,p2: sorted(cs,key=lambda kv: abs(p2[0]-kv[1][0]))[0][1]))
    R.append(('colfirst',lambda cs,last,c,p2: sorted(cs,key=lambda kv: abs(p2[1]-kv[1][1]))[0][1]))
    R.append(('maxpos',lambda cs,last,c,p2: max(cs,key=lambda kv:(kv[1][0],kv[1][1]))[0][1] if len(cs)>1 else cs[0][1]))
    R.append(('minpos',lambda cs,last,c,p2: min(cs,key=lambda kv:(kv[1][0],kv[1][1]))[0][1]))
    R.append(('momentum',lambda cs,last,c,p2: (sorted([kv for kv in cs if kv[0]==last] or cs)[0][1])))
    R.append(('anti_mom',lambda cs,last,c,p2: (sorted([kv for kv in cs if kv[0]!=last] or cs)[0][1])))
    return R

def replay(rule,Pmoves,C0):
    c=C0; last=None
    for i in range(1,len(Pmoves)):
        p2=Pmoves[i]
        cl=rule(cands8(c,p2),last,c,p2)
        if cl!=Cmoves[i]: return False
        last=[k for k,v in ADJ8[c].items() if v==cl][0]; c=cl
    return True

def rules8():
    R=[]
    for o in permutations(['UP','DOWN','LEFT','RIGHT']):
        oo=list(o); R.append(('ord:'+''.join(x[0] for x in oo),lambda cs,last,c,p2,oo=oo: sorted(cs,key=lambda kv: oo.index(kv[0]))[0][1]))
    R.append(('rowfirst',lambda cs,last,c,p2: sorted(cs,key=lambda kv: abs(p2[0]-kv[1][0]))[0][1]))
    R.append(('colfirst',lambda cs,last,c,p2: sorted(cs,key=lambda kv: abs(p2[1]-kv[1][1]))[0][1]))
    R.append(('maxpos',lambda cs,last,c,p2: max(cs,key=lambda kv:(kv[1][0],kv[1][1]))[0][1]))
    R.append(('minpos',lambda cs,last,c,p2: min(cs,key=lambda kv:(kv[1][0],kv[1][1]))[0][1]))
    R.append(('momentum',lambda cs,last,c,p2: (sorted([kv for kv in cs if kv[0]==last] or cs)[0][1])))
    R.append(('anti_mom',lambda cs,last,c,p2: (sorted([kv for kv in cs if kv[0]!=last] or cs)[0][1])))
    return R

def replay8(f):
    c=Cm[0]; last=None
    for i in range(1,len(Pm)):
        cl=f(cands8(c,Pm[i]),last,c,Pm[i])
        if cl!=Cm[i]: return False
        last=[k for k,v in ADJ8[c].items() if v==cl][0]; c=cl
    return True

def nextpos(f,i,c,p2,last):
    return f(cands8(c,p2),last,c,p2)

def graph9(G):
    nr,nc=len(G),len(G[0])
    def cell(i,j): return G[i][j] if 0<=i<nr and 0<=j<nc else 'B'
    Wn={(i,j) for i in range(nr) for j in range(nc) if cell(i,j) in 'WONRrb'}
    A={p:{} for p in Wn}
    D={'UP':(-1,0),'DOWN':(1,0),'LEFT':(0,-1),'RIGHT':(0,1)}
    for p in sorted(Wn):
        for k,(di,dj) in D.items():
            m=(p[0]+di,p[1]+dj); q=(p[0]+2*di,p[1]+2*dj)
            if cell(*m)=='g' and q in Wn: A[p][k]=q
    return A

def snap9(f):
    o=[];ey=[]
    for n in f.segmentation['nodes']:
        if n['color'] in ('b','R','r','N','O'):
            rr,cc=n['boundary'][0]; o.append((n['color'],round((rr-10)/3,2),round((cc-9)/3,2)))
        elif n['color'] in ('p','Y'):
            rr,cc=n['boundary'][0]; ey.append((n['color'],round((rr-10)/3,2),round((cc-9)/3,2)))
    return sorted(o),sorted(ey)

def face(i,t):
    a=OPS[i]; p0=a[t%PER[i]]; p1=a[(t+1)%PER[i]]
    dr=(p1[0]-p0[0],p1[1]-p0[1])
    for k,(di,dj) in DIRS.items():
        if (di*2,dj*2)==dr: return k
    return None

def bounce(seq,idx,step):
    out=[]; i=idx; s=step
    for t in range(80):
        out.append(seq[i])
        if i+s>=len(seq) or i+s<0: s=-s
        i+=s
    return out

def ocells(i,t):
    a=[O1,O2,O3][i]
    p=a[t%len(a)]; q=a[(t+1)%len(a)]
    cells={p,q}
    if abs(p[0]-q[0])==2 or abs(p[1]-q[1])==2: cells.add(((p[0]+q[0])//2,(p[1]+q[1])//2))
    return cells

def oset(i,t):
    a=OPS[i]; p0=a[t%len(a)]; p1=a[(t+1)%len(a)]
    s={p0,p1}
    if abs(p0[0]-p1[0])==2 or abs(p0[1]-p1[1])==2: s.add(((p0[0]+p1[0])//2,(p0[1]+p1[1])//2))
    return s,s-{p0,p1} if False else ({(p0[0]+p1[0])//2,(p0[1]+p1[1])//2} if (abs(p0[0]-p1[0])==2 or abs(p0[1]-p1[1])==2) else set())

def chaser_at(t, H):
    if t==1: return (4,6)
    if t==2: return (6,6)
    return H[t-2] if t-2 < len(H) else None
