try:
    from collections import deque
except Exception:
    pass
try:
    import json
except Exception:
    pass
try:
    from collections import deque, functools
except Exception:
    pass
try:
    import json, pickle
except Exception:
    pass
try:
    import itertools, json
except Exception:
    pass
try:
    import pickle
except Exception:
    pass
try:
    from collections import Counter
except Exception:
    pass
try:
    import functools
except Exception:
    pass

def player_cell(fr=None):
    fr = fr or current_frame
    for n in fr.segmentation['nodes']:
        if n['color']=='b':
            r0=min(p[0] for p in n['boundary']); c0=min(p[1] for p in n['boundary'])
            return ((r0-15)//3,(c0-15)//3)
    return None

def run_path(moves, start_cell):
    cur = start_cell
    done=[]
    mv={'UP':(-1,0),'DOWN':(1,0),'LEFT':(0,-1),'RIGHT':(0,1)}
    for m in moves:
        res = action([m])
        if res.get('level_completed') or res.get('done') or res.get('game_over'):
            done.append((m,'END:'+str({k:v for k,v in res.items() if k in ('level_completed','game_over','done','run_complete')})))
            return done, cur, res
        nxt = player_cell()
        exp = (cur[0]+mv[m][0], cur[1]+mv[m][1])
        done.append((m, nxt))
        if nxt != exp:
            return done, nxt, {'deviated':True,'expected':exp,'got':nxt}
        cur = nxt
    return done, cur, {'ok':True}

def cell_grid(fr=None):
    fr = fr or current_frame
    G=[['?']*11 for _ in range(11)]
    for n in fr.segmentation['nodes']:
        if n['pixels']>200: continue
        b=n['boundary']
        r0=min(p[0] for p in b); c0=min(p[1] for p in b)
        r1=max(p[0] for p in b); c1=max(p[1] for p in b)
        for r in range(r0,r1+1,3):
            for c in range(c0,c1+1,3):
                rr,cc=(r-15)//3,(c-15)//3
                if 0<=rr<11 and 0<=cc<11: G[rr][cc]=n['color']
    for r in range(11):
        for c in range(11):
            if G[r][c]=='?': G[r][c]='B'
    return G

def bfs(block):
    d={(10,0):0}; q=deque([(10,0)]); prev={}
    while q:
        u=q.popleft()
        for v in ADJ[u]:
            if v in block or v in d: continue
            d[v]=d[u]+1; prev[v]=u; q.append(v)
    return d,prev

def rule2(r,c,dr,dc,G):
    a=(r+dr,c+dc); b=(r+2*dr,c+2*dc)
    if 0<=b[0]<11 and 0<=b[1]<11 and G[a[0]][a[1]]!='B' and G[b[0]][b[1]]!='B': return [b]
    return []

def rule_slide(r,c,dr,dc,G):
    v=(r+dr,c+dc); last=None
    while 0<=v[0]<11 and 0<=v[1]<11 and G[v[0]][v[1]]!='B':
        last=v; v=(v[0]+dr,v[1]+dc)
    return [last] if last else []

def move2(r,c,dr,dc,G):
    a=(r+dr,c+dc); b=(r+2*dr,c+2*dc)
    if 0<=b[0]<11 and 0<=b[1]<11 and G[a[0]][a[1]]!='B' and G[b[0]][b[1]]!='B': return b
    return None

def bfs2(G,start,goal):
    from collections import deque
    prev={start:None}; q=deque([start])
    while q:
        u=q.popleft()
        if u==goal: break
        r,c=u
        for dr,dc,nm in ((-1,0,'UP'),(1,0,'DOWN'),(0,-1,'LEFT'),(0,1,'RIGHT')):
            v=move2(r,c,dr,dc,G)
            if v and v not in prev: prev[v]=u; q.append(v)
    if goal not in prev: return None
    p=[]; u=goal
    while u!=start: p.append((u,prev[u])); u=prev[u]
    return list(reversed(p))

def run2(moves, start):
    cur=start; log=[]
    d={'UP':(-2,0),'DOWN':(2,0),'LEFT':(0,-2),'RIGHT':(0,2)}
    for m in moves:
        res=action([m])
        nxt=player_cell()
        exp=(cur[0]+d[m][0], cur[1]+d[m][1])
        log.append((m,nxt))
        if res.get('level_completed') or res.get('done') or res.get('game_over'):
            return log,cur,{'END':{k:v for k,v in res.items() if k in ('level_completed','game_over','done','run_complete')}}
        if nxt!=exp:
            return log,nxt,{'deviated':True,'expected':exp,'got':nxt,'noop':res.get('no_op')}
        cur=nxt
    return log,cur,{'ok':True}

def grid2(fr=None):
    fr=fr or current_frame
    R0,C0=21,12; NR,NC=5,13
    G=[['.']*NC for _ in range(NR)]
    for n in fr.segmentation['nodes']:
        if n['pixels']>200: continue
        b=n['boundary']
        r0=min(p[0] for p in b); c0=min(p[1] for p in b)
        r1=max(p[0] for p in b); c1=max(p[1] for p in b)
        for r in range(r0,r1+1,3):
            for c in range(c0,c1+1,3):
                rr,cc=(r-R0)//3,(c-C0)//3
                if 0<=rr<NR and 0<=cc<NC: G[rr][cc]=n['color']
    return G

def pcell2(fr=None, org=(21,12)):
    fr=fr or current_frame
    for n in fr.segmentation['nodes']:
        if n['color'] in ('b','c'):
            b=n['boundary']; r0=min(p[0] for p in b); c0=min(p[1] for p in b)
            return ((r0-org[0])//3,(c0-org[1])//3)
    return None

def run_generic(moves, start, org=(21,12)):
    cur=start; log=[]; d={'UP':(-2,0),'DOWN':(2,0),'LEFT':(0,-2),'RIGHT':(0,2)}
    for m in moves:
        res=action([m]); nxt=pcell2(current_frame,org)
        exp=(cur[0]+d[m][0],cur[1]+d[m][1])
        log.append((m,nxt))
        if res.get('level_completed') or res.get('done') or res.get('game_over'):
            return log,cur,{'END':{k:v for k,v in res.items() if k in ('level_completed','game_over','done','run_complete')}}
        if nxt!=exp:
            return log,nxt,{'deviated':True,'expected':exp,'got':nxt,'noop':res.get('no_op'),'grid':'\n'+''.join('\n'+''.join(r) for r in grid2(current_frame))}
        cur=nxt
    return log,cur,{'ok':True}

def snapshot(fr=None):
    G=grid2(fr); pl=pcell2(fr)
    reds=[(i,j) for i,r in enumerate(G) for j,c in enumerate(r) if c in ('R','p','Y')]
    return pl, reds, G

def build(mode,tb,absorb):
    def step(st,a):
        p,(c,cf,lk),tur=st; dr,dc=DIRS[a]; np=(p[0]+2*dr,p[1]+2*dc)
        if np not in ADJ[p]: return None
        if tur and np==TUR:
            if (dr,dc) in PERP[TF]: tur=False
            else: return 'dead'
        elif tur and np==LUNGE: return 'dead'
        if c==np:
            if absorb and (dr,dc) in PERP[cf]: c=None
            else: return 'dead'
        if c is not None and lk:
            tgt = p if mode=='pre' else np
            blocked={TUR} if tur else set()
            dd=dist_to(tgt,blocked)
            if c in dd and dd[c]>0:
                opts=[v for v in ADJ[c] if v not in blocked and v in dd and dd[v]==dd[c]-1]
                if opts:
                    v=sorted(opts,key=tb)[0]
                    if v==tgt or v==np: return 'dead'
                    c=v; cf=((v[0]-c[0])//2,(v[1]-c[1])//2)
        elif c is not None:
            lk=(np[0]==c[0] or np[1]==c[1])
        return (np,(c,cf,lk),tur)
    return step

def cell(r,c,fr=None):
    A=(fr or current_frame).ascii.split('\n')
    return A[18+3*r][12+3*c]

def objs(fr=None):
    A=(fr or current_frame).ascii.split('\n')
    out={'R':[],'O':[],'b':[],'N':[],'p':[]}
    for r in range(16,42):
        for c in range(21,48):
            ch=A[r][c]
            if ch in out: out[ch].append(((r-16)//3,(c-21)//3))
    return {k:sorted(set(v)) for k,v in out.items()}

def walkable(r,c):
    return 0<=r<9 and 0<=c<9 and G[r][c]!='B'

def path_cells(p,d):
    return [(p[0]+d[0]//2,p[1]+d[1]//2), (p[0]+d[0],p[1]+d[1])]

def ok(p): return walkable(p[0],p[1])

W=lambda r,c: 0<=r<NR and 0<=c<NC and G[r][c] not in 'B.'

def run_timed(moves, start, ph, cyc):
    p=start; log=[]
    MD={'UP':(-2,0),'DOWN':(2,0),'LEFT':(0,-2),'RIGHT':(0,2)}
    for m in moves:
        res=action([m])
        o=objs()
        P=o['b'][0] if o['b'] else None; O=o['O'][0] if o['O'] else None
        ph=(ph+1)%6
        exp=(p[0]+MD[m][0], p[1]+MD[m][1])
        log.append((m,P,O))
        if res.get('game_over'): return log,P,{'death':True},ph
        if res.get('level_completed'): return log,P,{'win':True},ph
        if P!=exp: return log,P,{'deviated':True,'expected':exp},ph
        if O!=cyc[ph]: return log,P,{'patroller_deviated':True,'expected':cyc[ph],'got':O},ph
        p=P
    return log,p,{'ok':True},ph

def search(mode,tb,absorb,forbid,maxd=24):
    init=(START,(CH0,CH0F,False),True); q=deque([(init,[])]); seen={init}
    while q:
        st,path=q.popleft()
        if len(path)>maxd: continue
        for a in ['UP','DOWN','LEFT','RIGHT']:
            ns=step(st,a,mode,tb,absorb,forbid)
            if ns is None or ns=='dead': continue
            if ns[0]==GOAL: return path+[a]
            if ns in seen: continue
            seen.add(ns); q.append((ns,path+[a]))
    return None

def run_timed2(moves, start, ph, cyc, mod):
    p=start; log=[]; MD={'UP':(-2,0),'DOWN':(2,0),'LEFT':(0,-2),'RIGHT':(0,2)}
    for m in moves:
        res=action([m]); o=objs()
        P=o['b'][0] if o['b'] else None; O=o['O'][0] if o['O'] else None
        ph=(ph+1)%mod; exp=(p[0]+MD[m][0],p[1]+MD[m][1])
        log.append((m,P,O))
        if res.get('game_over'): return log,P,{'death':True},ph
        if res.get('level_completed'): return log,P,{'win':True},ph
        if P!=exp: return log,P,{'deviated':exp},ph
        if O!=cyc[ph]: return log,P,{'patroller':(cyc[ph],O)},ph
        p=P
    return log,p,{'ok':True},ph

def objs5(fr=None):
    A=(fr or current_frame).ascii.split('\n')
    d={'O':[],'b':[],'N':[]}
    for r in range(14,48):
        for c in range(9,55):
            ch=A[r][c]
            if ch in d: d[ch].append(((r-14)//3,(c-9)//3))
    return {k:sorted(set(v)) for k,v in d.items()}

occ=lambda n: sorted({cyc[n%per] for cyc,per in CYCS})

def occ_path(n):
    s=set()
    for cyc,per in CYCS:
        a=cyc[n%per]; b=cyc[(n+1)%per]
        s.add(a); s.add(b); s.add(((a[0]+b[0])//2,(a[1]+b[1])//2))
    return s

def expected(n): return sorted({cyc[n%per] for cyc,per in CYCS})

opath=lambda n: {x for cyc,per in CYCS for x in (cyc[n%per],cyc[(n+1)%per],(((cyc[n%per][0]+cyc[(n+1)%per][0])//2),((cyc[n%per][1]+cyc[(n+1)%per][1])//2)))}

def solve(maxd=24):
    init=(START,init_ch(),True)
    q=deque([(init,[])]); seen={init}
    while q:
        st,path=q.popleft()
        if len(path)>maxd: continue
        for a in ['UP','DOWN','LEFT','RIGHT']:
            ns=step_all(st,a)
            if ns is None or ns=='dead': continue
            if ns[0]==GOAL: return path+[a]
            if ns in seen: continue
            seen.add(ns); q.append((ns,path+[a]))
    return None,len(seen)

Wl=lambda r,c: 0<=r<NR and 0<=c<NC and G[r][c] not in 'B.'

def occP(n): return {PA[n%4],PB[n%12]}

def occPpath(n):
    s=set()
    for cyc,per in ((PA,4),(PB,12)):
        a=cyc[n%per]; b=cyc[(n+1)%per]
        s|={a,b,((a[0]+b[0])//2,(a[1]+b[1])//2)}
    return s

def o6(fr=None):
    A=(fr or current_frame).ascii.split('\n')
    d={'O':[],'b':[],'N':[],'R':[]}
    for r in range(12,52):
        for c in range(12,52):
            ch=A[r][c]
            if ch in d: d[ch].append(((r-12)//3,(c-12)//3))
    return {k:sorted(set(v)) for k,v in d.items()}

def o7(fr=None):
    A=(fr or current_frame).ascii.split('\n')
    d={'O':[],'b':[],'N':[],'R':[],'r':[]}
    for r in range(18,46):
        for c in range(12,52):
            ch=A[r][c]
            if ch in d: d[ch].append(((r-18)//3,(c-12)//3))
    return {k:sorted(set(v)) for k,v in d.items()}

def eyes7(fr=None):
    out=[]
    for n in (fr or current_frame).segmentation['nodes']:
        if n['pixels']==1 and n['color'] in ('p','c'):
            r,c=n['boundary'][0]; out.append((n['color'],((r-18)//3,(c-12)//3),((r-18)%3,(c-12)%3)))
    return sorted(out)

def allEyes(fr=None):
    out=[]
    for n in (fr or current_frame).segmentation['nodes']:
        if n['pixels']==1 and n['color'] in ('p','c','Y','y','W','O'):
            r,c=n['boundary'][0]; out.append((n['color'],((r-18)//3,(c-12)//3),((r-18)%3,(c-12)%3)))
    return sorted(out)

def bfs_dist(src):
    d={src:0}; q=deque([src])
    while q:
        u=q.popleft()
        for v in ADJ[u]:
            if v not in d: d[v]=d[u]+1; q.append(v)
    return d

def next_step(pos,target):
    # one 2-cell step toward target along shortest path
    d=bfs_dist(target)
    if pos not in d: return pos
    best=None
    for v in ADJ[pos]:
        if v in d and d[v]==d[pos]-1:
            if best is None or (v[0],v[1])<(best[0],best[1]): best=v
    return best if best else pos

def o8(fr=None):
    A=(fr or current_frame).ascii.split('\n')
    d={'b':[],'N':[],'R':[],'r':[]}
    for r in range(14,48):
        for c in range(14,48):
            ch=A[r][c]
            if ch in d: d[ch].append(((r-14)//3,(c-14)//3))
    return {k:sorted(set(v)) for k,v in d.items()}

def eyes8(fr=None):
    out=[]
    for n in (fr or current_frame).segmentation['nodes']:
        if n['pixels']==1 and n['color'] in ('p','c','Y'):
            r,c=n['boundary'][0]; out.append((n['color'],((r-14)//3,(c-14)//3)))
    return sorted(out)

def bfs_steps(src, blocked):
    # dist + prev options from all nodes toward src
    dist={src:0}; q=deque([src])
    while q:
        u=q.popleft()
        for v in ADJ[u]:
            if v==src or v in blocked: continue
            if v not in dist: dist[v]=dist[u]+1; q.append(v)
    return dist

def dist_to(src,blocked):
    dd={src:0}; q=deque([src])
    while q:
        u=q.popleft()
        for v in ADJ[u]:
            if v in blocked or v in dd: continue
            dd[v]=dd[u]+1; q.append(v)
    return dd

def step_belief(bel, np, turret_alive):
    # returns (new_bel_or_None, absorbed_chaser_set_flags) ; None if any branch dies
    out=set()
    blocked={TUR} if turret_alive else set()
    for (cp,cf,lk) in bel:
        if cp==np: return None,None   # player walked into chaser non-perp handled earlier
        if not lk:
            if np[0]==cp[0] or np[1]==cp[1]: out.add((cp,cf,True))
            else: out.add((cp,cf,False))
            continue
        d=dist_to(np,blocked|{cp})
        if cp not in d: out.add((cp,cf,lk)); continue
        opts=[v for v in ADJ[cp] if v not in blocked and v in d and d[v]==d[cp]-1]
        if not opts: out.add((cp,cf,lk)); continue
        for v in opts:
            if v==np: return None,None
            f=(v[0]-cp[0],v[1]-cp[1]); out.add((v,f,lk))
    return out, None

def apply(bel, turret_alive, p, a):
    dr,dc=DIRS[a]; np=(p[0]+dr,p[1]+dc)
    if np not in ADJ[p]: return ('noop',p,bel,turret_alive)
    if turret_alive and np==TUR:
        if (dr,dc) in PERP[TF]: return ('ok',np,bel,False)
        return ('dead',p,bel,turret_alive)
    if turret_alive and np==LUNGE: return ('dead',p,bel,turret_alive)
    # entering chaser cell?
    nb=set(); new_turret=turret_alive; killed=set()
    blocked={TUR} if turret_alive else set()
    for (cp,cf,lk) in bel:
        if cp==np:
            if (dr,dc) in PERP[cf]: killed.add((cp,cf,lk)); continue
            return ('dead',p,bel,turret_alive)
    rest=set(x for x in bel if x not in killed)
    if killed:
        return ('absorb',np,rest,False if False else turret_alive) if False else ('absorb',np,rest,turret_alive)
    res,_=step_belief(bel,np,turret_alive)
    if res is None: return ('dead',p,bel,turret_alive)
    return ('ok',np,res,turret_alive)

def S():
    d=DATA; ADJ={tuple(map(int,k[1:-1].split(', '))):v for k,v in d['ADJ'].items()}; return ADJ

tb1=lambda v:(v[0],v[1])

tb2=lambda v:(-v[0],v[1])

tb3=lambda v:(v[1],v[0])

def sim(cfg, acts, P1,P2,P3):
    model,tie=cfg
    W,A=graph9()
    from collections import deque
    def dd(src):
        d={src:0}; q=deque([src])
        while q:
            u=q.popleft()
            for v in A[u]:
                if v in d: continue
                d[v]=d[u]+1; q.append(v)
        return d
    p=(4,4); c=(2,3); aw=False; ph=[0,0,0]
    MOV={'UP':(-1,0),'DOWN':(1,0),'LEFT':(0,-1),'RIGHT':(0,1)}
    traj=[(p,c,aw,[P1[0],P2[0],P3[0]])]
    for a in acts:
        dr,dc=MOV[a]; np_=(p[0]+dr,p[1]+dc)
        if np_ not in A[p]: return None
        ph=[(ph[0]+1)%len(P1),(ph[1]+1)%len(P2),(ph[2]+1)%len(P3)]
        nn=[P1[ph[0]],P2[ph[1]],P3[ph[2]]]
        if np_==c or np_ in nn: return None
        naw = aw or (np_[0]==c[0] or np_[1]==c[1])
        if aw:
            tgt = p if model=='pre' else np_
            d=dd(tgt)
            if c in d and d[c]>0:
                o=[v for v in A[c] if v in d and d[v]==d[c]-1]
            else: o=[c]
            o.sort(key=lambda x:( tie=='row_asc' and x[0] or -x[0], tie=='col_asc' and x[1] or -x[1]))
            nc=o[0]
        else: nc=c
        if nc==np_ or nc in nn: return None
        p,c,aw=np_,nc,naw
        traj.append((p,c,aw,nn))
    return traj

def dcell(k): return tuple(map(int,k[1:-1].split(',')))

def make(ADJ):
    def dist_to(src,blocked):
        dd={src:0}; q=deque([src])
        while q:
            u=q.popleft()
            for v in ADJ[u]:
                if v in blocked or v in dd: continue
                dd[v]=dd[u]+1; q.append(v)
        return dd
    def advance(bel,np,tur):
        blocked={TUR} if tur else set()
        out=set()
        for (cp,cf,lk) in bel:
            if not lk:
                out.add((cp,cf,True) if (np[0]==cp[0] or np[1]==cp[1]) else (cp,cf,False)); continue
            dd=dist_to(np,blocked)
            if cp not in dd or dd[cp]==0: out.add((cp,cf,lk)); continue
            opts=[v for v in ADJ[cp] if v not in blocked and v in dd and dd[v]==dd[cp]-1]
            if not opts: out.add((cp,cf,lk)); continue
            for v in opts:
                if v==np: return None
                out.add((v,(v[0]-cp[0],v[1]-cp[1]),lk))
        return frozenset(out)
    return dist_to, advance

def apply_adv(st,a,mode,absorb):
    p,bel,tur=st; dr,dc=DIRS[a]; np=(p[0]+2*dr,p[1]+2*dc)
    if np not in ADJ[p]: return None
    if tur and np==TUR:
        if (dr,dc) in PERP[TF]: tur=False
        else: return 'dead'
    elif tur and np==LUNGE: return 'dead'
    keep=set()
    for (cp,cf,lk) in bel:
        if cp==np:
            if absorb and (dr,dc) in PERP[cf]: continue
            return 'dead'
        keep.add((cp,cf,lk))
    tgt = p if mode=='pre' else np
    blocked={TUR} if tur else set(); out=set()
    for (cp,cf,lk) in keep:
        if not lk:
            out.add((cp,cf,True) if (tgt[0]==cp[0] or tgt[1]==cp[1]) else (cp,cf,False)); continue
        dd=dist_to(tgt,blocked)
        if cp not in dd or dd[cp]==0: out.add((cp,cf,lk)); continue
        opts=[v for v in ADJ[cp] if v not in blocked and v in dd and dd[v]==dd[cp]-1]
        if not opts: out.add((cp,cf,lk)); continue
        for v in opts:
            if v==tgt or v==np: return 'dead'
            out.add((v,((v[0]-cp[0])//2,(v[1]-cp[1])//2),lk))
    return (np,frozenset(out),tur)

def solve_adv(maxd=30):
    init=(START,frozenset([(CH0,CH0F,False)]),True)
    q=deque([(init,())]); seen={init}
    while q:
        st,path=q.popleft()
        if len(path)>maxd: continue
        for a in ['UP','DOWN','LEFT','RIGHT']:
            ns=apply_adv(st,a)
            if ns is None or ns=='dead': continue
            if ns[0]==GOAL: return path+(a,)
            if ns in seen: continue
            seen.add(ns); q.append((ns,path+(a,)))
    return None, len(seen)

def advance(bel,tgt,tur):
    blocked={TUR} if tur else set(); out=set()
    for (cp,cf,lk) in bel:
        if not lk:
            out.add((cp,cf,True) if (tgt[0]==cp[0] or tgt[1]==cp[1]) else (cp,cf,False)); continue
        dd=dist_to(tgt,blocked)
        if cp not in dd or dd[cp]==0: out.add((cp,cf,lk)); continue
        opts=[v for v in ADJ[cp] if v not in blocked and v in dd and dd[v]==dd[cp]-1]
        if not opts: out.add((cp,cf,lk)); continue
        for v in opts:
            if v==tgt: return None
            out.add((v,(v[0]-cp[0],v[1]-cp[1]),lk))
    return frozenset(out)

def apply_st(st,a,mode):
    p,bel,tur=st; dr,dc=DIRS[a]; np=(p[0]+dr,p[1]+dc)
    if np not in ADJ[p]: return None
    if tur and np==TUR:
        if (dr,dc) in PERP[TF]: tur=False
        else: return 'dead'
    elif tur and np==LUNGE: return 'dead'
    for (cp,cf,lk) in bel:
        if cp==np: return 'dead'
    tgt = p if mode=='pre' else np
    nb=advance(bel,tgt,tur)
    if nb is None: return 'dead'
    for (cp,cf,lk) in nb:
        if (cp[0],cp[1])==(dr*-1+0,0): pass
        if cp==p and np!=p:  # swapped through player's old cell => crossed
            return 'dead'
    return (np,nb,tur)

def verify(plan, verbose=True):
    p=START; bel=frozenset([(CH0,CH0F,False)]); tur=True
    for i,a in enumerate(plan,1):
        dr,dc=DIRS[a]; np=(p[0]+dr,p[1]+dc)
        if np not in ADJ[p]: return 'step %d %s BLOCKED'%(i,a)
        if tur and np==TUR:
            if (dr,dc) in PERP[TF]: tur=False
            else: return 'step %d front-entry turret'%i
        elif tur and np==LUNGE: return 'step %d lunge cell'%i
        for (cp,cf,lk) in bel:
            if cp==np: return 'step %d walked into chaser'%i
        blocked={TUR} if tur else set(); out=set()
        for (cp,cf,lk) in bel:
            if not lk:
                out.add((cp,cf,True) if (np[0]==cp[0] or np[1]==cp[1]) else (cp,cf,False)); continue
            dd=dist_to(np,blocked)
            if cp not in dd or dd[cp]==0: out.add((cp,cf,lk)); continue
            opts=[v for v in ADJ[cp] if v not in blocked and v in dd and dd[v]==dd[cp]-1]
            if not opts: out.add((cp,cf,lk)); continue
            for v in opts:
                if v==np: return 'step %d %s CAUGHT chaser %s->%s player->%s'%(i,a,cp,v,np)
                out.add((v,(v[0]-cp[0],v[1]-cp[1]),lk))
        bel=frozenset(out); p=np
        if verbose: print(i,a,'P',p,'C',sorted(set(c for c,f,l in bel)),'tur',tur)
        if p==GOAL: return 'WIN at step %d'%i
    return 'no goal, ended at %s'%p

def run(plan,mode,absorb=True,verbose=False):
    st=(START,frozenset([(CH0,CH0F,False)]),True)
    for i,a in enumerate(plan,1):
        ns=apply_adv(st,a,mode,absorb)
        if ns is None: return 'step %d noop'%i
        if ns=='dead': return 'step %d DEAD (%s)'%(i,a)
        st=ns
        if verbose: print(i,a,'P',st[0],'C',sorted(set(c for c,f,l in st[1])),'tur',st[2])
        if st[0]==GOAL: return 'WIN at %d'%i
    return 'no goal'

def step_det(p,cpl,cf,lk,np,tur,mode):
    """single deterministic chaser; returns (c,f,lock) or 'dead'"""
    if lk:
        tgt = p if mode=='pre' else np
        dd=dist_to(tgt,{TUR} if tur else set())
        if cpl in dd and dd[cpl]>0:
            opts=[v for v in ADJ[cpl] if v not in ({TUR} if tur else set()) and v in dd and dd[v]==dd[cpl]-1]
            if opts:
                cands=sorted(opts)
                for v in cands:
                    if v==tgt or v==np: continue
                    return (v,((v[0]-cpl[0])//2,(v[1]-cpl[1])//2),lk)
                return 'dead'
        return (cpl,cf,lk)
    return (cpl,cf, True if (np[0]==cpl[0] or np[1]==cpl[1]) else False)

def trace(mode,tb):
    p=(10,0); c=(2,8); cf=(1,0); lk=False; tur=True; out=[]
    P=['RIGHT','RIGHT','UP','UP','LEFT','RIGHT','RIGHT','RIGHT','LEFT','LEFT','DOWN','DOWN','LEFT','UP','UP','RIGHT','RIGHT','RIGHT','UP','UP','UP','LEFT','LEFT']
    for i,a in enumerate(P,1):
        dr,dc=DIRS[a]; np=(p[0]+2*dr,p[1]+2*dc)
        if tur and np==(6,2): tur=False
        if c==np: out.append((i,a,np,c,'DEATH-walk')); break
        if lk:
            tgt = p if mode=='pre' else np
            dd=dist_to(tgt,set())
            if c in dd and dd[c]>0:
                opts=[v for v in ADJ[c] if v in dd and dd[v]==dd[c]-1]
                if opts:
                    v=sorted(opts,key=TBS[tb])[0]
                    if v==tgt or v==np: out.append((i,a,np,c,'CAUGHT')); break
                    cf=((v[0]-c[0])//2,(v[1]-c[1])//2); c=v
        else:
            lk=(np[0]==c[0] or np[1]==c[1])
        p=np; out.append((i,a,p,c,'lock' if lk else ''))
        if p==(0,4): break
    return out

tb=lambda v:(v[0],v[1])

def init_ch(): return frozenset((m,tb,(CH0,CH0F,False)) for m in MODELS for tb in range(4))

def step_all(st,a):
    p,ch,tur=st; dr,dc=DIRS[a]; np=(p[0]+2*dr,p[1]+2*dc)
    if np not in ADJ[p]: return None
    if tur and np==TUR:
        if (dr,dc) in PERP[TF]: tur=False
        else: return 'dead'
    elif tur and np==LUNGE: return 'dead'
    nch=set()
    for (m,tb,(c,cf,lk)) in ch:
        if c==np: return 'dead'
        if not lk:
            nch.add((m,tb,(c,cf,(np[0]==c[0] or np[1]==c[1])))); continue
        tgt = p if m=='pre' else np
        blocked={TUR} if tur else set()
        dd=dist_to(tgt,blocked)
        if c not in dd or dd[c]==0: nch.add((m,tb,(c,cf,lk))); continue
        opts=[v for v in ADJ[c] if v not in blocked and v in dd and dd[v]==dd[c]-1]
        if not opts: nch.add((m,tb,(c,cf,lk))); continue
        v=sorted(opts,key=TBS[tb])[0]
        if v==tgt or v==np: return 'dead'
        nch.add((m,tb,(v,((v[0]-c[0])//2,(v[1]-c[1])//2),True)))
    return (np,frozenset(nch),tur)

def scan(fr):
    rows=fr.ascii.split('\n')
    def find(ch):
        return [(r,c) for r in range(64) for c in range(64) if rows[r][c]==ch]
    return find('r'),find('b'),find('R'),find('N')

def to_cell(px): return ((px[0]-15)//3,(px[1]-15)//3)

def state8(fr=None):
    fr = fr or current_frame
    rows=fr.ascii.split('\n')
    def find(ch):
        p=[(r,c) for r in range(64) for c in range(64) if rows[r][c]==ch]
        return p
    def cen(p):
        if not p: return None
        return (sum(r for r,c in p)//len(p), sum(c for r,c in p)//len(p))
    pl=cen(find('b')); ch=cen(find('r')); tu=cen(find('R')); gl=cen(find('N'))
    eye=None
    for ch_,nm in [('p','purple'),('Y','yellow'),('O','orange'),('W','white')]:
        pass
    # eye color: look at cell just inside chaser body bottom / any non-red pixel inside its 3x3
    if ch:
        r0,c0=ch[0]-1,ch[1]-1
        for dr in range(3):
            for dc in range(3):
                v=rows[r0+dr][c0+dc]
                if v not in 'r': eye=(v,dr,dc)
    st={'player':to_cell(pl) if pl else None,'chaser':to_cell(ch) if ch else None,
        'turret':to_cell(tu) if tu else None,'goal':to_cell(gl) if gl else None,'eye':eye}
    return st

def to_cell2(px): return ((px[0]-15)//3,(px[1]-15)//3)

def bbox_center(p):
    if not p: return None
    rs=[r for r,c in p]; cs=[c for r,c in p]
    return ((min(rs)+max(rs))//2,(min(cs)+max(cs))//2)

def state9(fr=None):
    fr = fr or current_frame
    rows=fr.ascii.split('\n')
    def find(*chs): return [(r,c) for r in range(64) for c in range(64) if rows[r][c] in chs]
    pl=bbox_center(find('b')); ch=bbox_center(find('r')); tu=bbox_center(find('R')); gl=bbox_center(find('N'))
    eye=None
    if ch:
        r0=ch[0]-1; c0=ch[1]-1
        cells={}
        for dr in range(3):
            for dc in range(3):
                v=rows[r0+dr][c0+dc]
                cells[(dr,dc)]=v
        nonred={k:v for k,v in cells.items() if v!='r'}
        eye=sorted(nonred.items())
    return {'P':to_cell2(pl) if pl else None,'C':to_cell2(ch) if ch else None,
            'T':to_cell2(tu) if tu else None,'G':to_cell2(gl) if gl else None,'eye':eye}

def step(st,a,mode,tb,absorb,forbid):
    p,(c,cf,lk),tur=st; dr,dc=DIRS[a]; np=(p[0]+2*dr,p[1]+2*dc)
    if np not in ADJ[p] or np in forbid: return None
    if tur and np==TUR:
        if absorb and (dr,dc) in PERP[TF]: tur=False
        else: return 'dead'
    elif tur and np==LUNGE: return 'dead'
    if c==np: return 'dead'
    if lk:
        tgt = p if mode=='pre' else np
        blocked={TUR} if tur else set()
        dd=dist_to(tgt,blocked)
        if c in dd and dd[c]>0:
            opts=[v for v in ADJ[c] if v not in blocked and v in dd and dd[v]==dd[c]-1]
            if opts:
                v=sorted(opts,key=tb)[0]
                if v==tgt or v==np: return 'dead'
                return (np,(v,((v[0]-c[0])//2,(v[1]-c[1])//2),True),tur)
        return (np,(c,cf,lk),tur)
    return (np,(c,cf,(np[0]==c[0] or np[1]==c[1])),tur)

def nxt(st,a,mode,tb):
    p,(c,cf,lk),tur=st; dr,dc=DIRS[a]; np=(p[0]+2*dr,p[1]+2*dc)
    if np not in ADJ[p]: return None
    if tur and np==TUR:
        if (dr,dc) in PERP[TF]: tur=False
        else: return 'dead'
    elif tur and np==LUNGE: return 'dead'
    if c==np: return 'dead'
    if lk:
        tgt = p if mode=='pre' else np
        blocked={TUR} if tur else set()
        dd=dist_to(tgt,blocked)
        if c in dd and dd[c]>0:
            opts=[v for v in ADJ[c] if v not in blocked and v in dd and dd[v]==dd[c]-1]
            if opts:
                v=sorted(opts,key=TBS[tb])[0]
                if v==tgt or v==np: return 'dead'
                return (np,(v,((v[0]-c[0])//2,(v[1]-c[1])//2),True),tur)
        return (np,(c,cf,lk),tur)
    return (np,(c,cf,(np[0]==c[0] or np[1]==c[1])),tur)

def belief_plan(models,maxd=24):
    init=((START,(CH0,CH0F,False),True))
    bl=frozenset((init,m,t) for m,t in models)
    q=deque([(bl,[])]); seen={bl}
    while q:
        b,path=q.popleft()
        if len(path)>maxd: continue
        for a in ['UP','DOWN','LEFT','RIGHT']:
            nb=set(); ok=True
            for (st,m,t) in b:
                ns=nxt(st,a,m,t)
                if ns is None or ns=='dead': ok=False; break
                nb.add((ns,m,t))
            if not ok: continue
            if any(s[0]==GOAL for s,m,t in nb): return path+[a]
            k=frozenset(nb)
            if k in seen: continue
            seen.add(k); q.append((k,path+[a]))
    return None

def obs_key(st): return (st[0],st[1][0],st[1][1],st[2])

def choose(st,models):
    res=[]
    for a in ['UP','DOWN','LEFT','RIGHT']:
        nb=set(); ok=True
        for (m,t) in models:
            ns=nxt(st,a,m,t)
            if ns is None or ns=='dead': ok=False; break
            nb.add((ns,m,t))
        if not ok: continue
        groups={}
        for s,m,t in nb: groups.setdefault(obs(s),set()).add((m,t))
        good=True; worst=0; tot=0
        for k,ms in groups.items():
            base=[s for s,m,t in nb if obs(s)==k][0]
            pl=belief_plan(frozenset((base,m,t) for m,t in ms))
            if pl is None: good=False; break
            worst=max(worst,len(pl)); tot+=len(pl)
        if good: res.append((worst,tot,a))
    res.sort(); return res

def obs(st): return (st[0],st[1][0],st[1][1],st[1][2],st[2])

def run_policy(tm,tt,verbose=False):
    st=(START,(CH0,CH0F,False),True); models=list(ALL); path=[]
    for i in range(26):
        r=choose(st,models)
        if not r: return 'stuck at %s after %d'%(st[0],len(path)),path
        a=r[0][2]; ns=nxt(st,a,tm,tt)
        if ns=='dead': return 'DIED step %d (%s) path=%s'%(i+1,a,path),path
        if ns is None: return 'noop step %d'%i,path
        st=ns; path.append(a)
        o=obs(st); models=[(m,t) for (m,t) in models if obs(nxt_ok) if False] if False else models
        # narrow models: keep those for which SOME prior state could produce this obs -- approx: keep models consistent by re-deriving
        if st[0]==GOAL: return 'WIN in %d'%len(path),path
    return 'timeout',path

def bar(fr):
    rows=fr.ascii.split('\n')
    top=rows[0]; bot=rows[63]
    cnt=lambda s: sum(1 for ch in s if ch not in 'B.')
    return cnt(top),cnt(bot),[ (i,ch) for i,ch in enumerate(bot) if ch not in 'B.'][:3]

def nb(fr_rows):
    pts=[(r,c) for r in range(63) for c in range(64) if fr_rows[r][c] not in 'B.']
    rs=[p[0] for p in pts]; cs=[p[1] for p in pts]
    return min(rs),max(rs),min(cs),max(cs)

def fine(r0=1,c0=0,ni=21,nj=21):
    g=[]
    for i in range(ni):
        row=''
        for j in range(nj):
            r=r0+3*i; c=c0+3*j
            if r+2>62 or c+2>63: row+='?'; continue
            blk={A[r+dr][c+dc] for dr in range(3) for dc in range(3)}
            row+= next(iter(blk)) if len(blk)==1 else '#'
        g.append(row)
    return g

def fineblk(i,j,r0=1,c0=0):
    r=r0+3*i; c=c0+3*j
    blk={}
    for dr in range(3):
        for dc in range(3):
            v=A[r+dr][c+dc]; blk[v]=blk.get(v,0)+1
    return blk

def blk(i,j,rows=A):
    r=1+3*i; c=0+3*j
    cols={}
    for dr in range(3):
        for dc in range(3):
            v=rows[r+dr][c+dc]; cols[v]=cols.get(v,0)+1
    m=max(cols,key=cols.get)
    return m

def cellwalk(i,j):
    if i<3 or j<3 or i>15 or j>17: return False
    b=blk(i,j)
    cols={v for row in b for v in row}
    return bool(cols & {'W','g','b','N','R','r','O'})

def st9(fr=None):
    fr=fr or current_frame; rows=fr.ascii.split('\n')
    def b(i,j):
        r=1+3*i; c=0+3*j
        return [[rows[r+dr][c+dc] for dc in range(3)] for dr in range(3)]
    out={}
    for i in range(3,16,2):
        for j in range(3,18,2):
            blk=b(i,j); cols={}
            for row in blk:
                for v in row: cols[v]=cols.get(v,0)+1
            main=max(cols,key=cols.get)
            if main in 'bNrRO':
                eye=[(dr,dc) for dr in range(3) for dc in range(3) if blk[dr][dc]!=main]
                out[((i-3)//2,(j-3)//2)]=(main,cols.get('p',0) and 'p' or [v for k,v in cols.items() if k not in (main,'W','g','c')], eye)
    return out

def mkpat(track,start,dir):
    k=track.index(start)
    step={ 'L':(0,-1),'R':(0,1),'U':(-1,0),'D':(1,0)}[dir]
    d = -1 if track[k+1]!=(track[k][0],track[k][1]+ (1 if dir=='R' else -1)) and dir in 'LR' else (1 if dir in 'R' else (-1 if dir=='L' else None))
    return None

def bounce(track,ki,delta):
    ph=[]; k=ki; d=delta
    for _ in range(2*(len(track)-1)):
        ph.append(track[k]); k+=d
        if k==len(track) or k<0: d=-d; k+=2*d
    return ph

def dists(src,blocked):
    dd={src:0}; q=deque([src])
    while q:
        u=q.popleft()
        for v in ADJ[u]:
            if v in blocked or v in dd: continue
            dd[v]=dd[u]+1; q.append(v)
    return dd

def chaser_opts(c,p,pre,tur_alive):
    tgt = p
    blocked={t[0] for t in TUR if t[0] in tur_alive}
    dd=dists(tgt,blocked)
    if c not in dd: return [c]
    if dd[c]==0: return [c]
    o=[v for v in ADJ[c] if v not in blocked and v in dd and dd[v]==dd[c]-1]
    return o or [c]

def sw_pos(ph1,ph2,ph3): return (P1[ph1%len(P1)],P2[ph2%len(P2)],P3[ph3%len(P3)])

def turret_lane_hit(p,alive):
    # turret lunges 1 cell forward if player is in its ray
    for (t,f),al in zip(TUR,alive):
        if not al: continue
        dr,dc=f; r,c=t
        # ray
        k=1
        while True:
            cell=(r+dr*k,c+dc*k)
            if cell not in walk: break
            if cell==p: return (r+dr,c+dc)==p
            k+=1
    return False

def blkA(i,j,rows=None):
    rows=rows or current_frame.ascii.split('\n')
    r=1+3*i; c=0+3*j
    cols={}
    for dr in range(3):
        for dc in range(3):
            v=rows[r+dr][c+dc]; cols[v]=cols.get(v,0)+1
    return max(cols,key=cols.get)

def inray(p,t):
    (r,c),(dr,dc)=t; k=1
    while True:
        cell=(r+dr*k,c+dc*k)
        if cell not in WALK: return False
        if cell==p: return True
        k+=1

def lunge_kill(p,alive):
    return any(al and inray(p,t) and (t[0][0]+t[1][0],t[0][1]+t[1][1])==p for t,al in zip((T1,T2),alive))

def dd_from(src,blocked):
    dd={src:0}; q=deque([src])
    while q:
        u=q.popleft()
        for v in ADJ[u]:
            if v in blocked or v in dd: continue
            dd[v]=dd[u]+1; q.append(v)
    return dd

def copts(c,p,np_,alive,lock):
    if not lock: return [c]
    blocked={t for t,al in zip((T1[0],T2[0]),alive) if al}
    res=set()
    for tgt in (p,np_):
        dd=dd_from(tgt,blocked)
        if c in dd and dd[c]>0:
            for v in ADJ[c]:
                if v not in blocked and v in dd and dd[v]==dd[c]-1: res.add(v)
        else: res.add(c)
    return sorted(res) or [c]

def absorb_ok(src,dst,alive):
    for t,al in zip((T1,T2),alive):
        if al and dst==t[0] and (dst[0]-src[0],dst[1]-src[1])==t[1]: return False
    return True

def win(state,k):
    p,c,lock,ph1,ph2,ph3,a1,a2=state
    if p==GOAL: return True
    if k<=0: return False
    key=(state,k)
    if key in memo: return memo[key]
    for a,(dr,dc) in MOVES.items():
        np_=(p[0]+dr,p[1]+dc)
        if np_ not in ADJ[p]: continue
        if (a1 and np_==T1[0]) or (a2 and np_==T2[0]):
            if not absorb_ok(p,np_,(a1,a2)): continue
            na1=a1 and np_!=T1[0]; na2=a2 and np_!=T2[0]
        else: na1,na2=a1,a2
        nph=((ph1+1)%len(P1),(ph2+1)%len(P2),(ph3+1)%len(P3))
        n1,n2,n3=P1[nph[0]],P2[nph[1]],P3[nph[2]]
        if np_ in (c,n1,n2,n3): continue
        if lunge_kill(np_,(na1,na2)): continue
        nlock=lock or (np_[0]==c[0] or np_[1]==c[1])
        ok=True
        for nc in copts(c,p,np_,(na1,na2),nlock):
            if nc==np_ or nc in (n1,n2,n3): ok=False; break
            if not win((np_,nc,nlock,nph[0],nph[1],nph[2],na1,na2),k-1): ok=False; break
        if ok:
            memo[key]=True; return True
    memo[key]=False; return False

def best(state,k=24):
    p,c,lock,ph1,ph2,ph3,a1,a2=state
    for a,(dr,dc) in MOVES.items():
        np_=(p[0]+dr,p[1]+dc)
        if np_ not in ADJ[p]: continue
        if (a1 and np_==T1[0]) or (a2 and np_==T2[0]):
            if not absorb_ok(p,np_,(a1,a2)): continue
            na1=a1 and np_!=T1[0]; na2=a2 and np_!=T2[0]
        else: na1,na2=a1,a2
        nph=((ph1+1)%len(P1),(ph2+1)%len(P2),(ph3+1)%len(P3))
        n1,n2,n3=P1[nph[0]],P2[nph[1]],P3[nph[2]]
        if np_ in (c,n1,n2,n3): continue
        if lunge_kill(np_,(na1,na2)): continue
        nlock=lock or (np_[0]==c[0] or np_[1]==c[1])
        if all(nc!=np_ and nc not in (n1,n2,n3) and win((np_,nc,nlock,nph[0],nph[1],nph[2],na1,na2),k-1) for nc in copts(c,p,np_,(na1,na2),nlock)):
            return a,(np_,copts(c,p,np_,(na1,na2),nlock),(na1,na2),nph,nlock)
    return None,None

def conn(r,c,dr,dc):
    return blkA(3+2*r+dr, 3+2*c+dc)!='B'

def graph9():
    from collections import deque
    W=set()
    for i in range(3,16,2):
        for j in range(3,18,2):
            if blkA(i,j)!='B': W.add(((i-3)//2,(j-3)//2))
    A={}
    for (r,c) in W:
        A[(r,c)]=[(r+dr,c+dc) for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]
                  if (r+dr,c+dc) in W and blkA(3+2*r+dr,3+2*c+dc)!='B']
    return W,A

def tracks9():
    S1=[(1,0),(1,1),(1,2),(1,3),(1,4)]; S2=[(1,5),(1,6),(1,7)]; S3=[(0,6),(1,6),(2,6),(3,6),(4,6)]
    return bounce(S1,3,-1),bounce(S2,1,-1),bounce(S3,1,1)

def solve9(state,k):
    from collections import deque
    W,A=graph9(); P1,P2,P3=tracks9()
    GOAL=(5,4); T1=((5,3),(0,1)); T2=((6,3),(-1,0))
    MOV={'UP':(-1,0),'DOWN':(1,0),'LEFT':(0,-1),'RIGHT':(0,1)}
    memo={}
    def inray(p,t):
        (r,c),(dr,dc)=t; n=1
        while True:
            cell=(r+dr*n,c+dc*n)
            if cell not in W: return False
            if cell==p: return True
            n+=1
    def lunk(p,al):
        return any(a and inray(p,t) and (t[0][0]+t[1][0],t[0][1]+t[1][1])==p for t,a in zip((T1,T2),al))
    def dd(src,bl):
        d={src:0}; q=deque([src])
        while q:
            u=q.popleft()
            for v in A[u]:
                if v in bl or v in d: continue
                d[v]=d[u]+1; q.append(v)
        return d
    def co(c,p,np_,al,lock):
        if not lock: return [c]
        bl={t for t,a in zip((T1[0],T2[0]),al) if a}; res=set()
        for tgt in (p,np_):
            d=dd(tgt,bl)
            if c in d and d[c]>0:
                for v in A[c]:
                    if v not in bl and v in d and d[v]==d[c]-1: res.add(v)
            else: res.add(c)
        return sorted(res) or [c]
    def absok(s,d_,al):
        return not any(a and d_==t[0] and (d_[0]-s[0],d_[1]-s[1])==t[1] for t,a in zip((T1,T2),al))
    def win(st,k):
        p,c,lock,ph1,ph2,ph3,a1,a2=st
        if p==GOAL: return True
        if k<=0: return False
        key=(st,k)
        if key in memo: return memo[key]
        for a,(dr,dc) in MOV.items():
            np_=(p[0]+dr,p[1]+dc)
            if np_ not in A[p]: continue
            if (a1 and np_==T1[0]) or (a2 and np_==T2[0]):
                if not absok(p,np_,(a1,a2)): continue
                na1=a1 and np_!=T1[0]; na2=a2 and np_!=T2[0]
            else: na1,na2=a1,a2
            nph=((ph1+1)%len(P1),(ph2+1)%len(P2),(ph3+1)%len(P3))
            n=[P1[nph[0]],P2[nph[1]],P3[nph[2]]]
            if np_ in [c]+n: continue
            if lunk(np_,(na1,na2)): continue
            nl=lock or (np_[0]==c[0] or np_[1]==c[1])
            if all(nc!=np_ and nc not in n and win((np_,nc,nl,nph[0],nph[1],nph[2],na1,na2),k-1) for nc in co(c,p,np_,(na1,na2),nl)):
                memo[key]=True; return True
        memo[key]=False; return False
    if win(state,k):
        p,c,lock,ph1,ph2,ph3,a1,a2=state
        out=[]
        for a,(dr,dc) in MOV.items():
            np_=(p[0]+dr,p[1]+dc)
            if np_ not in A[p]: continue
            if (a1 and np_==T1[0]) or (a2 and np_==T2[0]):
                if not absok(p,np_,(a1,a2)): continue
                na1=a1 and np_!=T1[0]; na2=a2 and np_!=T2[0]
            else: na1,na2=a1,a2
            nph=((ph1+1)%len(P1),(ph2+1)%len(P2),(ph3+1)%len(P3))
            n=[P1[nph[0]],P2[nph[1]],P3[nph[2]]]
            if np_ in [c]+n: continue
            if lunk(np_,(na1,na2)): continue
            nl=lock or (np_[0]==c[0] or np_[1]==c[1])
            if all(nc!=np_ and nc not in n and win((np_,nc,nl,nph[0],nph[1],nph[2],na1,na2),k-1) for nc in co(c,p,np_,(na1,na2),nl)):
                out.append(a)
        return out
    return []

def solve9v(state,k,model,tie,maxnodes=200000):
    from collections import deque
    W,A=graph9(); P1,P2,P3=tracks9()
    GOAL=(5,4); T1=((5,3),(0,1)); T2=((6,3),(-1,0))
    MOV=[('UP',(-1,0)),('DOWN',(1,0)),('LEFT',(0,-1)),('RIGHT',(0,1))]
    def inray(p,t):
        (r,c),(dr,dc)=t; n=1
        while True:
            cell=(r+dr*n,c+dc*n)
            if cell not in W: return False
            if cell==p: return True
            n+=1
    def lunk(p,al):
        return any(a and inray(p,t) and (t[0][0]+t[1][0],t[0][1]+t[1][1])==p for t,a in zip((T1,T2),al))
    def dd(src,bl):
        d={src:0}; q=deque([src])
        while q:
            u=q.popleft()
            for v in A[u]:
                if v in bl or v in d: continue
                d[v]=d[u]+1; q.append(v)
        return d
    def cmove(c,p,np_,al,lock):
        if not lock: return c
        bl={t for t,a in zip((T1[0],T2[0]),al) if a}
        tgt = p if model=='pre' else np_
        d=dd(tgt,bl)
        if c not in d or d[c]==0: return c
        o=[v for v in A[c] if v not in bl and v in d and d[v]==d[c]-1]
        if not o: return c
        o.sort(key=lambda x:( tie=='row_asc' and x[0] or -x[0], tie=='col_asc' and x[1] or -x[1]))
        return o[0]
    def absok(s,d_,al):
        return not any(a and d_==t[0] and (d_[0]-s[0],d_[1]-s[1])==t[1] for t,a in zip((T1,T2),al))
    seen=set(); stack=[(state,())]
    n=0
    while stack:
        st,path=stack.pop()
        p,c,lock,ph1,ph2,ph3,a1,a2=st
        if p==GOAL: return list(path)
        if len(path)>=k: continue
        if st in seen: continue
        seen.add(st); n+=1
        if n>maxnodes: return 'TOOBIG'
        for a,(dr,dc) in MOV:
            np_=(p[0]+dr,p[1]+dc)
            if np_ not in A[p]: continue
            if (a1 and np_==T1[0]) or (a2 and np_==T2[0]):
                if not absok(p,np_,(a1,a2)): continue
                na1=a1 and np_!=T1[0]; na2=a2 and np_!=T2[0]
            else: na1,na2=a1,a2
            nph=((ph1+1)%len(P1),(ph2+1)%len(P2),(ph3+1)%len(P3))
            nn=[P1[nph[0]],P2[nph[1]],P3[nph[2]]]
            if np_ in [c]+nn: continue
            if lunk(np_,(na1,na2)): continue
            nl=lock or (np_[0]==c[0] or np_[1]==c[1])
            nc=cmove(c,p,np_,(na1,na2),nl)
            if nc==np_ or nc in nn: continue
            stack.append(((np_,nc,nl,nph[0],nph[1],nph[2],na1,na2),path+(a,)))
    return None

def dom(i,j):
    r=1+3*i; c=0+3*j
    cc=Counter()
    for dr in range(3):
        for dc in range(3):
            cc[rows[r+dr][c+dc]]+=1
    return cc.most_common(1)[0]

def solve9w(state,k,model,tie,variant,maxnodes=400000):
    from collections import deque
    W,A=graph9(); P1,P2,P3=tracks9()
    GOAL=(5,4); T1=((5,3),(0,1)); T2=((6,3),(-1,0))
    MOV=[('UP',(-1,0)),('DOWN',(1,0)),('LEFT',(0,-1)),('RIGHT',(0,1))]
    def inray(p,t):
        (r,c),(dr,dc)=t; n=1
        while True:
            cell=(r+dr*n,c+dc*n)
            if cell not in W: return False
            if cell==p: return True
            n+=1
    def lunk(p,al):
        return any(a and inray(p,t) and (t[0][0]+t[1][0],t[0][1]+t[1][1])==p for t,a in zip((T1,T2),al))
    def dd(src,bl):
        d={src:0}; q=deque([src])
        while q:
            u=q.popleft()
            for v in A[u]:
                if v in bl or v in d: continue
                d[v]=d[u]+1; q.append(v)
        return d
    def absok(s,d_,al):
        return not any(a and d_==t[0] and (d_[0]-s[0],d_[1]-s[1])==t[1] for t,a in zip((T1,T2),al))
    seen=set(); stack=[(state,())]; n=0
    while stack:
        st,path=stack.pop()
        p,c,lock,ph1,ph2,ph3,a1,a2,chdead=st
        if p==GOAL: return list(path)
        if len(path)>=k: continue
        if st in seen: continue
        seen.add(st); n+=1
        if n>maxnodes: return 'TOOBIG'
        for a,(dr,dc) in MOV:
            np_=(p[0]+dr,p[1]+dc)
            if np_ not in A[p]: continue
            if (a1 and np_==T1[0]) or (a2 and np_==T2[0]):
                if not absok(p,np_,(a1,a2)): continue
                na1=a1 and np_!=T1[0]; na2=a2 and np_!=T2[0]
            else: na1,na2=a1,a2
            nph=((ph1+1)%len(P1),(ph2+1)%len(P2),(ph3+1)%len(P3))
            nn=[P1[nph[0]],P2[nph[1]],P3[nph[2]]]
            if np_ in ([c] if not chdead else [])+nn: continue
            if lunk(np_,(na1,na2)): continue
            nl=lock or (np_[0]==c[0] or np_[1]==c[1])
            nch=chdead
            if not chdead:
                bl={t for t,a_ in zip((T1[0],T2[0]),(na1,na2)) if a_}
                if variant=='wall': bl=set(bl)|set(nn)
                tgt = p if model=='pre' else np_
                d=dd(tgt,bl)
                if c in d and d[c]>0:
                    o=[v for v in A[c] if v not in bl and v in d and d[v]==d[c]-1]
                else: o=[c]
                o.sort(key=lambda x:( tie=='row_asc' and x[0] or -x[0], tie=='col_asc' and x[1] or -x[1]))
                nc=o[0] if o else c
                if variant=='kill' and nc in nn: nch=True
                if not nch and (nc==np_ or nc in nn): continue
            else: nc=c
            stack.append(((np_,nc,nl,nph[0],nph[1],nph[2],na1,na2,nch),path+(a,)))
    return None

def cols_at(fr, letters='brROrN'):
    rows=fr.ascii.split('\n')
    pos={}
    for L in letters:
        pts=[(r,c) for r in range(63) for c in range(64) if rows[r][c]==L]
        if pts: pos[L]= (min(p[0] for p in pts)//6, min(p[1] for p in pts)//6, len(pts))
    return pos

def escape9(k,model,tie,maxnodes=600000):
    from collections import deque
    W,A=graph9(); P1,P2,P3=tracks9()
    MOV=[('UP',(-1,0)),('DOWN',(1,0)),('LEFT',(0,-1)),('RIGHT',(0,1))]
    def dd(src,bl):
        d={src:0}; q=deque([src])
        while q:
            u=q.popleft()
            for v in A[u]:
                if v in bl or v in d: continue
                d[v]=d[u]+1; q.append(v)
        return d
    seen=set(); stack=[(((4,4),(2,3),False,0,0,0),())]; n=0
    while stack:
        (p,c,lock,ph1,ph2,ph3),path=stack.pop()
        if p[0]==0: return list(path)
        if len(path)>=k: continue
        key=(p,c,lock,ph1,ph2,ph3)
        if key in seen: continue
        seen.add(key); n+=1
        if n>maxnodes: return 'BIG'
        for a,(dr,dc) in MOV:
            np_=(p[0]+dr,p[1]+dc)
            if np_ not in A[p]: continue
            nph=((ph1+1)%len(P1),(ph2+1)%len(P2),(ph3+1)%len(P3))
            nn=[P1[nph[0]],P2[nph[1]],P3[nph[2]]]
            if np_ in [c]+nn: continue
            nl=lock or (np_[0]==c[0] or np_[1]==c[1])
            if nl:
                bl=set()
                tgt = p if model=='pre' else np_
                d=dd(tgt,bl)
                if c in d and d[c]>0:
                    o=[v for v in A[c] if v in d and d[v]==d[c]-1]
                else: o=[c]
                o.sort(key=lambda x:( tie=='row_asc' and x[0] or -x[0], tie=='col_asc' and x[1] or -x[1]))
                nc=o[0]
            else: nc=c
            if nc==np_ or nc in nn: continue
            stack.append(((np_,nc,nl,nph[0],nph[1],nph[2]),path+(a,)))
    return None

def esc_adv(k=26):
    from collections import deque
    W,A=graph9(); P1,P2,P3=tracks9()
    MOV={'UP':(-1,0),'DOWN':(1,0),'LEFT':(0,-1),'RIGHT':(0,1)}
    def dd(src):
        d={src:0}; q=deque([src])
        while q:
            u=q.popleft()
            for v in A[u]:
                if v in d: continue
                d[v]=d[u]+1; q.append(v)
        return d
    def co(c,p,np_,lock):
        if not lock: return [c]
        res=set()
        for tgt in (p,np_):
            d=dd(tgt)
            if c in d and d[c]>0:
                for v in A[c]:
                    if v in d and d[v]==d[c]-1: res.add(v)
            else: res.add(c)
        return sorted(res) or [c]
    memo={}
    def win(st,k):
        p,c,lock,ph1,ph2,ph3=st
        if p[0]==0: return True
        if k<=0: return False
        key=(st,k)
        if key in memo: return memo[key]
        for a,(dr,dc) in MOV.items():
            np_=(p[0]+dr,p[1]+dc)
            if np_ not in A[p]: continue
            nph=((ph1+1)%len(P1),(ph2+1)%len(P2),(ph3+1)%len(P3))
            nn=[P1[nph[0]],P2[nph[1]],P3[nph[2]]]
            if np_ in [c]+nn: continue
            nl=lock or (np_[0]==c[0] or np_[1]==c[1])
            if all(nc!=np_ and nc not in nn and win((np_,nc,nl,nph[0],nph[1],nph[2]),k-1) for nc in co(c,p,np_,nl)):
                memo[key]=True; return True
        memo[key]=False; return False
    init=((4,4),(2,3),False,0,0,0)
    if not win(init,k): return None
    # extract a robust policy tree for first few plies
    def tree(st,k,d):
        p,c,lock,ph1,ph2,ph3=st
        out=[]
        for a,(dr,dc) in MOV.items():
            np_=(p[0]+dr,p[1]+dc)
            if np_ not in A[p]: continue
            nph=((ph1+1)%len(P1),(ph2+1)%len(P2),(ph3+1)%len(P3))
            nn=[P1[nph[0]],P2[nph[1]],P3[nph[2]]]
            if np_ in [c]+nn: continue
            nl=lock or (np_[0]==c[0] or np_[1]==c[1])
            opts=co(c,p,np_,nl)
            if all(nc!=np_ and nc not in nn and win((np_,nc,nl,nph[0],nph[1],nph[2]),k-1) for nc in opts):
                out.append((a,[ (nc, str(tuple(nn))) for nc in opts]))
        return out
    return tree(init,k,0)

def solve9r(state,k):
    from collections import deque
    W,A=graph9(); P1,P2,P3=tracks9()
    GOAL=(5,4); T1=((5,3),(0,1)); T2=((6,3),(-1,0))
    MOV=[('UP',(-1,0)),('DOWN',(1,0)),('LEFT',(0,-1)),('RIGHT',(0,1))]
    memo={}
    def inray(p,t):
        (r,c),(dr,dc)=t; n=1
        while True:
            cell=(r+dr*n,c+dc*n)
            if cell not in W: return False
            if cell==p: return True
            n+=1
    def lunk(p,al):
        return any(a and inray(p,t) and (t[0][0]+t[1][0],t[0][1]+t[1][1])==p for t,a in zip((T1,T2),al))
    def dd(src,bl):
        d={src:0}; q=deque([src])
        while q:
            u=q.popleft()
            for v in A[u]:
                if v in bl or v in d: continue
                d[v]=d[u]+1; q.append(v)
        return d
    def co(c,p,np_,al,awake):
        if not awake: return [c]
        bl={t for t,a in zip((T1[0],T2[0]),al) if a}; res=set()
        for tgt in (p,np_):
            d=dd(tgt,bl)
            if c in d and d[c]>0:
                for v in A[c]:
                    if v not in bl and v in d and d[v]==d[c]-1: res.add(v)
            else: res.add(c)
        return sorted(res) or [c]
    def absok(s,d_,al):
        return not any(a and d_==t[0] and (d_[0]-s[0],d_[1]-s[1])==t[1] for t,a in zip((T1,T2),al))
    def win(st,k):
        p,c,aw,ph1,ph2,ph3,a1,a2=st
        if p==GOAL: return True
        if k<=0: return False
        key=(st,k)
        if key in memo: return memo[key]
        for a,(dr,dc) in MOV:
            np_=(p[0]+dr,p[1]+dc)
            if np_ not in A[p]: continue
            if (a1 and np_==T1[0]) or (a2 and np_==T2[0]):
                if not absok(p,np_,(a1,a2)): continue
                na1=a1 and np_!=T1[0]; na2=a2 and np_!=T2[0]
            else: na1,na2=a1,a2
            nph=((ph1+1)%len(P1),(ph2+1)%len(P2),(ph3+1)%len(P3))
            nn=[P1[nph[0]],P2[nph[1]],P3[nph[2]]]
            if np_ in [c]+nn: continue
            if lunk(np_,(na1,na2)): continue
            nsw = aw or (np_[0]==c[0] or np_[1]==c[1])
            if all(nc!=np_ and nc not in nn and win((np_,nc,nsw,nph[0],nph[1],nph[2],na1,na2),k-1) for nc in co(c,p,np_,(na1,na2),aw)):
                memo[key]=True; return True
        memo[key]=False; return False
    if not win(state,k): return None
    p,c,aw,ph1,ph2,ph3,a1,a2=state
    good=[]
    for a,(dr,dc) in MOV:
        np_=(p[0]+dr,p[1]+dc)
        if np_ not in A[p]: continue
        if (a1 and np_==T1[0]) or (a2 and np_==T2[0]):
            if not absok(p,np_,(a1,a2)): continue
            na1=a1 and np_!=T1[0]; na2=a2 and np_!=T2[0]
        else: na1,na2=a1,a2
        nph=((ph1+1)%len(P1),(ph2+1)%len(P2),(ph3+1)%len(P3))
        nn=[P1[nph[0]],P2[nph[1]],P3[nph[2]]]
        if np_ in [c]+nn: continue
        if lunk(np_,(na1,na2)): continue
        nsw = aw or (np_[0]==c[0] or np_[1]==c[1])
        if all(nc!=np_ and nc not in nn and win((np_,nc,nsw,nph[0],nph[1],nph[2],na1,na2),k-1) for nc in co(c,p,np_,(na1,na2),aw)):
            good.append(a)
    return good

def solve9s(state,k,lunge=False):
    from collections import deque
    W,A=graph9(); P1,P2,P3=tracks9()
    GOAL=(5,4); T1=((5,3),(0,1)); T2=((6,3),(-1,0))
    MOV=[('UP',(-1,0)),('DOWN',(1,0)),('LEFT',(0,-1)),('RIGHT',(0,1))]
    memo={}
    def inray(p,t):
        (r,c),(dr,dc)=t; n=1
        while True:
            cell=(r+dr*n,c+dc*n)
            if cell not in W: return False
            if cell==p: return True
            n+=1
    def lunk(p,al):
        return lunge and any(a and inray(p,t) and (t[0][0]+t[1][0],t[0][1]+t[1][1])==p for t,a in zip((T1,T2),al))
    def dd(src,bl):
        d={src:0}; q=deque([src])
        while q:
            u=q.popleft()
            for v in A[u]:
                if v in bl or v in d: continue
                d[v]=d[u]+1; q.append(v)
        return d
    def co(c,p,np_,al,awake):
        if not awake: return [c]
        bl={t for t,a in zip((T1[0],T2[0]),al) if a}; res=set()
        for tgt in (p,np_):
            d=dd(tgt,bl)
            if c in d and d[c]>0:
                for v in A[c]:
                    if v not in bl and v in d and d[v]==d[c]-1: res.add(v)
            else: res.add(c)
        return sorted(res) or [c]
    def absok(s,d_,al):
        return not any(a and d_==t[0] and (d_[0]-s[0],d_[1]-s[1])==t[1] for t,a in zip((T1,T2),al))
    def win(st,k):
        p,c,aw,ph1,ph2,ph3,a1,a2=st
        if p==GOAL: return True
        if k<=0: return False
        key=(st,k)
        if key in memo: return memo[key]
        for a,(dr,dc) in MOV:
            np_=(p[0]+dr,p[1]+dc)
            if np_ not in A[p]: continue
            if (a1 and np_==T1[0]) or (a2 and np_==T2[0]):
                if not absok(p,np_,(a1,a2)): continue
                na1=a1 and np_!=T1[0]; na2=a2 and np_!=T2[0]
            else: na1,na2=a1,a2
            nph=((ph1+1)%len(P1),(ph2+1)%len(P2),(ph3+1)%len(P3))
            nn=[P1[nph[0]],P2[nph[1]],P3[nph[2]]]
            if np_ in [c]+nn: continue
            if lunk(np_,(na1,na2)): continue
            nsw = aw or (np_[0]==c[0] or np_[1]==c[1])
            if all(nc!=np_ and nc not in nn and win((np_,nc,nsw,nph[0],nph[1],nph[2],na1,na2),k-1) for nc in co(c,p,np_,(na1,na2),aw)):
                memo[key]=True; return True
        memo[key]=False; return False
    if not win(state,k): return None
    p,c,aw,ph1,ph2,ph3,a1,a2=state
    good=[]
    for a,(dr,dc) in MOV:
        np_=(p[0]+dr,p[1]+dc)
        if np_ not in A[p]: continue
        if (a1 and np_==T1[0]) or (a2 and np_==T2[0]):
            if not absok(p,np_,(a1,a2)): continue
            na1=a1 and np_!=T1[0]; na2=a2 and np_!=T2[0]
        else: na1,na2=a1,a2
        nph=((ph1+1)%len(P1),(ph2+1)%len(P2),(ph3+1)%len(P3))
        nn=[P1[nph[0]],P2[nph[1]],P3[nph[2]]]
        if np_ in [c]+nn: continue
        if lunk(np_,(na1,na2)): continue
        nsw = aw or (np_[0]==c[0] or np_[1]==c[1])
        if all(nc!=np_ and nc not in nn and win((np_,nc,nsw,nph[0],nph[1],nph[2],na1,na2),k-1) for nc in co(c,p,np_,(na1,na2),aw)):
            good.append(a)
    return good

def det9(state,k,model,tie,maxnodes=400000):
    from collections import deque
    W,A=graph9(); P1,P2,P3=tracks9()
    GOAL=(5,4); T1=((5,3),(0,1)); T2=((6,3),(-1,0))
    MOV=[('UP',(-1,0)),('DOWN',(1,0)),('LEFT',(0,-1)),('RIGHT',(0,1))]
    def dd(src,bl):
        d={src:0}; q=deque([src])
        while q:
            u=q.popleft()
            for v in A[u]:
                if v in bl or v in d: continue
                d[v]=d[u]+1; q.append(v)
        return d
    def absok(s,d_,al):
        return not any(a and d_==t[0] and (d_[0]-s[0],d_[1]-s[1])==t[1] for t,a in zip((T1,T2),al))
    seen=set(); stack=[(state,())]; n=0
    while stack:
        st,path=stack.pop()
        p,c,aw,ph1,ph2,ph3,a1,a2=st
        if p==GOAL: return list(path)
        if len(path)>=k: continue
        if st in seen: continue
        seen.add(st); n+=1
        if n>maxnodes: return 'BIG'
        for a,(dr,dc) in MOV:
            np_=(p[0]+dr,p[1]+dc)
            if np_ not in A[p]: continue
            if (a1 and np_==T1[0]) or (a2 and np_==T2[0]):
                if not absok(p,np_,(a1,a2)): continue
                na1=a1 and np_!=T1[0]; na2=a2 and np_!=T2[0]
            else: na1,na2=a1,a2
            nph=((ph1+1)%len(P1),(ph2+1)%len(P2),(ph3+1)%len(P3))
            nn=[P1[nph[0]],P2[nph[1]],P3[nph[2]]]
            if np_ in [c]+nn: continue
            naw = aw or (np_[0]==c[0] or np_[1]==c[1])
            if aw:
                bl={t for t,a_ in zip((T1[0],T2[0]),(na1,na2)) if a_}
                tgt = p if model=='pre' else np_
                d=dd(tgt,bl)
                if c in d and d[c]>0:
                    o=[v for v in A[c] if v not in bl and v in d and d[v]==d[c]-1]
                else: o=[c]
                o.sort(key=lambda x:( tie=='row_asc' and x[0] or -x[0], tie=='col_asc' and x[1] or -x[1]))
                nc=o[0]
            else: nc=c
            if nc==np_ or nc in nn: continue
            stack.append(((np_,nc,naw,nph[0],nph[1],nph[2],na1,na2),path+(a,)))
    return None

def det9b(state,k,model,tie,maxnodes=400000):
    from collections import deque
    W,A=graph9(); P1,P2,P3=tracks9()
    GOAL=(5,4); T1=((5,3),(0,1)); T2=((6,3),(-1,0))
    MOV=[('UP',(-1,0)),('DOWN',(1,0)),('LEFT',(0,-1)),('RIGHT',(0,1))]
    def dd(src,bl):
        d={src:0}; q=deque([src])
        while q:
            u=q.popleft()
            for v in A[u]:
                if v in bl or v in d: continue
                d[v]=d[u]+1; q.append(v)
        return d
    def front(d_,al):
        return any(a and d_==(t[0][0]+t[1][0],t[0][1]+t[1][1]) for t,a in zip((T1,T2),al))
    def absok(s,d_,al):
        return not any(a and d_==t[0] and (d_[0]-s[0],d_[1]-s[1])==t[1] for t,a in zip((T1,T2),al))
    seen=set(); stack=[(state,())]; n=0
    while stack:
        st,path=stack.pop()
        p,c,aw,ph1,ph2,ph3,a1,a2=st
        if p==GOAL: return list(path)
        if len(path)>=k: continue
        if st in seen: continue
        seen.add(st); n+=1
        if n>maxnodes: return 'BIG'
        for a,(dr,dc) in MOV:
            np_=(p[0]+dr,p[1]+dc)
            if np_ not in A[p]: continue
            if (a1 and np_==T1[0]) or (a2 and np_==T2[0]):
                if not absok(p,np_,(a1,a2)): continue
                na1=a1 and np_!=T1[0]; na2=a2 and np_!=T2[0]
            else: na1,na2=a1,a2
            if front(np_,(na1,na2)): continue
            nph=((ph1+1)%len(P1),(ph2+1)%len(P2),(ph3+1)%len(P3))
            nn=[P1[nph[0]],P2[nph[1]],P3[nph[2]]]
            if np_ in [c]+nn: continue
            naw = aw or (np_[0]==c[0] or np_[1]==c[1])
            if aw:
                bl={t for t,a_ in zip((T1[0],T2[0]),(na1,na2)) if a_}
                tgt = p if model=='pre' else np_
                d=dd(tgt,bl)
                if c in d and d[c]>0:
                    o=[v for v in A[c] if v not in bl and v in d and d[v]==d[c]-1]
                else: o=[c]
                o.sort(key=lambda x:( tie=='row_asc' and x[0] or -x[0], tie=='col_asc' and x[1] or -x[1]))
                nc=o[0]
            else: nc=c
            if nc==np_ or nc in nn: continue
            stack.append(((np_,nc,naw,nph[0],nph[1],nph[2],na1,na2),path+(a,)))
    return None

def pieces(fr=None):
    fr = fr or current_frame
    rows=fr.ascii.split('\n')
    out={}
    for L in 'bOrRN':
        pts=[(r,c) for r in range(64) for c in range(64) if rows[r][c]==L]
        if not pts: out[L]=[]; continue
        groups=[]
        for (r,c) in pts:
            for g in groups:
                if any(abs(r-a)+abs(c-b)<=2 for a,b in g):
                    g.append((r,c)); break
            else: groups.append([(r,c)])
        res=[]
        for g in groups:
            i=(min(p[0] for p in g)-1)//3; j=(min(p[1] for p in g)-0)//3
            # eye: sample center pixel of the 3x3 block
            rr=1+3*i; cc=0+3*j
            eye=rows[rr+1][cc+1]
            res.append(((i,j),eye))
        out[L]=res
    return out

def tocell(fr,pts): return [(((fi-3)//2,(fj-3)//2),e) for (fi,fj),e in pts]

def det9c(state,k,model,tie,P1,maxnodes=400000):
    from collections import deque
    W,A=graph9()
    GOAL=(5,4); T1=((5,3),(0,1)); T2=((6,3),(-1,0))
    MOV=[('UP',(-1,0)),('DOWN',(1,0)),('LEFT',(0,-1)),('RIGHT',(0,1))]
    def dd(src,bl):
        d={src:0}; q=deque([src])
        while q:
            u=q.popleft()
            for v in A[u]:
                if v in bl or v in d: continue
                d[v]=d[u]+1; q.append(v)
        return d
    def front(d_,al):
        return any(a and d_==(t[0][0]+t[1][0],t[0][1]+t[1][1]) for t,a in zip((T1,T2),al))
    def absok(s,d_,al):
        return not any(a and d_==t[0] and (d_[0]-s[0],d_[1]-s[1])==t[1] for t,a in zip((T1,T2),al))
    seen=set(); stack=[(state,())]; n=0
    while stack:
        st,path=stack.pop()
        p,c,aw,ph1,ph2,ph3,a1,a2=st
        if p==GOAL: return list(path)
        if len(path)>=k: continue
        if st in seen: continue
        seen.add(st); n+=1
        if n>maxnodes: return 'BIG'
        for a,(dr,dc) in MOV:
            np_=(p[0]+dr,p[1]+dc)
            if np_ not in A[p]: continue
            if (a1 and np_==T1[0]) or (a2 and np_==T2[0]):
                if not absok(p,np_,(a1,a2)): continue
                na1=a1 and np_!=T1[0]; na2=a2 and np_!=T2[0]
            else: na1,na2=a1,a2
            if front(np_,(na1,na2)): continue
            nph=((ph1+1)%len(P1),(ph2+1)%len(P2),(ph3+1)%len(P3))
            nn=[P1[nph[0]],P2[nph[1]],P3[nph[2]]]
            if np_ in [c]+nn: continue
            naw = aw or (np_[0]==c[0] or np_[1]==c[1])
            if aw:
                bl={t for t,a_ in zip((T1[0],T2[0]),(na1,na2)) if a_}
                tgt = p if model=='pre' else np_
                d=dd(tgt,bl)
                if c in d and d[c]>0:
                    o=[v for v in A[c] if v not in bl and v in d and d[v]==d[c]-1]
                else: o=[c]
                o.sort(key=lambda x:( tie=='row_asc' and x[0] or -x[0], tie=='col_asc' and x[1] or -x[1]))
                nc=o[0]
            else: nc=c
            if nc==np_ or nc in nn: continue
            stack.append(((np_,nc,naw,nph[0],nph[1],nph[2],na1,na2),path+(a,)))
    return None

def esc9c(state,k,model,tie,P1,P2,P3,maxnodes=400000):
    from collections import deque
    W,A=graph9()
    MOV=[('UP',(-1,0)),('DOWN',(1,0)),('LEFT',(0,-1)),('RIGHT',(0,1))]
    def dd(src):
        d={src:0}; q=deque([src])
        while q:
            u=q.popleft()
            for v in A[u]:
                if v in d: continue
                d[v]=d[u]+1; q.append(v)
        return d
    seen=set(); stack=[(state,())]; n=0; best=(0,None)
    while stack:
        st,path=stack.pop()
        p,c,aw,ph1,ph2,ph3=st
        if p[0]==0: return list(path)
        if len(path)>=k: continue
        if st in seen: continue
        seen.add(st); n+=1
        if n>maxnodes: break
        for a,(dr,dc) in MOV:
            np_=(p[0]+dr,p[1]+dc)
            if np_ not in A[p]: continue
            nph=((ph1+1)%len(P1),(ph2+1)%len(P2),(ph3+1)%len(P3))
            nn=[P1[nph[0]],P2[nph[1]],P3[nph[2]]]
            if np_ in [c]+nn: continue
            naw = aw or (np_[0]==c[0] or np_[1]==c[1])
            if aw:
                tgt = p if model=='pre' else np_
                d=dd(tgt)
                if c in d and d[c]>0:
                    o=[v for v in A[c] if v in d and d[v]==d[c]-1]
                else: o=[c]
                o.sort(key=lambda x:( tie=='row_asc' and x[0] or -x[0], tie=='col_asc' and x[1] or -x[1]))
                nc=o[0]
            else: nc=c
            if nc==np_ or nc in nn: continue
            stack.append(((np_,nc,naw,nph[0],nph[1],nph[2]),path+(a,)))
    return None

def hist9():
    acts=[]
    for t in reversed(transitions):
        if t.after_frame is None or t.after_frame.level!=9: continue
        a=str(t.action)
        if a=='RESET': break
        acts.append(a)
    return list(reversed(acts))

def obs9(fr):
    p=pieces(fr); tc=lambda L: tocell(fr,p[L])
    pl=tc('b'); ch=tc('r')
    eye=None
    if ch: eye=ch[0][1]
    return {'p': pl[0][0] if pl else None,
            'c': ch[0][0] if ch else None,
            'awake': (eye=='Y') if eye else None,
            'S': sorted(x[0] for x in tc('O')),
            'R': sorted(x[0] for x in tc('R')),
            'alive_R': len(tc('R'))}

def win_from(state,cfg,P1,P2,P3,k=40,maxnodes=120000):
    from collections import deque
    W,A=graph9()
    GOAL=(5,4); T1=((5,3),(0,1)); T2=((6,3),(-1,0))
    model,tie=cfg
    MOV=[('UP',(-1,0)),('DOWN',(1,0)),('LEFT',(0,-1)),('RIGHT',(0,1))]
    def dd(src,bl):
        d={src:0}; q=deque([src])
        while q:
            u=q.popleft()
            for v in A[u]:
                if v in bl or v in d: continue
                d[v]=d[u]+1; q.append(v)
        return d
    def front(d_,al): return any(a and d_==(t[0][0]+t[1][0],t[0][1]+t[1][1]) for t,a in zip((T1,T2),al))
    def absok(s,d_,al): return not any(a and d_==t[0] and (d_[0]-s[0],d_[1]-s[1])==t[1] for t,a in zip((T1,T2),al))
    seen=set(); stack=[(state,())]; n=0
    while stack:
        st,path=stack.pop()
        p,c,aw,ph1,ph2,ph3,a1,a2=st
        if p==GOAL: return len(path)
        if len(path)>=k: continue
        if st in seen: continue
        seen.add(st); n+=1
        if n>maxnodes: return None
        for a,(dr,dc) in MOV:
            np_=(p[0]+dr,p[1]+dc)
            if np_ not in A[p]: continue
            if (a1 and np_==T1[0]) or (a2 and np_==T2[0]):
                if not absok(p,np_,(a1,a2)): continue
                na1=a1 and np_!=T1[0]; na2=a2 and np_!=T2[0]
            else: na1,na2=a1,a2
            if front(np_,(na1,na2)): continue
            nph=((ph1+1)%len(P1),(ph2+1)%len(P2),(ph3+1)%len(P3))
            nn=[P1[nph[0]],P2[nph[1]],P3[nph[2]]]
            if np_ in [c]+nn: continue
            naw=aw or (np_[0]==c[0] or np_[1]==c[1])
            if aw:
                bl={t for t,a_ in zip((T1[0],T2[0]),(na1,na2)) if a_}
                tgt=p if model=='pre' else np_
                d=dd(tgt,bl)
                if c in d and d[c]>0: o=[v for v in A[c] if v not in bl and v in d and d[v]==d[c]-1]
                else: o=[c]
                o.sort(key=lambda x:( tie=='row_asc' and x[0] or -x[0], tie=='col_asc' and x[1] or -x[1]))
                nc=o[0]
            else: nc=c
            if nc==np_ or nc in nn: continue
            stack.append(((np_,nc,naw,nph[0],nph[1],nph[2],na1,na2),path+(a,)))
    return None

def state_from(cfg,acts,P1,P2,P3):
    tr=sim(cfg,acts,P1,P2,P3)
    if tr is None: return None
    p,c,aw,nn=tr[-1]
    ph=[len(acts)%len(P1),len(acts)%len(P2),len(acts)%len(P3)]
    return (p,c,aw,ph[0],ph[1],ph[2],True,True)

def trk9():
    P1A=[(1,3),(1,2),(1,1),(1,0),(1,1),(1,2),(1,3),(1,4)]
    P1B=[(1,3),(1,2),(1,1),(1,0),(1,1),(1,2)]
    P2 =[(1,6),(1,5),(1,6),(1,7)]
    P3 =[(2,6),(3,6),(4,6),(3,6)]
    return P1A,P1B,P2,P3

def win_from2(state,cfg,P1,P2,P3,k=40,maxnodes=150000,front_rule=True,absorb=True):
    from collections import deque
    W,A=graph9()
    GOAL=(5,4); T1=((5,3),(0,1)); T2=((6,3),(-1,0))
    model,tie=cfg
    MOV=[('UP',(-1,0)),('DOWN',(1,0)),('LEFT',(0,-1)),('RIGHT',(0,1))]
    def dd(src,bl):
        d={src:0}; q=deque([src])
        while q:
            u=q.popleft()
            for v in A[u]:
                if v in bl or v in d: continue
                d[v]=d[u]+1; q.append(v)
        return d
    def front(d_,al): return front_rule and any(a and d_==(t[0][0]+t[1][0],t[0][1]+t[1][1]) for t,a in zip((T1,T2),al))
    def absok(s,d_,al): return not any(a and d_==t[0] and (d_[0]-s[0],d_[1]-s[1])==t[1] for t,a in zip((T1,T2),al))
    seen=set(); stack=[(state,())]; n=0
    while stack:
        st,path=stack.pop()
        p,c,aw,ph1,ph2,ph3,a1,a2=st
        if p==GOAL: return len(path)
        if len(path)>=k: continue
        if st in seen: continue
        seen.add(st); n+=1
        if n>maxnodes: return None
        for a,(dr,dc) in MOV:
            np_=(p[0]+dr,p[1]+dc)
            if np_ not in A[p]: continue
            if absorb and ((a1 and np_==T1[0]) or (a2 and np_==T2[0])):
                if not absok(p,np_,(a1,a2)): continue
                na1=a1 and np_!=T1[0]; na2=a2 and np_!=T2[0]
            else:
                na1,na2=a1,a2
                if absorb and ((a1 and np_==T1[0]) or (a2 and np_==T2[0])): continue
            if front(np_,(na1,na2)): continue
            nph=((ph1+1)%len(P1),(ph2+1)%len(P2),(ph3+1)%len(P3))
            nn=[P1[nph[0]],P2[nph[1]],P3[nph[2]]]
            if np_ in [c]+nn: continue
            naw=aw or (np_[0]==c[0] or np_[1]==c[1])
            if aw:
                bl={t for t,a_ in zip((T1[0],T2[0]),(na1,na2)) if a_} if absorb else set()
                tgt=p if model=='pre' else np_
                d=dd(tgt,bl)
                if c in d and d[c]>0: o=[v for v in A[c] if v not in bl and v in d and d[v]==d[c]-1]
                else: o=[c]
                o.sort(key=lambda x:( tie=='row_asc' and x[0] or -x[0], tie=='col_asc' and x[1] or -x[1]))
                nc=o[0]
            else: nc=c
            if nc==np_ or nc in nn: continue
            stack.append(((np_,nc,naw,nph[0],nph[1],nph[2],na1,na2),path+(a,)))
    return None

def sweeper_only(P1,P2,P3,k=60):
    from collections import deque
    W,A=graph9()
    GOAL=(5,4)
    MOV=[('UP',(-1,0)),('DOWN',(1,0)),('LEFT',(0,-1)),('RIGHT',(0,1))]
    start=((4,4),0,0,0); q=deque([(start,())]); seen={start}
    while q:
        (p,ph1,ph2,ph3),path=q.popleft()
        if p==GOAL: return list(path)
        if len(path)>=k: continue
        for a,(dr,dc) in MOV:
            np_=(p[0]+dr,p[1]+dc)
            if np_ not in A[p]: continue
            nph=((ph1+1)%len(P1),(ph2+1)%len(P2),(ph3+1)%len(P3))
            nn=[P1[nph[0]],P2[nph[1]],P3[nph[2]]]
            if np_ in nn: continue
            st=(np_,nph[0],nph[1],nph[2])
            if st in seen: continue
            seen.add(st); q.append((st,path+(a,)))
    return None

def reach9(P1,P2,P3,k=60):
    from collections import deque
    W,A=graph9()
    MOV=[('UP',(-1,0)),('DOWN',(1,0)),('LEFT',(0,-1)),('RIGHT',(0,1))]
    start=((4,4),0,0,0); q=deque([(start,0)]); seen={start}; cells={}
    while q:
        (p,ph1,ph2,ph3),d=q.popleft()
        cells.setdefault(p,min(cells.get(p,99),d))
        if d>=k: continue
        for a,(dr,dc) in MOV:
            np_=(p[0]+dr,p[1]+dc)
            if np_ not in A[p]: continue
            nph=((ph1+1)%len(P1),(ph2+1)%len(P2),(ph3+1)%len(P3))
            nn=[P1[nph[0]],P2[nph[1]],P3[nph[2]]]
            if np_ in nn: continue
            st=(np_,nph[0],nph[1],nph[2])
            if st in seen: continue
            seen.add(st); q.append((st,d+1))
    return cells

def pat(track,start,dr):
    L=len(track); pos=[]; dirs=[]
    for k in range(2*L):
        i=(start+k)%(2*(L-1)) if L>1 else 0
        j=i if i<L else 2*(L-1)-i
        pos.append(track[j])
    per=len(pos)
    for k in range(per):
        a=pos[k]; b=pos[(k+1)%per]
        dirs.append((b[0]-a[0],b[1]-a[1]))
    return pos,dirs

def win_H2(state,cfg,P1,P2,P3,k=40,maxnodes=200000):
    from collections import deque
    W,A=graph9()
    GOAL=(5,4); T1=((5,3),(0,1)); T2=((6,3),(-1,0))
    model,tie=cfg
    MOV=[('UP',(-1,0)),('DOWN',(1,0)),('LEFT',(0,-1)),('RIGHT',(0,1))]
    S=[pat(*tr) for tr in [P1,P2,P3]]
    def dd(src,bl):
        d={src:0}; q=deque([src])
        while q:
            u=q.popleft()
            for v in A[u]:
                if v in bl or v in d: continue
                d[v]=d[u]+1; q.append(v)
        return d
    def absok(s,d_,al): return not any(a and d_==t[0] and (d_[0]-s[0],d_[1]-s[1])==t[1] for t,a in zip((T1,T2),al))
    seen=set(); stack=[(state,())]; n=0
    while stack:
        st,path=stack.pop()
        p,c,aw,ph1,ph2,ph3,a1,a2,s1,s2,s3=st
        if p==GOAL: return list(path)
        if len(path)>=k: continue
        if st in seen: continue
        seen.add(st); n+=1
        if n>maxnodes: return None
        for a,(dr,dc) in MOV:
            np_=(p[0]+dr,p[1]+dc)
            if np_ not in A[p]: continue
            if (a1 and np_==T1[0]) or (a2 and np_==T2[0]):
                if not absok(p,np_,(a1,a2)): continue
                na1=a1 and np_!=T1[0]; na2=a2 and np_!=T2[0]
            else: na1,na2=a1,a2
            nph=((ph1+1)%len(S[0][0]),(ph2+1)%len(S[1][0]),(ph3+1)%len(S[2][0]))
            nsw=[s1,s2,s3]; blocked=[]
            ok=True
            for idx,(alive,(pos,dirs)) in enumerate(zip([s1,s2,s3],S)):
                if not alive: continue
                i=nph[idx]; cell=pos[i]
                if np_==cell:
                    behind=(cell[0]-dirs[i][0],cell[1]-dirs[i][1])
                    if p==behind: nsw[idx]=False
                    else: ok=False; break
                else: blocked.append(cell)
            if not ok: continue
            if np_==c or np_ in blocked: continue
            naw=aw or (np_[0]==c[0] or np_[1]==c[1])
            if aw:
                bl={t for t,a_ in zip((T1[0],T2[0]),(na1,na2)) if a_}
                tgt=p if model=='pre' else np_
                d=dd(tgt,bl)
                if c in d and d[c]>0: o=[v for v in A[c] if v not in bl and v in d and d[v]==d[c]-1]
                else: o=[c]
                o.sort(key=lambda x:( tie=='row_asc' and x[0] or -x[0], tie=='col_asc' and x[1] or -x[1]))
                nc=o[0]
            else: nc=c
            if nc==np_ or nc in blocked: continue
            stack.append(((np_,nc,naw,nph[0],nph[1],nph[2],na1,na2,nsw[0],nsw[1],nsw[2]),path+(a,)))
    return None

def dd(src):
    d={src:0}; q=deque([src])
    while q:
        u=q.popleft()
        for v in A[u]:
            if v in d: continue
            d[v]=d[u]+1; q.append(v)
    return d
