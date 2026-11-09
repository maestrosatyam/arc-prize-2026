try:
    import json
except Exception:
    pass
try:
    import collections
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
try:
    import heapq
except Exception:
    pass
try:
    import time
except Exception:
    pass

def crop(fr,r0,r1,c0,c1,tag=''):
    b=fr.ascii.split('\n')
    print(tag)
    for r in range(r0,r1+1):
        print(r, b[r][c0:c1+1])

def brief(d):
    out={}
    for k,v in d.items():
        if k=='changed_cell_count': out[k]=v
        else:
            out[k]=[{kk:vv for kk,vv in e.items() if kk in ('at','at_before','at_after','pixels','before_color','after_color','color','hash')} for e in v][:12]
    return out

def trail_extent(fr):
    a = fr.ascii.split('\n')
    pts=[]
    for r in range(10,45):
        row=a[r]
        for c,ch in enumerate(row):
            if ch in 'w g'.replace(' ','') and (r<53):
                pass
    # simpler: collect w/g cells within rows 10..45
    for r in range(10,46):
        for c,ch in enumerate(a[r]):
            if ch in ('w','g'):
                pts.append((r,c,ch))
    return pts

def summarize(fr):
    a=fr.ascii.split('\n')
    # cursor bbox = magenta M cluster rows 30-45
    mr=[r for r in range(44,52) for c in range(64) if a[r][c]=='M']
    # trail cells w/g in rows 12..42
    tr=[(r,c) for r in range(12,43) for c in range(64) if a[r][c] in ('w','g')]
    trv=[(r,c) for r in range(12,43) for c in range(0,17) if a[r][c] in ('w','g')]
    trh=[(r,c) for r in range(12,43) for c in range(17,47) if a[r][c] in ('w','g')]
    # blocks
    blocks=[(a[r][c],r,c) for r in range(12,43) for c in range(17,47) if a[r][c] in 'RNb']
    return dict(cursor_rows=(min(mr),max(mr)) if mr else None, vtrail=(min(x[0] for x in trv),max(x[0] for x in trv)) if trv else None,
                htrail_cols=(min(x[1] for x in trh),max(x[1] for x in trh)) if trh else None,
                htrail_rows=(min(x[0] for x in trh),max(x[0] for x in trh)) if trh else None,
                blocks=sorted(set([(c,r) for c,r,_ in [(b[0],b[1],b[2]) for b in blocks]])))

def curs(fr):
    a=fr.ascii.split('\n')
    pts=[(r,c) for r in range(10,52) for c in range(0,20) if a[r][c]=='M']
    if not pts: return None
    rs=[p[0] for p in pts]; cs=[p[1] for p in pts]
    return (min(rs),max(rs),min(cs),max(cs))

def state(fr):
    a=fr.ascii.split('\n')
    cs=curs(fr)
    beam=[(r,c) for r in range(10,45) for c in range(17,50) if a[r][c] in ('w','g')]
    blocks={}
    for r in range(10,45):
        for c in range(17,50):
            if a[r][c] in 'RNb':
                blocks.setdefault(a[r][c],[]).append((r,c))
    bl={k:(min(p[0] for p in v),max(p[0] for p in v),min(p[1] for p in v),max(p[1] for p in v)) for k,v in blocks.items()}
    bm = (min(p[0] for p in beam),max(p[0] for p in beam),min(p[1] for p in beam),max(p[1] for p in beam)) if beam else None
    return cs, bm, bl

def blocks_of(fr):
    a=fr.ascii.split('\n')
    out={}
    for r in range(10,45):
        for c in range(17,50):
            ch=a[r][c]
            if ch in 'RNb':
                out.setdefault(ch,[]).append(r)
    return {k:(min(v),max(v)) for k,v in out.items()}

def snap(fr):
    a=fr.ascii.split('\n')
    arm=[(r,c) for r in range(10,45) for c in range(17,50) if a[r][c] in ('w','g')]
    am=(min(x[0] for x in arm),max(x[0] for x in arm),min(x[1] for x in arm),max(x[1] for x in arm)) if arm else None
    return curs(fr), am, blocks_of(fr)

def full(fr):
    a=fr.ascii.split('\n')
    out={}
    for r in range(10,45):
        for c in range(0,52):
            ch=a[r][c]
            if ch in 'RNb':
                out.setdefault(ch,[]).append((r,c))
    return {k:(min(p[0] for p in v),max(p[0] for p in v),min(p[1] for p in v),max(p[1] for p in v)) for k,v in out.items()}

def armcols(L): return (17,16+6*L) if L>0 else None

def armrows(ci): return (14+6*ci,15+6*ci)

def push_chain(blocks, idxs, d, axis):
    # blocks: dict name->(ri,cj). move set idxs by d along axis ('r' or 'c'), with cascade, clamp if axis c
    pos={k:v for k,v in blocks.items()}
    order=sorted(idxs, key=lambda k: pos[k][0] if axis=='r' else pos[k][1], reverse=(d>0))
    moved=set()
    def free(p, me):
        for k,v in pos.items():
            if k!=me and v==p: return False
        return True
    def trymove(k,d):
        ri,cj=pos[k]
        if axis=='r':
            nri=ri+d
            if nri<0 or nri>4: return False
            p=(nri,cj)
        else:
            ncj=cj+d
            if axis=='cw':
                if ncj>4: return 'clamp'
                p=(ri,ncj)
            else:
                if ncj<0: return 'clamp'
                p=(ri,ncj)
        # cascade: if occupied, move occupant first
        occ=[k2 for k2,v in pos.items() if k2!=k and v==p]
        for o in occ:
            r=trymove(o,d)
            if r is False: return False
        pos[k]=p; moved.add(k); return True
    ok=True
    for k in order:
        r=trymove(k,d)
        if r is False: ok=False
    return ok, pos, moved

def step(state, act):
    ci,L,blocks = state
    ar=armrows(ci); ac=armcols(L)
    if act=='UP' or act=='DOWN':
        d=-1 if act=='UP' else 1
        if not (0<=ci+d<=4): return None
        nci=ci+d
        nar=armrows(nci)
        swept=(min(ar[0],nar[0]), max(ar[1],nar[1]))
        if L==0: return (nci,L,dict(blocks))
        hit=[k for k,(ri,cj) in blocks.items() if COLS[cj]>=ac[0] and COLS[cj]<=ac[1] and ROWS[ri]<=swept[1] and ROWS[ri]+3>=swept[0]]
        if not hit: return (nci,L,dict(blocks))
        ok,newp,_=push_chain(blocks,hit,d,'r')
        if not ok: return None
        return (nci,L,newp)
    if act=='RIGHT':
        if L>=5: return None
        nL=L+1
        nac=armcols(nL); newcells=(16+6*L+1, 16+6*nL)
        hit=[k for k,(ri,cj) in blocks.items() if ri==ci and COLS[cj]+3>=newcells[0] and COLS[cj]<=newcells[1]]
        if not hit: return (ci,nL,dict(blocks))
        ok,newp,_=push_chain(blocks,hit,1,'cw')
        return (ci,nL,newp)
    if act=='LEFT':
        if L<=0: return None
        nL=L-1
        removed=(17+6*nL, 16+6*L)
        hit=[k for k,(ri,cj) in blocks.items() if ri==ci and COLS[cj]+3>=removed[0] and COLS[cj]<=removed[1]]
        if not hit: return (ci,nL,dict(blocks))
        ok,newp,_=push_chain(blocks,hit,-1,'ccw')
        return (ci,nL,newp)
    return None

def goal(s):
    ci,L,b=s
    if L!=7: return False
    r=ci
    return b[3]==(r,3) and b[2]==(r,4) and b[1]==(r,5) and b[0]==(r,6)

def sim_step(state, act, COLS=(18,24,30,36,42), ROWS=(13,19,25,31,37)):
    ci,L,blocks = state  # blocks tuple (R,b,N) each (ri,cj)
    names=('R','b','N')
    B={names[i]:blocks[i] for i in range(3)}
    ar=(14+6*ci,15+6*ci); ac=(17,16+6*L) if L>0 else None
    def chain(idxs,d,axis):
        pos=dict(B); moved=set()
        def trymove(k,dd,vis):
            if k in vis: return True
            vis=vis|{k}
            ri,cj=pos[k]
            if axis=='r':
                n=ri+d
                if n<0 or n>4: return False
                p=(n,cj)
            else:
                n=cj+d
                if axis=='cw' and n>4: return 'clamp'
                if axis=='ccw' and n<0: return 'clamp'
                p=(ri,n)
            for k2,v in list(pos.items()):
                if k2!=k and v==p:
                    r=trymove(k2,d,vis)
                    if r is False: return False
            pos[k]=p; moved.add(k); return True
        order=sorted(idxs,key=lambda k:pos[k][0] if axis=='r' else pos[k][1],reverse=(d>0))
        ok=True
        for k in order:
            if trymove(k,d,frozenset()) is False: ok=False
        return ok,pos
    if act in ('UP','DOWN'):
        d=-1 if act=='UP' else 1
        if not (0<=ci+d<=4): return None
        nci=ci+d; nar=(14+6*nci,15+6*nci)
        if L==0: return (nci,L,blocks)
        s0=min(ar[0],nar[0]); s1=max(ar[1],nar[1])
        hit=[k for k,(ri,cj) in B.items() if COLS[cj]<=ac[1] and COLS[cj]+3>=ac[0] and ROWS[ri]<=s1 and ROWS[ri]+3>=s0]
        if not hit: return (nci,L,blocks)
        ok,pos=chain(hit,d,'r')
        if not ok: return None
        return (nci,L,tuple(pos[n] for n in names))
    if act=='RIGHT':
        if L>=5: return None
        nL=L+1; ncell=(17+6*L,16+6*nL)
        hit=[k for k,(ri,cj) in B.items() if ri==ci and COLS[cj]+3>=ncell[0] and COLS[cj]<=ncell[1]]
        if not hit: return (ci,nL,blocks)
        ok,pos=chain(hit,1,'cw')
        return (ci,nL,tuple(pos[n] for n in names))
    if act=='LEFT':
        if L<=0: return None
        nL=L-1; rem=(17+6*nL,16+6*L)
        hit=[k for k,(ri,cj) in B.items() if ri==ci and COLS[cj]+3>=rem[0] and COLS[cj]<=rem[1]]
        if not hit: return (ci,nL,blocks)
        ok,pos=chain(hit,-1,'ccw')
        return (ci,nL,tuple(pos[n] for n in names))
    return None

def solve(start, goaltest):
    q=collections.deque([(start,())]); seen={start}
    while q:
        s,p=q.popleft()
        if goaltest(s): return s,p
        for a in ('UP','DOWN','LEFT','RIGHT'):
            ns=sim_step(s,a)
            if ns is None or ns in seen: continue
            seen.add(ns); q.append((ns,p+(a,)))
    return None,len(seen)

def read_state(fr):
    a=fr.ascii.split('\n')
    c=curs(fr); ci=(c[0]-12)//6
    arm=[(r,cc) for r in range(10,45) for cc in range(17,50) if a[r][cc] in ('w','g')]
    L=0 if not arm else (max(x[1] for x in arm)-16)//6
    pos={}
    for r in range(10,45):
        for cc in range(0,52):
            ch=a[r][cc]
            if ch in 'RNb': pos.setdefault(ch,[]).append((r,cc))
    def ij(ch):
        v=pos[ch]; return ((min(p[0] for p in v)-13)//6,(min(p[1] for p in v)-18)//6)
    return (ci,L,(ij('R'),ij('b'),ij('N')))

def sim2(state, act, COLS=(18,24,30,36,42), ROWS=(13,19,25,31,37)):
    ci,L,blocks=state; names=('R','b','N'); B={names[i]:blocks[i] for i in range(3)}
    ar=(14+6*ci,15+6*ci); ac=(17,16+6*L) if L>0 else None
    def chain(idxs,d,axis):
        pos=dict(B); moved=set()
        def trymove(k,vis):
            if k in vis: return False
            vis=vis|{k}
            ri,cj=pos[k]
            if axis=='r': n=(ri+d,cj)
            else:
                nc=cj+d
                if nc<0 or nc>4: return False
                n=(ri,nc)
            if n[0]<0 or n[0]>4: return False
            for k2,v in list(pos.items()):
                if k2!=k and v==n:
                    if not trymove(k2,vis): return False
            pos[k]=n; moved.add(k); return True
        order=sorted(idxs,key=lambda k:pos[k][0] if axis=='r' else pos[k][1],reverse=(d>0))
        allok=True
        for k in order:
            if not trymove(k,frozenset()): allok=False
        return allok,pos
    if act in ('UP','DOWN'):
        d=-1 if act=='UP' else 1
        if not (0<=ci+d<=4): return None
        nci=ci+d; nar=(14+6*nci,15+6*nci)
        if L==0: return (nci,L,blocks)
        s0=min(ar[0],nar[0]); s1=max(ar[1],nar[1])
        hit=[k for k,(ri,cj) in B.items() if COLS[cj]<=ac[1] and COLS[cj]+3>=ac[0] and ROWS[ri]<=s1 and ROWS[ri]+3>=s0]
        if not hit: return (nci,L,blocks)
        ok,pos=chain(hit,d,'r')
        if not ok: return None
        return (nci,L,tuple(pos[n] for n in names))
    if act=='RIGHT':
        if L>=5: return None
        nL=L+1; nc=(17+6*L,16+6*nL)
        hit=[k for k,(ri,cj) in B.items() if ri==ci and COLS[cj]+3>=nc[0] and COLS[cj]<=nc[1]]
        if hit: _,pos=chain(hit,1,'cw'); blocks=tuple(pos[n] for n in names)
        return (ci,nL,blocks)
    if act=='LEFT':
        if L<=0: return None
        nL=L-1; rem=(17+6*nL,16+6*L)
        hit=[k for k,(ri,cj) in B.items() if ri==ci and COLS[cj]+3>=rem[0] and COLS[cj]<=rem[1]]
        if hit: _,pos=chain(hit,-1,'ccw'); blocks=tuple(pos[n] for n in names)
        return (ci,nL,blocks)
    return None

def solve2(start,goaltest,sim=sim2):
    q=collections.deque([(start,())]); seen={start}
    while q:
        s,p=q.popleft()
        if goaltest(s): return p
        for a in ('UP','DOWN','LEFT','RIGHT'):
            ns=sim(s,a)
            if ns is None or ns in seen: continue
            seen.add(ns); q.append((ns,p+(a,)))
    return None

g=lambda st: st[2]==((0,0),(0,2),(0,1)) and st[0]==0 and st[1]==3

def sim3(state,act,N=7,MAXL=7):
    ci,L,blocks=state; names=('N','b','O','R'); B={names[i]:blocks[i] for i in range(4)}
    ar=(8+6*ci,9+6*ci)
    def chain(idxs,d,axis):
        pos=dict(B); 
        def tm(k,vis):
            if k in vis: return False
            vis=vis|{k}; ri,cj=pos[k]
            if axis=='r':
                nr=ri+d
                if nr<0 or nr>N-1: return False
                n=(nr,cj)
            else:
                nc=cj+d
                if nc<0 or nc>N-1: return False
                n=(ri,nc)
            for k2,v in list(pos.items()):
                if k2!=k and v==n:
                    if not tm(k2,vis): return False
            pos[k]=n; return True
        order=sorted(idxs,key=lambda k:pos[k][0] if axis=='r' else pos[k][1],reverse=(d>0))
        allok=True
        for k in order:
            if not tm(k,frozenset()): allok=False
        return allok,pos
    if act in('UP','DOWN'):
        d=-1 if act=='UP' else 1
        if not(0<=ci+d<N): return None
        nci=ci+d; nar=(8+6*nci,9+6*nci)
        if L==0: return (nci,L,blocks)
        s0=min(ar[0],nar[0]); s1=max(ar[1],nar[1])
        hit=[k for k,(ri,cj) in B.items() if 12+6*cj<=10+6*L and 15+6*cj>=10 and 7+6*ri<=s1 and 10+6*ri>=s0]
        if not hit: return (nci,L,blocks)
        ok,pos=chain(hit,d,'r')
        if not ok: return None
        return (nci,L,tuple(pos[n] for n in names))
    if act=='RIGHT':
        if L>=MAXL: return None
        nL=L+1; n0,n1=11+6*L,16+6*L
        hit=[k for k,(ri,cj) in B.items() if ri==ci and 12+6*cj<=n1 and 15+6*cj>=n0]
        if hit:
            _,pos=chain(hit,1,'c'); blocks=tuple(pos[n] for n in names)
        return (ci,nL,blocks)
    if act=='LEFT':
        if L<=0: return None
        nL=L-1; n0,n1=5+6*L,10+6*L
        hit=[k for k,(ri,cj) in B.items() if ri==ci and 12+6*cj<=n1 and 15+6*cj>=n0]
        if hit:
            _,pos=chain(hit,-1,'c'); blocks=tuple(pos[n] for n in names)
        return (ci,nL,blocks)
    return None

def read2(fr):
    a=fr.ascii.split('\n')
    c=None
    for r in range(0,64):
        for cc in range(0,12):
            if a[r][cc]=='M': c=(r,cc); break
        if c: break
    ci=(c[0]-6)//6
    arm=[(r,cc) for r in range(6,48) for cc in range(10,54) if a[r][cc] in ('w','g')]
    L=0 if not arm else (max(x[1] for x in arm)-10)//6
    pos={}
    for r in range(6,48):
        for cc in range(11,53):
            ch=a[r][cc]
            if ch in 'NbOR': pos.setdefault(ch,[]).append((r,cc))
    def ij(ch):
        v=pos[ch]; return ((min(p[0] for p in v)-7)//6,(min(p[1] for p in v)-12)//6)
    return (ci,L,tuple(ij(k) for k in 'NbOR'))

def bfs2(start,goaltest,ROWS=5,CAP=400000):
    from collections import deque
    MAXL=7
    def enc(s):
        ci,L,b=s; v=ci*8+L
        for i in range(4): v=(v<<6)|(b[i][0]*7+b[i][1])
        return v
    def dec(v):
        b=[]
        for i in range(3,-1,-1): b.append(((v>>6*i)&63)); 
        # careful ordering
        return None
    def step(s,act):
        ci,L,blocks=s; names=(0,1,2,3); B={i:blocks[i] for i in range(4)}
        ar=(8+6*ci,9+6*ci)
        def chain(idxs,d,axis):
            pos=dict(B); 
            def tm(k,vis):
                if k in vis: return False
                vis=vis|{k}; ri,cj=pos[k]
                if axis=='r':
                    nr=ri+d
                    if nr<0 or nr>=ROWS: return False
                    n=(nr,cj)
                else:
                    nc=cj+d
                    if nc<0 or nc>6: return False
                    n=(ri,nc)
                for k2,v2 in list(pos.items()):
                    if k2!=k and v2==n:
                        if not tm(k2,vis): return False
                pos[k]=n; return True
            order=sorted(idxs,key=lambda k:pos[k][0] if axis=='r' else pos[k][1],reverse=(d>0))
            allok=True
            for k in order:
                if not tm(k,frozenset()): allok=False
            return allok,pos
        if act in (0,1):
            d=-1 if act==0 else 1
            if not(0<=ci+d<ROWS): return None
            nci=ci+d; nar=(8+6*nci,9+6*nci)
            if L==0: return (nci,L,blocks)
            s0=min(ar[0],nar[0]); s1=max(ar[1],nar[1])
            hit=[k for k,(ri,cj) in B.items() if 12+6*cj<=10+6*L and 15+6*cj>=10 and 7+6*ri<=s1 and 10+6*ri>=s0]
            if not hit: return (nci,L,blocks)
            ok,pos=chain(hit,d,'r')
            if not ok: return None
            return (nci,L,tuple(pos[i] for i in names))
        if act==2:
            if L>=MAXL: return None
            nL=L+1; n0,n1=11+6*L,16+6*L
            hit=[k for k,(ri,cj) in B.items() if ri==ci and 12+6*cj<=n1 and 15+6*cj>=n0]
            if hit:
                _,pos=chain(hit,1,'c'); blocks=tuple(pos[i] for i in names)
            return (ci,nL,blocks)
        nL=L-1; n0,n1=5+6*L,10+6*L
        hit=[k for k,(ri,cj) in B.items() if ri==ci and 12+6*cj<=n1 and 15+6*cj>=n0]
        if hit:
            _,pos=chain(hit,-1,'c'); blocks=tuple(pos[i] for i in names)
        return (ci,nL,blocks)
    start=tuple(start)
    q=deque([(start,())]); seen={enc(start)}; popped=0
    while q:
        s,p=q.popleft(); popped+=1
        if popped>CAP: return ('cap',popped,len(seen))
        if goaltest(s): return (p,popped,len(seen))
        for a in range(4):
            ns=step(s,a)
            if ns is None: continue
            e=enc(ns)
            if e in seen: continue
            seen.add(e); q.append((ns,p+(a,)))
    return ('exhaust',popped,len(seen))

def runplan(acts,s,limit=None):
    for i,a in enumerate(acts):
        if limit and i>=limit: print('limit reached at',i); return s,i,'limit'
        pred=sim3(s,a)
        if pred is None: print('SIM-ILLEGAL',i,a,s); return s,i,'illegal'
        r=action([a]) or {}
        ns=read2(current_frame)
        if pred!=ns: print('DIVERGE',i,a,'pred',pred,'real',ns); return ns,i,'diverge'
        s=ns
        if r.get('level_completed') or r.get('game_over') or r.get('run_complete') or r.get('done'):
            print('END',i,a,{k:v for k,v in r.items() if k in('level_completed','game_over','run_complete','done')})
            return s,i,'end'
    return s,len(acts),'ok'

def rest_plan():
    plan=[('LEFT',1),('DOWN',3),
          ('DOWN',1),('RIGHT',3),('UP',3),('LEFT',3),('UP',1),('RIGHT',4),('LEFT',4),('DOWN',3),
          ('DOWN',1),('RIGHT',4),('UP',3),('LEFT',4),('UP',1),('RIGHT',7)]
    acts=[]
    for a,n in plan: acts+=[a]*n
    return acts

def read3(fr):
    a=fr.ascii.split('\n')
    c=None
    for r in range(0,64):
        for cc in range(0,12):
            if a[r][cc]=='M': c=(r,cc); break
        if c: break
    ci=(c[0]-6)//6
    r0,r1=8+6*ci,9+6*ci
    L=0
    for cc in range(10,54):
        if a[r0][cc] in ('w','W','g','G') or a[r1][cc] in ('w','W','g','G'): L=(cc-10)//6
        else: break
    pos={}
    for r in range(6,48):
        for cc in range(11,53):
            ch=a[r][cc]
            if ch in 'NbOR': pos.setdefault(ch,[]).append((r,cc))
    def ij(ch):
        v=pos[ch]; return ((min(p[0] for p in v)-7)//6,(min(p[1] for p in v)-12)//6)
    return (ci,L,tuple(ij(k) for k in 'NbOR'))

def read4(fr):
    a=fr.ascii.split('\n')
    c=None
    for r in range(0,64):
        for cc in range(0,12):
            if a[r][cc]=='M': c=(r,cc); break
        if c: break
    ci=(c[0]-6)//6
    r0,r1=8+6*ci,9+6*ci
    last=-1; cc=10
    while cc<54:
        if a[r0][cc] in ('w','W','g','G') or a[r1][cc] in ('w','W','g','G'): last=cc; cc+=1
        elif a[r0][cc] in 'NbOR' or a[r1][cc] in 'NbOR': cc+=1
        else: break
    L=0 if last<10 else (last-10)//6
    pos={}
    for r in range(6,48):
        for cc2 in range(11,53):
            ch=a[r][cc2]
            if ch in 'NbOR': pos.setdefault(ch,[]).append((r,cc2))
    def ij(ch):
        v=pos.get(ch)
        if not v: return None
        return ((min(p[0] for p in v)-7)//6,(min(p[1] for p in v)-12)//6)
    return (ci,L,tuple(ij(k) for k in 'NbOR'))

def sim4(state,act,N=7,MAXL=7):
    ci,L,blocks=state; nm=('N','b','O','R'); B={nm[i]:blocks[i] for i in range(4)}
    ar=(8+6*ci,9+6*ci)
    def rail(k): 
        r,c=B[k]; return (c==3 and r<=3)
    def chain(idxs,d,axis):
        pos=dict(B)
        def tm(k,vis):
            if k in vis: return False
            vis=vis|{k}; ri,cj=pos[k]
            if axis=='r':
                nr=ri+d
                if nr<0 or nr>=N: return False
                n=(nr,cj)
            else:
                if rail(k): return False
                nc=cj+d
                if nc<0 or nc>=N: return False
                n=(ri,nc)
            for k2,v in list(pos.items()):
                if k2!=k and v==n:
                    if not tm(k2,vis): return False
            pos[k]=n; return True
        order=sorted(idxs,key=lambda k:pos[k][0] if axis=='r' else pos[k][1],reverse=(d>0))
        ok=True
        for k in order:
            if not tm(k,frozenset()): ok=False
        return ok,pos
    if act in('UP','DOWN'):
        d=-1 if act=='UP' else 1
        if not(0<=ci+d<N): return None
        nci=ci+d; nar=(8+6*nci,9+6*nci)
        if L==0: return (nci,L,blocks)
        s0=min(ar[0],nar[0]); s1=max(ar[1],nar[1])
        hit=[k for k,(ri,cj) in B.items() if 12+6*cj<=10+6*L and 15+6*cj>=10 and 7+6*ri<=s1 and 10+6*ri>=s0]
        if not hit: return (nci,L,blocks)
        ok,pos=chain(hit,d,'r')
        if not ok: return None
        return (nci,L,tuple(pos[n] for n in nm))
    if act=='RIGHT':
        if L>=MAXL: return None
        nL=L+1; n0,n1=11+6*L,16+6*L
        hit=[k for k,(ri,cj) in B.items() if ri==ci and 12+6*cj<=n1 and 15+6*cj>=n0]
        if hit:
            _,pos=chain(hit,1,'c'); blocks=tuple(pos[n] for n in nm)
        return (ci,nL,blocks)
    if act=='LEFT':
        if L<=0: return None
        nL=L-1; n0,n1=5+6*L,10+6*L
        hit=[k for k,(ri,cj) in B.items() if ri==ci and 12+6*cj<=n1 and 15+6*cj>=n0]
        if hit:
            _,pos=chain(hit,-1,'c'); blocks=tuple(pos[n] for n in nm)
        return (ci,nL,blocks)
    return None

def sim5(state,act,N=7,MAXL=7):
    ci,L,blocks=state; nm=('N','b','O','R'); pos={nm[i]:blocks[i] for i in range(4)}
    def locked(p): r,c=p; return (c==3 and r<=3)
    def move_chain(keys,d,axis):
        p=dict(pos)
        def tm(k,vis):
            if k in vis: return False
            vis=vis|{k}; r,c=p[k]
            nr,nc=(r+d,c) if axis=='r' else (r,c+d)
            if not(0<=nr<N and 0<=nc<N): return False
            if axis=='c' and locked(p[k]): return False
            tgt=(nr,nc)
            for k2,v in list(p.items()):
                if k2!=k and v==tgt:
                    if not tm(k2,vis): return False
            p[k]=tgt; return True
        ok=all(tm(k,frozenset()) for k in keys)
        return ok,p
    if act in ('UP','DOWN'):
        d=-1 if act=='UP' else 1
        if not(0<=ci+d<N): return None
        hc=(ci,L-1) if L>0 else None
        if hc is not None:
            hit=[k for k,v in pos.items() if v==hc]
            if hit:
                ok,p=move_chain(hit,d,'r')
                if not ok: return None
                pos=p
        return (ci+d,L,tuple(pos[n] for n in nm))
    if act=='RIGHT':
        if L>=MAXL: return None
        cells=[(ci,L-1),(ci,L)] if L>0 else [(ci,0)]
        hit=[k for k,v in pos.items() if v in cells]
        if hit:
            ok,p=move_chain(hit,1,'c'); pos=p
        return (ci,L+1,tuple(pos[n] for n in nm))
    if act=='LEFT':
        if L<=0: return None
        hc=(ci,L-1); hit=[k for k,v in pos.items() if v==hc]
        if hit:
            ok,p=move_chain(hit,-1,'c'); pos=p
        return (ci,L-1,tuple(pos[n] for n in nm))
    return None

def st_from(fr):
    return read4(fr)

def dbg(state,act):
    ci,L,blocks=state; nm=('N','b','O','R'); pos={nm[i]:blocks[i] for i in range(4)}
    print('pos',pos)
    hc=(ci,L-1); print('hc',hc)
    hit=[k for k,v in pos.items() if v==hc]; print('hit',hit)
    return hit

def dbgmove(state):
    ci,L,blocks=state; nm=('N','b','O','R'); pos={nm[i]:blocks[i] for i in range(4)}
    N=7
    hc=(ci,L-1)
    hit=[k for k,v in pos.items() if v==hc]
    print('pos',pos,'hc',hc,'hit',hit)
    p=dict(pos)
    def locked(q): r,c=q; return (c==3 and r<=3)
    def tm(k,vis):
        if k in vis: return False
        vis=vis|{k}; r,c=p[k]
        nr,nc=(r-1,c) if False else (r,c-1)
        if not(0<=nr<N and 0<=nc<N): print('oob',k); return False
        if locked(p[k]): print('locked',k); return False
        tgt=(nr,nc)
        for k2,v in list(p.items()):
            if k2!=k and v==tgt:
                if not tm(k2,vis): print('blocked',k,'by',k2); return False
        p[k]=tgt; return True
    ok=all(tm(k,frozenset()) for k in hit)
    print('ok',ok,'p',p)

def railset(fr):
    a=fr.ascii.split('\n')
    cnt={}
    for c in range(11,53):
        n=0
        for r in range(4,52):
            if a[r][c] in ('G','g'): n+=1
        cnt[c]=n
    cols=[c for c,n in cnt.items() if n>=8]
    cells=set()
    for c in cols:
        for r in range(4,52):
            if a[r][c] in ('G','g'): cells.add((r,c))
    return frozenset(cells),tuple(cols)

def sim6(state,act,rail,N=7,MAXL=7):
    ci,L,blocks=state; nm=('N','b','O','R'); pos={nm[i]:blocks[i] for i in range(4)}
    def locked(q):
        r,c=q
        for rr in range(7+6*r,11+6*r):
            for cc in range(12+6*c,16+6*c):
                if (rr,cc) in rail: return True
        return False
    def move_chain(keys,d,axis):
        p=dict(pos)
        def tm(k,vis):
            if k in vis: return False
            vis=vis|{k}; r,c=p[k]
            nr,nc=(r+d,c) if axis=='r' else (r,c+d)
            if not(0<=nr<N and 0<=nc<N): return False
            if axis=='c' and locked(p[k]): return False
            tgt=(nr,nc)
            for k2,v in list(p.items()):
                if k2!=k and v==tgt:
                    if not tm(k2,vis): return False
            p[k]=tgt; return True
        ok=all(tm(k,frozenset()) for k in keys)
        return ok,p
    if act in ('UP','DOWN'):
        d=-1 if act=='UP' else 1
        if not(0<=ci+d<N): return None
        if L>0:
            hc=(ci,L-1); hit=[k for k,v in pos.items() if v==hc]
            if hit:
                ok,p=move_chain(hit,d,'r')
                if not ok: return None
                pos=p
        return (ci+d,L,tuple(pos[n] for n in nm))
    if act=='RIGHT':
        if L>=MAXL: return None
        cells=[(ci,L-1),(ci,L)] if L>0 else [(ci,0)]
        hit=[k for k,v in pos.items() if v in cells]
        if hit:
            ok,p=move_chain(hit,1,'c'); pos=p
        return (ci,L+1,tuple(pos[n] for n in nm))
    if act=='LEFT':
        if L<=0: return None
        hc=(ci,L-1); hit=[k for k,v in pos.items() if v==hc]
        if hit:
            ok,p=move_chain(hit,-1,'c'); pos=p
        return (ci,L-1,tuple(pos[n] for n in nm))
    return None

def railset2(fr):
    a=fr.ascii.split('\n')
    cols=[]
    for c in range(11,53):
        n=sum(1 for r in range(4,52) if a[r][c] in ('G','g'))
        if n>=8: cols.append(c)
    cells=set()
    for c in cols:
        rs=[r for r in range(4,52) if a[r][c] in ('G','g')]
        for r in range(min(rs),max(rs)+1): cells.add((r,c))
    return frozenset(cells),tuple(cols)

def sim7(state,act,rail,N=7,MAXL=7):
    ci,L,blocks=state; nm=('N','b','O','R'); pos={nm[i]:blocks[i] for i in range(4)}
    def locked(q):
        r,c=q
        return any((rr,cc) in rail for rr in range(7+6*r,11+6*r) for cc in range(12+6*c,16+6*c))
    def chain(keys,d,axis):
        p=dict(pos)
        def tm(k,vis):
            if k in vis: return False
            vis=vis|{k}; r,c=p[k]
            nr,nc=(r+d,c) if axis=='r' else (r,c+d)
            if not(0<=nr<N and 0<=nc<N): return False
            if axis=='c' and locked(p[k]): return False
            tgt=(nr,nc)
            for k2,v in list(p.items()):
                if k2!=k and v==tgt:
                    if not tm(k2,vis): return False
            p[k]=tgt; return True
        ok=all(tm(k,frozenset()) for k in keys)
        return ok,p
    if act in ('UP','DOWN'):
        d=-1 if act=='UP' else 1
        if not(0<=ci+d<N): return None
        if L>0:
            s0=min(8+6*ci,8+6*(ci+d)); s1=max(9+6*ci,9+6*(ci+d))
            hit=[k for k,(r,c) in pos.items() if c<=L-1 and 7+6*r<=s1 and 10+6*r>=s0]
            if hit:
                ok,p=chain(hit,d,'r')
                if not ok: return None
                pos=p
        return (ci+d,L,tuple(pos[n] for n in nm))
    if act=='RIGHT':
        if L>=MAXL: return None
        cells=[(ci,L-1),(ci,L)] if L>0 else [(ci,0)]
        hit=[k for k,v in pos.items() if v in cells]
        if hit:
            ok,p=chain(hit,1,'c'); pos=p
        return (ci,L+1,tuple(pos[n] for n in nm))
    if act=='LEFT':
        if L<=0: return None
        hit=[k for k,v in pos.items() if v==(ci,L-1)]
        if hit:
            ok,p=chain(hit,-1,'c'); pos=p
        return (ci,L-1,tuple(pos[n] for n in nm))
    return None

def astar(start,goal_test,heuristic,rail,maxn=120000,acts=('UP','DOWN','LEFT','RIGHT')):
    h=heuristic(start)
    pq=[(h,0,start,())]; seen={start:(0,())}; cnt=0
    while pq and cnt<maxn:
        f,g,s,path=heapq.heappop(pq); cnt+=1
        if goal_test(s): return path,s,cnt
        if seen.get(s,(None,))[0]<g: continue
        for a in acts:
            n=sim7(s,a,rail)
            if n is None: continue
            ng=g+1
            if ng>=seen.get(n,(1e9,))[0]: continue
            seen[n]=(ng,path+(a,))
            heapq.heappush(pq,(ng+heuristic(n),ng,n,path+(a,)))
    return None,None,cnt

def mk_target(d):  # d: dict key->(r,c)
    base={'N':None,'b':None,'O':None,'R':None}; base.update(d)
    return base

def state_matches(s,t):
    return all(s[2][i]==t[k] for i,k in enumerate('NbOR') if k in t)

def hA(s):
    p=s[2][2]; return abs(p[0]-4)+abs(p[1]-0)+ (0 if s[0]==p[0] else 1)

def goal1(s): return s[2][2]==(1,0) and s[1]==0

def h1(s):
    r,c=s[2][2]; return abs(r-1)+abs(c-0)+(1 if s[1]>0 else 0)

def run_stage(s,goal,h,maxn=45000):
    p,g,n=astar(s,goal,h,RAIL,maxn=maxn)
    return p,g,n

def g2(s): return s[2][3]==(6,0) and s[2][2]==(1,0) and s[1]==0

def h2(s):
    r,c=s[2][3]; ro,co=s[2][2]; return abs(r-6)+abs(c)+abs(ro-1)+abs(co)+(1 if s[1]>0 else 0)

def RAILF(): return railset2(current_frame)[0]

def stage(s,goal,h,rail,maxn=45000):
    p,g,n=astar(s,goal,h,rail,maxn=maxn); return p,g,n

def g4(s): return s[2][1]==(6,2) and s[2][2]==(6,1) and s[2][3]==(6,0) and s[1]==0

def h4(s):
    rb,cb=s[2][1]; return abs(rb-6)+abs(cb-2)+(1 if s[1]>0 else 0)

def bar(fr):
    a=fr.ascii.split('\n')[53]
    return sum(1 for ch in a if ch not in ('c','B'))

def build_plan(s0,rail):
    P=[]
    def g1(s): return s[2][2]==(1,0) and s[1]==0
    def h1(s):
        r,c=s[2][2]; return abs(r-1)+abs(c)+(1 if s[1]>0 else 0)
    p,s1,_=stage(s0,g1,h1,rail); P+=p; 
    p,s2,_=stage(s1,g2,h2,rail); P+=p
    def g3(s): return s[2][2]==(6,1) and s[2][3]==(6,0) and s[1]==0
    def h3(s):
        ro,co=s[2][2]; rr,cr=s[2][3]; return abs(ro-6)+abs(co-1)+abs(rr-6)+abs(cr)+(1 if s[1]>0 else 0)
    p,s3,_=stage(s2,g3,h3,rail); P+=p
    p,s4,_=stage(s3,g4,h4,rail); P+=p
    def g5(s): return s[2]==((6,3),(6,2),(6,1),(6,0)) and s[1]==0
    def h5(s):
        rn,cn=s[2][0]; return abs(rn-6)+abs(cn-3)+(1 if s[1]>0 else 0)
    p,s5,_=stage(s4,g5,h5,rail); P+=p
    P+=['DOWN']+['RIGHT']*7
    return P,s5

def exec_chunk(acts,st,rail):
    for i,a in enumerate(acts):
        pred=sim7(st,a,rail)
        if pred is None: return ('PRED_NONE',i,st,None)
        res=action([a]) or {}
        real=read4(current_frame)
        flags={k:res.get(k) for k in ('level_completed','game_over','run_complete','done') if res.get(k)}
        if pred!=real: return ('MISMATCH',i,pred,real,flags)
        st=pred
        if flags: return ('END',i,st,flags)
    return ('OK',len(acts),st,None)
