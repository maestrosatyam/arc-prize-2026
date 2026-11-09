try:
    import json
except Exception:
    pass
try:
    import collections
except Exception:
    pass

def crop(r0,r1,c0,c1):
    for r in range(r0,r1+1):
        print(r, a[r][c0:c1+1])

def summ(label):
    d = frame_diff()
    out=[label,'changed',d['changed_cell_count']]
    for e in d.get('moved',[]):
        out.append(f"{e['color']} {e['from']}->{e['to']}")
    for e in d.get('appeared',[]):
        out.append("app "+e['color']+str(e['at']))
    for e in d.get('disappeared',[]):
        out.append("dis "+e['color']+str(e['at']))
    for e in d.get('resized',[]):
        if e['color']!='B': out.append("resz "+e['color']+str(e['before_pixels'])+'->'+str(e['after_pixels']))
    print(' | '.join(out))

def geo():
    seg=current_frame.segmentation
    out={}
    for n in seg['nodes']:
        if n['color']=='W' and n['pixels']==1:
            out['sel']=tuple(n['boundary'][0])
    return out

def run(acts):
    log=[]
    for a in acts:
        r=action([a])
        d=frame_diff()
        gp=r.get('gameplay_changed')
        moved=[(e['color'],e['from'],e['to']) for e in d.get('moved',[]) if e['color'] in 'bWY']
        log.append((a,gp,moved))
        if r.get('level_completed') or r.get('game_over') or r.get('run_complete'):
            log.append(('STOP',r.get('state'),r.get('level_completed'),r.get('game_over'),r.get('run_complete')))
            break
    for l in log: print(l)
    print('sel', geo(), 'timer', [n['pixels'] for n in current_frame.segmentation['nodes'] if n['color']=='p'])

def centers():
    seg=current_frame.segmentation
    W=[tuple(n['boundary'][0]) for n in seg['nodes'] if n['color']=='W' and n['pixels']==1]
    pts={'O':[], 'r':[], 'b':[]}
    for n in seg['nodes']:
        if n['color'] in pts:
            if n['pixels']==1: pts[n['color']].append(tuple(n['boundary'][0]))
            else:
                b=n['boundary']
                # for multi-pixel comps add all cells via bbox midpoints approx
                pts[n['color']].extend([tuple(p) for p in b])
    return W, pts

def classify(W):
    r,c=W
    a=current_frame.ascii.split('\n')
    def cell(rr,cc):
        if 0<=rr<64 and 0<=cc<64: return a[rr][cc]
        return '?'
    if cell(r,c+1)=='b' or cell(r,c-1)=='b' or cell(r+1,c)=='b' or cell(r-1,c)=='b': return 'b'
    if cell(r+1,c+1)=='O' or cell(r-1,c-1)=='O' or cell(r+1,c-1)=='O' or cell(r-1,c+1)=='O': return 'O'
    return 'r'

def moves_for(dpr,dpc):
    acts=[]
    vr,vc = dpr//3, dpc//3
    acts += ['UP']*(-vr) + ['DOWN']*vr + ['LEFT']*(-vc) + ['RIGHT']*vc
    return acts

def runs(row,ch='R'):
    s=a[row]; out=[]; c0=None
    for i in range(64):
        if s[i]==ch:
            if c0 is None: c0=i
        else:
            if c0 is not None: out.append((c0,i-1)); c0=None
    if c0 is not None: out.append((c0,63))
    return out

def masks(kind):
    r0,c0=cur[kind]
    out=[]
    for r in range(r0%3,64,3):
        for c in range(c0%3,64,3):
            m=0
            for i,(dr,dc) in enumerate(dots):
                if kind=='X':
                    if abs(dr-r)==abs(dc-c) and abs(dr-r)<=11: m|=1<<i
                elif kind=='D':
                    if abs(dr-r)+abs(dc-c)==12: m|=1<<i
                else:
                    if dr==r and abs(dc-c)<=21: m|=1<<i
            cost=(abs(r-r0)+abs(c-c0))//3
            out.append((m,cost,(r,c)))
    return out

def best_per_mask(ms):
    d={}
    for m,c,p in ms:
        if m not in d or c<d[m][0]: d[m]=(c,p)
    return d

def classify3(W):
    r,c=W
    a=current_frame.ascii.split('\n')
    def cell(rr,cc):
        return a[rr][cc] if (0<=rr<64 and 0<=cc<64) else '?'
    orth = cell(r,c+1)=='R' or cell(r,c-1)=='R' or cell(r+1,c)=='R' or cell(r-1,c)=='R'
    diag = cell(r+1,c+1)=='R' or cell(r-1,c-1)=='R' or cell(r+1,c-1)=='R' or cell(r-1,c+1)=='R'
    if orth and not diag: return 'B'
    if diag: return 'X'
    return 'D'

def solve_to(kind, target):
    W,_=centers()
    k=classify3(W[0])
    assert k==kind, (k,kind,W)
    dpr=target[0]-W[0][0]; dpc=target[1]-W[0][1]
    print(kind,'at',W[0],'->',target,'presses',(abs(dpr)+abs(dpc))//3)
    acts=moves_for(dpr,dpc)
    run(acts)
    W2,_=centers()
    ok = W2[0]==target
    print('  landed',W2[0],'ok',ok,'completed',last_action_call_result.get('level_completed'))
    return ok

def cover(kind,r,c):
    m=0
    for i,(dr,dc) in enumerate(dots):
        if kind=='X':
            if abs(dr-r)==abs(dc-c) and abs(dr-r)<=10: m|=1<<i
        else:
            if (dr==r and abs(dc-c)<=13) or (dc==c and abs(dr-r)<=13): m|=1<<i
    return m

def enum(kind,r0,c0):
    out=[]
    for r in range(r0%3,64,3):
        for c in range(c0%3,64,3):
            out.append((cover(kind,r,c),(abs(r-r0)+abs(c-c0))//3,(r,c)))
    return out

def classify4(W):
    r,c=W
    a=current_frame.ascii.split('\n')
    def cell(rr,cc): return a[rr][cc] if (0<=rr<64 and 0<=cc<64) else '?'
    orth=[cell(r,c+1),cell(r,c-1),cell(r+1,c),cell(r-1,c)]
    diag=[cell(r+1,c+1),cell(r-1,c-1),cell(r+1,c-1),cell(r-1,c+1)]
    oc=[x for x in orth if x not in 'BwW']
    dc=[x for x in diag if x not in 'BwW']
    if oc and not dc: return ('plus',oc[0])
    if dc and not oc: return ('X',dc[0])
    return ('?',oc+dc)

def xcells(r0,c0):
    s=set()
    for k in range(-10,11):
        s.add((r0+k,c0+k)); s.add((r0+k,c0-k))
    return s

def fit(pts):
    best=None
    for r0 in range(-20,84):
        for c0 in range(-20,84):
            for typ in ('X','+','D'):
                hit=0; need=0
                for (r,c) in pts:
                    d=abs(r-r0)+abs(c-c0) if typ=='D' else (max(abs(r-r0),abs(c-c0)) if typ=='X' else (0 if (r==r0 or c==c0) else 999))
                    pass
                # simpler: compute for each type the implied size from pts
    return None

def fitobj(pts):
    pts=list(pts); res=[]
    for r0 in range(64):
        for c0 in range(64):
            if all(abs(r-r0)==abs(c-c0) for r,c in pts):
                res.append(('X',(r0,c0),max(max(abs(r-r0),abs(c-c0)) for r,c in pts)))
            if all(r==r0 or c==c0 for r,c in pts):
                res.append(('+',(r0,c0),max(max(abs(r-r0),abs(c-c0)) for r,c in pts)))
            vals=set(abs(r-r0)+abs(c-c0) for r,c in pts)
            if len(vals)==1 and list(vals)[0]>0:
                res.append(('D',(r0,c0),list(vals)[0]))
    return res

def levelinfo():
    a=current_frame.ascii.split('\n')
    g=[[a[r][c] for c in range(64)] for r in range(64)]
    mask=[[False]*64 for _ in range(64)]
    sw=[]
    for r in range(0,58):
        for c in range(0,58):
            if g[r][c]=='g' and g[r+5][c]=='g' and g[r][c+5]=='g' and g[r+5][c+5]=='g':
                sw.append(((r+1,c+1),(r+4,c+4),g[r+1][c+1]))
                for rr in range(r,r+6):
                    for cc in range(c,c+6):
                        mask[rr][cc]=True
    dots=[]
    for r in range(63):
        for c in range(63):
            if g[r][c]=='c' and g[r+2][c]=='c' and g[r][c+2]=='c' and g[r+2][c+2]=='c':
                dots.append((g[r+1][c+1],r+1,c+1))
                for rr in range(r-1,r+3):
                    for cc in range(c-1,c+3):
                        if 0<=rr<64 and 0<=cc<64: mask[rr][cc]=True
    pix={}
    for r in range(64):
        for c in range(64):
            ch=g[r][c]
            if ch in 'MNObYSrp' and not mask[r][c]:
                pix.setdefault(ch,set()).add((r,c))
    return sw,dots,pix,g

def cells_of(typ,ctr,arm):
    r0,c0=ctr; s=set()
    if typ=='X':
        for k in range(-arm,arm+1): s.add((r0+k,c0+k)); s.add((r0+k,c0-k))
    elif typ=='+':
        for k in range(-arm,arm+1): s.add((r0,c0+k)); s.add((r0+k,c0))
    else:
        for k in range(-arm,arm+1):
            s.add((r0+k,c0+(arm-abs(k)))); s.add((r0+k,c0-(arm-abs(k))))
    return {(r,c) for r,c in s if 0<=r<64 and 0<=c<64}

def touches(cols):
    out=set()
    for (a,b),ch in SW:
        if cols & {(r,c) for r in range(a[0],a[1]+1) for c in range(a[0],a[1]+1)}: out.add(ch)
    return out

def build(typ,ctr,arm,mod):
    r0,c0=ctr
    out=[]
    for r in range(r0%3,64,3):
        for c in range(c0%3,64,3):
            cl=cells_of(typ,(r,c),arm)
            m=0
            for i,col in enumerate('bRO'):
                for d in DOTS[col]:
                    if d in cl: m|=1<<(i*8+len(DOTS[col][:0]))  # placeholder
            out.append(((r,c),cl,touches(cl)))
    return out

def maskof(cl,col):
    return sum(1 for d in DOTS[col] if d in cl)

def swtouch(cl):
    out=set()
    for (r0,c0),(r1,c1),ch in SWR:
        for (r,c) in cl:
            if r0<=r<=r1 and c0<=c<=c1: out.add(ch); break
    return out

def solve_obj(name):
    typ,cur,arm,col0=OBJ[name]
    r0,c0=cur
    placements=[(r,c) for r in range(r0%3,64,3) for c in range(c0%3,64,3)]
    idx={p:i for i,p in enumerate(placements)}
    cells={}; touch={}; cover={}
    for p in placements:
        cl=cells_of(typ,p,arm); cells[p]=cl; touch[p]=swtouch(cl)
        cover[p]={col:sum(1 for d in DOTS[col] if d in cl) for col in DOTS}
    from collections import deque
    start=(r0,c0,col0)
    def goal(p,col): return cover[p][col]>0
    dist={start:0}; prev={start:None}; q=deque([start])
    best=None
    while q:
        st=q.popleft()
        r,c,col=st
        if goal((r,c),col):
            best=st; break
        for dr,dc in ((-3,0),(3,0),(0,-3),(0,3)):
            nr,nc=r+dr,c+dc
            if (nr,nc) in idx:
                ncol=col
                t=touch[(nr,nc)]
                if t: ncol=sorted(t)[0] if len(t)==1 else None
                if ncol is None: continue
                nst=(nr,nc,ncol)
                if nst not in dist:
                    dist[nst]=dist[st]+1; prev[nst]=st; q.append(nst)
    if best is None: return None
    path=[]; st=best
    while prev[st] is not None:
        p=prev[st]; dr=st[0]-p[0]; dc=st[1]-p[1]
        path.append({-3:'UP',3:'DOWN'}[dr] if dr else { -3:'LEFT',3:'RIGHT'}[dc])
        st=p
    path.reverse()
    return dict(end=best,cost=dist[best],path=path)

def swtouch2(cl):
    out=set()
    for (r0,c0),(r1,c1),ch in SWR:
        for (r,c) in cl:
            if r0<=r<=r1 and c0<=c<=c1: out.add(ch); break
    return out

def solve_obj2(name):
    typ,cur,arm,col0=OBJ[name]
    r0,c0=cur
    placements=[(r,c) for r in range(r0%3,64,3) for c in range(c0%3,64,3)]
    idx={p:i for i,p in enumerate(placements)}
    touch={}; cover={}
    for p in placements:
        cl=cells_of(typ,p,arm); touch[p]=swtouch2(cl)
        cover[p]={col:sum(1 for d in DOTS[col] if d in cl) for col in DOTS}
    from collections import deque
    start=(r0,c0,col0); dist={start:0}; prev={start:None}; q=deque([start]); best=None
    while q:
        st=q.popleft(); r,c,col=st
        if cover[(r,c)].get(col,0)>0: best=st; break
        for dr,dc in ((-3,0),(3,0),(0,-3),(0,3)):
            nr,nc=r+dr,c+dc
            if (nr,nc) in idx:
                t=touch[(nr,nc)]
                if len(t)>1: continue
                ncol = t.pop() if t else col
                nst=(nr,nc,ncol)
                if nst not in dist:
                    dist[nst]=dist[st]+1; prev[nst]=st; q.append(nst)
    if best is None: return None
    path=[]; st=best
    while prev[st] is not None:
        p=prev[st]; dr=st[0]-p[0]; dc=st[1]-p[1]
        path.append({-3:'UP',3:'DOWN',0:''}[dr] or {-3:'LEFT',3:'RIGHT'}[dc]); st=p
    path.reverse()
    return dict(end=best,cost=dist[best],path=path,cov=cover[best[:2]])

def bfs_obj(typ,cur,arm,col0,swr,dots,goalcov=None,goalpos=None,goalcol=None,maxk=8):
    r0,c0=cur
    placements=[(r,c) for r in range(r0%3,64,3) for c in range(c0%3,64,3)]
    idx={p:1 for p in placements}
    touch={}; cover={}
    for p in placements:
        cl=cells_of(typ,p,arm)
        t=set()
        for (a,b),(a1,b1),ch in swr:
            for (r,c) in cl:
                if a<=r<=a1 and c<=b<=b1: t.add(ch); break
        touch[p]=t
        cover[p]={col:sum(1 for d in dots[col] if d in cl) for col in dots}
    from collections import deque
    start=(r0,c0,col0); dist={start:0}; prev={start:None}; q=deque([start]); best=None
    def isgoal(p,col):
        if goalpos and p!=tuple(goalpos): return False
        if goalcol and col!=goalcol: return False
        if goalcov is not None and cover[p].get(col,0)<goalcov: return False
        return cover[p].get(col,0)>0
    while q:
        st=q.popleft(); r,c,col=st
        if isgoal((r,c),col): best=st; break
        for dr,dc in ((-3,0),(3,0),(0,-3),(0,3)):
            nr,nc=r+dr,c+dc
            if (nr,nc) in idx:
                t=touch[(nr,nc)]
                if len(t)>1: continue
                ncol=t.pop() if t else col
                nst=(nr,nc,ncol)
                if nst not in dist:
                    dist[nst]=dist[st]+1; prev[nst]=st; q.append(nst)
    if best is None: return None
    path=[]; st=best
    while prev[st] is not None:
        p=prev[st]; dr=st[0]-p[0]; dc=st[1]-p[1]
        path.append({-3:'UP',3:'DOWN',0:''}[dr] or {-3:'LEFT',3:'RIGHT'}[dc]); st=p
    path.reverse()
    return dict(end=best,cost=dist[best],path=path,cov=cover[best[:2]])

def bfs2(typ,cur,arm,col0,swr,dots,goalcov=None,goalpos=None,goalcol=None):
    r0,c0=cur
    placements=[(r,c) for r in range(r0%3,64,3) for c in range(c0%3,64,3)]
    idx={p:1 for p in placements}; touch={}; cover={}
    for p in placements:
        cl=cells_of(typ,p,arm); t=frozenset()
        for (a,b),(a1,b1),ch in swr:
            if any(a<=r<=a1 and c0_<=b1 for (r,c0_) in cl if a<=r<=a1) and any(a<=r<=a1 and c<=b1 and a0<=c<=b0 for (r,c) in cl):
                pass
        ts=set()
        for (ar,ac),(br,bc),ch in swr:
            for (r,c) in cl:
                if ar<=r<=br and ac<=c<=bc: ts.add(ch); break
        touch[p]=frozenset(ts)
        cover[p]={col:sum(1 for d in dots[col] if d in cl) for col in dots}
    start=(r0,c0,col0); dist={start:0}; prev={start:None}; q=collections.deque([start]); best=None
    def isgoal(p,col):
        if goalpos and p!=tuple(goalpos): return False
        if goalcol and col!=goalcol: return False
        k=cover[p].get(col,0)
        if goalcov and k<goalcov: return False
        return k>0
    while q:
        st=q.popleft(); r,c,col=st
        if isgoal((r,c),col): best=st; break
        for dr,dc in ((-3,0),(3,0),(0,-3),(0,3)):
            nr,nc=r+dr,c+dc
            if (nr,nc) in idx:
                t=touch[(nr,nc)]
                if len(t)!=1: continue
                nst=(nr,nc,next(iter(t)))
                if nst not in dist:
                    dist[nst]=dist[st]+1; prev[nst]=st; q.append(nst)
    if best is None: return None
    path=[]; st=best
    while prev[st] is not None:
        p=prev[st]; dr=st[0]-p[0]; dc=st[1]-p[1]
        path.append({-3:'UP',3:'DOWN',0:''}[dr] or {-3:'LEFT',3:'RIGHT'}[dc]); st=p
    path.reverse()
    return dict(end=best,cost=dist[best],path=path,cov=cover[best[:2]])

def bfs3(typ,cur,arm,col0,swr,dots,goalcov=None,goalpos=None,goalcol=None):
    r0,c0=cur
    placements=[(r,c) for r in range(r0%3,64,3) for c in range(c0%3,64,3)]
    idx={p:1 for p in placements}; touch={}; cover={}
    for p in placements:
        cl=cells_of(typ,p,arm); ts=set()
        for (ar,ac),(br,bc),ch in swr:
            for (r,c) in cl:
                if ar<=r<=br and ac<=c<=bc:
                    ts.add(ch); break
        touch[p]=frozenset(ts)
        cover[p]={col:sum(1 for d in dots[col] if d in cl) for col in dots}
    start=(r0,c0,col0); dist={start:0}; prev={start:None}; q=collections.deque([start]); best=None
    def isgoal(p,col):
        if goalpos and p!=tuple(goalpos): return False
        if goalcol and col!=goalcol: return False
        k=cover[p].get(col,0)
        if goalcov and k<goalcov: return False
        return k>0
    while q:
        st=q.popleft(); r,c,col=st
        if isgoal((r,c),col): best=st; break
        for dr,dc in ((-3,0),(3,0),(0,-3),(0,3)):
            nr,nc=r+dr,c+dc
            if (nr,nc) in idx:
                t=touch[(nr,nc)]
                if len(t)!=1: continue
                nst=(nr,nc,next(iter(t)))
                if nst not in dist:
                    dist[nst]=dist[st]+1; prev[nst]=st; q.append(nst)
    if best is None: return None
    path=[]; st=best
    while prev[st] is not None:
        p=prev[st]; dr=st[0]-p[0]; dc=st[1]-p[1]
        path.append({-3:'UP',3:'DOWN',0:''}[dr] or {-3:'LEFT',3:'RIGHT'}[dc]); st=p
    path.reverse()
    return dict(end=best,cost=dist[best],path=path,cov=cover[best[:2]])

def reach(typ,cur,arm,col0,swr,dots):
    r0,c0=cur
    placements=[(r,c) for r in range(r0%3,64,3) for c in range(c0%3,64,3)]
    idx={p:1 for p in placements}; touch={}; cover={}
    for p in placements:
        cl=cells_of(typ,p,arm); ts=set()
        for (ar,ac),(br,bc),ch in swr:
            for (r,c) in cl:
                if ar<=r<=br and ac<=c<=bc:
                    ts.add(ch); break
        touch[p]=frozenset(ts)
        cover[p]={col:sum(1 for d in dots[col] if d in cl) for col in dots}
    start=(r0,c0,col0); dist={start:0}; q=collections.deque([start])
    while q:
        r,c,col=q.popleft()
        for dr,dc in ((-3,0),(3,0),(0,-3),(0,3)):
            nr,nc=r+dr,c+dc
            if (nr,nc) in idx:
                t=touch[(nr,nc)]
                ncol = next(iter(t)) if len(t)==1 else (None if len(t)>1 else col)
                if ncol is None: continue
                nst=(nr,nc,ncol)
                if nst not in dist:
                    dist[nst]=dist[(r,c,col)]+1; q.append(nst)
    good={}
    for (r,c,col) in dist:
        k=cover[(r,c)].get(col,0)
        if k>0: good.setdefault((r,c,col),(dist[(r,c,col)],cover[(r,c)][col]))
    return good, len(dist)

def bfs4(typ,cur,arm,col0,swr,dots,goalcov=None,goalpos=None,goalcol=None):
    r0,c0=cur
    placements=[(r,c) for r in range(r0%3,64,3) for c in range(c0%3,64,3)]
    idx={p:1 for p in placements}; touch={}; cover={}
    for p in placements:
        cl=cells_of(typ,p,arm); ts=set()
        for (ar,ac),(br,bc),ch in swr:
            for (r,c) in cl:
                if ar<=r<=br and ac<=c<=bc:
                    ts.add(ch); break
        touch[p]=frozenset(ts)
        cover[p]={col:sum(1 for d in dots[col] if d in cl) for col in dots}
    start=(r0,c0,col0); dist={start:0}; prev={start:None}; q=collections.deque([start]); best=None
    def isgoal(p,col):
        if goalpos and p!=tuple(goalpos): return False
        if goalcol and col!=goalcol: return False
        k=cover[p].get(col,0)
        if goalcov and k<goalcov: return False
        return k>0
    while q:
        st=q.popleft(); r,c,col=st
        if isgoal((r,c),col): best=st; break
        for dr,dc in ((-3,0),(3,0),(0,-3),(0,3)):
            nr,nc=r+dr,c+dc
            if (nr,nc) in idx:
                t=touch[(nr,nc)]
                if len(t)>1: continue
                ncol=next(iter(t)) if len(t)==1 else col
                nst=(nr,nc,ncol)
                if nst not in dist:
                    dist[nst]=dist[st]+1; prev[nst]=st; q.append(nst)
    if best is None: return None
    path=[]; st=best
    while prev[st] is not None:
        p=prev[st]; dr=st[0]-p[0]; dc=st[1]-p[1]
        path.append({-3:'UP',3:'DOWN',0:''}[dr] or {-3:'LEFT',3:'RIGHT'}[dc]); st=p
    path.reverse()
    return dict(end=best,cost=dist[best],path=path,cov=cover[best[:2]])

def wpos():
    a=current_frame.ascii.split('\n')
    for r in range(64):
        c=a[r].find('W')
        if c>=0: return (r,c)

def colr(p):
    a=current_frame.ascii.split('\n')
    r,c=p
    return a[r-1][c-1:c+2]+a[r][c-1]+a[r][c+1]+a[r+1][c-1:c+2]

def route(typ,cur,arm,col0,swr,mydots,cap=None):
    r0,c0=cur
    placements=[(r,c) for r in range(r0%3,64,3) for c in range(c0%3,64,3)]
    idx={p:i for i,p in enumerate(placements)}
    colr={}
    for p in placements:
        cl=cells_of(typ,p,arm); ts=set()
        for (ar,ac),(br,bc),ch in swr:
            for (r,c) in cl:
                if ar<=r<=br and ac<=c<=bc:
                    ts.add(ch); break
        m=0
        for i,d in enumerate(mydots):
            if d in cl: m|=1<<i
        colr[p]=(frozenset(ts),m)
    n=len(mydots); full=(1<<n)-1
    start=(r0,c0,col0,0)
    dist={start:0}; prev={start:None}; q=collections.deque([start]); best=None
    while q:
        st=q.popleft(); r,c,col,m=st
        if m==full: best=st; break
        for dr,dc in ((-3,0),(3,0),(0,-3),(0,3)):
            nr,nc=r+dr,c+dc
            if (nr,nc) in idx:
                t,mm=colr[(nr,nc)]
                if len(t)>1: continue
                ncol=next(iter(t)) if len(t)==1 else col
                nm=m|mm
                nst=(nr,nc,ncol,nm)
                if nst not in dist:
                    dist[nst]=dist[st]+1; prev[nst]=st; q.append(nst)
                    if cap and len(dist)>cap: return dict(too_big=len(dist))
    if best is None: return None
    path=[]; st=best
    while prev[st] is not None:
        p=prev[st]; dr=st[0]-p[0]; dc=st[1]-p[1]
        path.append({-3:'UP',3:'DOWN',0:''}[dr] or {-3:'LEFT',3:'RIGHT'}[dc]); st=p
    path.reverse()
    return dict(end=best,cost=dist[best],path=path)

def route2(typ,cur,arm,col0,swr,dotcols,cap=400000):
    r0,c0=cur
    placements=[(r,c) for r in range(r0%3,64,3) for c in range(c0%3,64,3)]
    idx={p:i for i,p in enumerate(placements)}
    info={}
    for p in placements:
        cl=cells_of(typ,p,arm); ts=set()
        for (ar,ac),(br,bc),ch in swr:
            for (r,c) in cl:
                if ar<=r<=br and ac<=c<=bc:
                    ts.add(ch); break
        masks={}
        for i,(d,dcol) in enumerate(dotcols):
            if d in cl: masks[dcol]=masks.get(dcol,0)|(1<<i)
        info[p]=(frozenset(ts),masks)
    n=len(dotcols); full=(1<<n)-1
    start=(r0,c0,col0,0); dist={start:0}; prev={start:None}; q=collections.deque([start]); best=None
    while q:
        st=q.popleft(); r,c,col,m=st
        if m==full: best=st; break
        for dr,dc in ((-3,0),(3,0),(0,-3),(0,3)):
            nr,nc=r+dr,c+dc
            if (nr,nc) in idx:
                t,masks=info[(nr,nc)]
                if len(t)>1: continue
                ncol=next(iter(t)) if len(t)==1 else col
                nm=m|masks.get(ncol,0)
                nst=(nr,nc,ncol,nm)
                if nst not in dist:
                    dist[nst]=dist[st]+1; prev[nst]=st; q.append(nst)
                    if len(dist)>cap: return dict(too_big=len(dist))
    if best is None: return None
    path=[]; st=best
    while prev[st] is not None:
        p=prev[st]; dr=st[0]-p[0]; dc=st[1]-p[1]
        path.append({-3:'UP',3:'DOWN',0:''}[dr] or {-3:'LEFT',3:'RIGHT'}[dc]); st=p
    path.reverse()
    return dict(end=best,cost=dist[best],path=path)

def exec_path(path,label):
    for i,act in enumerate(path):
        action([act])
        st=last_action_call_result
        if st.get('level_completed') or st.get('run_complete') or st.get('done') or st.get('game_over'):
            print(label,'FINISHED at',i,act,'lvl',st.get('level_completed'),'run',st.get('run_complete'),'done',st.get('done'),'over',st.get('game_over'))
            return True
    print(label,'path done; dots left',len(levelinfo()[1]))
    return False

def ring_cells(c,arm=9):
    r0,c0=c; out=set()
    for d in range(-arm,arm+1):
        out.add((r0-arm,c0+d)); out.add((r0+arm,c0+d)); out.add((r0+d,c0-arm)); out.add((r0+d,c0+arm))
    return {p for p in out if 0<=p[0]<64 and 0<=p[1]<64}

def plus_cells(c,arm=12):
    r0,c0=c
    out={(r0,c) for c in range(max(0,c0-arm),min(64,c0+arm+1))} | {(r,c0) for r in range(max(0,r0-arm),min(64,r0+arm+1))}
    return out

def matchcov(p,q):
    s=set()
    for i in pm[p]:
        if DOTS[i][2]=='b': s.add(i)
    for i in rm[q]:
        if DOTS[i][2]=='Y': s.add(i)
    return s

def bars():
    sw,dots,pix,g=levelinfo()
    B=pix['b']
    byrow=collections.defaultdict(list); bycol=collections.defaultdict(list)
    for r,c in B: byrow[r].append(c); bycol[c].append(r)
    hr=[r for r,v in byrow.items() if len(v)>12]; hc=[c for c,v in bycol.items() if len(v)>12]
    hspan=(min(byrow[hr[0]]),max(byrow[hr[0]])) if hr else None
    vspan=(min(bycol[hc[0]]),max(bycol[hc[0]])) if hc else None
    return wpos(),hr,hspan,hc,vspan,len(B)

def ring_edges():
    sw,dots,pix,g=levelinfo()
    Y=pix['Y']
    byrow=collections.defaultdict(list); bycol=collections.defaultdict(list)
    for r,c in Y: byrow[r].append(c); bycol[c].append(r)
    hr=sorted([(r,min(v),max(v),len(v)) for r,v in byrow.items() if len(v)>5])
    hc=sorted([(c,min(v),max(v),len(v)) for c,v in bycol.items() if len(v)>5])
    return wpos(),hr,hc,len(Y)

def redges():
    st=ring_edges()
    return st

def analyze(fr):
    a=fr.ascii.split('\n')
    R,C=len(a),len(a[0])
    grid=[[a[r][c] for c in range(C)] for r in range(R)]
    seen=[[False]*C for _ in range(R)]
    comps=[]
    for r in range(R):
        for c in range(C):
            if seen[r][c] or grid[r][c]=='B': continue
            col=grid[r][c]; st=[(r,c)]; seen[r][c]=True; cells=[]
            while st:
                y,x=st.pop(); cells.append((y,x))
                for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
                    ny,nx=y+dy,x+dx
                    if 0<=ny<R and 0<=nx<C and not seen[ny][nx] and grid[ny][nx]==col:
                        seen[ny][nx]=True; st.append((ny,nx))
            comps.append((col,cells))
    dots=[]; objs=[]
    for col,cells in comps:
        rs=[p[0] for p in cells]; cs=[p[1] for p in cells]
        h=max(rs)-min(rs)+1; w=max(cs)-min(cs)+1
        if h<=3 and w<=3 and len(cells)<=9 and col not in ('w','G','c','p','W'):
            dots.append((col,min(rs),min(cs)))
        else:
            objs.append((col,len(cells),min(rs),max(rs),min(cs),max(cs)))
    return sorted(dots),sorted(objs,key=lambda z:-z[1])

def objcols(fr):
    d,o=analyze(fr)
    big=[x for x in o if x[1]>=20 and not (x[2]==63 or x[3]==63)]
    return sorted(set(x[0] for x in big)), [ (x[0],x[1],x[2],x[3],x[4],x[5]) for x in big]

def dots_boxes(fr):
    a=fr.ascii.split('\n'); R,C=64,64
    g=[[a[r][c] for c in range(C)] for r in range(R)]
    out=[]
    for r in range(R-2):
        for c in range(C-2):
            blk=[g[r+dr][c+dc] for dr in range(3) for dc in range(3)]
            if blk[4]!='c' and blk[0]=='c' and blk[2]=='c' and blk[6]=='c' and blk[8]=='c':
                out.append((r+1,c+1,blk[4]))
    return out

def objs2(fr):
    a=fr.ascii.split('\n'); R,C=64,64
    cnt=collections.Counter()
    for r in range(R):
        for c in range(C):
            ch=a[r][c]
            if ch not in ('B','c','W','p','w','g','G'): cnt[ch]+=1
    return sorted([(k,v) for k,v in cnt.items() if v>=15])
