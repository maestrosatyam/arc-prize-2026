try:
    from collections import defaultdict
except Exception:
    pass
try:
    import re
except Exception:
    pass
try:
    from collections import deque
except Exception:
    pass
try:
    import json
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

def cells(color):
    return [(r,c) for r in range(64) for c in range(64) if grid[r][c]==color]

def ch(r,c): return g[r][c] if 0<=r<64 and 0<=c<64 else 'W'

def blockfree(r,c,blocked):
    return all(ch(r+dr,c+dc) not in blocked for dr in range(5) for dc in range(5))

def hlink(r,c1,c2):
    lo=min(c1,c2)-1; hi=max(c1,c2)+6
    return all(not blocked(r+dr,cc) for dr in range(5) for cc in range(lo,hi))

def vlink(c,r1,r2):
    lo=min(r1,r2)-1; hi=max(r1,r2)+6
    return all(not blocked(rr,c+dc) for rr in range(lo,hi) for dc in range(5))

def blocked(r,c):
    if br0<=r<=br1 and bc0<=c<=bc1: return False
    return ch(r,c)=='B'

def free5(r,c): return all(not blocked(r+dr,c+dc) for dr in range(5) for dc in range(5))

def blockbb(fr):
    g=[list(r) for r in fr.ascii.split('\n')]
    bl=[(r,c) for r in range(6,48) for c in range(64) if g[r][c]=='b']
    if not bl: return None,g
    return (min(x[0] for x in bl),min(x[1] for x in bl)),g

def cellcomp(g,r,c):
    cnt={}
    for dr in range(5):
        for dc in range(5):
            ch=g[r+dr][c+dc]; cnt[ch]=cnt.get(ch,0)+1
    return ''.join(f'{k}{v}' for k,v in sorted(cnt.items()))

def bars(fr):
    g=[list(r) for r in fr.ascii.split('\n')]
    row63=sum(1 for c in range(64) if g[63][c]!='W')
    top=[(r,c) for r in range(7) for c in range(10) if g[r][c] not in ('W','w')]
    return row63, top

def comp(r,c): return [ch(r+dr,c+dc) for dr in range(5) for dc in range(5)]

def passable(r,c):
    return all(g[r+dr][c+dc]!='W' for dr in range(5) for dc in range(5))

def ok(r,c):
    cc=comp(r,c)
    return all(x!='W' for x in cc)

def bridge(r,c,nr,nc):
    if nr==r:
        lo,hi=min(c,nc),max(c,nc)
        bc = lo-1 if nc<c else hi+1
        return [ch(r+dr,bc) for dr in range(5)]
    else:
        lo,hi=min(r,nr),max(r,nr)
        br = lo-1 if nr<r else hi+1
        return [ch(br,c+dc) for dc in range(5)]

def gridof(fr): return [list(r) for r in fr.ascii.split('\n')]

def chg(g,r,c): return g[r][c] if 0<=r<64 and 0<=c<64 else 'W'

def compg(g,r,c): return [chg(g,r+dr,c+dc) for dr in range(5) for dc in range(5)]

def bridgeg(g,r,c,nr,nc):
    if nr==r:
        bc = (min(c,nc)-1) if nc<c else (max(c,nc)+1)
        return [chg(g,r+dr,bc) for dr in range(5)]
    br = (min(r,nr)-1) if nr<r else (max(r,nr)+1)
    return [chg(g,br,c+dc) for dc in range(5)]

def bfs(g,start,goal,allowR=True):
    def ok(r,c):
        cc=compg(g,r,c)
        return all(x!='W' and (allowR or x!='R') for x in cc)
    seen={start}; dq=deque([(start,[])])
    while dq:
        (r,c),path=dq.popleft()
        if (r,c)==goal: return path,seen
        for dr,dc,mv in ((6,0,'DOWN'),(-6,0,'UP'),(0,6,'RIGHT'),(0,-6,'LEFT')):
            n=(r+dr,c+dc)
            if not(2<=n[0]<=56 and 2<=n[1]<=56) or n in seen: continue
            if ok(n[0],n[1]) and all(x!='W' for x in bridgeg(g,r,c,*n)):
                seen.add(n); dq.append((n,path+[mv]))
    return None,seen

def redinfo(fr):
    g=gridof(fr)
    R=[(r,c) for r in range(64) for c in range(64) if g[r][c]=='R']
    return len(R), (min(x[0] for x in R),max(x[0] for x in R),min(x[1] for x in R),max(x[1] for x in R))

def sym(r,c):
    if r+4>63 or c+4>63: return ' '
    cc=set(compg(g,r,c))
    if 'b' in cc and 'B' in cc: return '@'
    if 'R' in cc: return 'R'
    if cc=={'B'}: return '.'
    if 'W' in cc and 'B' in cc: return '~'
    if cc=={'W'}: return ' '
    if 'b' in cc: return 'F'
    return '?'

def grayinfo(fr):
    g=gridof(fr)
    GR=[(r,c) for r in range(6,64) for c in range(64) if g[r][c]=='g']
    if not GR: return None,0
    return (min(x[0] for x in GR),min(x[1] for x in GR)),len(GR)

def state(fr):
    g=gridof(fr)
    bl=[(r,c) for r in range(6,48) for c in range(64) if g[r][c]=='b']
    bb=(min(x[0] for x in bl),min(x[1] for x in bl)) if bl else None
    gr=[(r,c) for r in range(6,64) for c in range(64) if g[r][c]=='g']
    gb=(min(x[0] for x in gr),min(x[1] for x in gr)) if gr else None
    # red blob = red cells not in col40/row40 line: find red bbox excluding line
    R=[(r,c) for r in range(64) for c in range(64) if g[r][c]=='R']
    blob=[p for p in R if p[0] in range(36,44)]
    blobc = (min(p[1] for p in blob), max(p[1] for p in blob)) if blob else None
    return bb,gb,len(R),blobc,sum(1 for c in range(64) if g[63][c]=='b')

def okcell(r,c,allowR=False):
    cc=compg(g,r,c)
    for x in cc:
        if x=='W': return False
        if x=='R' and not allowR: return False
    return True

def bfs2(start,goal,allowR=False):
    seen={start}; dq=deque([(start,[])])
    while dq:
        (r,c),path=dq.popleft()
        if (r,c)==goal: return path,sorted(seen)
        for dr,dc,mv in ((6,0,'DOWN'),(-6,0,'UP'),(0,6,'RIGHT'),(0,-6,'LEFT')):
            n=(r+dr,c+dc)
            if not(2<=n[0]<=56 and 2<=n[1]<=56) or n in seen: continue
            if okcell(*n,allowR) and all(x!='W' for x in bridgeg(g,r,c,*n)):
                seen.add(n); dq.append((n,path+[mv]))
    return None,sorted(seen)

def kind(g,r,c):
    cc=C5(g,r,c)
    if any(x in cc for x in 'RY'): return 'X'
    if 'W' in cc: return '#'
    if 'b' in cc: return '@'
    return '.'

def link(a,b):
    return all(x!='W' for x in bridgeg(g,a[0],a[1],b[0],b[1]))

def compof(p):
    for i,c in enumerate(comps):
        if p in c: return i
    return None

def full(fr):
    g=gridof(fr)
    def bbox(col):
        p=[(r,c) for r in range(6,64) for c in range(64) if g[r][c]==col]
        return (min(x[0] for x in p),min(x[1] for x in p),max(x[0] for x in p),max(x[1] for x in p)) if p else None
    # block = blue comp of 24 px in maze; frame is separate but let's just list comps
    b=[(r,c) for r in range(6,64) for c in range(64) if g[r][c]=='b']
    S=set(b); comps=[]
    while S:
        s=S.pop(); comp={s}; dq=deque([s])
        while dq:
            r,c=dq.popleft()
            for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
                n=(r+dr,c+dc)
                if n in S: S.remove(n); comp.add(n); dq.append(n)
        comps.append(comp)
    blk=[sorted(c) [0] for c in comps if len(c)>=19 and len(c)<=25]
    R=[(r,c) for r in range(64) for c in range(64) if g[r][c]=='R']
    # red blob heads: 5x5 clusters -> report rows where >=5 red in a row
    from collections import Counter
    rc=Counter(r for r,c in R)
    heads=[r for r,n in rc.items() if n>=5]
    return dict(blk=blk, nR=len(R), redrowheads=heads, bar=sum(1 for c in range(64) if g[63][c]=='b'))

def comps_of(col):
    S=set((r,c) for r in range(64) for c in range(64) if g[r][c]==col); out=[]
    while S:
        s=S.pop(); comp={s}; dq=deque([s])
        while dq:
            r,c=dq.popleft()
            for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
                n=(r+dr,c+dc)
                if n in S: S.remove(n); comp.add(n); dq.append(n)
        out.append(comp)
    return sorted([(len(c),min(x[0] for x in c),min(x[1] for x in c),max(x[0] for x in c),max(x[1] for x in c)) for c in out],key=lambda t:(t[1],t[2]))

def graycomps(fr):
    g=gridof(fr)
    S=[(r,c) for r in range(6,64) for c in range(64) if g[r][c]=='g']
    S=set(S); out=[]
    while S:
        s=S.pop(); comp={s}; dq=deque([s])
        while dq:
            r,c=dq.popleft()
            for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
                n=(r+dr,c+dc)
                if n in S: S.remove(n); comp.add(n); dq.append(n)
        out.append((len(comp),min(x[0] for x in comp),min(x[1] for x in comp)))
    return out

def blk(fr):
    g=gridof(fr)
    S=[(r,c) for r in range(6,64) for c in range(64) if g[r][c]=='b']
    S=set(S); out=[]
    while S:
        s=S.pop(); comp={s}; dq=deque([s])
        while dq:
            r,c=dq.popleft()
            for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
                n=(r+dr,c+dc)
                if n in S: S.remove(n); comp.add(n); dq.append(n)
        if 18<=len(comp)<=25: out.append((min(x[0] for x in comp),min(x[1] for x in comp),len(comp)))
    return out

def okc(r,c):
    return kind(r,c) in ('open','gray')

def lnk(a,b): return all(x!='W' for x in bridgeg(g,a[0],a[1],b[0],b[1]))

def C5(g,r,c): return set(g[r+i][c+j] for i in range(5) for j in range(5))

def BR(g,r,c,nr,nc):
    if nr==r:  # horizontal move: strip col = between
        c0=min(c,nc)+5
        return set(g[r+i][c0] for i in range(5))
    r0=min(r,nr)+5
    return set(g[r0][c+j] for j in range(5))

def lv3(fr):
    g=gridof(fr)
    def box(col):
        p=[(r,c) for r in range(6,64) for c in range(64) if g[r][c]==col]
        return None if not p else (min(x[0] for x in p),min(x[1] for x in p),max(x[0] for x in p),max(x[1] for x in p))
    # block = blue comp of size 24 inside maze
    S=set((r,c) for r in range(6,60) for c in range(64) if g[r][c]=='b'); bl=None
    while S:
        s=S.pop(); comp={s}; dq=deque([s])
        while dq:
            r,c=dq.popleft()
            for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
                n=(r+dr,c+dc)
                if n in S: S.remove(n); comp.add(n); dq.append(n)
        if len(comp)==24: bl=(min(x[0] for x in comp),min(x[1] for x in comp))
    return dict(blk=bl, Y=box('Y'), R1=box('R'), gray=graycomps(fr), bar=sum(1 for c in range(64) if g[63][c]=='b'))

def redboxes(fr):
    g=gridof(fr)
    S=set((r,c) for r in range(6,64) for c in range(64) if g[r][c]=='R'); out=[]
    while S:
        s=S.pop(); comp={s}; dq=deque([s])
        while dq:
            r,c=dq.popleft()
            for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
                n=(r+dr,c+dc)
                if n in S: S.remove(n); comp.add(n); dq.append(n)
        out.append((min(x[0] for x in comp),min(x[1] for x in comp),max(x[0] for x in comp),max(x[1] for x in comp),len(comp)))
    return sorted(out)

def hud(fr):
    g=gridof(fr)
    return '\n'.join(''.join(g[r][0:14]) for r in range(0,7))

def blobstate(fr):
    rb=redboxes(fr)
    # unit A comp bbox top row: 32 = home, 37/38 = moved down ; unit B: 33 = home, 37 = moved up
    return rb[0][0], rb[1][0]

def step(mv):
    res=action([mv]); st=lv3(current_frame)
    return st['blk'], res.get('no_op'), lv3(current_frame)['gray'], blobstate(current_frame), st['bar']

def yellow_open(fr=None):
    fr=fr or current_frame
    g=gridof(fr)
    return not any(g[r][c]=='Y' for r in range(20,25) for c in range(50,55))

def blobs(fr=None):
    rb=redboxes(fr or current_frame)
    return {'A_moved': rb[0][0]>=37, 'B_moved': rb[1][0]>=37}

def go(mv,expect=None):
    global pos
    res=action([mv]); st=lv3(current_frame)
    np=[p for p in blk(current_frame) if p[2]==24][0][:2]
    return np, bool(res.get('no_op')), st['bar']

def curblk(): return [p for p in blk(current_frame) if p[2]==24 and p[0]<60][0][:2]

def redmap(fr,label,r0=30,r1=56,c0=18,c1=46):
    g=gridof(fr)
    print(label)
    for r in range(r0,r1):
        cols=[c for c in range(c0,c1) if g[r][c]=='R']
        if cols: print(' r%d:'%r, cols)

def grayboxes(fr):
    g=gridof(fr)
    seen=set(); out=[]
    for r in range(6,60):
        for c in range(64):
            if g[r][c] in 'wGg' and (r,c) not in seen:
                # flood
                st=[(r,c)]; comp=[]
                seen.add((r,c))
                while st:
                    y,x=st.pop(); comp.append((y,x))
                    for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        ny,nx=y+dy,x+dx
                        if 0<=ny<64 and 0<=nx<64 and g[ny][nx] in 'wGg' and (ny,nx) not in seen:
                            seen.add((ny,nx)); st.append((ny,nx))
                out.append((min(p[0] for p in comp),max(p[0] for p in comp),min(p[1] for p in comp),max(p[1] for p in comp),len(comp)))
    return out

def gap(r,c,r2,c2):
    mr,mc=(r+r2)//2,(c+c2)//2
    return g[mr][mc]!='W'

def sp(adj,s,t,blocked=()):
    from collections import deque
    if s in blocked or t in blocked: return None
    dq=deque([[s]]); seen={s}
    while dq:
        p=dq.popleft()
        if p[-1]==t: return p
        for n in adj.get(p[-1],[]):
            if n not in seen and n not in blocked:
                seen.add(n); dq.append(p+[n])
    return None

def dirs(p):
    D={(-6,0):'U',(6,0):'D',(0,-6):'L',(0,6):'R'}
    return [D[(b[0]-a[0],b[1]-a[1])] for a,b in zip(p,p[1:])]

def maze(fr=None):
    g=gridof(fr or current_frame)
    cells=[8,14,20,26,32,38,44,50]
    def ok(r,c): return all(g[r+dr][c+dc]!='W' for dr in range(5) for dc in range(5))
    adj={}
    for r in cells:
        for c in cells:
            if ok(r,c):
                for dr,dc in ((-6,0),(6,0),(0,-6),(0,6)):
                    r2,c2=r+dr,c+dc
                    if r2 in cells and c2 in cells and ok(r2,c2) and g[(r+r2)//2][(c+c2)//2]!='W':
                        adj.setdefault((r,c),[]).append((r2,c2))
    return adj

def cellch(r,c): return gridof(current_frame)[r][c]

def wait_pair(a,b):
    for d in (a,b):
        np,no,bar=go(M[d])
        if no: print('  WAIT NOOP',d)

def walk(plan, risky):
    # risky: dict target_cell -> (wait_dirs tuple)
    n=0
    for d in plan:
        tgt=None
        cur=lv3(current_frame)['blk']
        dr={'U':(-6,0),'D':(6,0),'L':(0,-6),'R':(0,6)}[d]
        tgt=(cur[0]+dr[0],cur[1]+dr[1])
        if tgt in risky:
            tries=0
            while True:
                ch=cellch(*tgt)
                if ch!='R' and ch!='Y': break
                tries+=1
                print('  wait for',tgt,'cell',ch,'try',tries)
                if tries>10: return False
                wait_pair(*risky[tgt])
        np,no,bar=go(M[d]); n+=1
        print(n,d,np,'bar',bar,'NOOP' if no else '')
        if no: return False
    return True

def wait_pair2(a,b):
    for d in (a,b):
        np,no,bar=go(MM[d])
        if no: print('  WAITNOOP',d)

def walk2(plan,risky):
    n=0
    for d in plan:
        cur=lv3(current_frame)['blk']
        dr={'U':(-6,0),'D':(6,0),'L':(0,-6),'R':(0,6)}[d]
        tgt=(cur[0]+dr[0],cur[1]+dr[1])
        if tgt in risky:
            tries=0
            while True:
                ch=cellch(*tgt)
                if ch not in ('R','Y'): break
                tries+=1
                print('  wait',tgt,ch,'try',tries)
                if tries>10: return False
                wait_pair2(*risky[tgt])
        np,no,bar=go(MM[d]); n+=1
        print(n,d,np,'bar',bar,'NOOP' if no else '','done' if (last_action_call_result.get('level_completed') or last_action_call_result.get('game_over')) else '')
        if no: return False
        if last_action_call_result.get('level_completed') or last_action_call_result.get('game_over'): return True
    return True

def cellinfo(r,c):
    from collections import Counter
    cnt=Counter(g[r+dr][c+dc] for dr in range(5) for dc in range(5))
    return dict(cnt)

def walk_to(t, stop_on_noop=True, log=True):
    a=maze(); s=lv3(current_frame)['blk']
    p=sp(a,s,t)
    if not p: 
        print('no path',s,t); return False
    for d in dirs(p):
        np,no,bar=go(MM[d])
        if log: print(d,np,'bar',bar,'NOOP' if no else '')
        if no and stop_on_noop: return False
    return True

def walk_adapt(plan, risky, maxwait=12):
    M={'U':'UP','D':'DOWN','L':'LEFT','R':'RIGHT'}
    n=0
    for d in plan:
        cur=lv3(current_frame)['blk']
        dr={'U':(-6,0),'D':(6,0),'L':(0,-6),'R':(0,6)}[d]
        tgt=(cur[0]+dr[0],cur[1]+dr[1])
        if tgt in risky:
            tries=0
            while True:
                ch=gridof(current_frame)[tgt[0]][tgt[1]]
                if ch not in ('R','Y'): break
                tries+=1
                print('   wait for',tgt,ch,'try',tries)
                if tries>maxwait: return False
                for w in risky[tgt]:
                    go(M[w])
        np,no,bar=go(M[d]); n+=1
        print(n,d,np,'bar',bar,'NOOP' if no else '')
        if no: return False
        r=last_action_call_result
        if r.get('level_completed') or r.get('game_over'): return True
    return True
