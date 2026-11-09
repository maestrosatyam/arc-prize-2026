try:
    from collections import deque
except Exception:
    pass
try:
    from collections import Counter
except Exception:
    pass
try:
    from itertools import combinations
except Exception:
    pass
try:
    import json
except Exception:
    pass

def cell(r,c): return a[r][c]

def coarse_map(fr):
    A = fr.ascii.split('\n')
    g=[]
    for i in range(11):
        row=''
        for j in range(11):
            r0,c0=4+5*i,4+5*j
            vals={}
            for r in range(r0,r0+5):
                for c in range(c0,c0+5):
                    ch=A[r][c]
                    vals[ch]=vals.get(ch,0)+1
            # classify: B or S => floor; Y/O => wall
            floor = vals.get('B',0)+vals.get('S',0)
            tot = sum(vals.values())
            row += '.' if floor>tot*0.5 else '#'
        g.append(row)
    return g

def deg(u):
    return [(u[0]+dr,u[1]+dc) for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)) if (u[0]+dr,u[1]+dc) in F]

def bfs(A0,B0,m0):
    from collections import deque
    start=(A0,B0,m0,0)
    seen={start:None}; q=deque([start])
    while q:
        s=q.popleft()
        A,B,m,mode=s
        for d in 'UDLR':
            if mode==0:
                na=(A[0]+D[d][0],A[1]+D[d][1]); nb=(B[0]+D[MR[d]][0],B[1]+D[MR[d]][1])
                if na in HAZ or nb in HAZ: continue   # death
                if na not in FLOOR: na=A
                if nb not in FLOOR or nb==m: nb=B
                if na==m: na=A
                ns=(na,nb,m,0)
                if na==B and nb==A:
                    path=[];cur=s
                    while seen[cur] is not None:
                        dd,pv=seen[cur];path.append(dd);cur=pv
                    return path[::-1]+[d]
            else:
                nm=(m[0]+D[d][0],m[1]+D[d][1])
                if nm in HAZ: continue
                if nm not in FLOOR or nm in (A,B): nm=m
                ns=(A,B,nm,1)
            if ns==s or ns in seen: continue
            seen[ns]=(d,s); q.append(ns)
        # clicks
        for nmode in (0,1):
            if nmode==mode: continue
            ns=(A,B,m,nmode)
            if ns in seen: continue
            seen[ns]=('CLICK%d'%nmode,s); q.append(ns)
    return None

def nxt(s,d,F):
    L,R,k=s
    nL=mv(L,d) if mv(L,d) in F else L
    nR=mvR(R,d) if mvR(R,d) in F else R
    nk=set(k)
    for col,pos in KEYS.items():
        if (nL==pos or nR==pos): nk.add(col)
    return (nL,nR,frozenset(nk))

def joint_bfs(start, goal_fn):
    st=start; seen={st:None}; q=deque([st])
    while q:
        a,b=q.popleft()
        for d in 'UDLR':
            na,nb=nxt(a,d),nxt(b,d)
            if na==nb or (na,nb)==(b,a): continue
            s=(na,nb)
            if s in seen: continue
            seen[s]=(d,(a,b))
            if goal_fn(s):
                path=[]; cur=s
                while seen[cur] is not None:
                    dd,prev=seen[cur]; path.append(dd); cur=prev
                return s, path[::-1]
            q.append(s)
    return None,None

g1=lambda s: s[0][0]==1 and s[1][0]==1

g2=lambda s: s[0]==(9,7) and s[1]==(9,3)

def solo_bfs(p0, goal_fn):
    seen={p0:0}; q=deque([p0])
    while q:
        u=q.popleft()
        if goal_fn(u): return u, seen[u]
        for d in 'UDLR':
            v=nxt(u,d)
            if v not in seen: seen[v]=seen[u]+1; q.append(v)
    return None,None

def spos(fr):
    A=fr.ascii.split('\n')
    out=[]
    for i in range(11):
        for j in range(11):
            r0,c0=4+5*i,4+5*j
            if A[r0+2][c0+2]=='S': out.append((i,j))
    return tuple(out)

def track(fr):
    p=spos(fr)
    return (p[0],p[1]) if len(p)==2 else None

def run_plan(plan, mapstr, n=13, cell=4, off=6, mirrored=True):
    OK={(i,j) for i in range(n) for j in range(n) if mapstr[i][j]=='.'}
    D={'U':(-1,0),'D':(1,0),'L':(0,-1),'R':(0,1)}
    MR={'U':'U','D':'D','L':'R','R':'L'} if mirrored else {'U':'U','D':'D','L':'L','R':'R'}
    NAME={'U':'UP','D':'DOWN','L':'LEFT','R':'RIGHT'}
    def sp(fr):
        A=fr.ascii.split('\n'); out=[]
        for i in range(n):
            for j in range(n):
                r0,c0=off+cell*i,off+cell*j
                if A[r0+1][c0+1]=='S': out.append((i,j))
        return tuple(out)
    def stp(pos,d):
        out=[]
        for k,p in enumerate(pos):
            dd=d if k==0 else MR[d]
            q=(p[0]+D[dd][0],p[1]+D[dd][1])
            out.append(q if q in OK else p)
        return tuple(out)
    pos=sp(current_frame); log=[]
    for d in plan:
        cross = (stp(pos,d)==(pos[1],pos[0]))
        pred=stp(pos,d)
        r=action([NAME[d]])
        act=sp(current_frame)
        term=bool(r.get('level_completed') or r.get('game_over') or r.get('done'))
        tag='ok'
        if term: tag='TERM'
        elif act!=pred: tag='MISMATCH pred=%s'%(pred,)
        log.append((d,tag,act))
        if term:
            log.append(('TERMINAL',r.get('state'),'LVL' if r.get('level_completed') else 'OVER' if r.get('game_over') else 'DONE'))
            return log,act
        if act!=pred: return log,act
        pos=pred
        if cross: log.append(('CROSSING-MOVE done',))
    return log,pos

def probe(acts):
    out=[]
    for a in acts:
        p0=track(current_frame)
        r=action([a])
        p1=track(current_frame)
        delta=[None,None]
        for k in range(2):
            if p0 and p1: delta[k]=(p1[k][0]-p0[k][0], p1[k][1]-p0[k][1])
        out.append((a,p0,p1,delta,r.get('level_completed'),r.get('game_over')))
        if r.get('level_completed') or r.get('game_over') or r.get('done'): break
    return out

def blk(i,j):
    r0,c0=4+5*i,4+5*j
    return A[r0+2][c0+2]

def step(p,d,F):
    i,j=p
    q={'U':(i-1,j),'D':(i+1,j),'L':(i,j+1),'R':(i,j-1)}[d]
    return q if q in F else p

def run_mirror(plan):
    MAZE = ["###########","###......##","##...#....#","#...###...#","#..#####..#","##.######.#","##.######.#","#...####..#","#....#....#","#....#....#","#....#...##"]
    FL={(i,j) for i in range(11) for j in range(11) if MAZE[i][j]=='.'}
    D={'U':(-1,0),'D':(1,0),'L':(0,-1),'R':(0,1)}
    MR={'U':'U','D':'D','L':'R','R':'L'}
    NAME={'U':'UP','D':'DOWN','L':'LEFT','R':'RIGHT'}
    def stp(pos,d):
        out=[]
        for k,p in enumerate(pos):
            dd = d if k==0 else MR[d]
            q=(p[0]+D[dd][0],p[1]+D[dd][1])
            out.append(q if q in FL else p)
        return tuple(out)
    pos=track(current_frame)
    log=[]
    for d in plan:
        pred=stp(pos,d)
        r=action([NAME[d]])
        act=track(current_frame)
        term=bool(r.get('level_completed') or r.get('game_over') or r.get('done'))
        tag='ok'
        if term: tag='TERM'
        elif act!=pred: tag='MISMATCH pred=%s'%pred
        log.append((d,tag,act))
        if term:
            log.append(('TERMINAL',r.get('state'),r.get('level_completed'),r.get('game_over'),r.get('done')))
            return log,act
        if act!=pred: return log,act
        pos=pred
    return log,pos

def budget():
    A=current_frame.ascii.split('\n')
    return sum(1 for c in A[0] if c=='B'), sum(1 for c in A[63] if c=='B')

def cmap(A):
    g=[]
    for i in range(13):
        row=''
        for j in range(13):
            r0,c0=6+4*i,6+4*j
            cnt={}
            for r in range(r0,r0+4):
                for c in range(c0,c0+4):
                    ch=A[r][c]; cnt[ch]=cnt.get(ch,0)+1
            red=cnt.get('R',0)
            if red>4: row+='X'
            elif cnt.get('B',0)>=12: row+='.'
            else: row+='#'
        g.append(row)
    return g

def spos4(fr):
    A=fr.ascii.split('\n')
    out=[]
    for i in range(13):
        for j in range(13):
            r0,c0=6+4*i,6+4*j
            if A[r0+1][c0+1]=='S': out.append((i,j))
    return tuple(out)

def run2(plan, mapstr, n=13, cell=4, off=6, mirrored=True):
    FREE={(i,j) for i in range(n) for j in range(n) if mapstr[i][j]=='.'}
    HAZ={(i,j) for i in range(n) for j in range(n) if mapstr[i][j]=='X'}
    D={'U':(-1,0),'D':(1,0),'L':(0,-1),'R':(0,1)}
    MR={'U':'U','D':'D','L':'R','R':'L'} if mirrored else {'U':'U','D':'D','L':'L','R':'R'}
    NAME={'U':'UP','D':'DOWN','L':'LEFT','R':'RIGHT'}
    def sp(fr):
        A=fr.ascii.split('\n'); out=[]
        for i in range(n):
            for j in range(n):
                r0,c0=off+cell*i,off+cell*j
                if A[r0+1][c0+1]=='S': out.append((i,j))
        return tuple(out)
    def stp(pos,d):
        out=[];dead=False
        for k,p in enumerate(pos):
            dd=d if k==0 else MR[d]
            q=(p[0]+D[dd][0],p[1]+D[dd][1])
            if q in FREE: out.append(q)
            elif q in HAZ: dead=True; out.append(q)
            else: out.append(p)
        return tuple(out),dead
    pos=sp(current_frame); log=[]
    for d in plan:
        pred,dead=stp(pos,d)
        if dead: return log+([('ABORT-HAZARD',d,pred)],pos)
        cross = (pred==(pos[1],pos[0]))
        r=action([NAME[d]])
        act=sp(current_frame)
        term=bool(r.get('level_completed') or r.get('game_over') or r.get('done'))
        tag='ok'
        if term: tag='TERM'
        elif act!=pred: tag='MISMATCH pred=%s'%(pred,)
        log.append((d,tag,act))
        if term:
            log.append(('TERMINAL','LVL' if r.get('level_completed') else 'OVER' if r.get('game_over') else 'DONE'))
            return log,act
        if act!=pred: return log,act
        pos=pred
        if cross: log.append(('CROSS',))
    return log,pos

def cmap3(A):
    g=[];info=[]
    for i in range(13):
        row=''
        for j in range(13):
            r0,c0=6+4*i,6+4*j
            cnt={}
            for r in range(r0,r0+4):
                for c in range(c0,c0+4):
                    ch=A[r][c]; cnt[ch]=cnt.get(ch,0)+1
            blk=cnt.get('B',0); s=cnt.get('S',0); bb=cnt.get('b',0)
            bg=cnt.get('p',0)+cnt.get('R',0)
            if s: cls='S'
            elif bb: cls='k'
            elif blk==16: cls='.'
            elif blk==0: cls='#'
            else: cls='?'
            row+=cls; info.append((i,j,cls,cnt))
        g.append(row)
    return g,info

def run3(plan, mapstr, n=13, cell=4, off=6):
    FREE={(i,j) for i in range(n) for j in range(n) if mapstr[i][j]!='#'}
    HAZ={(i,j) for i in range(n) for j in range(n) if mapstr[i][j]=='X'}
    D={'U':(-1,0),'D':(1,0),'L':(0,-1),'R':(0,1)}
    MR={'U':'U','D':'D','L':'R','R':'L'}
    NAME={'U':'UP','D':'DOWN','L':'LEFT','R':'RIGHT'}
    def sp(fr):
        A=fr.ascii.split('\n'); out=[]
        for i in range(n):
            for j in range(n):
                r0,c0=off+cell*i,off+cell*j
                if A[r0+1][c0+1]=='S': out.append((i,j))
        return set(out)
    def stp(pos,d):
        out=[];dead=False
        for k,p in enumerate(pos):
            dd=d if k==0 else MR[d]
            q=(p[0]+D[dd][0],p[1]+D[dd][1])
            if q in FREE: out.append(q)
            elif q in HAZ: dead=True; out.append(q)
            else: out.append(p)
        return tuple(out),dead
    # initialize identity: assume index0 = leftmost (smaller col) square
    obs=sorted(sp(current_frame))
    pos=(obs[0],obs[1])
    log=[]
    for d in plan:
        pred,dead=stp(pos,d)
        if dead: return log+[('ABORT-HAZARD',d,pred)],pos
        r=action([NAME[d]])
        obs=sp(current_frame)
        term=bool(r.get('level_completed') or r.get('game_over') or r.get('done'))
        tag='ok' if set(pred)==obs else 'MISMATCH pred=%s obs=%s'%(pred,obs)
        log.append((d,tag,pred))
        if term:
            log.append(('TERMINAL','LVL' if r.get('level_completed') else 'OVER' if r.get('game_over') else 'DONE'))
            return log,pred
        if set(pred)!=obs: return log,pred
        pos=pred
    return log,pos

def reach(p0,walls):
    FREE=mk_free(walls); s={p0}; q=deque([p0])
    while q:
        u=q.popleft()
        for dr,dc in D.values():
            v=(u[0]+dr,u[1]+dc)
            if v in FREE and v not in s: s.add(v);q.append(v)
    return s

def run4(plan, pos, mapstr, wall=(), n=13, cell=4, off=6):
    FREE={(i,j) for i in range(n) for j in range(n) if mapstr[i][j]!='#' and (i,j) not in set(wall)}
    D={'U':(-1,0),'D':(1,0),'L':(0,-1),'R':(0,1)}
    MR={'U':'U','D':'D','L':'R','R':'L'}
    NAME={'U':'UP','D':'DOWN','L':'LEFT','R':'RIGHT'}
    def sp(fr):
        A=fr.ascii.split('\n'); out=[]
        for i in range(n):
            for j in range(n):
                r0,c0=off+cell*i,off+cell*j
                if A[r0+1][c0+1]=='S': out.append((i,j))
        return set(out)
    def st(p,d):
        out=[]
        for k,p0 in enumerate(p):
            dd=d if k==0 else MR[d]
            q=(p0[0]+D[dd][0],p0[1]+D[dd][1])
            out.append(q if q in FREE else p0)
        return tuple(out)
    log=[]
    for d in plan:
        pred=st(pos,d)
        r=action([NAME[d]])
        obs=sp(current_frame)
        term=bool(r.get('level_completed') or r.get('game_over') or r.get('done'))
        tag='ok' if obs==set(pred) else 'MISMATCH pred=%s obs=%s'%(pred,obs)
        log.append((d,tag,pred))
        if term:
            log.append(('TERMINAL','LVL' if r.get('level_completed') else 'OVER' if r.get('game_over') else 'DONE'))
            return log,pred
        if obs!=set(pred): return log,pred
        pos=pred
    return log,pos

def step2(pos,FREE,d):
    MR={'U':'U','D':'D','L':'R','R':'L'}; D={'U':(-1,0),'D':(1,0),'L':(0,-1),'R':(0,1)}
    out=[]
    for k,p in enumerate(pos):
        dd=d if k==0 else MR[d]
        q=(p[0]+D[dd][0],p[1]+D[dd][1])
        out.append(q if q in FREE else p)
    return tuple(out)

def scan(start):
    seen={start}; q=deque([start]); mind=99; states=[]
    while q:
        s=q.popleft()
        dd=abs(s[0][0]-s[1][0])+abs(s[0][1]-s[1][1])
        if dd<mind: mind=dd; states=[s]
        elif dd==mind: states.append(s)
        for d in 'UDLR':
            n=step(s,d)
            if n[0]==n[1]: continue
            if n in seen: continue
            seen.add(n); q.append(n)
    return mind,states,len(seen)

def snap(tag):
    seg=current_frame.segmentation
    print(tag,[(n['color'],n['pixels'],n['boundary'][0]) for n in seg['nodes'] if n['color'] in ('S','w','Y','b')])

def free_from(mk):
    bad=set(mk)
    return {(i,j) for i in range(13) for j in range(13) if BASE[i][j]!='#' and (i,j) not in bad}

def cross(start,walls):
    FREE=mk_free(walls)
    def step(pos,d):
        out=[]
        for k,p in enumerate(pos):
            dd=d if k==0 else MR[d]
            q=(p[0]+D[dd][0],p[1]+D[dd][1])
            out.append(q if q in FREE else p)
        return tuple(out)
    seen={start:None}; q=deque([start])
    while q:
        s=q.popleft()
        for d in 'UDLR':
            n=step(s,d)
            if n[0]==n[1]: continue
            if n==(s[1],s[0]):
                path=[];cur=s
                while seen[cur] is not None:
                    dd,pv=seen[cur];path.append(dd);cur=pv
                return path[::-1]+[d]
            if n in seen: continue
            seen[n]=(d,s);q.append(n)
    return None

def mk_free(walls):
    return {(i,j) for i in range(13) for j in range(13) if BASE[i][j]!='#' and (i,j) not in set(walls)}

def comp(p0):
    s={p0};q=deque([p0])
    while q:
        u=q.popleft()
        for dr,dc in D:
            v=(u[0]+dr,u[1]+dc)
            if v in FREE and v not in s: s.add(v);q.append(v)
    return s

def solve(L0,R0):
    start=(L0,R0)
    seen={start}; par={start:None}; Q=deque([start]); found=None
    while Q:
        s=Q.popleft()
        F=freer(held(s[0],s[1]))
        for d in 'UDLR':
            a=s[0]; b=s[1]
            na=mv(a,d) if mv(a,d) in F else a
            nb=mvR(b,d) if mvR(b,d) in F else b
            ns=(na,nb)
            if ns==s: continue
            if na==nb or (na==b and nb==a): return par,s,ns,d
            if ns not in seen: seen.add(ns); par[ns]=(s,d); Q.append(ns)
    return None

def explore(start,walls):
    FREE={(i,j) for i in range(13) for j in range(13) if BASE[i][j]!='#' and (i,j) not in set(walls)}
    def step(pos,d):
        out=[]
        for k,p in enumerate(pos):
            dd=d if k==0 else MR[d]
            q=(p[0]+D[dd][0],p[1]+D[dd][1])
            out.append(q if q in FREE else p)
        return tuple(out)
    seen={start}; q=deque([start]); mind=99; ex=[]
    while q:
        s=q.popleft()
        dd=abs(s[0][0]-s[1][0])+abs(s[0][1]-s[1][1])
        if dd<mind: mind=dd; ex=[s]
        elif dd==mind: ex.append(s)
        for d in 'UDLR':
            n=step(s,d)
            if n[0]==n[1] or n in seen: continue
            seen.add(n); q.append(n)
    return seen,mind,ex

def paths(src,blocked):
    blocked=set(blocked)
    dist={src:('')}
    q=deque([src])
    while q:
        u=q.popleft()
        for dn,(dr,dc) in D.items():
            v=(u[0]+dr,u[1]+dc)
            if v in CELLS and v not in blocked and v not in dist:
                dist[v]=dist[u]+dn; q.append(v)
    return dist

def crossing(walls, s0):
    FREE=CELLS-set(walls)
    def step(pos,d):
        out=[]
        for k,p in enumerate(pos):
            dd=d if k==0 else MR[d]
            q=(p[0]+D[dd][0],p[1]+D[dd][1])
            out.append(q if q in FREE else p)
        return tuple(out)
    seen={s0:None}; q=deque([s0])
    while q:
        s=q.popleft()
        for d in 'UDLR':
            n=step(s,d)
            if n[0]==n[1]: continue
            if n==(s[1],s[0]):
                path=[];cur=s
                while seen[cur] is not None:
                    dd,pv=seen[cur];path.append(dd);cur=pv
                return path[::-1]+[d]
            if n in seen: continue
            seen[n]=(d,s);q.append(n)
    return None

def rd(fr):
    out={}
    for n in fr.segmentation['nodes']:
        if n['color'] in ('S','w','Y','b'):
            r,c=n['boundary'][0]
            cell=((r-6)//4,(c-6)//4) if n['pixels']==16 else ((r-6+1)//4,(c-6+1)//4)
            out.setdefault(n['color'],set()).add(cell)
    return out

def cellpx(cell,mk=False):
    i,j=cell; r=6+4*i; c=6+4*j
    return (r+2,c+2) if mk else (r+1,c+1)

def do(steps, expect, tag=''):
    acts=[]
    for s in steps:
        acts.append(s if isinstance(s,dict) else s)
    r=action(acts)
    st=rd(current_frame)
    ok=True; msgs=[]
    for k,v in expect.items():
        got=st.get(k,set())
        if got!=set(v): ok=False; msgs.append('%s exp %s got %s'%(k,sorted(v),sorted(got)))
    term=bool(r.get('level_completed') or r.get('game_over') or r.get('done'))
    print(tag,'OK' if ok else 'MISMATCH', msgs, 'exec',r.get('executed'),
          'LVL' if r.get('level_completed') else ('OVER' if r.get('game_over') else ''),
          'stop',r.get('stop_reason'),'budget',budget())
    return ok,term,st

def stp(pos,d):
    out=[]
    for k,p in enumerate(pos):
        dd=d if k==0 else MR[d]
        q=(p[0]+D[dd][0],p[1]+D[dd][1])
        out.append(q if q in FREE else p)
    return tuple(out)

def classify(i,j):
    from collections import Counter
    c=Counter()
    for r in range(6+4*i,6+4*i+4):
        for cc in range(6+4*j,6+4*j+4):
            c[A[r][cc]]+=1
    R=c['R']+c['O']+c['r']; Bk=c['B']+c['c']+c['G']
    tot=sum(c.values())
    if R>=3 and Bk>=3: return 'X'   # checkerboard hazard
    if Bk>=tot-1: return '#'
    return '.'

def cellinfo(i,j,cell=5,off=4):
    c=Counter(A[r][cc] for r in range(off+cell*i,off+cell*i+cell) for cc in range(off+cell*j,off+cell*j+cell))
    tot=sum(c.values()); Bk=c['B']+c['c']+c['G']
    top=max(c,key=lambda k:c[k])
    if Bk>=tot*0.7: return '#'
    return {'P':'P','N':'N','O':'O','p':'p','S':'S','M':'M','Y':'Y','R':'R','b':'b','w':'w','W':'W','G':'G','g':'g','c':'c'}.get(top,'?')

def rd4(fr):
    out={}
    for n in fr.segmentation['nodes']:
        if n['color'] in ('S','w','Y','b'):
            r,c=n['boundary'][0]; px=n['pixels']
            if px>=20: cell=((r-4)//5,(c-4)//5)
            elif px>=6: cell=((r-4+1)//5,(c-4+1)//5)
            else: cell=None
            if cell: out.setdefault(n['color'],set()).add(cell)
    return out

def stepall(s,d):
    A,B,m,mode=s
    if mode==0:
        na=(A[0]+D[d][0],A[1]+D[d][1]); nb=(B[0]+D[MR[d]][0],B[1]+D[MR[d]][1])
        dead = na in HAZ or nb in HAZ
        if na not in FLOOR or na==m or na==B: na=A
        if nb not in FLOOR or nb==m or nb==A: nb=B
        return (na,nb,m,0),dead
    nm=(m[0]+D[d][0],m[1]+D[d][1])
    dead = nm in HAZ
    if nm not in FLOOR or nm in (A,B): nm=m
    return (A,B,nm,mode),dead

def px4(cell): i,j=cell; return (4+5*i+2, 4+5*j+2)

def rdL4(fr):
    out={}
    for n in fr.segmentation['nodes']:
        if n['color'] in ('S','w','Y','b') and n['pixels']<40:
            b=n['boundary']; r=min(p[0] for p in b); c=min(p[1] for p in b)
            k=n['pixels']
            cell=((r-4)//5,(c-4)//5) if k>=20 else ((r-5)//5,(c-5)//5)
            out.setdefault(n['color'],set()).add(cell)
    return out

def do4(steps,expect,tag=''):
    r=action(steps); st=rdL4(current_frame); ok=True; msgs=[]
    for k,v in expect.items():
        got=st.get(k,set())
        if got!=set(v): ok=False; msgs.append('%s exp %s got %s'%(k,sorted(v),sorted(got)))
    print(tag,'OK' if ok else 'MISMATCH',msgs,'exec',r.get('executed'),
          'LVL' if r.get('level_completed') else ('OVER' if r.get('game_over') else ''),
          'stop',r.get('stop_reason'),'budget',budget())
    return ok,bool(r.get('level_completed') or r.get('game_over') or r.get('done')),st

def rdL5(fr):
    out={}
    for n in fr.segmentation['nodes']:
        if n['color'] in ('S','w','Y') and 10<n['pixels']<30:
            b=n['boundary']; r=min(p[0] for p in b); c=min(p[1] for p in b)
            out.setdefault(n['color'],set()).add(((r-2)//4,(c-2)//4))
    return out

def ci(A,i,j,cell=4,off=2):
    c=Counter(A[r][cc] for r in range(off+cell*i,off+cell*i+cell) for cc in range(off+cell*j,off+cell*j+cell))
    tot=sum(c.values()); Bk=c['B']+c['c']+c['G']
    if Bk>=tot*0.75: return '#'
    return max((k for k in c if k!='B'),key=lambda k:c[k])

def px5(cell): i,j=cell; return (2+4*i+2, 2+4*j+2)

def snap5():
    out=[]
    for n in current_frame.segmentation['nodes']:
        if n['pixels']<60 and n['pixels']>5 and n['color'] not in ('B',):
            bb=n['boundary']; rs=[x[0] for x in bb]; cs=[x[1] for x in bb]
            out.append((n['color'],n['pixels'],((min(rs)-2)//4,(min(cs)-2)//4)))
    return sorted(out,key=lambda t:(t[2],t[0]))

def stepR(p,d,F):
    i,j=p
    q={'U':(i-1,j),'D':(i+1,j),'L':(i,j-1),'R':(i,j+1)}[d]
    return q if q in F else p

def goal(s): return s[0]==s[1]

def inb(p): return 0<=p[0]<N and 0<=p[1]<N

def free_with(keys):
    F=set(FREE0)
    for col,cells in DOOR.items():
        if col in keys: F|=cells
    F|=set(KEYS.values())
    return F

def mv(p,d):
    i,j=p; return {'U':(i-1,j),'D':(i+1,j),'L':(i,j+1),'R':(i,j-1)}[d]

def mvR(p,d):
    i,j=p; return {'U':(i-1,j),'D':(i+1,j),'L':(i,j-1),'R':(i,j+1)}[d]

def sim(plan, start):
    s=start
    F=free_with(set(s[2]))
    trace=[s]
    for d in plan:
        F=free_with(set(s[2]))
        s=nxt(s,d,F); trace.append(s)
    return trace

def freeL(keys,DOOR=DOOR,FREE0=FREE0,KEYS=KEYS):
    F=set(FREE0)
    for col,cells in DOOR.items():
        if col in keys: F|=cells
    return F

def nxtL(s,d,F,KEYS=KEYS,mv=mv,mvR=mvR):
    L,R,k=s
    nL=mv(L,d) if mv(L,d) in F else L
    nR=mvR(R,d) if mvR(R,d) in F else R
    nk=set(k)
    for col,pos in KEYS.items():
        if nL==pos or nR==pos: nk.add(col)
    return (nL,nR,frozenset(nk))

def planL(start,goalcell=(4,7)):
    seen={start}; par={start:None}; Q=deque([start]); found=None
    while Q:
        s=Q.popleft()
        if s[0]==s[1]==goalcell: found=s; break
        F=freeL(set(s[2]))
        for d in 'UDLR':
            ns=nxtL(s,d,F)
            if ns not in seen: seen.add(ns); par[ns]=(s,d); Q.append(ns)
    if found is None: return None
    out=[]; s=found
    while par[s] is not None:
        p,d=par[s]; out.append(d); s=p
    return out[::-1]

def held(L,R): return set(c for c,ps in KEYC.items() if L in ps or R in ps)

def freer(h,DOOR=DOOR,GROUND=GROUND):
    F=set(GROUND)
    for col,cells in DOOR.items():
        if col in h: F|=cells
    return F

def solve5(L0,R0,maxd=200):
    ROOMS=[(1,4,0,14),(6,8,0,6),(6,8,8,14),(10,13,0,6),(10,13,8,14)]
    G=set()
    for (r0,r1,c0,c1) in ROOMS:
        for i in range(r0,r1+1):
            for j in range(c0,c1+1): G.add((i,j))
    G.discard((11,3))
    DOOR={'N':{(5,3),(5,4),(5,5)},'O':{(9,2),(9,3),(9,4)},'p':{(5,9),(5,10),(5,11),(9,9),(9,10),(9,11)}}
    KEYC={'N':{(6,8)},'O':{(6,14)},'p':{(1,3),(12,3)}}
    G|=set().union(*KEYC.values())
    def held(L,R):
        return frozenset(c for c,ps in KEYC.items() if L in ps or R in ps)
    def free(h):
        F=set(G)
        for c,cells in DOOR.items():
            if c in h: F|=cells
        return F
    def go(p,d,mir):
        i,j=p
        if mir: q={'U':(i-1,j),'D':(i+1,j),'L':(i,j+1),'R':(i,j-1)}[d]
        else:   q={'U':(i-1,j),'D':(i+1,j),'L':(i,j-1),'R':(i,j+1)}[d]
        return q
    start=(L0,R0)
    seen={start}; par={start:None}; Q=deque([start])
    while Q:
        s=Q.popleft()
        F=free(held(s[0],s[1]))
        for d in 'UDLR':
            a,b=s
            qa=go(a,d,True); qb=go(b,d,False)
            na=qa if qa in F else a
            nb=qb if qb in F else b
            if (na,nb)==(a,b): continue
            if na==nb or (na==b and nb==a):
                out=[d]; t=s
                while par[t] is not None:
                    p2,dd=par[t]; out.append(dd); t=p2
                return ''.join(out[::-1])
            if (na,nb) not in seen:
                seen.add((na,nb)); par[(na,nb)]=(s,d); Q.append((na,nb))
    return None

def obs5(fr):
    ss=[]
    for n in fr.segmentation['nodes']:
        if n['color']=='S' and 10<n['pixels']<30:
            b=n['boundary']; ss.append(((min(p[0] for p in b)-2)//4,(min(p[1] for p in b)-2)//4))
    return ss

def model5():
    ROOMS=[(1,4,0,14),(6,8,0,6),(6,8,8,14),(10,13,0,6),(10,13,8,14)]
    G=set()
    for (r0,r1,c0,c1) in ROOMS:
        for i in range(r0,r1+1):
            for j in range(c0,c1+1): G.add((i,j))
    G.discard((11,3))
    DOOR={'N':{(5,3),(5,4),(5,5)},'O':{(9,2),(9,3),(9,4)},'p':{(5,9),(5,10),(5,11),(9,9),(9,10),(9,11)}}
    KEYC={'N':{(6,8)},'O':{(6,14)},'p':{(1,3),(12,3)}}
    G|=set().union(*KEYC.values())
    return G,DOOR,KEYC

def run5(plan,L0,R0):
    G,DOOR,KEYC=model5()
    def held(L,R): return set(c for c,ps in KEYC.items() if L in ps or R in ps)
    def free(h):
        F=set(G)
        for c,cells in DOOR.items():
            if c in h: F|=cells
        return F
    def go(p,d,mir):
        i,j=p
        if mir: return {'U':(i-1,j),'D':(i+1,j),'L':(i,j+1),'R':(i,j-1)}[d]
        return {'U':(i-1,j),'D':(i+1,j),'L':(i,j-1),'R':(i,j+1)}[d]
    acts={'U':'UP','D':'DOWN','L':'LEFT','R':'RIGHT'}
    L,R=L0,R0
    for k,d in enumerate(plan):
        F=free(held(L,R))
        qa=go(L,d,True); qb=go(R,d,False)
        nL=qa if qa in F else L; nR=qb if qb in F else R
        r=action([{'action':acts[d]}])
        done=r.get('level_completed') or r.get('game_over') or r.get('done')
        ss=obs5(current_frame)
        # assign
        if len(ss)==2:
            import math
            d1=abs(ss[0][0]-nL[0])+abs(ss[0][1]-nL[1])+abs(ss[1][0]-nR[0])+abs(ss[1][1]-nR[1])
            d2=abs(ss[1][0]-nL[0])+abs(ss[1][1]-nL[1])+abs(ss[0][0]-nR[0])+abs(ss[0][1]-nR[1])
            if d1<=d2: oL,oR=ss[0],ss[1]
            else: oL,oR=ss[1],ss[0]
        else:
            print('k',k,d,'odd obs',ss,'over',r.get('game_over'),'lvl',r.get('level_completed')); return None
        if (oL,oR)!=(nL,nR) and not done:
            print('MISMATCH k',k,d,'exp',(nL,nR),'got',(oL,oR),'noop',r.get('no_op')); return (oL,oR)
        L,R=oL,oR
        if done:
            print('END at k',k,d,'lvl',r.get('level_completed'),'over',r.get('game_over'),'done',r.get('done'))
            return 'DONE'
    print('plan finished, no completion. pos',L,R)
    return (L,R)

def heldX(L,R): return set(c for c,ps in KEYC.items() if L in ps or R in ps)

def freeX(h):
    F=set(G)
    for c,cells in DOOR.items():
        if c in h: F|=cells
    return F

def goX(p,d,mir):
    i,j=p
    if mir: return {'U':(i-1,j),'D':(i+1,j),'L':(i,j+1),'R':(i,j-1)}[d]
    return {'U':(i-1,j),'D':(i+1,j),'L':(i,j-1),'R':(i,j+1)}[d]

def solve_to(L0,R0,tgtL,tgtR,upblocked=('p',)):
    G,DOOR,KEYC=model5()
    def held(L,R): return set(c for c,ps in KEYC.items() if L in ps or R in ps)
    def free(h):
        F=set(G)
        for c,cells in DOOR.items():
            if c in h: F|=cells
        return F
    def go(p,d,mir):
        i,j=p
        if mir: return {'U':(i-1,j),'D':(i+1,j),'L':(i,j+1),'R':(i,j-1)}[d]
        return {'U':(i-1,j),'D':(i+1,j),'L':(i,j-1),'R':(i,j+1)}[d]
    start=(L0,R0); seen={start}; par={start:None}; Q=deque([start])
    while Q:
        s=Q.popleft()
        if s[0]==tgtL and s[1]==tgtR:
            out=[]; t=s
            while par[t] is not None:
                p2,dd=par[t]; out.append(dd); t=p2
            return ''.join(out[::-1])
        F=free(held(s[0],s[1]))
        for d in 'UDLR':
            a,b=s
            qa=go(a,d,True); qb=go(b,d,False)
            na=qa if qa in F else a
            nb=qb if qb in F else b
            if (na,nb)==s: continue
            if (na,nb) not in seen: seen.add((na,nb)); par[(na,nb)]=(s,d); Q.append((na,nb))
    return None

def solve_any(L0,R0,tgtsL,tgtR):
    G,DOOR,KEYC=model5()
    def held(L,R): return set(c for c,ps in KEYC.items() if L in ps or R in ps)
    def free(h):
        F=set(G)
        for c,cells in DOOR.items():
            if c in h: F|=cells
        return F
    def go(p,d,mir):
        i,j=p
        if mir: return {'U':(i-1,j),'D':(i+1,j),'L':(i,j+1),'R':(i,j-1)}[d]
        return {'U':(i-1,j),'D':(i+1,j),'L':(i,j-1),'R':(i,j+1)}[d]
    start=(L0,R0); seen={start}; par={start:None}; Q=deque([start])
    while Q:
        s=Q.popleft()
        if s[1]==tgtR and s[0] in tgtsL:
            out=[]; t=s
            while par[t] is not None:
                p2,dd=par[t]; out.append(dd); t=p2
            return ''.join(out[::-1])
        F=free(held(s[0],s[1]))
        for d in 'UDLR':
            a,b=s
            qa=go(a,d,True); qb=go(b,d,False)
            na=qa if qa in F else a
            nb=qb if qb in F else b
            if (na,nb)==s: continue
            if (na,nb) not in seen: seen.add((na,nb)); par[(na,nb)]=(s,d); Q.append((na,nb))
    return None

def free(h):
    F=set(G)
    for c,cells in DOOR.items():
        if c in h: F|=cells
    return F

def go(p,d,mir):
    i,j=p
    if mir: return {'U':(i-1,j),'D':(i+1,j),'L':(i,j+1),'R':(i,j-1)}[d]
    return {'U':(i-1,j),'D':(i+1,j),'L':(i,j-1),'R':(i,j+1)}[d]

def solve_any2(L0,R0,tgtsL,tgtsR):
    G,DOOR,KEYC=model5()
    def held(L,R): return set(c for c,ps in KEYC.items() if L in ps or R in ps)
    def free(h):
        F=set(G)
        for c,cells in DOOR.items():
            if c in h: F|=cells
        return F
    def go2(p,d,mir):
        i,j=p
        if mir: return {'U':(i-1,j),'D':(i+1,j),'L':(i,j+1),'R':(i,j-1)}[d]
        return {'U':(i-1,j),'D':(i+1,j),'L':(i,j-1),'R':(i,j+1)}[d]
    start=(L0,R0); seen={start}; par={start:None}; Q=deque([start])
    while Q:
        s=Q.popleft()
        if s[0] in tgtsL and s[1] in tgtsR:
            out=[]; t=s
            while par[t] is not None:
                p2,dd=par[t]; out.append(dd); t=p2
            return ''.join(out[::-1])
        F=free(held(s[0],s[1]))
        for d in 'UDLR':
            a,b=s
            qa=go2(a,d,True); qb=go2(b,d,False)
            na=qa if qa in F else a; nb=qb if qb in F else b
            if (na,nb)!=s and (na,nb) not in seen:
                seen.add((na,nb)); par[(na,nb)]=(s,d); Q.append((na,nb))
    return None

def geo5():
    A=current_frame.ascii.split('\n')
    G=set()
    for i in range(15):
        for j in range(15):
            if ci(A,i,j)=='#': G.add((i,j))
    keys={'N':{(6,8)},'O':{(6,14)},'p':{(1,3),(12,3)}}
    DOOR={'N':{(5,3),(5,4),(5,5)},'O':{(9,2),(9,3),(9,4)},'p':{(5,10),(5,11),(5,12),(9,10),(9,11),(9,12)}}
    for ps in keys.values(): G|=ps
    for ps in DOOR.values(): G|=ps
    return G,DOOR,keys

def solve5b(L0,R0):
    G,DOOR,KEYC=geo5()
    def held(L,R): return set(c for c,ps in KEYC.items() if L in ps or R in ps)
    def free(h):
        F=set(G)
        for c,cells in DOOR.items():
            if c in h: F|=cells
        return F
    def g5(p,d,mir):
        i,j=p
        if mir: return {'U':(i-1,j),'D':(i+1,j),'L':(i,j+1),'R':(i,j-1)}[d]
        return {'U':(i-1,j),'D':(i+1,j),'L':(i,j-1),'R':(i,j+1)}[d]
    start=(L0,R0); seen={start}; par={start:None}; Q=deque([start])
    while Q:
        s=Q.popleft()
        F=free(held(s[0],s[1]))
        for d in 'UDLR':
            a,b=s
            qa=g5(a,d,True); qb=g5(b,d,False)
            na=qa if qa in F else a; nb=qb if qb in F else b
            if (na,nb)==s: continue
            if na==nb or (na==b and nb==a):
                out=[d]; t=s
                while par[t] is not None:
                    p2,dd=par[t]; out.append(dd); t=p2
                return ''.join(out[::-1])
            if (na,nb) not in seen: seen.add((na,nb)); par[(na,nb)]=(s,d); Q.append((na,nb))
    return None

def h5(L,R): return set(c for c,ps in KEYC.items() if L in ps or R in ps)

def f5(h):
    F=set(G)
    for c,cells in DOOR.items():
        if c in h: F|=cells
    return F

def t5(L,R,plan):
    print('start',L,R)
    for k,d in enumerate(plan):
        F=f5(h5(L,R))
        qa=go((L[0],L[1]),d,True); qb=go((R[0],R[1]),d,False)
        nL=qa if qa in F else L; nR=qb if qb in F else R
        L,R=nL,nR
        print(k,d,L,R,'held',sorted(h5(L,R)))

def geo5b():
    A=current_frame.ascii.split('\n')
    G=set()
    for i in range(15):
        for j in range(15):
            if ci(A,i,j)=='#': G.add((i,j))
    keys={'N':{(6,8)},'O':{(6,14)},'p':{(1,3),(12,3)}}
    DOOR={'N':{(5,3),(5,4),(5,5)},'O':{(9,2),(9,3),(9,4)},'p':{(5,10),(5,11),(5,12),(9,10),(9,11),(9,12)}}
    for ps in keys.values(): G|=ps
    return G,DOOR,keys

def solve5c(L0,R0):
    G,DOOR,KEYC=geo5b()
    def held(L,R): return set(c for c,ps in KEYC.items() if L in ps or R in ps)
    def free(h):
        F=set(G)
        for c,cells in DOOR.items():
            if c in h: F|=cells
        return F
    def g5(p,d,mir):
        i,j=p
        if mir: return {'U':(i-1,j),'D':(i+1,j),'L':(i,j+1),'R':(i,j-1)}[d]
        return {'U':(i-1,j),'D':(i+1,j),'L':(i,j-1),'R':(i,j+1)}[d]
    start=(L0,R0); seen={start}; par={start:None}; Q=deque([start])
    while Q:
        s=Q.popleft()
        F=free(held(s[0],s[1]))
        for d in 'UDLR':
            a,b=s
            qa=g5(a,d,True); qb=g5(b,d,False)
            na=qa if qa in F else a; nb=qb if qb in F else b
            if (na,nb)==s: continue
            if na==nb or (na==b and nb==a):
                out=[d]; t=s
                while par[t] is not None:
                    p2,dd=par[t]; out.append(dd); t=p2
                return ''.join(out[::-1])
            if (na,nb) not in seen: seen.add((na,nb)); par[(na,nb)]=(s,d); Q.append((na,nb))
    return None

def tr(L,R,plan):
    print('start',L,R)
    for k,d in enumerate(plan):
        F=set(G)
        for c,cells in DOOR.items():
            if L in KEYC[c] or R in KEYC[c]: F|=cells
        qa=go(L,d,True); qb=go(R,d,False)
        nL=qa if qa in F else L; nR=qb if qb in F else R
        L,R=nL,nR
        hd=sorted(c for c,ps in KEYC.items() if L in ps or R in ps)
        print(k,d,L,R,'held',hd)

def geo5d():
    A=current_frame.ascii.split('\n')
    DOOR={'N':{(5,3),(5,4),(5,5)},'O':{(9,2),(9,3),(9,4)},'p':{(5,10),(5,11),(5,12),(9,10),(9,11),(9,12)}}
    alld=set().union(*DOOR.values())
    G=set()
    for i in range(15):
        for j in range(15):
            if ci(A,i,j)=='#' and (i,j) not in alld: G.add((i,j))
    keys={'N':{(6,8)},'O':{(6,14)},'p':{(1,3),(12,3)}}
    for ps in keys.values(): G|=ps
    return G,DOOR,keys

def solve5d(L0,R0):
    G,DOOR,KEYC=geo5d()
    def held(L,R): return set(c for c,ps in KEYC.items() if L in ps or R in ps)
    def free(h):
        F=set(G)
        for c,cells in DOOR.items():
            if c in h: F|=cells
        return F
    def g5(p,d,mir):
        i,j=p
        if mir: return {'U':(i-1,j),'D':(i+1,j),'L':(i,j+1),'R':(i,j-1)}[d]
        return {'U':(i-1,j),'D':(i+1,j),'L':(i,j-1),'R':(i,j+1)}[d]
    start=(L0,R0); seen={start}; par={start:None}; Q=deque([start])
    while Q:
        s=Q.popleft()
        F=free(held(s[0],s[1]))
        for d in 'UDLR':
            a,b=s
            qa=g5(a,d,True); qb=g5(b,d,False)
            na=qa if qa in F else a; nb=qb if qb in F else b
            if (na,nb)==s: continue
            if na==nb or (na==b and nb==a):
                out=[d]; t=s
                while par[t] is not None:
                    p2,dd=par[t]; out.append(dd); t=p2
                return ''.join(out[::-1])
            if (na,nb) not in seen: seen.add((na,nb)); par[(na,nb)]=(s,d); Q.append((na,nb))
    return None

def tr5(L,R,plan):
    G,DOOR,KEYC=geo5d()
    print('start',L,R)
    for k,d in enumerate(plan):
        F=set(G)
        for c,cells in DOOR.items():
            if L in KEYC[c] or R in KEYC[c]: F|=cells
        qa=go(L,d,True); qb=go(R,d,False)
        nL=qa if qa in F else L; nR=qb if qb in F else R
        L,R=nL,nR
        hd=sorted(c for c,ps in KEYC.items() if L in ps or R in ps)
        print(k,d,L,R,'held',hd)

def geo5e():
    A=current_frame.ascii.split('\n')
    DOOR={'N':{(5,3),(5,4),(5,5)},'O':{(9,2),(9,3),(9,4)},'p':{(5,10),(5,11),(5,12),(9,10),(9,11),(9,12)}}
    alld=set().union(*DOOR.values())
    G=set()
    for i in range(15):
        for j in range(15):
            if ci(A,i,j) in ('#','S') and (i,j) not in alld: G.add((i,j))
    keys={'N':{(6,8)},'O':{(6,14)},'p':{(1,3),(12,3)}}
    for ps in keys.values(): G|=ps
    return G,DOOR,keys

def solve5e(L0,R0):
    G,DOOR,KEYC=geo5e()
    def held(L,R): return set(c for c,ps in KEYC.items() if L in ps or R in ps)
    def free(h):
        F=set(G)
        for c,cells in DOOR.items():
            if c in h: F|=cells
        return F
    def g5(p,d,mir):
        i,j=p
        if mir: return {'U':(i-1,j),'D':(i+1,j),'L':(i,j+1),'R':(i,j-1)}[d]
        return {'U':(i-1,j),'D':(i+1,j),'L':(i,j-1),'R':(i,j+1)}[d]
    start=(L0,R0); seen={start}; par={start:None}; Q=deque([start])
    while Q:
        s=Q.popleft()
        F=free(held(s[0],s[1]))
        for d in 'UDLR':
            a,b=s
            qa=g5(a,d,True); qb=g5(b,d,False)
            na=qa if qa in F else a; nb=qb if qb in F else b
            if (na,nb)==s: continue
            if na==nb or (na==b and nb==a):
                out=[d]; t=s
                while par[t] is not None:
                    p2,dd=par[t]; out.append(dd); t=p2
                return ''.join(out[::-1])
            if (na,nb) not in seen: seen.add((na,nb)); par[(na,nb)]=(s,d); Q.append((na,nb))
    return None

def tr5e(L,R,plan):
    G,DOOR,KEYC=geo5e()
    out=[]
    for k,d in enumerate(plan):
        F=set(G)
        for c,cells in DOOR.items():
            if L in KEYC[c] or R in KEYC[c]: F|=cells
        qa=go(L,d,True); qb=go(R,d,False)
        nL=qa if qa in F else L; nR=qb if qb in F else R
        L,R=nL,nR
        out.append((k,d,L,R))
    return out

def run5e(plan,L0,R0):
    G,DOOR,KEYC=geo5e()
    acts={'U':'UP','D':'DOWN','L':'LEFT','R':'RIGHT'}
    L,R=L0,R0
    for k,d in enumerate(plan):
        F=set(G)
        for c,cells in DOOR.items():
            if L in KEYC[c] or R in KEYC[c]: F|=cells
        qa=go(L,d,True); qb=go(R,d,False)
        nL=qa if qa in F else L; nR=qb if qb in F else R
        r=action([{'action':acts[d]}])
        done=r.get('level_completed') or r.get('game_over') or r.get('done')
        ss=obs5(current_frame)
        if done:
            print('END k',k,d,'lvl',r.get('level_completed'),'over',r.get('game_over'),'done',r.get('done')); return 'DONE'
        if len(ss)!=2:
            print('k',k,d,'odd obs',ss); return None
        d1=abs(ss[0][0]-nL[0])+abs(ss[0][1]-nL[1])+abs(ss[1][0]-nR[0])+abs(ss[1][1]-nR[1])
        d2=abs(ss[1][0]-nL[0])+abs(ss[1][1]-nL[1])+abs(ss[0][0]-nR[0])+abs(ss[0][1]-nR[1])
        oL,oR=(ss[0],ss[1]) if d1<=d2 else (ss[1],ss[0])
        if (oL,oR)!=(nL,nR):
            print('MISMATCH k',k,d,'exp',(nL,nR),'got',(oL,oR),'noop',r.get('no_op')); return (oL,oR)
        L,R=oL,oR
    print('finished plan no completion',L,R)
    return (L,R)

def Ss(fr):
    out=[]
    for n in fr.segmentation['nodes']:
        if n['color'] in ('S','b'):
            b=n['boundary']
            cells=[]
            r0=(min(p[0] for p in b)-2)//4; c0=(min(p[1] for p in b)-2)//4
            r1=(max(p[0] for p in b)-3)//4; c1=(max(p[1] for p in b)-3)//4
            out.append((n['color'],n['pixels'],r0,c0,r1,c1))
    return out

def solve_collide(L0,R0):
    G,DOOR,KEYC=geo5e()
    def held(L,R): return set(c for c,ps in KEYC.items() if L in ps or R in ps)
    def free(h):
        F=set(G)
        for c,cells in DOOR.items():
            if c in h: F|=cells
        return F
    def g5(p,d,mir):
        i,j=p
        if mir: return {'U':(i-1,j),'D':(i+1,j),'L':(i,j+1),'R':(i,j-1)}[d]
        return {'U':(i-1,j),'D':(i+1,j),'L':(i,j-1),'R':(i,j+1)}[d]
    start=(L0,R0); seen={start}; par={start:None}; Q=deque([start])
    while Q:
        s=Q.popleft()
        L,R=s
        # collision setup: L directly below R, and cell above R blocked -> press U wins
        if L[0]==R[0]+1 and L[1]==R[1]:
            F=free(held(L,R))
            if (R[0]-1,R[1]) not in F:
                out=['U']; t=s
                while par[t] is not None:
                    p2,dd=par[t]; out.append(dd); t=p2
                return ''.join(out[::-1])
        # also: L directly above R, cell below R blocked -> press D
        if L[0]==R[0]-1 and L[1]==R[1]:
            F=free(held(L,R))
            if (R[0]+1,R[1]) not in F:
                out=['D']; t=s
                while par[t] is not None:
                    p2,dd=par[t]; out.append(dd); t=p2
                return ''.join(out[::-1])
        F=free(held(L,R))
        for d in 'UDLR':
            a,b=s
            qa=g5(a,d,True); qb=g5(b,d,False)
            na=qa if qa in F else a; nb=qb if qb in F else b
            if (na,nb)==s or (na,nb) in seen: continue
            if na==nb: continue
            seen.add((na,nb)); par[(na,nb)]=(s,d); Q.append((na,nb))
    return None

def geo6():
    A=current_frame.ascii.split('\n')
    G=set()
    for i in range(1,13):
        for j in range(1,13):
            if ci(A,i,j)=='#': G.add((i,j))
    keys={'O':{(3,4)},'N':{(3,10),(10,4)}}
    DOOR={'N':{(7,3),(7,4),(7,5)},'O':{(7,9),(7,10),(7,11)}}
    for ps in keys.values(): G|=ps
    return G,DOOR,keys

def solve6(L0,R0,adj_ok=True):
    G,DOOR,KEYC=geo6()
    def held(L,R): return set(c for c,ps in KEYC.items() if L in ps or R in ps)
    def free(h):
        F=set(G)
        for c,cells in DOOR.items():
            if c in h: F|=cells
        return F
    def g6(p,d,mir):
        i,j=p
        if mir: return {'U':(i-1,j),'D':(i+1,j),'L':(i,j+1),'R':(i,j-1)}[d]
        return {'U':(i-1,j),'D':(i+1,j),'L':(i,j-1),'R':(i,j+1)}[d]
    start=(L0,R0); seen={start}; par={start:None}; Q=deque([start])
    while Q:
        s=Q.popleft(); L,R=s
        if R==(L[0],L[1]+1):
            out=['R','R']; t=s
            while par[t] is not None:
                p2,dd=par[t]; out.append(dd); t=p2
            return ''.join(out[::-1])
        F=free(held(L,R))
        for d in 'UDLR':
            qa=g6(L,d,False); qb=g6(R,d,True)
            na=qa if qa in F else L; nb=qb if qb in F else R
            if (na,nb)==s or (na,nb) in seen: continue
            if (na[0],na[1]+1)==nb or (nb[0],nb[1]+1)==na:
                pass
            seen.add((na,nb)); par[(na,nb)]=(s,d); Q.append((na,nb))
    return None

def g6(p,d,mir):
    i,j=p
    if mir: return {'U':(i-1,j),'D':(i+1,j),'L':(i,j+1),'R':(i,j-1)}[d]
    return {'U':(i-1,j),'D':(i+1,j),'L':(i,j-1),'R':(i,j+1)}[d]

def room(p):
    if p is None: return '?'
    r,c=p
    if r<7: return 'T'+('L' if c<7 else 'R')
    return 'B'+('L' if c<7 else 'R')

def geo6b():
    A=current_frame.ascii.split('\n')
    G=set()
    for i in range(1,13):
        for j in range(1,13):
            if ci(A,i,j) in ('#','S'): G.add((i,j))
    keys={'O':{(3,4)},'N':{(3,10),(10,4)}}
    DOOR={'N':{(7,3),(7,4),(7,5)},'O':{(7,9),(7,10),(7,11)}}
    for ps in keys.values(): G|=ps
    return G,DOOR,keys

def solve6b(L0,R0,verbose=False):
    G,DOOR,KEYC=geo6b()
    def held(L,R): return set(c for c,ps in KEYC.items() if L in ps or R in ps)
    def free(h):
        F=set(G)
        for c,cells in DOOR.items():
            if c in h: F|=cells
        return F
    start=(L0,R0); seen={start}; par={start:None}; Q=deque([start])
    best=0
    while Q:
        s=Q.popleft(); L,R=s
        if R==(L[0],L[1]+1):
            out=['R','R']; t=s
            while par[t] is not None:
                p2,dd=par[t]; out.append(dd); t=p2
            return ''.join(out[::-1])
        F=free(held(L,R))
        for d in 'UDLR':
            qa=g6(L,d,False); qb=g6(R,d,True)
            na=qa if qa in F else L; nb=qb if qb in F else R
            if (na,nb)==s or (na,nb) in seen: continue
            seen.add((na,nb)); par[(na,nb)]=(s,d); Q.append((na,nb))
    if verbose: print('states',len(seen))
    return None

def free6(h):
    F=set(G)
    for c,cells in DOOR.items():
        if c in h: F|=cells
    return F

def mp():
    A=current_frame.ascii.split('\n')
    return ''.join(ci(A,7,j) for j in range(1,13)), ''.join(ci(A,3,j) for j in range(1,13)), Ss(current_frame)

def cellvar(i,j):
    s=set()
    for dr in range(4):
        for dc in range(4):
            s.add(A[2+i*4+dr][2+j*4+dc])
    return ''.join(sorted(s))
