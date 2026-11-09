try:
    from collections import Counter
except Exception:
    pass
try:
    from collections import defaultdict
except Exception:
    pass
try:
    import itertools
except Exception:
    pass
try:
    from collections import deque
except Exception:
    pass

def bbox(n):
    ps = n['boundary']
    rs=[p[0] for p in ps]; cs=[p[1] for p in ps]
    return min(rs),min(cs),max(rs),max(cs)

def centers():
    nodes = current_frame.segmentation['nodes']
    w = [n for n in nodes if n['color']=='W' and n['pixels']==1]
    sel = tuple(w[0]['boundary'][0]) if w else None
    def big(col):
        ns=[n for n in nodes if n['color']==col and n['pixels']>=8]
        if not ns: return None
        b=[n['boundary'] for n in ns]
        rs=[p[0] for x in b for p in x]; cs=[p[1] for x in b for p in x]
        return ((min(rs)+max(rs))//2, (min(cs)+max(cs))//2, min(rs),max(rs),min(cs),max(cs))
    return sel, big('Y'), big('b')

def snap():
    nodes = current_frame.segmentation['nodes']
    out={}
    for col in ('O','r','b'):
        ns=[n for n in nodes if n['color']==col and n['pixels']==1]
        out[col]=[tuple(n['boundary'][0]) for n in ns]
    out['W']=[tuple(n['boundary'][0]) for n in nodes if n['color']=='W']
    out['bar']=[n['pixels'] for n in nodes if n['color']=='p']
    return out

def wpos():
    ns=current_frame.segmentation['nodes']
    w=[n for n in ns if n['color']=='W' and n['pixels']==1]
    return tuple(w[0]['boundary'][0]) if w else None

def dots(col):
    ns=current_frame.segmentation['nodes']
    # box centers: charcoal 3x3 boxes
    boxes=[tuple(n['boundary'][0]) for n in ns if n['color']=='c' and n['pixels']==8]
    centers=[(r+1,c+1) for (r,c) in boxes]
    out={}
    for (r,c) in centers:
        for n in ns:
            if n['pixels']==1 and tuple(n['boundary'][0])==(r,c):
                out.setdefault(n['color'],[]).append((r,c))
    return out

def move_to(tr,tc,maxsteps=25):
    for i in range(maxsteps):
        cr,cc = wpos()
        if (cr,tc)==(tr,tc) and cc==tc: return 'done'
        if cr!=tr:
            a = 'UP' if cr>tr else 'DOWN'
        else:
            a = 'LEFT' if cc>tc else 'RIGHT'
        r=action([a])
        if r.get('game_over') or r.get('level_completed') or r.get('done'):
            return ('END',a,r.get('state'),wpos())
        ncr,ncc=wpos()
        if (ncr,ncc)==(cr,cc):
            return ('BLOCKED',a,(cr,cc))
    return 'maxsteps'

def big_bbox(col):
    ns=[n for n in nodes if n['color']==col and n['pixels']>=8]
    if not ns: return None
    b=[n['boundary'] for n in ns]
    rs=[p[0] for x in b for p in x]; cs=[p[1] for x in b for p in x]
    return (min(rs),max(rs),min(cs),max(cs))

def xfit(col):
    from collections import Counter
    ns=current_frame.segmentation['nodes']
    pts=[]
    for n in ns:
        if n['color']==col:
            for (r,c) in n['boundary']:
                pts.append((r,c))
            # boundary only gives corners; need actual pixels -> use ascii crop? fallback
    return pts

def pixmap(col):
    a=current_frame.ascii.split('\n')
    pts=set()
    for r,row in enumerate(a):
        for c,ch in enumerate(row):
            if ch==col: pts.add((r,c))
    return pts

def xcenter(col):
    P=pixmap(col)
    if not P: return None
    from collections import Counter
    d=Counter(r-c for r,c in P); s=Counter(r+c for r,c in P)
    dd=d.most_common(1)[0][0]; ss=s.most_common(1)[0][0]
    return ((dd+ss)//2,(ss-dd)//2)

def cover(name,rc,cc):
    s=set()
    for i,(r,c) in enumerate(DOTS):
        if name=='line':
            if r==rc and abs(c-cc)<=21: s.add(i)
        elif name=='X':
            if abs(r-rc)==abs(c-cc)<=11: s.add(i)
        else:
            if abs(r-rc)+abs(c-cc)==12: s.add(i)
    return s

def expected(name,rc,cc):
    pts=set()
    if name=='X':
        for k in range(12):
            for dr in (-k,k):
                for dc in (-k,k):
                    pts.add((rc+dr,cc+dc))
    elif name=='dia':
        for k in range(-12,13):
            pts.add((rc+k,cc+12-abs(k))); pts.add((rc+k,cc-12+abs(k)))
    else:
        for c in range(cc-21,cc+22): pts.add((rc,c))
    return {(r,c) for (r,c) in pts if 0<=r<64 and 0<=c<64}

def locate(name, prev, win=6):
    P=pixmap('R')
    best=None
    for dr in range(-win,win+1):
        for dc in range(-win,win+1):
            rc,cc=prev[0]+dr,prev[1]+dc
            if not(0<=rc<64 and 0<=cc<64): continue
            e=expected(name,rc,cc)
            sc=len(e&P)/len(e)
            key=(-sc,abs(dr)+abs(dc))
            if best is None or key<best[0]: best=(key,(rc,cc),sc,len(e))
    return best[1],round(best[2],3),best[3]

def move_obj(name, cur, target, maxsteps=30):
    log=[]
    for _ in range(maxsteps):
        if cur==tuple(target): return ('done',cur,log)
        dr=target[0]-cur[0]; dc=target[1]-cur[1]
        a = ('UP' if dr<0 else 'DOWN') if dr!=0 else ('LEFT' if dc<0 else 'RIGHT')
        r=action([a])
        nxt,sc,sz = locate(name,cur,5)
        log.append((a,nxt,sc))
        if r.get('game_over'): return ('GAMEOVER',cur,log)
        if r.get('level_completed') or r.get('done'): return ('END',nxt,log)
        if nxt==cur: return ('BLOCKED',cur,log)
        cur=nxt
    return ('maxsteps',cur,log)

def cov_plus(rc,cc):
    s=set()
    for i,(r,c) in enumerate(D6):
        if r==rc and abs(c-cc)<=13: s.add(i)
        if c==cc and abs(r-rc)<=13: s.add(i)
    return s

def cov_x(rc,cc):
    s=set()
    for i,(r,c) in enumerate(D6):
        if abs(r-rc)==abs(c-cc)<=10: s.add(i)
    return s

def exp2(name,rc,cc,arm):
    pts=set()
    if name=='plus':
        for k in range(-arm,arm+1):
            pts.add((rc,cc+k)); pts.add((rc+k,cc))
    else:
        for k in range(arm+1):
            for dr in (-k,k):
                for dc in (-k,k): pts.add((rc+dr,cc+dc))
    return {(r,c) for (r,c) in pts if 0<=r<64 and 0<=c<64}

def locate2(name,color,prev,arm,win=5):
    P=pixmap(color); best=None
    for dr in range(-win,win+1):
        for dc in range(-win,win+1):
            rc,cc=prev[0]+dr,prev[1]+dc
            if not(0<=rc<64 and 0<=cc<64): continue
            e=exp2(name,rc,cc,arm)
            if not e: continue
            sc=len(e&P)/len(e)
            key=(-sc,abs(dr)+abs(dc))
            if best is None or key<best[0]: best=(key,(rc,cc),sc)
    return best[1],round(best[2],3)

def move2(name,color,arm,cur,target,maxsteps=30):
    log=[]
    for _ in range(maxsteps):
        if tuple(cur)==tuple(target): return ('done',cur,log)
        dr=target[0]-cur[0]; dc=target[1]-cur[1]
        a=('UP' if dr<0 else 'DOWN') if dr!=0 else ('LEFT' if dc<0 else 'RIGHT')
        r=action([a])
        nxt,sc=locate2(name,color,cur,arm,5)
        log.append((a,nxt,sc))
        if r.get('game_over'): return ('GAMEOVER',cur,log)
        if r.get('level_completed') or r.get('done'): return ('END',nxt,log)
        if nxt==cur: return ('BLOCKED',cur,log)
        cur=nxt
    return ('maxsteps',cur,log)

def find_obj(name,cur,arm,win=4):
    best=None
    for col in COLS2:
        c,sc=locate2(name,col,cur,arm,win)
        if best is None or sc>best[2]+1e-9: best=(col,c,round(sc,3))
    return best

def step(st,a):
    rh,ch,dx,dy=st; dr,dc=D[a]
    rv=rh+dy; cv=ch-dx
    hb=hov(rh+dr,ch+dc); vb=vov(rv+dr,cv+dc)
    nrh,nch=(rh+dr,ch+dc) if not hb else (rh,ch)
    nrv,ncv=(rv+dr,cv+dc) if not vb else (rv,cv)
    return (nrh,nch,nch-ncv,nrv-nrh)

def track(name, cur, arm, win=4):
    cols=('S','O','N','M','Y','r','R','b','P')
    best=None
    for col in cols:
        c,sc=locate2(name,col,cur,arm,win)
        if best is None or sc>best[2]+1e-9: best=(col,c,round(sc,3))
    return best

def go(name, arm, cur, seq):
    log=[]
    for a,n in seq:
        for _ in range(n):
            r=action([a])
            col,c,sc=track(name,cur,arm)
            if r.get('game_over'): return ('GAMEOVER',cur,col,log)
            if r.get('level_completed') or r.get('done'): return ('END',c,col,log)
            if c==cur: return ('BLOCKED',cur,col,log)
            cur=c; log.append((a,col,c))
    return ('ok',cur,col,log)

def cov(kind,rc,cc,dots):
    s=set()
    for d in dots:
        r,c=d
        if kind=='dia':
            if abs(r-rc)+abs(c-cc)==9: s.add(bi[d])
        elif kind=='x':
            if abs(r-rc)==abs(c-cc)<=11: s.add(bi[d])
        else:
            if (r==rc and abs(c-cc)<=14) or (c==cc and abs(r-rc)<=14): s.add(bi[d])
    return frozenset(s)

def outline_offers(kind):
    offs=[]
    if kind=='dia':
        for t in range(-9,10):
            offs.append((t,9-abs(t))); offs.append((t,-(9-abs(t))))
    elif kind=='x':
        for t in range(-11,12): offs.append((t,t)); offs.append((t,-t))
    else:
        for t in range(-14,15): offs.append((0,t)); offs.append((t,0))
    return offs

def touchmap(kind):
    tm={}
    for col,(r0,r1,c0,c1) in FILLS.items():
        for r in range(r0,r1+1):
            for c in range(c0,c1+1):
                for (dr,dc) in outline_offers(kind):
                    rr,cc=r-dr,c-dc
                    if 0<=rr<64 and 0<=cc<64: tm.setdefault((rr,cc),set()).add(col)
    return tm

def bfs(kind,start,target,curcol,desired):
    tm=touchmap(kind)
    st=(start[0],start[1],curcol)
    prev={st:None}; q=deque([st]); best=None
    while q:
        r,c,col=q.popleft()
        if (r,c)==target and col==desired: best=(r,c,col); break
        for dr,dc in ((3,0),(-3,0),(0,3),(0,-3)):
            nr,nc=r+dr,c+dc
            if not(0<=nr<64 and 0<=nc<64): continue
            t=tm.get((nr,nc))
            if t:
                if desired in t: ncol=desired
                elif len(t)==1: ncol=list(t)[0]
                else: continue
            else: ncol=col
            ns=(nr,nc,ncol)
            if ns not in prev:
                prev[ns]=(r,c,col); q.append(ns)
    if best is None: return None
    path=[]; s=best
    while s: path.append(s); s=prev[s]
    path.reverse()
    acts=['DOWN' if b[0]>a[0] else 'UP' if b[0]<a[0] else ('RIGHT' if b[1]>a[1] else 'LEFT')
          for a,b in zip(path,path[1:])]
    return acts

def touchmap2(kind):
    tm={}
    for col,(r0,r1,c0,c1) in FILLS.items():
        for r in range(r0,r1+1):
            for c in range(c0,c1+1):
                for (dr,dc) in outline_offers(kind):
                    rr,cc=r-dr,c-dc
                    if 0<=rr<64 and 0<=cc<64: tm.setdefault((rr,cc),set()).add(col)
    return tm

def bfs2(kind,start,target,curcol,desired):
    tm=touchmap2(kind)
    st=(start[0],start[1],curcol)
    prev={st:None}; q=deque([st]); best=None
    while q:
        r,c,col=q.popleft()
        if (r,c)==target and col==desired: best=(r,c,col); break
        for dr,dc in ((3,0),(-3,0),(0,3),(0,-3)):
            nr,nc=r+dr,c+dc
            if not(0<=nr<64 and 0<=nc<64): continue
            t=tm.get((nr,nc))
            if t:
                if desired in t: ncol=desired
                elif len(t)==1: ncol=list(t)[0]
                else: continue
            else: ncol=col
            ns=(nr,nc,ncol)
            if ns not in prev: prev[ns]=(r,c,col); q.append(ns)
    if best is None: return None
    path=[]; s=best
    while s: path.append(s); s=prev[s]
    path.reverse(); return path

def tm3(kind):
    tm={}
    for col,(r0,r1,c0,c1) in FILLS2.items():
        for r in range(r0,r1+1):
            for c in range(c0,c1+1):
                for (dr,dc) in outline_offers(kind):
                    rr,cc=r-dr,c-dc
                    if 0<=rr<64 and 0<=cc<64: tm.setdefault((rr,cc),set()).add(col)
    return tm

def bfs3(kind,start,target,curcol,desired,strict=True):
    tm=tm3(kind); st=(start[0],start[1],curcol)
    prev={st:None}; q=deque([st]); best=None
    while q:
        r,c,col=q.popleft()
        if (r,c)==target and col==desired: best=(r,c,col); break
        for dr,dc in ((3,0),(-3,0),(0,3),(0,-3)):
            nr,nc=r+dr,c+dc
            if not(0<=nr<64 and 0<=nc<64): continue
            t=tm.get((nr,nc))
            if t:
                if len(t)>1 and strict: continue
                ncol=desired if desired in t else list(t)[0]
            else: ncol=col
            ns=(nr,nc,ncol)
            if ns not in prev: prev[ns]=(r,c,col); q.append(ns)
    if best is None: return None
    path=[]; s=best
    while s: path.append(s); s=prev[s]
    path.reverse(); return path

def dia_pos(cur):
    best=None
    for col in ('N','R','Y','S','b'):
        c,sc=locate2('dia',col,cur,9,5)
        if best is None or sc>best[2]: best=(col,c,sc)
    return best

def press_to(tr,tc,maxs=12):
    cur=wpos()
    while cur!=(tr,tc) and maxs>0:
        a='DOWN' if tr>cur[0] else 'UP' if tr<cur[0] else ('RIGHT' if tc>cur[1] else 'LEFT')
        r=action([a])
        if r.get('game_over') or r.get('level_completed') or r.get('done'):
            return ('END',cur,r.get('state'))
        new=wpos()
        if new==cur: return ('stuck',cur)
        cur=new; maxs-=1
    return ('ok',cur)

def sq():
    for n in current_frame.segmentation['nodes']:
        if n['color']=='Y' and n['pixels']>10: return bbox(n),n['pixels']

def plusinfo():
    P=pixmap('b')
    rows={}; cols={}
    for r,c in P:
        rows.setdefault(r,[]).append(c); cols.setdefault(c,[]).append(r)
    lr=[ (r,sorted(v)) for r,v in rows.items() if len(v)>3]
    lc=[ (c,sorted(v)) for c,v in cols.items() if len(v)>3]
    return len(P), lr, lc

def plus2():
    P={p for p in pixmap('b')} - DOTB - {wpos()}
    rows={}; cols={}
    for r,c in P:
        rows.setdefault(r,[]).append(c); cols.setdefault(c,[]).append(r)
    out={}
    for r,v in rows.items():
        if len(v)>3: out['h']=(r,min(v),max(v),len(v))
    for c,v in cols.items():
        if len(v)>3: out['v']=(min(v),max(v),c,len(v))
    return out

def hov(rh,ch): return M[0]<=rh<=M[1] and (ch-12)<=M[3] and (ch+12)>=M[2]

def vov(rv,cv): return M[2]<=cv<=M[3] and (rv-12)<=M[1] and (rv+12)>=M[0]

def goal(s):
    rh,hl,cv,vt,t=s
    return t and rh==15 and cv==9 and 0<=hl<=3 and vt==9

def plus3():
    DOT={(6,12),(9,9),(9,30),(27,12)}
    P={p for p in pixmap('b')}-DOT-{wpos()}
    rows={};cols={}
    for r,c in P:
        rows.setdefault(r,[]).append(c); cols.setdefault(c,[]).append(r)
    h=[(r,min(v),max(v)) for r,v in rows.items() if len(v)>3]
    v=[(min(v),max(v),c) for c,v in cols.items() if len(v)>3]
    return (h[0] if h else None),(v[0] if v else None)

def stp(s,a):
    rh,hl,cv,vt,t=s; dr,dc=Dm[a]
    if not hblk(rh+dr,hl+dc): rh,hl=rh+dr,hl+dc
    if not vblk(cv+dc,vt+dr): cv,vt=cv+dc,vt+dr
    return (rh,hl,cv,vt,t or touched2(rh,hl,cv,vt))

def hblk(rh,hl): return 28<=rh<=35 and hl<=35 and hl+36>=28

def vblk(cv,vt): return 28<=cv<=35 and vt<=35 and vt+18>=28

def touched(rh,hl,cv,vt):
    a=M[0]<=rh<=M[1] and hl<=BIN[3] and hl+36>=BIN[2]
    return a

def touched2(rh,hl,cv,vt): return (3<=rh<=5 and hl<=38 and hl+36>=36) or (36<=cv<=38 and vt<=5 and vt+18>=3)

def plusobs(col):
    P=set(pixmap(col))
    rows={};cols={}
    for r,c in P:
        rows.setdefault(r,[]).append(c); cols.setdefault(c,[]).append(r)
    h=max([(len(v),r,min(v),max(v)) for r,v in rows.items() if len(v)>2],default=None)
    v=max([(len(v),min(v),max(v),c) for c,v in cols.items() if len(v)>2],default=None)
    return h,v

def poplus(cols=('P','R')):
    P=set()
    for c in cols: P|=set(pixmap(c))
    P-={(9,9),(15,3),(15,36),(27,9),(3,36),(3,37),(3,38),(4,36),(4,37),(4,38),(5,36),(5,37),(5,38)}
    rows={};cl={};w=wpos()
    for r,c in P:
        rows.setdefault(r,[]).append(c); cl.setdefault(c,[]).append(r)
    h=max([(len(v),r,min(v),max(v)) for r,v in rows.items() if len(v)>=18],default=None)
    v=max([(len(v),min(v),max(v),c) for c,v in cl.items() if len(v)>=12],default=None)
    return (h[1],h[2],h[3]) if h else None, (v[1],v[2],v[3]) if v else None
