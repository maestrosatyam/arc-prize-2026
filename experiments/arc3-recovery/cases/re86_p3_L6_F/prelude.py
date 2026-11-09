try:
    import json
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

def brief(d):
    out={}
    for k,v in d.items():
        if k=='changed_cell_count': out[k]=v
        else:
            out[k]=[{kk:vv for kk,vv in e.items() if kk in ('from','to','at','pixels','before_color','after_color','bbox')} for e in v][:8]
    return out

def whites():
    return [tuple(n['boundary'][0]) for n in current_frame.segmentation['nodes'] if n['color']=='W']

def runs(s, ch):
    out=[]; start=None
    for i,c in enumerate(s):
        if c==ch and start is None: start=i
        elif c!=ch and start is not None: out.append((start,i-1)); start=None
    if start is not None: out.append((start,len(s)-1))
    return out

def xcells(r0,c0,L=11):
    s={(r0,c0)}
    for k in range(1,L+1):
        for dr,dc in ((k,k),(k,-k),(-k,k),(-k,-k)):
            s.add((r0+dr,c0+dc))
    return s

def dcells(r0,c0,R=12):
    s=set()
    for dr in range(-R,R+1):
        dc = R-abs(dr)
        s.add((r0+dr,c0+dc)); s.add((r0+dr,c0-dc))
    return s

def lcells(r0,c0,H=21):
    return {(r0,c) for c in range(c0-H,c0+H+1)}

def onb(cs): return all(0<=r<64 and 0<=c<64 for r,c in cs)

def cost(p, start): return abs(p[0]-start[0])//3 + abs(p[1]-start[1])//3

def rel(cells, ctr):
    return sorted((r-ctr[0], c-ctr[1]) for r,c in cells)

def plus(r0,c0,L=13):
    s={(r0,c0)}
    for k in range(1,L+1):
        s|={(r0+k,c0),(r0-k,c0),(r0,c0+k),(r0,c0-k)}
    return s

def Xc(r0,c0,L=10):
    s={(r0,c0)}
    for k in range(1,L+1):
        s|={(r0+k,c0+k),(r0+k,c0-k),(r0-k,c0+k),(r0-k,c0-k)}
    return s

def pluscenter(fr):
    # find M pixels not in slot fill rows55-58 cols29-32
    a = fr.ascii.split('\n')
    pts = [(r,c) for r in range(64) for c in range(64) if a[r][c]=='M' and not (55<=r<=58 and 29<=c<=32)]
    if not pts: return None, 0
    rs = [p[0] for p in pts]; cs=[p[1] for p in pts]
    return (min(rs),max(rs),min(cs),max(cs)), len(pts)

def snap():
    a = current_frame.ascii.split('\n')
    pts = [(r,c) for r in range(64) for c in range(64) if a[r][c] in 'MN' and not (55<=r<=58 and 53<=c<=56)]
    Ws = [(r,c) for r in range(64) for c in range(64) if a[r][c]=='W']
    Ss = [(r,c) for r in range(64) for c in range(64) if a[r][c]=='S']
    return len(pts), Ws, len(Ss), a[63].count('p'), a[63].count('w')

def crop(r0,c0,R,C):
    for r in range(r0,r0+R): print(r, ''.join(a[r][c0:c0+C]))

def dia(r0,c0,R=9):
    s=set()
    for k in range(-R,R+1):
        m = R-abs(k)
        s|={(r0+k,c0+m),(r0+k,c0-m)}
    return s

def Xa(r0,c0,L=11):
    s={(r0,c0)}
    for k in range(1,L+1): s|={(r0+k,c0+k),(r0+k,c0-k),(r0-k,c0+k),(r0-k,c0-k)}
    return s

def Pl(r0,c0,L=14):
    s={(r0,c0)}
    for k in range(1,L+1): s|={(r0+k,c0),(r0-k,c0),(r0,c0+k),(r0,c0-k)}
    return s

def solve5():
    blue = [(6,21),(6,39),(45,33),(51,24),(51,45),(60,33)]
    red  = [(27,51),(36,42),(33,57),(42,54)]
    dots = blue+red
    shapes = {'D':(dia,(18,30)),'X':(Xa,(42,24)),'P':(Pl,(30,51))}
    allopts={}
    for name,(f,start) in shapes.items():
        lst=[]
        for r in range(0,64):
            for c in range(0,64):
                if (r-start[0])%3 or (c-start[1])%3: continue
                cells=f(r,c)
                if not onb(cells): continue
                cov=frozenset(d for d in dots if d in cells)
                cols = set('B' if d in blue else 'R' for d in cov)
                if len(cols)>1: continue
                col = cols.pop() if cols else None
                mv = abs(r-start[0])//3 + abs(c-start[1])//3
                lst.append((mv,r,c,cov,col))
        lst.append((0,start[0],start[1],frozenset(),None))
        allopts[name]=lst
    best=[]
    D,X,P = allopts['D'],allopts['X'],allopts['P']
    for d in D:
        for x in X:
            if d[4] and x[4] and d[4]!=x[4] and (d[3]&x[3]): continue
            u1 = d[3]|x[3]
            if len(u1)==10:
                best.append((d[0]+x[0],('D',d[1],d[2],d[4]),('X',x[1],x[2],x[4]),('P',None,None,None)))
                continue
            for p in P:
                if (d[4] and p[4] and d[4]!=p[4] and (d[3]&p[3])) or (x[4] and p[4] and x[4]!=p[4] and (x[3]&p[3])): continue
                if len(u1|p[3])==10:
                    best.append((d[0]+x[0]+p[0],('D',d[1],d[2],d[4]),('X',x[1],x[2],x[4]),('P',p[1],p[2],p[4])))
    best.sort(key=lambda t:t[0])
    return best

def solve5b(clip=True):
    blue = [(6,21),(6,39),(45,33),(51,24),(51,45),(60,33)]
    red  = [(27,51),(36,42),(33,57),(42,54)]
    dots = blue+red
    shapes = {'D':(dia,(18,30)),'X':(Xa,(42,24)),'P':(Pl,(30,51))}
    allopts={}
    for name,(f,start) in shapes.items():
        lst=[]
        for r in range(0,64):
            for c in range(0,64):
                if (r-start[0])%3 or (c-start[1])%3: continue
                cells=f(r,c)
                if not clip and not onb(cells): continue
                cov=frozenset(d for d in dots if d in cells)
                cols = set('B' if d in blue else 'R' for d in cov)
                if len(cols)>1: continue
                col = cols.pop() if cols else None
                mv = abs(r-start[0])//3 + abs(c-start[1])//3
                lst.append((mv,r,c,cov,col))
        lst.append((0,start[0],start[1],frozenset(),None))
        allopts[name]=lst
    best=[]
    D,X,P = allopts['D'],allopts['X'],allopts['P']
    for d in D:
        for x in X:
            if d[4] and x[4] and d[4]!=x[4] and (d[3]&x[3]): continue
            u1 = d[3]|x[3]
            if len(u1)==10:
                best.append((d[0]+x[0],('D',d[1],d[2],d[4]),('X',x[1],x[2],x[4]),('P',None,None,None)))
                continue
            for p in P:
                if (d[4] and p[4] and d[4]!=p[4] and (d[3]&p[3])) or (x[4] and p[4] and x[4]!=p[4] and (x[3]&p[3])): continue
                if len(u1|p[3])==10:
                    best.append((d[0]+x[0]+p[0],('D',d[1],d[2],d[4]),('X',x[1],x[2],x[4]),('P',p[1],p[2],p[4])))
    best.sort(key=lambda t:t[0])
    return best

def route(f, start, target, fill, clip=True):
    best=None
    for r in range(0,64):
        for c in range(0,64):
            if (r-start[0])%3 or (c-start[1])%3: continue
            cells=f(r,c)
            if not clip: 
                continue
            if not any((rf,cf) in cells for rf in range(fill[0],fill[1]+1) for cf in range(fill[2],fill[3]+1)): continue
            cost = abs(r-start[0])//3+abs(c-start[1])//3 + abs(target[0]-r)//3+abs(target[1]-c)//3
            if best is None or cost<best[0]: best=(cost,(r,c))
    return best

def touched(f, pos):
    cells = f(*pos)
    return [k for k,(r0,r1,c0,c1) in SLOTS.items() if any(r0<=r<=r1 and c0<=c<=c1 for r,c in cells)]

def path_positions(start, mid, target, o1, o2):
    ps=[start]
    r,c = start
    def step_to(tr,tc,order):
        nonlocal r,c
        seq=[]
        if order=='r': axes=[(0,tr-r),(1,tc-c)]
        else: axes=[(1,tc-c),(0,tr-r)]
        for ax,d in axes:
            while (r,c)!=(tr,tc) and ((ax==0 and r!=tr) or (ax==1 and c!=tc)):
                if ax==0: r += 3 if d>0 else -3
                else: c += 3 if d>0 else -3
                seq.append((r,c))
        return seq
    ps += step_to(mid[0],mid[1],o1)
    ps += step_to(target[0],target[1],o2)
    return ps

def plan_route(f, start, target, need, clip=True):
    cands=[]
    for r in range(0,64):
        for c in range(0,64):
            if (r-start[0])%3 or (c-start[1])%3: continue
            t = touched(f,(r,c))
            if need not in t or len(t)>1: continue
            cost = abs(r-start[0])//3+abs(c-start[1])//3+abs(target[0]-r)//3+abs(target[1]-c)//3
            cands.append((cost,(r,c)))
    cands.sort()
    for cost,q in cands[:25]:
        for o1 in 'rc':
            for o2 in 'rc':
                ps = path_positions(start,q,target,o1,o2)
                seq=[]
                for p in ps:
                    for k in touched(f,p):
                        if not seq or seq[-1]!=k: seq.append(k)
                if seq and seq[-1]==need and not touched(f,target):
                    return cost,q,o1+o2,seq
    return None

def touched2(f, pos, slots):
    cells = f(*pos)
    return [k for k,(r0,r1,c0,c1) in slots.items() if any(r0<=r<=r1 and c0<=c<=c1 for r,c in cells)]

def validate(f, start, mid, target, o1, o2, slots):
    ps = path_positions(start,mid,target,o1,o2)
    seq=[]
    for p in ps:
        for k in touched2(f,p,slots):
            if not seq or seq[-1]!=k: seq.append(k)
    return ps, seq

def acts_from_path(ps):
    out=[]
    for (r0,c0),(r1,c1) in zip(ps, ps[1:]):
        a = 'DOWN' if r1>r0 else 'UP' if r1<r0 else 'RIGHT' if c1>c0 else 'LEFT'
        out += [a]*(abs(r1-r0)//3 or abs(c1-c0)//3)
    return out

def shape_dump():
    a = current_frame.ascii.split('\n')
    pts = [(r,c) for r in range(64) for c in range(64) if a[r][c]=='b']
    return pts

def bars(pts):
    rows = {}
    cols = {}
    for r,c in pts:
        rows.setdefault(r,[]).append(c); cols.setdefault(c,[]).append(r)
    rl = [(r,min(v),max(v),len(v)) for r,v in sorted(rows.items()) if len(v)>3]
    cl = [(c,min(v),max(v),len(v)) for c,v in sorted(cols.items()) if len(v)>3]
    return rl,cl

def dots_state():
    seg = current_frame.segmentation
    out = {}
    for n in seg['nodes']:
        if n['pixels']<=8 and n['color'] in ('b','Y'): out[tuple(n['boundary'][0])] = n['color']
    return out

def free(r,c,box=BOX): return 0<=r<64 and 0<=c<64 and (r,c) not in box

def hseg(row,c0,L=25): return [(row,c) for c in range(c0,c0+L)]

def vseg(col,r0,L=25): return [(r,col) for r in range(r0,r0+L)]

def can(cells,dr,dc,box=BOX): return all(free(r+dr,c+dc,box) for r,c in cells)

def step_plus(st,a,box=BOX,S=3):
    hr,hc,vr,vc = st
    d = {'UP':(-S,0),'DOWN':(S,0),'LEFT':(0,-S),'RIGHT':(0,S)}[a]
    H = hseg(hr,hc); V = vseg(vc,vr)
    return ((hr+d[0],hc+d[1]) if can(H,*d,box) else (hr,hc)) + ((vr+d[0],vc+d[1]) if can(V,*d,box) else (vr,vc))

def segs(st,kind='plus'):
    if kind=='plus':
        hr,hc,vr,vc = st
        return [hseg(hr,hc), vseg(vc,vr)]
    return []

def fr(r,c): return 0<=r<64 and 0<=c<64 and (r,c) not in BOX2

def stp(st,a):
    hr,hc,vr,vc = st
    d = {'UP':(-3,0),'DOWN':(3,0),'LEFT':(0,-3),'RIGHT':(0,3)}[a]
    Hok = all(fr(hr+d[0],c+d[1]) for c in range(hc,hc+25))
    Vok = all(fr(r+d[0],vc+d[1]) for r in range(vr,vr+25))
    return ((hr+d[0],hc+d[1]) if Hok else (hr,hc)) + ((vr+d[0],vc+d[1]) if Vok else (vr,vc))

def goal(st):
    hr,hc,vr,vc = st
    return hr==9 and hc in (6,9) and vc==12 and vr in (3,6)

def obs_plus():
    pts = [(r,c) for r in range(64) for c in range(64) if current_frame.ascii.split('\n')[r][c]=='b' and not (dots_state().get((r,c)))]
    rows={};cols={}
    for r,c in pts: rows.setdefault(r,[]).append(c); cols.setdefault(c,[]).append(r)
    hr,hc,vr,vc=None,None,None,None
    for r,v in rows.items():
        if len(v)>=20 and max(v)-min(v)>=20: hr,hc=r,min(v)
    for c,v in cols.items():
        if len(v)>=20 and max(v)-min(v)>=20: vc,vr=c,min(v)
    return (hr,hc,vr,vc)

def blocked(r,c): return (r,c) in BOX2

def stp2(st,a):
    hr,hc,vr,vc = st
    d = {'UP':(-3,0),'DOWN':(3,0),'LEFT':(0,-3),'RIGHT':(0,3)}[a]
    Hok = not any(blocked(hr+d[0],c+d[1]) for c in range(hc,hc+25))
    Vok = not any(blocked(r+d[0],vc+d[1]) for r in range(vr,vr+25))
    return ((hr+d[0],hc+d[1]) if Hok else (hr,hc)) + ((vr+d[0],vc+d[1]) if Vok else (vr,vc))

def goal2(st):
    hr,hc,vr,vc = st
    return hr==9 and hc in (6,9) and vc==12 and vr in (3,6)
