try:
    from collections import defaultdict
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
    import itertools
except Exception:
    pass
try:
    import json
except Exception:
    pass

def centers():
    seg = current_frame.segmentation
    out = {}
    ys, xs = [], []
    for n in seg['nodes']:
        if n['color']=='W':
            out['W'] = tuple(n['boundary'][0])
    # yellow cross arms: union of yellow non-dot pixels (dots are 1px inside charcoal boxes)
    return out

def cross_center(color):
    seg = current_frame.segmentation
    pts=[]
    for n in seg['nodes']:
        if n['color']==color and n['pixels']>1:
            for b in n['boundary']:
                pts.append(tuple(b))
    return pts

def wpos():
    for n in current_frame.segmentation['nodes']:
        if n['color']=='W':
            return tuple(n['boundary'][0])
    return None

def move_seq(acts):
    log=[]
    for a in acts:
        before = wpos()
        res = action([a])
        after = wpos()
        log.append((a, before, after, res.get('level_completed'), res.get('game_over')))
        if before==after:
            log.append('BLOCKED')
            break
    return log

def crop(r0,r1,c0,c1): return '\n'.join(''.join(G[r][c] for c in range(c0,c1+1)) for r in range(r0,r1+1))

def cells(color):
    return {(r,c) for r in range(64) for c in range(64) if G[r][c]==color}

def grid():
    return [list(row) for row in current_frame.ascii.split('\n')]

def shape_center(color):
    G = grid()
    pts = {(r,c) for r in range(64) for c in range(64) if G[r][c]==color}
    # drop isolated 1px dots (no same-color 4-neighbor)
    keep = {p for p in pts if any((p[0]+dr,p[1]+dc) in pts for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)))}
    if not keep: return None, None, len(pts)
    rs=[p[0] for p in keep]; cs=[p[1] for p in keep]
    return (min(rs)+max(rs))/2, (min(cs)+max(cs))/2, len(pts)

def dots(color):
    G = grid()
    pts = {(r,c) for r in range(64) for c in range(64) if G[r][c]==color}
    return sorted(p for p in pts if not any((p[0]+dr,p[1]+dc) in pts for dr,dc in ((1,0),(-1,0),(0,1),(0,-1))))

def nbrs8(p):
    return [(p[0]+dr,p[1]+dc) for dr in (-1,0,1) for dc in (-1,0,1) if (dr,dc)!=(0,0)]

def analyze(color):
    G = grid()
    pts = {(r,c) for r in range(64) for c in range(64) if G[r][c]==color}
    keep = {p for p in pts if any(q in pts for q in nbrs8(p))}
    dots_ = sorted(pts - keep)
    ctr = None
    if keep:
        rs=[p[0] for p in keep]; cs=[p[1] for p in keep]
        ctr = ((min(rs)+max(rs))/2, (min(cs)+max(cs))/2)
    return ctr, dots_, len(keep)

def move_to(color, target, max_steps=25):
    log=[]
    for _ in range(max_steps):
        ctr,_,_ = analyze(color)
        if ctr is None: break
        dr, dc = target[0]-ctr[0], target[1]-ctr[1]
        if dr==0 and dc==0:
            log.append('ARRIVED')
            break
        if abs(dr)%3 or abs(dc)%3:
            log.append(('OFFGRID', dr, dc)); break
        a = 'DOWN' if dr>0 else 'UP' if dr<0 else ('RIGHT' if dc>0 else 'LEFT')
        res = action([a])
        nctr,_,_ = analyze(color)
        log.append((a, ctr, nctr, res.get('level_completed'), res.get('game_over')))
        if nctr == ctr:
            log.append('BLOCKED'); break
        if res.get('level_completed') or res.get('game_over'): break
    return log

def covers(shape, ctr):
    S = cells_for(shape, ctr)
    return frozenset(k for k,(p,col) in dots.items() if p in S)

def Rcells():
    G = grid()
    return {(r,c) for r in range(64) for c in range(64) if G[r][c]=='R'}, {(r,c) for r in range(64) for c in range(64) if G[r][c]=='W'}

def tmpl(kind, ctr):
    r,c = ctr
    if kind=='line': return {(r, c+d) for d in range(-21,22)}
    if kind=='X':
        s=set()
        for d in range(-11,12): s.add((r+d,c+d)); s.add((r+d,c-d))
        return s
    s=set()
    for d in range(-12,13):
        s.add((r+d, c+(12-abs(d)))); s.add((r+d, c-(12-abs(d))))
    return s

def fit(kind, R, approx=False):
    best=None
    for r in range(-24,88):
        for c in range(-24,88):
            t = tmpl(kind,(r,c))
            onb = {p for p in t if 0<=p[0]<64 and 0<=p[1]<64}
            miss = len(onb - R)
            if miss==0:
                return (r,c)
            if approx and (best is None or miss<best[0]): best=(miss,(r,c))
    return best[1] if (approx and best) else None

def bar():
    for n in current_frame.segmentation['nodes']:
        if n['color']=='p' and n['pixels']>3: return n['pixels']
    return None

def fit2(kind, R, W):
    P = R | W
    best = None
    for r in range(-24,88):
        for c in range(-24,88):
            t = tmpl(kind,(r,c))
            onb = {p for p in t if 0<=p[0]<64 and 0<=p[1]<64}
            if len(onb) < 20: continue
            miss = len(onb - P)
            if miss == 0:
                if best is None or len(onb) > best[0]:
                    best = (len(onb),(r,c))
    return best[1] if best else None

def state():
    R,W = Rcells()
    pos = {k: fit2(k,R,W) for k in ('line','X','dia')}
    w = tuple(W)[0] if W else None
    return pos, w

def sel_name(pos, w):
    for k,c in pos.items():
        if c==w: return k
    return None

def dots_left():
    R,W = Rcells()
    P = R | W
    ds=[(6,6),(6,45),(15,51),(21,21),(27,9),(30,48),(33,33),(39,21)]
    return [d for d in ds if d in R]

def select(name):
    for _ in range(4):
        pos,w = state()
        if sel_name(pos,w)==name: return True
        action(['SPACE'])
    return False

def drive(name, target):
    if not select(name): return 'NOSEL'
    log=[]
    for _ in range(30):
        pos,w = state()
        cur = pos[name]
        if cur is None: log.append('LOST'); break
        dr,dc = target[0]-cur[0], target[1]-cur[1]
        if dr==0 and dc==0: log.append('ARRIVED'); break
        a = 'DOWN' if dr>0 else 'UP' if dr<0 else ('RIGHT' if dc>0 else 'LEFT')
        res = action([a])
        log.append((a,cur))
        if res.get('level_completed') or res.get('game_over'): break
    return log

def cov_X(ctr):
    r,c=ctr; s=set()
    for i,p in enumerate(dots):
        dr,dc=p[0]-r,p[1]-c
        if abs(dr)==abs(dc) and abs(dr)<=arm: s.add(i)
    return frozenset(s)

def cov_P(ctr):
    r,c=ctr; s=set()
    for i,p in enumerate(dots):
        dr,dc=p[0]-r,p[1]-c
        if (dr==0 and abs(dc)<=parm) or (dc==0 and abs(dr)<=parm): s.add(i)
    return frozenset(s)

def enum(cur, covf):
    d={}
    for r in range(-20,84):
        for c in range(-20,84):
            if (r-cur[0])%3 or (c-cur[1])%3: continue
            cs=covf((r,c))
            if not cs: continue
            cost=(abs(r-cur[0])//3)+(abs(c-cur[1])//3)
            if cs not in d or cost<d[cs][0]: d[cs]=(cost,(r,c))
    return d

def cells_of(*colors):
    G = grid()
    return {(r,c) for r in range(64) for c in range(64) if G[r][c] in colors}

def tmpl_at(kind, ctr, arm):
    r,c=ctr
    if kind=='X':
        s=set()
        for d in range(-arm,arm+1): s.add((r+d,c+d)); s.add((r+d,c-d))
        return s
    s = {(r,c+d) for d in range(-arm,arm+1)} | {(r+d,c) for d in range(-arm,arm+1)}
    return s

def fitc(kind, avail, arm):
    best=None
    for r in range(64):
        for c in range(64):
            t = tmpl_at(kind,(r,c),arm)
            onb = {p for p in t if 0<=p[0]<64 and 0<=p[1]<64}
            if onb - avail: continue
            if best is None or len(onb)>best[0]: best=(len(onb),(r,c))
    return best[1] if best else None

def l4():
    return cells_of('W')

def pos4():
    return {'X': fitc('X', cells_of('S'), 10), 'P': fitc('plus', cells_of('M','W'), 13)}

def sel4(pos):
    W = cells_of('W')
    w = tuple(W)[0] if W else None
    for k,c in pos.items():
        if c==w: return k
    return None

def iso_dots():
    D = cells_of('O','N')
    return sorted(p for p in D if not any(q in D for q in nbrs8(p)))

def drive4(name, target):
    for _ in range(4):
        pos = pos4()
        if sel4(pos)==name: break
        action(['SPACE'])
    log=[]
    for _ in range(30):
        pos = pos4()
        cur = pos[name]
        if cur is None: log.append('LOST'); break
        dr,dc = target[0]-cur[0], target[1]-cur[1]
        if dr==0 and dc==0: log.append('ARRIVED'); break
        a = 'DOWN' if dr>0 else 'UP' if dr<0 else ('RIGHT' if dc>0 else 'LEFT')
        res = action([a])
        log.append((a,cur))
        if res.get('level_completed') or res.get('game_over'): break
    return log

def snap(f):
    G = [list(row) for row in f.ascii.split('\n')]
    W = [(r,c) for r in range(64) for c in range(64) if G[r][c]=='W']
    colcount = {}
    for r in range(64):
        for c in range(64):
            ch=G[r][c]
            if ch in ('M','N','S','O','r','Y'): colcount[ch]=colcount.get(ch,0)+1
    return W, colcount

def content():
    G = grid()
    return {(r,c) for r in range(64) for c in range(64) if G[r][c] not in ('B','g','c','p')}

def ok(kind, ctr, arm):
    C = content()
    t = tmpl_at(kind, ctr, arm)
    onb = {p for p in t if 0<=p[0]<64 and 0<=p[1]<64}
    return onb <= C

def count_color(ch): return len(cells_of(ch))

def press(name, act):
    m = MODEL[name]
    d = {'UP':(-3,0),'DOWN':(3,0),'LEFT':(0,-3),'RIGHT':(0,3)}[act]
    old = tuple(m['ctr'])
    res = action([act])
    m['ctr'] = [old[0]+d[0], old[1]+d[1]]
    if not ok2(m['kind'], m['ctr'], m['arm']) or res.get('no_op'):
        m['ctr'] = list(old); return ('FAIL', old)
    if res.get('level_completed'): return ('WIN',)
    if res.get('game_over'): return ('DEAD',)
    return ('ok', tuple(m['ctr']))

def run(name, acts):
    out=[]
    for a in acts:
        r = press(name,a); out.append(r)
        if r[0] in ('FAIL','WIN','DEAD'): break
    return out

def ok2(kind, ctr, arm):
    C = content() | {(63,c) for c in range(64)}
    t = tmpl_at(kind, ctr, arm)
    onb = {p for p in t if 0<=p[0]<64 and 0<=p[1]<64}
    return onb <= C

def okp(ctr):
    C = content() | {(63,c) for c in range(64)}
    t = tmpl_at('plus', ctr, 13)
    return {p for p in t if 0<=p[0]<64 and 0<=p[1]<64} <= C

def ppress(act):
    m = MODEL['P']
    d = {'UP':(-3,0),'DOWN':(3,0),'LEFT':(0,-3),'RIGHT':(0,3)}[act]
    old = tuple(m['ctr']); res = action([act]); m['ctr']=[old[0]+d[0],old[1]+d[1]]
    if not okp(tuple(m['ctr'])) or res.get('no_op'):
        m['ctr']=list(old); return ('FAIL',old)
    if res.get('level_completed'): return ('WIN',)
    if res.get('game_over'): return ('DEAD',)
    return ('ok',tuple(m['ctr']))

def prun(acts):
    out=[]
    for a in acts:
        r=ppress(a); out.append(r)
        if r[0]!='ok': break
    return out

def ppress2(act):
    m = MODEL['P']
    d = {'UP':(-3,0),'DOWN':(3,0),'LEFT':(0,-3),'RIGHT':(0,3)}[act]
    old = tuple(m['ctr']); res = action([act]); m['ctr']=[old[0]+d[0],old[1]+d[1]]
    if not okp(tuple(m['ctr'])) or res.get('no_op'):
        m['ctr']=list(old); return ('FAIL',old)
    if res.get('level_completed'): return ('WIN',)
    if res.get('game_over'): return ('DEAD',)
    return ('ok',tuple(m['ctr']))

def cc(ch): return {(r,c) for r in range(64) for c in range(64) if G[r][c]==ch}

def shape_of(ch):
    S = cc(ch)
    iso = {p for p in S if not any(q in S for q in nbrs8(p))}
    keep = S - iso
    rs=[p[0] for p in keep]; cs=[p[1] for p in keep]
    return (min(rs),max(rs),min(cs),max(cs),len(keep))

def comps(ch):
    S = {(r,c) for r in range(64) for c in range(64) if G[r][c]==ch}
    seen=set(); out=[]
    for p in S:
        if p in seen: continue
        st=[p]; comp=set()
        while st:
            q=st.pop()
            if q in comp or q not in S: continue
            comp.add(q); seen.add(q)
            for q2 in nbrs8(q): st.append(q2)
        out.append(comp)
    return sorted(out, key=len, reverse=True)

def tmpl3(kind, ctr, arm):
    r,c = ctr
    if kind=='X':
        s=set()
        for d in range(-arm,arm+1): s.add((r+d,c+d)); s.add((r+d,c-d))
        return s
    if kind=='plus':
        return {(r,c+d) for d in range(-arm,arm+1)} | {(r+d,c) for d in range(-arm,arm+1)}
    s=set()
    for d in range(-arm,arm+1):
        s.add((r+d, c+(arm-abs(d)))); s.add((r+d, c-(arm-abs(d))))
    return s

def okc(kind, ctr, arm):
    C = content() | {(63,c) for c in range(64)} | {(r,c) for r in range(64) for c in range(64) if grid()[r][c]=='W'}
    t = tmpl3(kind, ctr, arm)
    return {p for p in t if 0<=p[0]<64 and 0<=p[1]<64} <= C

def pressN(mod, name, act):
    kind, arm, ctr = mod[name]
    d = {'UP':(-3,0),'DOWN':(3,0),'LEFT':(0,-3),'RIGHT':(0,3)}[act]
    old = ctr; new = (old[0]+d[0], old[1]+d[1])
    res = action([act])
    if res.get('no_op') or not okc(kind, new, arm):
        return ('FAIL', old)
    mod[name] = (kind, arm, new)
    if res.get('level_completed'): return ('WIN',)
    if res.get('game_over'): return ('DEAD',)
    return ('ok', new)

def runN(mod, name, acts):
    out=[]
    for a in acts:
        r = pressN(mod,name,a); out.append((a,r[0],r[1]))
        if r[0] != 'ok': break
    return out

def boxes_small(G):
    # find 3x3 charcoal-bordered boxes
    out=[]
    for r in range(1,62):
        for c in range(1,62):
            ring = [(r-1,c-1),(r-1,c),(r-1,c+1),(r,c-1),(r,c+1),(r+1,c-1),(r+1,c),(r+1,c+1)]
            if all(G[x][y]=='c' for x,y in ring):
                out.append(((r,c), G[r][c]))
    # dedupe overlapping: keep those whose center is not itself ring of another
    res={}
    for (pos,ctr) in out: res[pos]=ctr
    return sorted(res.items())

def dots_of(G):
    out={}
    for (pos,ctr) in boxes_small(G):
        out[pos]=ctr
    return out

def sockets(G, thr=5):
    out={}
    for r in range(1,63):
        for c in range(1,63):
            ring=[G[r+dr][c+dc] for dr in (-1,0,1) for dc in (-1,0,1) if (dr,dc)!=(0,0)]
            n=sum(1 for x in ring if x=='c')
            if n>=thr: out[(r,c)]=(G[r][c], n)
    return out

def ringc(G,p):
    return sum(1 for dr in (-1,0,1) for dc in (-1,0,1) if (dr,dc)!=(0,0) and G[p[0]+dr][p[1]+dc]=='c')

def xcover(a,b,arm=11):
    s=set()
    for d in range(-arm,arm+1): s.add((a+d,b+d)); s.add((a+d,b-d))
    return s

def find(targets, arm=11):
    out=[]
    for a in range(64):
        for b in range(64):
            S = xcover(a,b,arm)
            if all(p in S for p in targets): out.append((a,b))
    return out

def cells_for(shape, ctr):
    return tmpl3(shape if shape!='D' else 'dia', ctr, arms[shape])

def cells_of_shape(sh,ctr): return tmpl3(KIND[sh],ctr,ARMS[sh])

def cover_of(sh,ctr,col):
    S=cells_of_shape(sh,ctr); return frozenset(k for k,(p,c) in DOTS.items() if c==col and p in S)

def dist(a,b): return (abs(a[0]-b[0])+abs(a[1]-b[1]))//3

def touch_positions(sh, col, r0, c0):
    res=[]
    for a in range(-15,80):
        for b in range(-15,80):
            if (a-r0)%3 or (b-c0)%3: continue
            R0,R1,C0,C1 = BOX[col]
            S = cells_of_shape(sh,(a,b))
            if any(R0<=p[0]<=R1 and C0<=p[1]<=C1 for p in S): res.append((a,b))
    return res

def options(sh):
    r0,c0=CUR[sh]; out={}
    for a in range(-15,80):
        for b in range(-15,80):
            if (a-r0)%3 or (b-c0)%3: continue
            for col in 'YSNbR':
                cv=cover_of(sh,(a,b),col)
                if not cv: continue
                if col==STARTCOL[sh]: cost,wp=dist(CUR[sh],(a,b)),None
                else:
                    bt=best_touch(sh,col,CUR[sh],(a,b))
                    if bt is None: continue
                    cost,wp=bt
                key=(col,cv)
                if key not in out or cost<out[key][0]: out[key]=(cost,(a,b),wp)
    return out

def touch_best(sh, col, r0, c0, target):
    R0,R1,C0,C1 = BOX[col]; best=None
    for a in range(-15,80):
        for b in range(-15,80):
            if (a-r0)%3 or (b-c0)%3: continue
            S = cells_of_shape(sh,(a,b))
            if any(R0<=p[0]<=R1 and C0<=p[1]<=C1 for p in S):
                c = dist((r0,c0),(a,b))+dist((a,b),target)
                if best is None or c<best: best=c
    return best

def touches(sh,ctr,col):
    R0,R1,C0,C1=FILL[col]; return any(R0<=p[0]<=R1 and C0<=p[1]<=C1 for p in cells_of_shape(sh,ctr))

def best_touch(sh,col,start,target):
    r0,c0=start; best=None
    for a in range(-15,80):
        for b in range(-15,80):
            if (a-r0)%3 or (b-c0)%3: continue
            if touches(sh,(a,b),col):
                c=dist(start,(a,b))+dist((a,b),target)
                if best is None or c<best[0]: best=(c,(a,b))
    return best

def fills_touched(sh,ctr):
    S=cells_of_shape(sh,ctr); out=[]
    for k,(r0,r1,c0,c1) in FILL.items():
        if any(r0<=p[0]<=r1 and c0<=p[1]<=c1 for p in S): out.append(k)
    return out

def selw(mod, name, maxp=4):
    tgt = mod[name][2]; log=[]
    for _ in range(maxp):
        if cells_of('W')=={tgt}: return ('selected', name, log)
        action(['SPACE']); log.append(('W', sorted(cells_of('W'))))
    return ('FAIL', name, sorted(cells_of('W')), tgt, log)

def go(mod, name, seq):
    res = runN(mod, name, seq)
    return res

def plus_cells(c,arm=12):
    r,b=c; return {(r,b+d) for d in range(-arm,arm+1)} | {(r+d,b) for d in range(-arm,arm+1)}

def frame_cells(c,h=9):
    r,b=c; S=set()
    for d in range(-h,h+1): S.add((r-h,b+d)); S.add((r+h,b+d)); S.add((r+d,b-h)); S.add((r+d,b+h))
    return S

def mv(n, act):
    out=[]
    for _ in range(n):
        before=sorted(cells_of('W'))
        action([act])
        after=sorted(cells_of('W'))
        out.append((act, before, after, before!=after))
    return out

def bboxY():
    Y=[p for p in cells_of('Y') if p not in {(30,45),(30,54),(57,45),(57,54)}]
    rs=[p[0] for p in Y]; cs=[p[1] for p in Y]
    return (min(rs),max(rs),min(cs),max(cs),len(Y))

def bboxY_f(f):
    g=G_of(f)
    EX={(30,45),(30,54),(57,45),(57,54)}
    Y=[(r,c) for r in range(64) for c in range(64) if g[r][c]=='Y' and (r,c) not in EX]
    W=[(r,c) for r in range(64) for c in range(64) if g[r][c]=='W']
    rs=[p[0] for p in Y]; cs=[p[1] for p in Y]
    return (min(rs),max(rs),min(cs),max(cs),len(Y),W)

def G_of(f): return [list(line) for line in f.ascii.split('\n')]

def bboxB(f=None):
    g = G_of(f) if f else grid()
    B=[(r,c) for r in range(64) for c in range(64) if g[r][c]=='b' and (r,c) not in {(6,12),(9,9),(9,30),(27,12)}]
    rs=[p[0] for p in B]; cs=[p[1] for p in B]
    return (min(rs),max(rs),min(cs),max(cs),len(B))

def rect_outline(r0,c0,r1,c1):
    S=set()
    for c in range(c0,c1+1): S.add((r0,c)); S.add((r1,c))
    for r in range(r0,r1+1): S.add((r,c0)); S.add((r,c1))
    return S

def plus(c,arm=12):
    r,b=c; return {(r,b+d) for d in range(-arm,arm+1)} | {(r+d,b) for d in range(-arm,arm+1)}
