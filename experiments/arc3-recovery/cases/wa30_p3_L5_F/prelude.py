try:
    import json
except Exception:
    pass
try:
    from collections import deque
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

def compact(d):
    out = {}
    for k,v in d.items():
        if isinstance(v, list):
            out[k] = v[:8]
        else:
            out[k]=v
    return out

def greens():
    seg = current_frame.segmentation
    out = []
    for n in seg['nodes']:
        if n['color'] in ('N','W','b','c','g','P'):
            b = n['boundary']
            out.append((n['color'], n['pixels'], (b[0][0],b[0][1],b[2][0],b[2][1])))
    return out

def bbox(b):
    rs=[p[0] for p in b]; cs=[p[1] for p in b]
    return (min(rs),min(cs),max(rs),max(cs))

def snap(fr):
    O=[];B=[]
    for n in fr.segmentation['nodes']:
        if n['color']=='O': O.append((bb(n)[0]//4,bb(n)[1]//4))
        if n['color']=='B' and n['pixels'] in (12,24): B.append((bb(n)[0]//4,bb(n)[1]//4,n['pixels']))
    return sorted(O),sorted(B)

def boxes():
    seg = current_frame.segmentation
    out=[]
    for n in seg['nodes']:
        if n['color']=='c' and n['pixels']>=8:
            out.append(bbox(n['boundary']))
    return sorted(out)

def lat():
    seg = current_frame.segmentation
    for n in seg['nodes']:
        if n['color']=='N':
            r,c,_,_ = bbox(n['boundary'])
            return (r//4, c//4)

def objat(fr, r0,c0):
    seg = fr.segmentation
    for n in seg['nodes']:
        b = bbox(n['boundary'])
        if b[0]<=r0<=b[2] and b[1]<=c0<=b[3]:
            return (n['color'], n['pixels'], b)
    return None

def objs_in(fr, r0,c0,r1,c1):
    seg = fr.segmentation
    out=[]
    for n in seg['nodes']:
        if n['pixels'] > 3000: continue
        b = bbox(n['boundary'])
        if not (b[2]<r0 or b[0]>r1 or b[3]<c0 or b[1]>c1):
            out.append((n['color'], n['pixels'], b))
    return sorted(out)

def state(regions={'A':(24,44,27,47),'B':(28,16,31,19),'C':(36,32,39,35)}):
    fr = current_frame
    seg = fr.segmentation
    N=None
    for n in seg['nodes']:
        if n['color']=='N': N=bbox(n['boundary'])
    st={'N':N}
    for k,(r0,c0,r1,c1) in regions.items():
        o = objs_in(fr, r0,c0,r1,c1)
        st[k] = ''.join(sorted(x[0] for x in o))
    return st

def tube(fr=current_frame):
    out=[]
    for n in fr.segmentation['nodes']:
        if n['color'] in ('g','b') and n['pixels']>4:
            out.append((n['color'], n['pixels'], bbox(n['boundary'])))
    return out

def blueboxes(fr):
    out=[]
    for n in fr.segmentation['nodes']:
        if n['color']=='b' and n['pixels']==4:
            out.append(bbox(n['boundary']))
    return sorted(out)

def inb(p): return 0<=p[0]<16 and 0<=p[1]<16

def bfs(start, obstacles, carry=None, goal_test=None):
    # state (pos, facing); returns dict of (pos,facing)->path
    start_state=(start, None)
    q=deque([start_state]); seen={start_state:[]}
    while q:
        s=q.popleft(); path=seen[s]
        if goal_test is not None and goal_test(s[0], s[1]):
            return path
        pos,f=s
        for name,(dr,dc) in MOVES.items():
            np_=(pos[0]+dr,pos[1]+dc)
            if not inb(np_): continue
            if np_ in obstacles: continue
            if carry is not None:
                box=(pos[0]+carry[0],pos[1]+carry[1])
                nb=(np_[0]+carry[0],np_[1]+carry[1])
                if not inb(nb): continue
                if nb in obstacles: continue
                if np_==box: continue   # would move into own carried box
            ns=(np_,name)
            if ns not in seen:
                seen[ns]=path+[name]
                q.append(ns)
    return None

def stage(start, facing, box_cell, target, obstacles):
    # obstacles: frozenset of static box cells (includes box_cell)
    best=None
    for side,(dr,dc) in MOVES.items():
        grab=(box_cell[0]-dr, box_cell[1]-dc)
        if not inb(grab): continue
        obs1=set(obstacles)
        p1=bfs(start, frozenset(obs1), None, lambda p,f,grab=grab,side=side: p==grab and f==side)
        if p1 is None: continue
        # after grab, box carried at offset (dr,dc)
        drop=(target[0]-dr, target[1]-dc)
        if not inb(drop): continue
        obs2=set(obstacles); obs2.discard(box_cell)
        p2=bfs(grab, frozenset(obs2), (dr,dc), lambda p,f,drop=drop: p==drop)
        if p2 is None: continue
        cost=len(p1)+1+len(p2)+1
        acts=p1+['SPACE']+p2+['SPACE']
        if best is None or cost<best[0]:
            best=(cost, acts, drop, side)
    return best

def full():
    out={}
    for n in current_frame.segmentation['nodes']:
        if n['pixels']<3000:
            out[(n['color'],tuple(bbox(n['boundary'])))] = n['pixels']
    return sorted([ (k[0],k[1],v) for k,v in out.items() ])

def agents():
    o=[]
    for n in current_frame.segmentation['nodes']:
        if n['pixels']<3000 and n['color'] in ('O','B','N'):
            b=bbox(n['boundary']); o.append((n['color'], b[0]//4, b[1]//4))
    return sorted(o)

def tube_occ():
    occ=set()
    for n in current_frame.segmentation['nodes']:
        if n['color'] in ('c','B') and n['pixels']==12:
            b=bbox(n['boundary'])
            r,c=b[0]//4,b[1]//4
            if (r,c) in [(7,3),(7,4),(8,3),(8,4),(9,3),(9,4)]: occ.add((r,c))
    return sorted(occ)

def timer():
    for n in current_frame.segmentation['nodes']:
        if n['color']=='P': return n['pixels']

def grid_state(fr=current_frame):
    occ=set(); P=None; O=None
    for n in fr.segmentation['nodes']:
        if n['pixels']>=3000: continue
        b=bbox(n['boundary']); r,c=b[0]//4,b[1]//4
        if n['color']=='N':
            P=(r,c)
            # player occupies 1 cell: use bbox top-left, but N may be 3x4 due to stripe; recompute
        if n['color']=='O': O=(r,c)
        if n['color'] in ('c','B','G'): occ.add((r,c))
    # fix player cell: N is 12px covering 3 rows x4 cols or 4x3; the true cell is where W stripe is not
    return P,O,occ

def player_cell(fr=current_frame):
    # player cell = the 4x4 cell containing most N pixels
    best=None
    for n in fr.segmentation['nodes']:
        if n['color']=='N':
            b=bbox(n['boundary'])
            for r in range(b[0]//4, b[2]//4+1):
                for c in range(b[1]//4, b[3]//4+1):
                    cnt=sum(1 for rr in range(r*4,r*4+4) for cc in range(c*4,c*4+4)
                            if fr.ascii.split('\n')[rr][cc]=='N')
                    if best is None or cnt>best[0]: best=(cnt,(r,c))
    return best[1]

def st(fr):
    occ=set(); P=None; O=None; W=False
    lines=fr.ascii.split('\n')
    for n in fr.segmentation['nodes']:
        if n['pixels']>=3000: continue
        b=bbox(n['boundary'])
        if b[0]>=60: continue
        r,c=b[0]//4,b[1]//4
        col=n['color']
        if col=='O': O=(r,c)
        elif col in ('c','B','G'): occ.add((r,c))
        elif col=='W':
            if r<15: W=True
    # player cell: max N density
    best=(0,None)
    for r in range(16):
        for c in range(16):
            cnt=sum(1 for rr in range(r*4,r*4+4) for cc in range(c*4,c*4+4) if lines[rr][cc]=='N')
            if cnt>best[0]: best=(cnt,(r,c))
    P=best[1]
    return P,O,occ|{O} if O else occ, W

def carried(fr):
    for n in fr.segmentation['nodes']:
        if n['color']=='W' and n['pixels']>=8 and bbox(n['boundary'])[0]<60: return True
    return False

def plan_deliver(start, box, targets, obstacles):
    MOVES={'U':(-1,0),'D':(1,0),'L':(0,-1),'R':(0,1)}
    NAME={'U':'UP','D':'DOWN','L':'LEFT','R':'RIGHT'}
    def inb(p): return 0<=p[0]<16 and 0<=p[1]<16
    def bfs(st0, obs, carry, goal):
        from collections import deque
        s0=(st0,None); q=deque([s0]); seen={s0:[]}
        while q:
            s=q.popleft(); path=seen[s]
            if goal(s[0],s[1]): return path
            pos,f=s
            for nm,(dr,dc) in MOVES.items():
                np_=(pos[0]+dr,pos[1]+dc)
                if not inb(np_) or np_ in obs: continue
                if carry is not None:
                    cur=(pos[0]+carry[0],pos[1]+carry[1]); nb=(np_[0]+carry[0],np_[1]+carry[1])
                    if not inb(nb) or nb in obs or np_==cur: continue
                ns=(np_,nm)
                if ns not in seen:
                    seen[ns]=path+[nm]; q.append(ns)
        return None
    best=None
    for side,(dr,dc) in MOVES.items():
        grab=(box[0]-dr,box[1]-dc)
        if not inb(grab) or grab in obstacles: continue
        p1=bfs(start, frozenset(obstacles), None, lambda p,f,grab=grab,side=side: p==grab and f==side)
        if p1 is None: continue
        obs2=set(obstacles); obs2.discard(box)
        for tgt in targets:
            if tgt in obs2: continue
            drop=(tgt[0]-dr,tgt[1]-dc)
            if not inb(drop) or drop in obs2: continue
            p2=bfs(grab, frozenset(obs2), (dr,dc), lambda p,f,drop=drop: p==drop)
            if p2 is None: continue
            acts=[NAME[x] for x in p1]+['SPACE']+[NAME[x] for x in p2]+['SPACE']
            if best is None or len(acts)<len(best[0]):
                best=(acts, drop, side, tgt)
    return best

def occ_cells(fr):
    s=set()
    for n in fr.segmentation['nodes']:
        if n['pixels']>=3000: continue
        b=bbox(n['boundary'])
        if b[0]>=60: continue
        if n['color'] in ('c','B','G','O'): s.add((b[0]//4,b[1]//4))
    return s

def pcell(fr):
    lines=fr.ascii.split('\n'); best=(0,None)
    for r in range(16):
        for c in range(16):
            cnt=sum(1 for rr in range(r*4,r*4+4) for cc in range(c*4,c*4+4) if lines[rr][cc]=='N')
            if cnt>best[0]: best=(cnt,(r,c))
    return best[1]

def plan_push(start, box, wallcells, obstacles):
    MOVES={'U':(-1,0),'D':(1,0),'L':(0,-1),'R':(0,1)}
    NAME={'U':'UP','D':'DOWN','L':'LEFT','R':'RIGHT'}
    def inb(p): return 0<=p[0]<16 and 0<=p[1]<16
    def bfs(st0, obsP, obsB, carry, goal):
        from collections import deque
        s0=(st0,None); q=deque([s0]); seen={s0:[]}
        while q:
            s=q.popleft(); path=seen[s]
            if goal(s[0],s[1]): return path
            pos,f=s
            for nm,(dr,dc) in MOVES.items():
                np_=(pos[0]+dr,pos[1]+dc)
                if not inb(np_) or np_ in obsP: continue
                if carry is not None:
                    cur=(pos[0]+carry[0],pos[1]+carry[1]); nb=(np_[0]+carry[0],np_[1]+carry[1])
                    if not inb(nb) or nb in obsB or np_==cur: continue
                ns=(np_,nm)
                if ns not in seen:
                    seen[ns]=path+[nm]; q.append(ns)
        return None
    best=None
    for side,(dr,dc) in MOVES.items():
        grab=(box[0]-dr,box[1]-dc)
        if not inb(grab) or grab in obstacles or grab in wallcells: continue
        obsP1=frozenset(set(obstacles)|wallcells)
        p1=bfs(start, obsP1, obsP1, None, lambda p,f,grab=grab,side=side: p==grab and f==side)
        if p1 is None: continue
        obsP=set(obstacles)|set(wallcells)
        for tgt in wallcells:
            if tgt==box: continue
            obsB=set(obstacles)|(wallcells-{tgt})   # box may enter only tgt among wall cells
            drop=(tgt[0]-dr,tgt[1]-dc)
            if not inb(drop) or drop in obsP: continue
            p2=bfs(grab, frozenset(obsP), frozenset(obsB), (dr,dc), lambda p,f,drop=drop: p==drop)
            if p2 is None: continue
            acts=[NAME[x] for x in p1]+['SPACE']+[NAME[x] for x in p2]+['SPACE']
            if best is None or len(acts)<len(best[0]): best=(acts,tgt,side)
    return best

def bar(fr):
    lines=fr.ascii.split('\n')
    row=lines[60]+lines[61]+lines[62]+lines[63]
    from collections import Counter
    c=Counter(row)
    return dict(c)

def plan_room(start, facing, box, interior, wall, others):
    MOVES={'U':(-1,0),'D':(1,0),'L':(0,-1),'R':(0,1)}
    NAME={'U':'UP','D':'DOWN','L':'LEFT','R':'RIGHT'}
    def bfs_push(st0,f0,box0,obsP,obsB):
        from collections import deque
        s0=(st0,f0,box0); q=deque([s0]); seen={s0:[]}
        while q:
            s=q.popleft(); path=seen[s]
            p,f,bx=s
            if bx in wall and p in interior: return path
            for nm,(dr,dc) in MOVES.items():
                np_=(p[0]+dr,p[1]+dc)
                if np_ not in interior: continue
                nb=(bx[0]+dr,bx[1]+dc)
                if np_==bx:
                    if nb in obsB or nb not in (interior|wall): continue
                else:
                    if nb in obsB or nb not in interior: continue
                    if nb==np_: continue
                ns=(np_,nm,nb)
                if ns not in seen:
                    seen[ns]=path+[nm]; q.append(ns)
        return None
    best=None
    obsP=set(others)|{box}
    # phase1: reach a cell adjacent to box with facing toward it, then SPACE to grab
    for side,(dr,dc) in MOVES.items():
        grab=(box[0]-dr,box[1]-dc)
        if grab not in interior or grab in others: continue
        from collections import deque
        s0=(start,facing); q=deque([s0]); seen={s0:[]}
        p1=None
        while q:
            s=q.popleft(); path=seen[s]
            if s[0]==grab and s[1]==side: p1=path; break
            p,f=s
            for nm,(dr2,dc2) in MOVES.items():
                np_=(p[0]+dr2,p[1]+dc2)
                if np_ not in interior or np_ in others: continue
                ns=(np_,nm)
                if ns not in seen:
                    seen[ns]=path+[nm]; q.append(ns)
        if p1 is None: continue
        p2=bfs_push(grab,side,box,frozenset(obsP-{box}),frozenset(others))
        if p2 is None: continue
        acts=[NAME[x] for x in p1]+['SPACE']+[NAME[x] for x in p2]+['SPACE']
        if best is None or len(acts)<len(best[0]): best=(acts,(grab,side))
    return best

def iboxes(fr,INT):
    out=set()
    for n in fr.segmentation['nodes']:
        if n['color']=='c' and n['pixels'] in (12,24):
            b=bbox(n['boundary']); r0,c0=b[0]//4,b[1]//4
            if n['pixels']==12:
                if (r0,c0) in INT: out.add((r0,c0))
            else:
                dr=(b[2]-b[0])//4; dc=(b[3]-b[1])//4
                if dc>dr:
                    for c in (c0,c0+1):
                        if (r0,c) in INT: out.add((r0,c))
                else:
                    for r in (r0,r0+1):
                        if (r,c0) in INT: out.add((r,c))
    return out

def do_one(face,INT,WALL):
    fr=current_frame; p=pcell(fr); bs=iboxes(fr,INT)
    if not bs: return ('none',)
    cands=[]
    for b in bs:
        pl=plan_room(p,face,b,INT,WALL,bs-{b})
        if pl: cands.append((len(pl[0]),b,pl[0]))
    if not cands: return ('nostart',sorted(bs))
    cands.sort()
    n,b,acts=cands[0]
    for a in acts:
        r=action([a])
        if r['level_completed'] or r['game_over']: return ('END',a,r['level_completed'],r['game_over'])
        if a in ('UP','DOWN','LEFT','RIGHT'): face=a
    return (b,acts,face,sorted(iboxes(current_frame,INT)))

def fill(fr):
    T=[(5,14),(6,1),(7,1),(7,13),(9,2),(14,9),(14,10)]
    cen={}
    for n in fr.segmentation['nodes']:
        if n['color'] in ('b','g') and n['pixels']<=12:
            b=bb(n); cen[(b[0]//4,b[1]//4)]=n['color']
    return sum(1 for t in T if cen.get(t)=='b')

def bb(n):
    pts=n['boundary']; rs=[p[0] for p in pts]; cs=[p[1] for p in pts]; return (min(rs),min(cs),max(rs),max(cs))

def free_cell(p, for_box=False):
    if not (0<=p[0]<16 and 0<=p[1]<16): return False
    if p in BL: return False
    if p in BOXES: return False
    if not for_box and p in PLAYER: return False
    return True

goal=lambda o,b,c: b==(14,10)

def pl(fr):
    for n in fr.segmentation['nodes']:
        if n['color']=='N':
            b=bb(n); return (b[0]//4,b[1]//4)

def cen(fr):
    out={}
    for n in fr.segmentation['nodes']:
        if n['color'] in ('c','B','G') and n['pixels'] in (12,24):
            b=bb(n); out[(b[0]//4,b[1]//4)]=n['color']
    return out

def cells(fr,r0,c0,r1,c1):
    l=fr.ascii.split('\n'); out=[]
    for r in range(r0*4,r1*4+4):
        out.append('%2d '%r + l[r][c0*4:c1*4+4])
    return '\n'.join(out)
