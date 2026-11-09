try:
    import json
except Exception:
    pass
try:
    import math
except Exception:
    pass
try:
    import math, heapq
except Exception:
    pass

def ppos():
    seg = current_frame.segmentation
    p = [n for n in seg['nodes'] if n['color']=='p' and n['pixels']==9 and n['boundary'][0][0]>10]
    b = p[0]['boundary'][0]
    return (b[0]+1, b[1]+1)

def state_sig(f):
    seg = f.segmentation
    dots = sorted(tuple(n['boundary'][0]) for n in seg['nodes'] if n['color']=='S' and n['pixels']==1)
    ms = sorted(tuple(n['boundary'][0]) for n in seg['nodes'] if n['color']=='M' and n['boundary'][0][0]>10)
    ps = sorted(tuple(n['boundary'][0]) for n in seg['nodes'] if n['color']=='p' and n['pixels']==9)
    return dots, ms, ps

def objs(seg): return sorted([(n['color'],n['pixels'],tuple(n['boundary'][0])) for n in seg['nodes'] if n['boundary'][0][0]>10 and n['color'] not in ('B','W')])

def bar(f): 
    n=[x for x in f.segmentation['nodes'] if x['color']=='W']; return n[0]['pixels'] if n else None

def snap(t): return (int(round(t[0])),int(round(t[1])))

def pcenter(Ppix):
    # Ppix are top-left of each sub-component; reconstruct full body via known shape offsets
    # instead: use min row/col of all P sub-parts and known diamond geometry
    rs=[p[0] for p in Ppix]; cs=[p[1] for p in Ppix]
    # the diamond spans 4 rows x 5 cols; top tip at (minr, minc+2)
    return (min(rs)+1.5, min(cs)+2.0)

def pbody(c):
    r,c0 = c
    rr = math.floor(r)
    # body rows rr-1..rr+2 approx; use radius-3 diamond on center
    return [(rr-1,c0),(rr,c0-2),(rr,c0+2),(rr+1,c0-1),(rr+1,c0),(rr+1,c0+1),(rr-2,c0)]

def cells(bbox):
    r0,c0,r1,c1 = bbox
    return [(r,c) for r in range(r0,r1+1) for c in range(c0,c1+1)]

def risk(click, Ppix, objs_):
    pc = pcenter(Ppix)
    d = math.dist(click, pc)
    if d <= 10.5:
        v = ((click[0]-pc[0])/d, (click[1]-pc[1])/d)
        newc = (pc[0]+13*v[0], pc[1]+13*v[1])
        kind='JUMP'
    else:
        # drift toward nearest object
        best=None;bd=99
        for col,bb,px in objs_:
            cc = ((bb[0]+bb[2])/2,(bb[1]+bb[3])/2)
            dd = math.dist(pc,cc)
            if dd<bd: bd=dd;best=cc
        if bd>20: return ('stay', pc, None)
        v=((best[0]-pc[0])/bd,(best[1]-pc[1])/bd)
        newc=(pc[0]+4.5*v[0], pc[1]+4.5*v[1]); kind='drift'
    body = pbody(newc)
    hits=[]
    for col,bb,px in objs_:
        cc = ((bb[0]+bb[2])/2,(bb[1]+bb[3])/2)
        if math.dist(newc,cc) <= 4.5: hits.append((col,bb))
    return (kind, (round(newc[0],1),round(newc[1],1)), hits)

def u(a,b):
    dx,dy=b[0]-a[0],b[1]-a[1]; d=math.hypot(dx,dy); return (dx/d,dy/d,d)

def plan(P0, Y0, goal, maxstep=5.8, jump=13.0, drift=4.5, safe=11.0, eat=4.5, maxnodes=60):
    # state: (Y pos, P pos); action: choose next Y center within maxstep of Y; P jumps if dist(P,click)<=jump else drifts drift toward new Y
    start=(Y0,P0)
    def h(Y): return math.hypot(Y[0]-goal[0],Y[1]-goal[1])
    pq=[(h(Y0),0,start,None)]
    seen={start:0}
    while pq:
        f,g,st,_=heapq.heappop(pq)
        Y,P=st
        if h(Y)<=4.0: 
            # reconstruct
            return g
        if g>=maxnodes: continue
        for dr in [-6,-5,-4,-3,0,3,4,5,6]:
            for dc in [-6,-5,-4,-3,0,3,4,5,6]:
                nY=(Y[0]+dr,Y[1]+dc)
                d=math.hypot(dr,dc)
                if d<3 or d>maxstep: continue
                if not(11<=nY[0]<=60 and 1<=nY[1]<=62): continue
                dp=math.hypot(P[0]-nY[0],P[1]-nY[1])
                if dp<=jump:  # P jumps toward click -> lands ~13 toward it -> likely eats
                    continue
                dd=max(dp-drift,0.0)
                if dd<=eat: continue
                u_,v_=P[0]-nY[0],P[1]-nY[1]
                nP=(nY[0]+u_/dp*dd, nY[1]+v_/dp*dd)
                nst=(nY,nP)
                ng=g+1
                if nst in seen and seen[nst]<=ng: continue
                seen[nst]=ng
                heapq.heappush(pq,(ng+h(nY),ng,nst,(Y,P,nY)))
    return None

def bb(n):
    b=n['boundary']; rs=[p[0] for p in b]; cs=[p[1] for p in b]
    return (min(rs),min(cs),max(rs),max(cs))

def bigP(f):
    out=[]
    for n in f.segmentation['nodes']:
        if n['color']=='P' and n['pixels']>=4:
            b=n['boundary']; rs=[p[0] for p in b]; cs=[p[1] for p in b]
            out.append((round((min(rs)+max(rs))/2,1),round((min(cs)+max(cs))/2,1)))
    return out

def Yc(f):
    for n in f.segmentation['nodes']:
        if n['color']=='Y' and n['pixels']==16:
            b=n['boundary']; rs=[p[0] for p in b]; cs=[p[1] for p in b]
            return ((min(rs)+max(rs))/2,(min(cs)+max(cs))/2)
    return None

def pclusters(f):
    pts=[]
    for n in f.segmentation['nodes']:
        if n['color']=='P':
            b=n['boundary']
            for p in b: pts.append((p[0],p[1]))
    pts=[p for p in pts if p[0]>9]
    # cluster by proximity (diamond spans 5x5)
    clusters=[]
    for p in pts:
        for cl in clusters:
            if abs(cl['r'][0]-p[0])<=4 and abs(cl['c'][0]-p[1])<=4:
                cl['r'].append(p[0]); cl['c'].append(p[1]); break
        else:
            clusters.append({'r':[p[0]],'c':[p[1]]})
    return [(round(sum(c['r'])/len(c['r']),1), round(sum(c['c'])/len(c['c']),1)) for c in clusters]

def col_at(f,r,c): return f.ascii.split('\n')[r][c]

def barf(f):
    row=f.ascii.split('\n')[63]
    return sum(1 for ch in row if ch not in ('B','W'))

def state(f):
    objs=[];Ps=[];blobs=[]
    pts={'P':[]}
    for n in f.segmentation['nodes']:
        b=n['boundary']; rs=[p[0] for p in b]; cs=[p[1] for p in b]
        r0,c0,r1,c1=min(rs),min(cs),max(rs),max(cs); ctr=((r0+r1)/2,(c0+c1)/2)
        if n['color']=='P':
            if r0>9: pts['P'].append(ctr)
        elif n['color']=='b' and n['pixels']>40: blobs.append(ctr)
        elif n['color'] in ('M','p','Y','O','R') and r0>9:
            objs.append({'c':n['color'],'ctr':ctr,'px':n['pixels'],'bb':(r0,c0,r1,c1)})
    cl=[]
    for p in pts['P']:
        for c in cl:
            if abs(c[0][0]-p[0])<=4 and abs(c[0][1]-p[1])<=4: c.append(p); break
        else: cl.append([p])
    for c in cl:
        Ps.append((sum(x[0] for x in c)/len(c), sum(x[1] for x in c)/len(c)))
    return objs,Ps,blobs

def D(a,b): return math.hypot(a[0]-b[0],a[1]-b[1])

def hurl_click(P, objs):
    best=None
    for dr in range(-10,11):
        for dc in range(-10,11):
            d=math.hypot(dr,dc)
            if d<3 or d>10.4: continue
            cr,cc=snap((P[0]+dr,P[1]+dc))
            if not(11<=cr<=62 and 0<=cc<=63): continue
            if any(D((cr,cc),o['ctr'])<=7.5 for o in objs): continue
            ux,uy=dr/d,dc/d
            land=(min(62,max(11,P[0]+13*ux)), min(63,max(0,P[1]+13*uy)))
            m=min(D(land,o['ctr']) for o in objs)
            if best is None or m>best[0]: best=(m,(cr,cc),land)
    return best

def away_click(o,P,dmin=10.6):
    v=(o['ctr'][0]-P[0], o['ctr'][1]-P[1]); d=math.hypot(*v)
    if d<0.1: v=(1,0); d=1
    for k in (6,5,4):
        t=snap((o['ctr'][0]+v[0]/d*k, o['ctr'][1]+v[1]/d*k))
        if 11<=t[0]<=60 and 0<=t[1]<=62 and D(t,P)>dmin: return t
    return None
