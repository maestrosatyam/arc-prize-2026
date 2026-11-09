try:
    import json
except Exception:
    pass
try:
    import re
except Exception:
    pass
try:
    import math
except Exception:
    pass
try:
    import math, collections
except Exception:
    pass
try:
    import itertools
except Exception:
    pass

def crop(r0,r1,c0,c1,step=1):
    for r in range(r0,r1,step):
        print(r, rows[r][c0:c1])

def click(r,c):
    res = action([{'action':'MOUSE','row':r,'col':c}])
    d = frame_diff()
    out = {'noop':res['no_op'],'gp':res['gameplay_changed'],'lvl':res['level_completed'],'over':res['game_over']}
    ch=[]
    for k,v in d.items():
        if k!='changed_cell_count':
            for e in v[:4]:
                if k=='moved': ch.append((e['color'],e['from'],e['to']))
                elif k in ('appeared','disappeared'): ch.append((k,e.get('color'),e.get('bbox')))
                elif k=='changed_color': ch.append((k,e.get('at'),e.get('before_color'),e.get('after_color')))
    out['ch']=ch
    return out

def sqs():
    return sorted((n['boundary'][0][0],n['boundary'][0][1]) for n in current_frame.segmentation['nodes'] if n['color']=='M' and n['pixels']==4)

def move_test(r,c):
    before = sqs()
    out = click(r,c)
    return before, sqs(), out['noop']

def tokens():
    out=[]
    for nd in current_frame.segmentation['nodes']:
        if nd['color'] in ('M','p','Y','S') and nd['pixels'] in (4,9,16):
            r0,c0 = nd['boundary'][0]
            if r0<10: continue
            k = {4:1,9:1,16:2}[nd['pixels']]
            out.append({'color':nd['color'],'tl':(r0,c0),'k':k,'anchor':(r0+k,c0+k),'px':nd['pixels']})
    return out

def disc():
    rows = current_frame.ascii.split('\n')
    cells=set()
    for r in range(20,36):
        for c in range(24,42):
            if rows[r][c] in 'bMpY': cells.add((r,c))
    # fill per-row spans
    out=set()
    for r in range(20,36):
        cs=[c for (rr,c) in cells if rr==r]
        if cs:
            for c in range(min(cs),max(cs)+1): out.add((r,c))
    return out

def aligned(a,c):
    dr,dc = c[0]-a[0], c[1]-a[1]
    return (dr==0 and dc!=0) or (dc==0 and dr!=0) or (abs(dr)==abs(dc) and dr!=0)

def mover_for(cell, toks, rng=8):
    best=None; tie=False
    for t in toks:
        dr,dc = cell[0]-t['anchor'][0], cell[1]-t['anchor'][1]
        d=max(abs(dr),abs(dc))
        if d==0 or d>rng or not aligned(t['anchor'],cell): continue
        if best is None or d<best: best=d; tie=False
        elif d==best: tie=True
    if best is None: return None
    if tie: return ('tie',None)
    for t in toks:
        dr,dc = cell[0]-t['anchor'][0], cell[1]-t['anchor'][1]
        if max(abs(dr),abs(dc))==best and aligned(t['anchor'],cell): return (best,t)
    return None

def plan(toks, mine, target, rng=8):
    import collections
    start = mine['anchor']
    q=collections.deque([(start,[])]); seen={start}
    while q:
        pos,path = q.popleft()
        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            for d in range(1,rng+1):
                cell=(pos[0]+dr*d,pos[1]+dc*d)
                if not (0<=cell[0]<64 and 0<=cell[1]<64): continue
                mv = mover_for(cell, toks, rng)
                if mv is None or mv[0]=='tie': continue
                if mv[1]['anchor']!=pos: continue
                if cell not in seen:
                    seen.add(cell); q.append((cell,path+[cell]))
                    if cell==target: return path+[cell]
    return None

def plan2(toks, mine, target, rng=8, maxhops=4):
    import collections
    others=[t for t in toks if t is not mine]
    start=mine['anchor']
    q=collections.deque([(start,[])]); seen={start}
    while q:
        pos,path=q.popleft()
        if len(path)>=maxhops: continue
        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            for d in range(1,rng+1):
                cell=(pos[0]+dr*d,pos[1]+dc*d)
                if not(0<=cell[0]<64 and 0<=cell[1]<64): continue
                dyn=[dict(t) for t in others]+[dict(mine,anchor=pos)]
                mv=mover_for(cell,dyn,rng)
                if mv is None or mv[0]=='tie' or mv[1]['anchor']!=pos: continue
                if cell not in seen:
                    seen.add(cell); np=path+[cell]
                    if cell==target: return np
                    q.append((cell,np))
    return None

def maxd_for(dr,dc): return 7 if (dr==0 or dc==0) else 6

def mover_for2(cell, toks):
    best=None; tie=False
    for t in toks:
        dr,dc = cell[0]-t['anchor'][0], cell[1]-t['anchor'][1]
        if dr==0 and dc==0: return (0,t)
        if not aligned(t['anchor'],cell): continue
        d=max(abs(dr),abs(dc))
        if d>maxd_for(dr,dc): continue
        if best is None or d<best: best=d; tie=False
        elif d==best: tie=True
    if best is None: return None
    if tie: return ('tie',None)
    for t in toks:
        dr,dc = cell[0]-t['anchor'][0], cell[1]-t['anchor'][1]
        if max(abs(dr),abs(dc))==best and aligned(t['anchor'],cell): return (best,t)
    return None

def plan3(toks, mine, target, maxhops=3):
    import collections
    others=[t for t in toks if t is not mine]
    start=mine['anchor']
    q=collections.deque([(start,[])]); seen={start}
    while q:
        pos,path=q.popleft()
        if len(path)>=maxhops: continue
        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            for d in range(1,maxd_for(dr,dc)+1):
                cell=(pos[0]+dr*d,pos[1]+dc*d)
                if not(0<=cell[0]<64 and 0<=cell[1]<64): continue
                dyn=[dict(t) for t in others]+[dict(mine,anchor=pos)]
                mv=mover_for2(cell,dyn)
                if mv is None or mv[0]=='tie' or mv[1]['anchor']!=pos: continue
                if cell not in seen:
                    seen.add(cell); np=path+[cell]
                    if cell==target: return np
                    q.append((cell,np))
    return None

def bb(nd):
    b=nd['boundary']; pts=b
    return (min(p[0] for p in pts),min(p[1] for p in pts),max(p[0] for p in pts),max(p[1] for p in pts))

def merge_cell(toks, a, b, r_in=8.4, r_out=9.7):
    best=None
    A=[t for t in toks if t['anchor']==a][0]; B=[t for t in toks if t['anchor']==b][0]
    others=[t for t in toks if t is not A and t is not B]
    for r in range(64):
        for c in range(64):
            da=math.hypot(r-A['anchor'][0],c-A['anchor'][1]); db=math.hypot(r-B['anchor'][0],c-B['anchor'][1])
            if da>r_in or db>r_in: continue
            m=min(math.hypot(r-t['anchor'][0],c-t['anchor'][1]) for t in others) if others else 99
            if m<r_out: continue
            score=min(r_in-da,r_in-db,m-r_out,da+db)
            if best is None or score>best[0]: best=(score,(r,c),round(da,2),round(db,2),round(m,2))
    return best

def hops_from(toks, start, others, maxd=6, r_in=8.4, r_out=9.7):
    q=collections.deque([(start,0)]); dist={start:0}
    while q:
        pos,d=q.popleft()
        if d>=maxd: continue
        for dr in range(-9,10):
            for dc in range(-9,10):
                dd=math.hypot(dr,dc)
                if dd<1 or dd>r_in: continue
                cell=(pos[0]+dr,pos[1]+dc)
                if not(1<=cell[0]<63 and 1<=cell[1]<63): continue
                if any(math.hypot(cell[0]-o[0],cell[1]-o[1])<=r_out for o in others): continue
                if cell not in dist: dist[cell]=d+1; q.append((cell,d+1))
    return dist

def best_merge(toks, a, b, ball, r_in=8.4, r_out=9.7):
    A=[t for t in toks if t['anchor']==a][0]; B=[t for t in toks if t['anchor']==b][0]
    others=[t['anchor'] for t in toks if t is not A and t is not B]
    distA=hops_from(toks,A['anchor'],others+[B['anchor']])
    distB=hops_from(toks,B['anchor'],others+[A['anchor']])
    best=None
    for r in range(1,63):
        for c in range(1,63):
            da=math.hypot(r-A['anchor'][0],c-A['anchor'][1]); db=math.hypot(r-B['anchor'][0],c-B['anchor'][1])
            if da>r_in or db>r_in: continue
            if any(math.hypot(r-o[0],c-o[1])<=r_out for o in others): continue
            cost=None; who=None
            cand=[v for k,v in distB.items() if math.hypot(k[0]-r,k[1]-c)<=r_in]
            if cand: cost=min(cand)+1; who='B'
            candA=[v for k,v in distA.items() if math.hypot(k[0]-r,k[1]-c)<=r_in]
            if candA and (cost is None or min(candA)+1<cost): cost=min(candA)+1; who='A'
            if cost is None: continue
            key=(cost,math.hypot(r-ball[0],c-ball[1]))
            if best is None or key<best[0]: best=(key,(r,c),who)
    return best

def best_merge2(toks, a, b, ball, r_in=8.4, r_out=9.7, maxd=4):
    A=[t for t in toks if t['anchor']==a][0]; B=[t for t in toks if t['anchor']==b][0]
    others=[t['anchor'] for t in toks if t is not A and t is not B]
    distA=hops_from(toks,A['anchor'],others+[B['anchor']],maxd,r_in,r_out)
    distB=hops_from(toks,B['anchor'],others+[A['anchor']],maxd,r_in,r_out)
    best=None
    for r in range(1,63):
        for c in range(1,63):
            if any(math.hypot(r-o[0],c-o[1])<=r_out for o in others): continue
            da=math.hypot(r-A['anchor'][0],c-A['anchor'][1]); db=math.hypot(r-B['anchor'][0],c-B['anchor'][1])
            hA=0 if da<=r_in else min([v for k,v in distA.items() if math.hypot(k[0]-r,k[1]-c)<=r_in] or [99])
            hB=0 if db<=r_in else min([v for k,v in distB.items() if math.hypot(k[0]-r,k[1]-c)<=r_in] or [99])
            if hA==99 or hB==99: continue
            cost=hA+hB+1
            key=(cost,math.hypot(r-ball[0],c-ball[1]))
            if best is None or key<best[0]: best=(key,(r,c))
    return best

def route_merge(toks, a, b, c, r_in=8.4, r_out=9.7, maxd=4):
    A=[t for t in toks if t['anchor']==a][0]; B=[t for t in toks if t['anchor']==b][0]
    others=[t['anchor'] for t in toks if t is not A and t is not B]
    def bfs(start, obs):
        q=collections.deque([start]); par={start:None}
        while q:
            pos=q.popleft()
            if math.hypot(pos[0]-c[0],pos[1]-c[1])<=r_in:
                p=[];cur=pos
                while par[cur] is not None: p.append(cur); cur=par[cur]
                return p[::-1]
            if len([x for x in [0]]) and sum(1 for _ in [0])>0: pass
            depth=0; cur=pos
            while par[cur] is not None: depth+=1; cur=par[cur]
            if depth>=maxd: continue
            for dr in range(-9,10):
                for dc in range(-9,10):
                    dd=math.hypot(dr,dc)
                    if dd<1 or dd>r_in: continue
                    cell=(pos[0]+dr,pos[1]+dc)
                    if not(1<=cell[0]<63 and 1<=cell[1]<63): continue
                    if any(math.hypot(cell[0]-o[0],cell[1]-o[1])<=r_out for o in obs): continue
                    if cell not in par: par[cell]=pos; q.append(cell)
        return None
    da=math.hypot(c[0]-a[0],c[1]-a[1]); db=math.hypot(c[0]-b[0],c[1]-b[1])
    if da<=r_in and db<=r_in: return [c]
    if db<=r_in:
        p=bfs(a,others+[b])
        if p is not None: return p+[c]
    if da<=r_in:
        p=bfs(b,others+[a])
        if p is not None: return p+[c]
    pa=bfs(a,others+[b]); pb=bfs(b,others+[a]) if False else None
    if pa: return pa+[c]
    return None

def pbox(fr):
    xs=[]; 
    for nd in fr.segmentation['nodes']:
        if nd['color']=='P' and nd['pixels']<=12: xs+= nd['boundary']
    if not xs: return None
    return (min(p[0] for p in xs),min(p[1] for p in xs),max(p[0] for p in xs),max(p[1] for p in xs))

def misc(fr):
    out={}
    for nd in fr.segmentation['nodes']:
        if nd['color'] in 'bg' and nd['pixels']>=40: out['ball']=nd['color']
        if nd['pixels']==16 and nd['boundary'][0][0]<=6: out['tgt']=nd['color']
        if nd['pixels']==9 and nd['boundary'][0][0]<=6: out['tgt2']=nd['color']
    return out

def pboxes(fr):
    P=[(nd['boundary'][0][0],nd['boundary'][0][1],nd['boundary'][-1][0] if nd['boundary'][-1][0]>nd['boundary'][0][0] else 0,0) for nd in fr.segmentation['nodes'] if nd['color']=='P' and nd['pixels']>1]
    rs=[nd['boundary'][0][0] for nd in fr.segmentation['nodes'] if nd['color']=='P' and nd['pixels']==1]+[nd['boundary'][0][0] for nd in fr.segmentation['nodes'] if nd['color']=='P' and nd['pixels']==3]
    cs=[nd['boundary'][0][1] for nd in fr.segmentation['nodes'] if nd['color']=='P']
    return min(rs),max(rs),min(cs),max(cs)

def inv_boxes(fr):
    pts=[(nd['boundary'][0][0],nd['boundary'][0][1]) for nd in fr.segmentation['nodes'] if nd['color']=='P']
    out=[]
    for r,c in pts: out.append((r,c))
    # cluster by col halves
    L=[p for p in pts if p[1]<32]; R=[p for p in pts if p[1]>=32]
    def box(g):
        if not g: return None
        rs=[p[0] for p in g]; cs=[p[1] for p in g]
        return (min(rs),min(cs),max(rs),max(cs))
    return box(L),box(R)

def dbox(cell,bx):
    if bx is None: return 99
    r,c=cell; r0,c0,r1,c1=bx
    dr=max(r0-r,0,r-r1); dc=max(c0-c,0,c-c1)
    return (dr*dr+dc*dc)**.5

def bx(g):
    if not g: return None
    rs=[p[0] for p in g]; cs=[p[1] for p in g]; return (min(rs),min(cs),max(rs),max(cs))

def state(fr):
    seg=fr.segmentation
    pts=[(nd['boundary'][0][0],nd['boundary'][0][1]) for nd in seg['nodes'] if nd['color']=='P']
    L=[p for p in pts if p[0]<64 and p[1]<32]; R=[p for p in pts if p[1]>=32]
    toks=[(t['color'],t['anchor']) for t in tokens()]
    dots=[(nd['boundary'][0][0],nd['boundary'][0][1]) for nd in seg['nodes'] if nd['color']=='S' and nd['pixels']==1]
    return bx(L),bx(R),toks,dots

def hunters(fr):
    pts=[(nd['boundary'][0][0],nd['boundary'][0][1]) for nd in fr.segmentation['nodes'] if nd['color']=='P']
    L=[p for p in pts if p[1]<32]; R=[p for p in pts if p[1]>=32]
    return bx(L),bx(R)

def dcent(t,cell):
    r,c=t['anchor']; return ((r-0.5-cell[0])**2+(c-0.5-cell[1])**2)**.5

def check(cell,expect_idx):
    tk=tokens(); out=[]
    for i,t in enumerate(tk):
        d=dcent(t,cell); out.append((t['color'],t['anchor'],round(d,2),'IN' if d<=8.2 else ('?edge' if d<=8.8 else '')))
    H=hunters(current_frame); hd=[round(dbox(cell,h),1) for h in H]
    ins=[i for i,o in enumerate(out) if o[3]=='IN']
    ok = sorted(ins)==sorted(expect_idx) and min(hd)>=9
    return ok,out,hd

def throw_cell(H, idx):
    b=H[idx]
    if b is None: return None
    r0,c0,r1,c1=b; cr=(r0+r1)/2.0; cc=(c0+c1)/2.0
    best=None
    for dr in (7,8,6,9,5):
        for dc in (0,1,-1,2,-2,3,-3,4,-4):
            cell=(int(round(r1+dr)), int(round(cc+dc)))
            if not (9<=cell[0]<=62 and 1<=cell[1]<=62): continue
            d=dbox(cell,b)
            if d>8.35: continue
            tk=tokens(); mind=min([dcent(t,cell) for t in tk]+[99])
            if mind<9.0: continue
            score=(d, -mind)
            if best is None or score<best[0]: best=(score,cell,round(d,2),round(mind,2))
        if best: break
    return best[1] if best else None

def run_step(cell,want_anchors):
    tk=tokens(); ins=[t['anchor'] for t in tk if dcent(t,cell)<=8.2]
    H=hunters(current_frame); hd=[round(dbox(cell,h),1) for h in H]
    if sorted(ins)!=sorted(want_anchors) or min(hd)<9:
        print('  BLOCK',cell,'in-range',ins,'Hdist',hd); return None
    res=click(cell[0],cell[1])
    print(cell,'noop' if res['noop'] else ('LEVEL' if res['lvl'] else ('OVER' if res['over'] else 'ok')),[(t['color'],t['anchor']) for t in tokens()],'H',hunters(current_frame),'misc',misc(current_frame))
    return res

def snap(fr):
    seg=fr.segmentation; out=[]
    for n in seg['nodes']:
        if n['color'] in ('P','R','b','Y','O','p','M','S') and n['pixels']>=4: out.append((n['color'],bb(n),n['pixels']))
        elif n['color']=='P': out.append(('P.',bb(n),n['pixels']))
    return out

def tok(fr):
    return [(n['color'],bb(n),n['pixels']) for n in fr.segmentation['nodes'] if n['color'] in ('R','O','Y','p','M','S') and n['pixels']>=4 and n['boundary'][0][0]>8]

def inv(fr):
    pts=[(n['boundary'][0][0],n['boundary'][0][1]) for n in fr.segmentation['nodes'] if n['color']=='P' and n['boundary'][0][0]>8]
    return bx(pts) if pts else None

def ocenter(fr):
    for n in fr.segmentation['nodes']:
        if n['color']=='O' and n['pixels']>=9:
            r0,c0,r1,c1=bb(n); return ((r0+r1)/2.0,(c0+c1)/2.0,(r0,c0,r1,c1))
    return None

def st(fr):
    seg=fr.segmentation
    o=[(bb(n),n['pixels']) for n in seg['nodes'] if n['color']=='O' and n['pixels']>10]
    y=[(bb(n),n['pixels']) for n in seg['nodes'] if n['color']=='Y' and n['pixels']>10]
    p=[n['boundary'][0][0:2] for n in seg['nodes'] if n['color']=='P' and n['boundary'][0][0]>8]
    bar=[x['pixels'] for x in seg['nodes'] if x['color']=='W']
    return o,y,(bx(p) if p else None),bar

def st2(fr):
    seg=fr.segmentation
    board=[(n['color'],bb(n),n['pixels']) for n in seg['nodes'] if n['color'] in ('O','Y','R','p','M','S','b') and n['pixels']>10 and n['boundary'][0][0]>8]
    p=[n['boundary'][0][0:2] for n in seg['nodes'] if n['color']=='P' and n['boundary'][0][0]>8]
    bar=[x['pixels'] for x in seg['nodes'] if x['color']=='W']
    return board,(bx(p) if p else None),bar

def toks7(fr):
    out=[]
    for n in fr.segmentation['nodes']:
        c=n['color']
        if c in ('M','p','Y','O','R','S') and n['boundary'][0][0]>8:
            r0,c0,r1,c1=bb(n); out.append({'color':c,'anchor':(r1,c1),'bbox':(r0,c0,r1,c1),'size':r1-r0+1})
    return out

def rng(sz): return {1:8,2:8,4:8,5:10,7:12}.get(sz,8)
