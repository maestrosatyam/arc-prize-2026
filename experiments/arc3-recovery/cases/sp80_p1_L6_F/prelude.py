try:
    import json
except Exception:
    pass
try:
    import random
except Exception:
    pass
try:
    import itertools
except Exception:
    pass
try:
    import random, time
except Exception:
    pass

def crop(r0,r1,c0,c1):
    for r in range(r0,r1+1):
        print(r, a[r][c0:c1+1])

def brief(d):
    out={}
    for k,v in d.items():
        if k=='changed_cell_count': out[k]=v
        elif k=='moved': out['moved']=[(m['color'],m['from'],m['to']) for m in v]
        elif k=='rotated': out['rot']=[(r.get('color'),r.get('rotated_by')) for r in v]
        elif k in ('appeared','disappeared'): out[k]=[(x['color'],x['bbox']) for x in v]
        elif k=='changed_color': out['cc']=[(c['at'],c['before_color'],c['after_color']) for c in v]
        elif k=='resized': out['resized']=[(r['color'],r['before_pixels'],r['after_pixels'],r['at_after']) for r in v]
    return out

def bars():
    out=[]
    for n in current_frame.segmentation['nodes']:
        if n['color'] in 'bR' and n['pixels']>=40:
            b=n['boundary']
            rs=[p[0] for p in b]; cs=[p[1] for p in b]
            out.append((n['color'], min(rs),min(cs),max(rs),max(cs), n['pixels']))
    return sorted(out, key=lambda x:(x[1],x[2]))

def simulate(bars, balls, gaps, nrows=64, cup_top=4, cup_bot=11, ceil_bot=3, cup_spans=None, maxiter=200):
    # bars: list of (r0, c0, c1) inclusive rows/cols
    # balls: list of (r0, c0) top-left of 4x4 ball, moving up
    # gaps: list of (c0,c1) gap columns; capture if ball cols within gap and reaches rows cup_bot-3..cup_bot
    captured = [0]*len(gaps)
    lost = 0
    pend = [(r,c) for (r,c) in balls]
    seen = set()
    it = 0
    while pend and it < maxiter:
        it += 1
        r,c = pend.pop()
        if (r,c) in seen: 
            continue
        seen.add((r,c))
        # move up step by step (4 rows) until collision
        cur = r
        result = None
        while True:
            nxt = cur - 4
            if nxt < 0:
                result = ('lost', cur, c); break
            # check bars overlapping rows nxt..nxt+3, cols c..c+3
            hitb = None
            for (br0, bc0, bc1) in bars:
                if br0 <= nxt+3 and br0+3 >= nxt and bc0 <= c+3 and bc1 >= c:
                    # bar occupies rows br0..br0+3; ball would overlap
                    if br0+3 < cur:  # bar above current ball top
                        hitb = (br0, bc0, bc1); break
            if hitb:
                br0, bc0, bc1 = hitb
                # ball stops with its top at br0+4
                stop = br0 + 4
                result = ('split', stop, c, bc0, bc1); break
            # check cups: ball rows overlap cup region
            if nxt+3 >= cup_top and nxt <= cup_bot:
                # ball in cup rows: check columns
                res = None
                for gi,(gc0,gc1) in enumerate(gaps):
                    if c >= gc0 and c+3 <= gc1:
                        res = ('cap', gi); break
                if res:
                    result = ('cap', res[1], nxt, c); break
                # blocked by cup solid or ceiling?
                blocked = False
                if cup_spans:
                    for (sc0,sc1) in cup_spans:
                        if not (c > sc1 or c+3 < sc0):
                            blocked = True
                if blocked or nxt+3 <= ceil_bot:
                    result = ('lost', nxt, c); break
                # check bars in cup rows
                hitb2 = None
                for (br0, bc0, bc1) in bars:
                    if br0 <= nxt+3 and br0+3 >= nxt and bc0 <= c+3 and bc1 >= c:
                        hitb2 = (br0,bc0,bc1); break
                if hitb2:
                    br0,bc0,bc1 = hitb2
                    result = ('split', br0+4, c, bc0, bc1); break
            cur = nxt
        if result is None: continue
        if result[0]=='lost': lost += 1
        elif result[0]=='cap': captured[result[1]] += 1
        else:
            _, sr, sc, bc0, bc1 = result
            for nc in (bc0-4, bc1+1):
                if nc < 0 or nc+3 > 63:
                    lost += 1
                else:
                    pend.append((sr, nc))
    return captured, lost

def positions(i):
    w = WID[i]
    return [(r,c) for r in range(12,53,4) for c in range(0,64-w+1,4)]

def to_bars(cfg): return [(cfg[i][0], cfg[i][1], cfg[i][1]+WID[i]-1) for i in range(4)]

def valid(cfg):
    bs = to_bars(cfg)
    for i in range(4):
        for j in range(i+1,4):
            r0,c0,c1 = bs[i]; R0,C0,C1 = bs[j]
            if not (r0+3 < R0 or R0+3 < r0) and not (c1 < C0 or C1 < c0):
                return False
    return True

def cost(cfg):
    t=0
    for i in range(4):
        r,c=cfg[i]
        if (r,c)!=INIT[i]: t += 1+(abs(r-INIT[i][0])+abs(c-INIT[i][1]))//4
    return t

def score(cfg):
    if not valid(cfg): return (-1,0)
    cap, lost = simulate(to_bars(cfg), BALLS3, GAPS3, cup_spans=SPANS3)
    return (sum(1 for x in cap if x>0), -cost(cfg))

def valid2(bs):
    for i in range(4):
        for j in range(i+1,4):
            r0,c0,c1=bs[i]; R0,C0,C1=bs[j]
            if not (r0+3<R0 or R0+3<r0) and not (c1<C0 or C1<c0): return False
    return True

def simulate2(bars, balls, gaps, cup_spans=None, maxiter=400):
    captured=[0]*len(gaps); lost=0; pend=list(balls); seen=set(); it=0
    def barhit(r_lo, r_hi, c0, c1, exclude=None):
        for k,(br0,bc0,bc1) in enumerate(bars):
            if k==exclude: continue
            if br0<=r_hi and br0+3>=r_lo and bc0<=c1 and bc1>=c0: return k
        return None
    while pend and it<maxiter:
        it+=1
        r,c = pend.pop()
        if (r,c) in seen: continue
        seen.add((r,c))
        cur=r
        while True:
            nxt=cur-4
            if nxt<0:
                lost+=1; break
            k = barhit(nxt, nxt+3, c, c+3)
            if k is not None:
                br0,bc0,bc1 = bars[k]
                if br0+3 >= cur:   # bar not above ball
                    lost+=1; break
                stop = br0+4
                blocked_any=False
                for nc in (bc0-4, bc1+1):
                    if nc<0 or nc+3>63:
                        lost+=1; continue
                    lo,hi = (nc, c+3) if nc<c else (c, nc+3)
                    if barhit(stop, stop+3, lo, hi, exclude=k) is not None:
                        lost+=1; continue
                    pend.append((stop,nc))
                break
            if nxt+3>=4 and nxt<=11:
                gi=None
                for i,(g0,g1) in enumerate(gaps):
                    if c>=g0 and c+3<=g1: gi=i;break
                if gi is not None:
                    captured[gi]+=1; break
                blocked = any(not(c> s1 or c+3< s0) for (s0,s1) in (cup_spans or []))
                if blocked or nxt+3<=3:
                    lost+=1; break
            cur=nxt
    return captured, lost

def tb(cfg): return [(cfg[i][0], cfg[i][1], cfg[i][1]+WID[i]-1) for i in range(4)]

def sc(cfg):
    bs=tb(cfg)
    if not valid2(bs): return (-1,0)
    cap,lost=simulate2(bs,BALLS3,GAPS3,SPANS3)
    return (sum(1 for x in cap if x>0), -cost(cfg))

def tb_(cfg): return [(cfg[i][0], cfg[i][1], cfg[i][1]+WID[i]-1) for i in range(4)]

def cost_(cfg):
    t=0
    for i in range(4):
        r,c=cfg[i]
        if (r,c)!=INIT[i]: t += 1+(abs(r-INIT[i][0])+abs(c-INIT[i][1]))//4
    return t

def sc_(cfg):
    bs=tb_(cfg)
    if not valid2(bs): return (-1,0)
    cap,lost=simulate2(bs,BALLS3,GAPS3,SPANS3)
    return (sum(1 for x in cap if x>0), -cost_(cfg))

def tb2(cfg): return [(cfg[i][0], cfg[i][1], cfg[i][1]+WID[i]-1) for i in range(4)]

def tb3(cfg): return [(cfg[i][0], cfg[i][1], cfg[i][1]+WID[i]-1) for i in range(4)]

def cost3(cfg):
    t=0
    for i in range(4):
        r,c=cfg[i]
        if (r,c)!=INIT[i]: t += 1+(abs(r-INIT[i][0])+abs(c-INIT[i][1]))//4
    return t

def sc3(cfg):
    bs=tb3(cfg)
    if not valid2(bs): return (-1,0,0)
    cap,lost=simulate2(bs,BALLS3,GAPS3,SPANS3)
    return (sum(1 for x in cap if x>0), -lost, -cost3(cfg))

def tb4(cfg): return [(cfg[i][0], cfg[i][1], cfg[i][1]+WID[i]-1) for i in range(4)]

def cost4(cfg):
    t=0
    for i in range(4):
        r,c=cfg[i]
        if (r,c)!=INIT[i]: t += 1+(abs(r-INIT[i][0])+abs(c-INIT[i][1]))//4
    return t

def sc4(cfg):
    bs=tb4(cfg)
    if not valid2(bs): return (-1,0,0)
    cap,lost=simulate2(bs,BALLS3,GAPS3,SPANS3)
    return (sum(1 for x in cap if x>0), -lost, -cost4(cfg))

def comps(ax, chars='Mr'):
    grid=[list(row) for row in ax]
    seen=[[False]*64 for _ in range(64)]
    out=[]
    for r in range(64):
        for c in range(64):
            if grid[r][c] in chars and not seen[r][c]:
                st=[(r,c)]; seen[r][c]=True; cells=[]
                while st:
                    y,x=st.pop(); cells.append((y,x))
                    for dy,dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        ny,nx=y+dy,x+dx
                        if 0<=ny<64 and 0<=nx<64 and not seen[ny][nx] and grid[ny][nx] in chars:
                            seen[ny][nx]=True; st.append((ny,nx))
                rs=[p[0] for p in cells]; cs=[p[1] for p in cells]
                out.append((min(rs),max(rs),min(cs),max(cs),len(cells),grid[min(rs)][min(cs)]))
    return out

def simulate3(bars, balls, gaps, cup_spans=None, maxiter=400):
    captured=[0]*len(gaps); stray=0; off=0; pend=list(balls); seen=set(); it=0
    def barhit(r_lo,r_hi,c0,c1,ex=None):
        for k,(b0,bc0,bc1) in enumerate(bars):
            if k==ex: continue
            if b0<=r_hi and b0+3>=r_lo and bc0<=c1 and bc1>=c0: return k
        return None
    while pend and it<maxiter:
        it+=1
        r,c=pend.pop()
        if (r,c) in seen: continue
        seen.add((r,c))
        cur=r
        while True:
            nxt=cur-4
            if nxt<0:
                stray+=1; break
            k=barhit(nxt,nxt+3,c,c+3)
            if k is not None:
                b0,bc0,bc1=bars[k]
                if b0+3>=cur:
                    stray+=1; break
                stop=b0+4
                for nc in (bc0-4, bc1+1):
                    if nc+3<0 or nc>63:
                        off+=1; continue
                    if nc<0 or nc+3>63:
                        stray+=1; continue
                    lo,hi=(nc,c+3) if nc<c else (c,nc+3)
                    if barhit(stop,stop+3,lo,hi,ex=k) is not None:
                        stray+=1; continue
                    pend.append((stop,nc))
                break
            if nxt+3>=4 and nxt<=11:
                gi=None
                for i,(g0,g1) in enumerate(gaps):
                    if c>=g0 and c+3<=g1: gi=i;break
                if gi is not None:
                    captured[gi]+=1; break
                blocked=any(not(c>s1 or c+3<s0) for (s0,s1) in (cup_spans or []))
                if blocked or nxt+3<=3:
                    stray+=1; break
            cur=nxt
    return captured, stray, off

def tb5(cfg): return [(cfg[i][0], cfg[i][1], cfg[i][1]+WID[i]-1) for i in range(4)]

def cost5(cfg):
    t=0
    for i in range(4):
        r,c=cfg[i]
        if (r,c)!=INIT[i]: t += 1+(abs(r-INIT[i][0])+abs(c-INIT[i][1]))//4
    return t

def sc5(cfg):
    bs=tb5(cfg)
    if not valid2(bs): return (-1,0,0)
    cap,stray,off=simulate3(bs,BALLS3,GAPS3,SPANS3)
    return (sum(1 for x in cap if x>0), -stray, -cost5(cfg))

def lvl3_search(iters=18000, seed=11, WID=(16,20,24,24), INIT=((20,8),(28,40),(32,8),(40,36)),
                BALLS=((56,4),(56,36),(56,56)), GAPS=((8,11),(28,31),(52,55)), SPANS=((4,15),(24,35),(48,59))):
    def tb(cfg): return [(cfg[i][0], cfg[i][1], cfg[i][1]+WID[i]-1) for i in range(4)]
    def cost(cfg):
        t=0
        for i in range(4):
            r,c=cfg[i]
            if (r,c)!=INIT[i]: t += 1+(abs(r-INIT[i][0])+abs(c-INIT[i][1]))//4
        return t
    def sc(cfg):
        bs=tb(cfg)
        if not valid2(bs): return (-1,0,0)
        cap,stray,off=simulate3(bs,BALLS,GAPS,SPANS)
        return (sum(1 for x in cap if x>0), -stray, -cost(cfg))
    random.seed(seed)
    best={}; n=0
    while n<iters:
        cfg=[(random.randrange(12,53,4), random.randrange(0,64-WID[i]+1,4)) for i in range(4)]
        s=sc(cfg); n+=1
        for _ in range(250):
            i=random.randrange(4); cand=list(cfg)
            if random.random()<0.4:
                cand[i]=(random.randrange(12,53,4), random.randrange(0,64-WID[i]+1,4))
            else:
                r,c=cand[i]
                nr=r+random.choice([-4,0,4]); nc=c+random.choice([-8,-4,0,4,8])
                if 12<=nr<=52 and 0<=nc<=64-WID[i]: cand[i]=(nr,nc)
            s2=sc(cand); n+=1
            if s2>=s: cfg,s=cand,s2
            if s[0]==3 and s[1]==0:
                best.setdefault(cost(cfg), tuple(cfg))
    out=[]
    for c in sorted(best)[:5]:
        cfg=best[c]
        out.append((c,cfg,simulate3(tb(cfg),BALLS,GAPS,SPANS)))
    return out

def sim4(bars, balls, gaps=(3,9,13,17), spans=((2,4),(8,10),(12,14),(16,18)), umax=19, vcup=17, vmax=19, maxiter=400):
    """Mirrored physics: balls descend. bars: list of (v,u0,u1). balls: list of (v,u)."""
    cap=[0]*len(gaps); stray=0; off=0
    pend=list(balls); seen=set(); it=0
    def barhit(v_lo,v_hi,u,ex=None):
        for k,(bv,b0,b1) in enumerate(bars):
            if k==ex: continue
            if v_lo<=bv<=v_hi and b0<=u<=b1: return k
        return None
    while pend and it<maxiter:
        it+=1
        v,u=pend.pop()
        if (v,u) in seen: continue
        seen.add((v,u))
        cur=v
        while True:
            nxt=cur+1
            if nxt>vmax:
                stray+=1; break
            k=barhit(nxt,nxt,u,u)
            if k is not None:
                bv,b0,b1=bars[k]
                if bv-3<=cur:
                    stray+=1; break
                stop=bv-1
                for nu in (b0-1, b1+1):
                    if nu<0 or nu>umax:
                        off+=1; continue
                    lo,hi=(nu,u) if nu<u else (u,nu)
                    if barhit(stop,stop,lo,hi,ex=k) is not None:
                        stray+=1; continue
                    pend.append((stop,nu))
                break
            if nxt==vcup:
                if u in gaps:
                    cap[gaps.index(u)]+=1; break
                if any(s0<=u<=s1 for (s0,s1) in spans):
                    stray+=1; break
            if nxt>vmax:
                stray+=1; break
            cur=nxt
    return cap,stray,off

def sim4t(bars, balls, gaps=(3,9,13,17), spans=((2,4),(8,10),(12,14),(16,18)), umax=19, vcup=17, vmax=19):
    cap=[0]*len(gaps); stray=0; off=0; log=[]
    pend=list(balls)
    while pend:
        v,u=pend.pop()
        cur=v
        while True:
            nxt=cur+1
            if nxt>vmax:
                stray+=1; log.append(f'({v},{u}) wall'); break
            k=None
            for j,(bv,b0,b1) in enumerate(bars):
                if bv==nxt and b0<=u<=b1: k=j;break
            if k is not None:
                bv,b0,b1=bars[k]
                stop=bv-1
                log.append(f'({v},{u}) hit bar{k} v{bv} u{b0}-{b1} -> exits {b0-1},{b1+1}')
                for nu in (b0-1,b1+1):
                    if nu<0 or nu>umax: off+=1; log.append('  off')
                    else: pend.append((stop,nu))
                break
            if nxt==vcup:
                if u in gaps:
                    cap[gaps.index(u)]+=1; log.append(f'({v},{u}) CAP cup{gaps.index(u)}'); break
                if any(s0<=u<=s1 for (s0,s1) in spans):
                    stray+=1; log.append(f'({v},{u}) stray on cup'); break
            cur=nxt
    return cap,stray,off,log

def sim5(bars, balls, gaps=(3,9,13,17), spans=((2,4),(8,10),(12,14),(16,18)), umax=19, vcup=17, vmax=19):
    cap=[0]*len(gaps); stray=0; off=0; pend=list(balls); seen=set(); it=0
    def hit(v,lo,hi,ex=None):
        for j,(bv,b0,b1) in enumerate(bars):
            if j==ex: continue
            if bv==v and not(b1<lo or b0>hi): return j
        return None
    while pend and it<500:
        it+=1
        v,u=pend.pop()
        if (v,u) in seen: continue
        seen.add((v,u))
        cur=v; steps=0
        while steps<40:
            steps+=1
            nxt=cur+1
            if nxt>vmax: stray+=1; break
            k=None
            for j,(bv,b0,b1) in enumerate(bars):
                if bv==nxt and b0<=u<=b1: k=j; break
            if k is not None:
                bv,b0,b1=bars[k]; stop=bv-1
                if stop<cur: stray+=1; break
                for nu in (b0-1,b1+1):
                    if nu<0 or nu>umax: off+=1; continue
                    lo,hi=(nu,u) if nu<u else (u,nu)
                    if hit(stop,lo,hi,ex=k) is not None: stray+=1; continue
                    pend.append((stop,nu))
                break
            if nxt==vcup:
                if u in gaps: cap[gaps.index(u)]+=1; break
                if any(s0<=u<=s1 for (s0,s1) in spans): stray+=1; break
            cur=nxt
    return cap,stray,off

def lvl4_search(iters=20000, seed=5, WID=(5,5,7,4,4), INIT=((5,5),(5,12),(9,2),(10,14),(13,12)), BALL=((1,7))):
    import random
    def tb(cfg): return [(cfg[i][0], cfg[i][1], cfg[i][1]+WID[i]-1) for i in range(5)]
    def cost(cfg):
        t=0
        for i in range(5):
            v,u=cfg[i]
            if (v,u)!=INIT[i]: t+=1+abs(v-INIT[i][0])+abs(u-INIT[i][1])
        return t
    def valid(cfg):
        bs=tb(cfg)
        for i in range(5):
            for j in range(i+1,5):
                if bs[i][0]==bs[j][0] and not(bs[i][2]<bs[j][1] or bs[j][2]<bs[i][1]): return False
        return True
    def sc(cfg):
        if not valid(cfg): return (-1,0,0)
        cap,stray,off=sim5(tb(cfg),[BALL])
        return (sum(1 for x in cap if x>0), -stray, -cost(cfg))
    random.seed(seed); best={}
    n=0
    while n<iters:
        cfg=[(random.randrange(1,17), random.randrange(0,20-WID[i])) for i in range(5)]
        s=sc(cfg); n+=1
        for _ in range(250):
            i=random.randrange(5); cand=list(cfg)
            if random.random()<0.4:
                cand[i]=(random.randrange(1,17), random.randrange(0,20-WID[i]))
            else:
                v,u=cand[i]
                nv=v+random.choice([-1,0,1]); nu=u+random.choice([-2,-1,0,1,2])
                if 1<=nv<=16 and 0<=nu<=19-WID[i]: cand[i]=(nv,nu)
            s2=sc(cand); n+=1
            if s2>=s: cfg,s=cand,s2
            if s[0]==4 and s[1]==0: best.setdefault(cost(cfg), tuple(cfg))
    out=[]
    for c in sorted(best)[:5]:
        cfg=best[c]; out.append((c,cfg,sim5(tb(cfg),[BALL])))
    return out

def lvl4_frontier(iters=20000, seed=9, WID=(5,5,7,4,4), INIT=((5,5),(5,12),(9,2),(10,14),(13,12)), BALL=(1,7)):
    import random
    def tb(cfg): return [(cfg[i][0], cfg[i][1], cfg[i][1]+WID[i]-1) for i in range(5)]
    def cost(cfg):
        t=0
        for i in range(5):
            v,u=cfg[i]
            if (v,u)!=INIT[i]: t+=1+abs(v-INIT[i][0])+abs(u-INIT[i][1])
        return t
    def valid(cfg):
        bs=tb(cfg)
        for i in range(5):
            for j in range(i+1,5):
                if bs[i][0]==bs[j][0] and not(bs[i][2]<bs[j][1] or bs[j][2]<bs[i][1]): return False
        return True
    def sc(cfg):
        if not valid(cfg): return (-1,0,0,0)
        cap,stray,off=sim5(tb(cfg),[BALL])
        return (sum(1 for x in cap if x>0), -stray, -off, -cost(cfg))
    random.seed(seed); rec={}
    n=0
    while n<iters:
        cfg=[(random.randrange(1,17), random.randrange(0,20-WID[i])) for i in range(5)]
        s=sc(cfg); n+=1
        for _ in range(250):
            i=random.randrange(5); cand=list(cfg)
            if random.random()<0.4: cand[i]=(random.randrange(1,17), random.randrange(0,20-WID[i]))
            else:
                v,u=cand[i]; nv=v+random.choice([-1,0,1]); nu=u+random.choice([-2,-1,0,1,2])
                if 1<=nv<=16 and 0<=nu<=19-WID[i]: cand[i]=(nv,nu)
            s2=sc(cand); n+=1
            if s2>=s: cfg,s=cand,s2
            if s[0]>=3: rec.setdefault((s[0],-s[1],-s[2]), (cost(cfg), tuple(cfg)))
    return sorted(rec.items())[:14]

def lvl4_opt(iters=25000, seed=17, WID=(5,5,7,4,4), INIT=((5,5),(5,12),(9,2),(10,14),(13,12)), BALL=(1,7), SEEDCFG=((14,4),(12,8),(4,4),(10,13),(6,10))):
    import random
    def tb(cfg): return [(cfg[i][0], cfg[i][1], cfg[i][1]+WID[i]-1) for i in range(5)]
    def cost(cfg):
        t=0
        for i in range(5):
            v,u=cfg[i]
            if (v,u)!=INIT[i]: t+=1+abs(v-INIT[i][0])+abs(u-INIT[i][1])
        return t
    def feas(cfg):
        bs=tb(cfg)
        for i in range(5):
            for j in range(i+1,5):
                if bs[i][0]==bs[j][0] and not(bs[i][2]<bs[j][1] or bs[j][2]<bs[i][1]): return False
        cap,stray,off=sim5(tb(cfg),[BALL])
        return all(x>0 for x in cap) and stray==0
    random.seed(seed)
    pool=[SEEDCFG] if feas(SEEDCFG) else []
    # random restarts to collect feasible configs
    n=0
    while n<iters:
        cfg=[(random.randrange(1,17), random.randrange(0,20-WID[i])) for i in range(5)]
        ok=feas(cfg); n+=1
        if ok: pool.append(tuple(cfg)); continue
        s=(sum(1 for x in sim5(tb(cfg),[BALL])[0] if x>0))
        for _ in range(120):
            i=random.randrange(5); cand=list(cfg)
            if random.random()<0.4: cand[i]=(random.randrange(1,17), random.randrange(0,20-WID[i]))
            else:
                v,u=cand[i]; nv=v+random.choice([-1,0,1]); nu=u+random.choice([-2,-1,0,1,2])
                if 1<=nv<=16 and 0<=nu<=19-WID[i]: cand[i]=(nv,nu)
            n+=1
            if feas(cand):
                pool.append(tuple(cand)); cfg=cand; break
    best=sorted(pool, key=cost)
    # local improvement on best
    improved=True
    cur=best[0]; bc=cost(cur)
    while improved:
        improved=False
        for i in range(5):
            for dv in range(-6,7):
                for du in range(-8,9):
                    v,u=cur[i]; nv,nu=v+dv,u+du
                    if not(1<=nv<=16) or not(0<=nu<=19-WID[i]): continue
                    cand=list(cur); cand[i]=(nv,nu)
                    nc=cost(cand)
                    if nc<bc and feas(cand):
                        cur=tuple(cand); bc=nc; improved=True
    return bc, cur, sim5(tb(cur),[BALL]), len(pool)

def tbk(cfg,WID=WID): return [(cfg[i][0], cfg[i][1], cfg[i][1]+WID[i]-1) for i in range(5)]

def barso(f=current_frame):
    out=[]
    for n in f.segmentation['nodes']:
        b=n['boundary']; r0=min(p[0] for p in b); r1=max(p[0] for p in b); c0=min(p[1] for p in b); c1=max(p[1] for p in b)
        if n['color'] in 'Rb' and r1-r0<=3:
            out.append((n['color'],(r0-2)//3,(c0-2)//3,(r1-2)//3,(c1-2)//3))
    return out

def barso2():
    out=[]
    for n in current_frame.segmentation['nodes']:
        b=n['boundary']; r0=min(p[0] for p in b); r1=max(p[0] for p in b); c0=min(p[1] for p in b); c1=max(p[1] for p in b)
        out.append((n['color'],(r0-2)//3,(c0-2)//3,(r1-2)//3,(c1-2)//3,n['pixels']))
    return out

def tbk2(cfg,WID=(5,5,7,4,4)): return [(cfg[i][0], cfg[i][1], cfg[i][1]+WID[i]-1) for i in range(5)]

def trace(bars, start=(1,7)):
    out=[]; pend=[start]
    while pend:
        v,u=pend.pop(); cur=v
        while True:
            nxt=cur+1
            if nxt>19: out.append(f'({v},{u})->wall'); break
            k=None
            for j,(bv,b0,b1) in enumerate(bars):
                if bv==nxt and b0<=u<=b1: k=j;break
            if k is not None:
                bv,b0,b1=bars[k]; out.append(f'({v},{u}) hits v{bv} u{b0}-{b1} -> {b0-1},{b1+1}')
                for nu in (b0-1,b1+1):
                    if 0<=nu<=19: pend.append((bv-1,nu))
                break
            if nxt==17:
                out.append(f'({v},{u}) -> cup@u{u}' if u in (3,9,13,17) else f'({v},{u}) -> stop u{u}'); break
            cur=nxt
    return out

def lvl4_robust(evals=6000, seed=3, WID=(5,5,7,4,4), INIT=((5,5),(5,12),(9,2),(10,14),(13,12))):
    import random
    def feas_both(cfg):
        bs=tbk2(cfg)
        # no overlaps
        for i in range(5):
            for j in range(i+1,5):
                if bs[i][0]==bs[j][0] and not(bs[i][2]<bs[j][1] or bs[j][2]<bs[i][1]): return False
        c1,s1,o1=sim5(bs,[(1,7)])
        bs2=[b for b in bs if b!=(bs[2])]
        v,u0,u1=bs[2]
        if u1-u0==6:
            bs2=bs2+[(v,u0,u0+2),(v,u0+4,u1)]
        c2,s2,o2=sim5(bs2,[(1,7)])
        return all(x>0 for x in c1) and s1==0 and all(x>0 for x in c2) and s2==0
    def cost(cfg):
        t=0
        for i in range(5):
            v,u=cfg[i]
            if (v,u)!=INIT[i]: t+=1+abs(v-INIT[i][0])+abs(u-INIT[i][1])
        return t
    random.seed(seed)
    found=[]
    n=0
    while n<evals:
        cfg=[(random.randrange(1,17), random.randrange(0,20-WID[i])) for i in range(5)]
        n+=1
        if feas_both(cfg): found.append((cost(cfg),tuple(cfg)))
    found.sort()
    return found[:6], n

def lvl4_robust2(evals=7000, seed=8, WID=(5,5,7,4,4), INIT=((5,5),(5,12),(9,2),(10,14),(13,12))):
    import random
    def both(cfg):
        bs=tbk2(cfg)
        for i in range(5):
            for j in range(i+1,5):
                if bs[i][0]==bs[j][0] and not(bs[i][2]<bs[j][1] or bs[j][2]<bs[i][1]): return None
        c1,s1,o1=sim5(bs,[(1,7)])
        v,u0,u1=bs[2]
        bs2=[b for k,b in enumerate(bs) if k!=2]
        if u1-u0==6: bs2=bs2+[(v,u0,u0+2),(v,u0+4,u1)]
        c2,s2,o2=sim5(bs2,[(1,7)])
        return (sum(1 for x in c1 if x>0), s1, sum(1 for x in c2 if x>0), s2)
    def cost(cfg):
        t=0
        for i in range(5):
            v,u=cfg[i]
            if (v,u)!=INIT[i]: t+=1+abs(v-INIT[i][0])+abs(u-INIT[i][1])
        return t
    def sc(cfg):
        b=both(cfg)
        if b is None: return (-99,0,0)
        f1,s1,f2,s2=b
        return (min(f1,f2)*10 - max(s1,s2)*3, -min(s1+s2,9), -cost(cfg))
    random.seed(seed)
    found=[]; n=0; best=None
    while n<evals:
        cfg=[(random.randrange(1,17), random.randrange(0,20-WID[i])) for i in range(5)]
        s=sc(cfg); n+=1
        for _ in range(200):
            i=random.randrange(5); cand=list(cfg)
            if random.random()<0.4: cand[i]=(random.randrange(1,17), random.randrange(0,20-WID[i]))
            else:
                v,u=cand[i]; nv=v+random.choice([-1,0,1]); nu=u+random.choice([-2,-1,0,1,2])
                if 1<=nv<=16 and 0<=nu<=19-WID[i]: cand[i]=(nv,nu)
            s2=sc(cand); n+=1
            if s2>=s: cfg,s=cand,s2
            if s[0]==40 and s[1]==0:
                found.append((cost(cfg),tuple(cfg)))
                if best is None or cost(cfg)<best[0]: best=(cost(cfg),tuple(cfg))
    found.sort()
    return found[:5], n

def lvl4_cheapest(evals=8000, seed=21, WID=(5,5,7,4,4), INIT=((5,5),(5,12),(9,2),(10,14),(13,12))):
    import random
    def ok(cfg):
        bs=tbk2(cfg)
        for i in range(5):
            for j in range(i+1,5):
                if bs[i][0]==bs[j][0] and not(bs[i][2]<bs[j][1] or bs[j][2]<bs[i][1]): return False
        c,s,o=sim5(bs,[(1,7)])
        return all(x>0 for x in c) and s==0
    def cost(cfg):
        t=0
        for i in range(5):
            v,u=cfg[i]
            if (v,u)!=INIT[i]: t+=1+abs(v-INIT[i][0])+abs(u-INIT[i][1])
        return t
    def sc(cfg):
        bs=tbk2(cfg)
        for i in range(5):
            for j in range(i+1,5):
                if bs[i][0]==bs[j][0] and not(bs[i][2]<bs[j][1] or bs[j][2]<bs[i][1]): return (-99,0,0)
        c,s,o=sim5(bs,[(1,7)])
        return (sum(1 for x in c if x>0)*10 - s*3, -s, -cost(cfg))
    random.seed(seed)
    pool=[]; n=0
    while n<evals and len(pool)<12:
        cfg=[(random.randrange(1,17), random.randrange(0,20-WID[i])) for i in range(5)]
        s=sc(cfg); n+=1
        for _ in range(200):
            i=random.randrange(5); cand=list(cfg)
            if random.random()<0.4: cand[i]=(random.randrange(1,17), random.randrange(0,20-WID[i]))
            else:
                v,u=cand[i]; nv=v+random.choice([-1,0,1]); nu=u+random.choice([-2,-1,0,1,2])
                if 1<=nv<=16 and 0<=nu<=19-WID[i]: cand[i]=(nv,nu)
            s2=sc(cand); n+=1
            if s2>=s:
                cfg,s=cand,s2
                if s[0]==40 and s[1]==0:
                    pool.append(tuple(cfg)); break
    best=None
    for cfg in pool:
        cur=cfg; bc=cost(cur)
        improved=True
        while improved:
            improved=False
            for i in range(5):
                for dv in range(-4,5):
                    for du in range(-6,7):
                        v,u=cur[i]; nv,nu=v+dv,u+du
                        if not(1<=nv<=16) or not(0<=nu<=19-WID[i]): continue
                        cand=list(cur); cand[i]=(nv,nu)
                        nc=cost(cand)
                        if nc<bc and ok(cand):
                            cur=tuple(cand); bc=nc; improved=True
        if best is None or bc<best[0]: best=(bc,cur)
    return len(pool), best

def lvl4_local(start, iters=4000, seed=1, WID=(5,5,7,4,4), INIT=((5,5),(5,12),(9,2),(10,14),(13,12))):
    import random
    def ok(cfg):
        bs=tbk2(cfg)
        for i in range(5):
            for j in range(i+1,5):
                if bs[i][0]==bs[j][0] and not(bs[i][2]<bs[j][1] or bs[j][2]<bs[i][1]): return False
        c,s,o=sim5(bs,[(1,7)])
        return all(x>0 for x in c) and s==0
    def cost(cfg):
        t=0
        for i in range(5):
            v,u=cfg[i]
            if (v,u)!=INIT[i]: t+=1+abs(v-INIT[i][0])+abs(u-INIT[i][1])
        return t
    random.seed(seed)
    cur=tuple(start); bc=cost(cur)
    assert ok(cur), 'start not feasible'
    n=0
    for _ in range(iters):
        i=random.randrange(5)
        dv=random.randint(-4,4); du=random.randint(-6,6)
        v,u=cur[i]; nv,nu=v+dv,u+du
        if not(1<=nv<=16) or not(0<=nu<=19-WID[i]): continue
        cand=list(cur); cand[i]=(nv,nu); n+=1
        nc=cost(cand)
        if nc<=bc and ok(cand):
            cur=tuple(cand); bc=nc
    return bc, cur, n

def barso3():
    out=[]
    for n in current_frame.segmentation['nodes']:
        b=n['boundary']; r0=min(p[0] for p in b); r1=max(p[0] for p in b); c0=min(p[1] for p in b); c1=max(p[1] for p in b)
        if n['color'] in 'Rbc' and (r1-r0)<=3:
            out.append((n['color'],(r0-2)//3,(c0-2)//3,(r1-2)//3,(c1-2)//3))
    return sorted(out,key=lambda x:(x[1],x[2]))

def objs5():
    out=[]
    for n in current_frame.segmentation['nodes']:
        b=n['boundary']; r0=min(p[0] for p in b); r1=max(p[0] for p in b); c0=min(p[1] for p in b); c1=max(p[1] for p in b)
        out.append((n['color'],(r0-5)//3,(c0-5)//3,(r1-5)//3,(c1-5)//3,n['pixels']))
    return sorted(out,key=lambda x:(x[1],x[2]))

def sim6(bars, blockers=(), balls=((17,5),(17,13)), gaps=(5,11,15), spans=((4,6),(10,12),(14,16)),
         mouth=(11,1), umax=18, vmax=18):
    cap=[0]*len(gaps); stray=0; off=0; side=0
    pend=[(v,u,'up') for (v,u) in balls]; seen=set(); it=0
    def barr(v,lo,hi,ex=None):
        for j,(bv,b0,b1) in enumerate(bars):
            if j==ex: continue
            if bv==v and not(b1<lo or b0>hi): return j
        return None
    def blk(v,lo,hi):
        for (bv,b0,b1) in blockers:
            if bv==v and not(b1<lo or b0>hi): return True
        return False
    while pend and it<400:
        it+=1
        v,u,mode=pend.pop()
        if (v,u,mode) in seen: continue
        seen.add((v,u,mode))
        if mode=='h':
            if (v,u)==mouth: side+=1; continue
            if blk(v,u,u) or any(b0<=u<=b1 for (bv,b0,b1) in bars if bv==v): stray+=1; continue
            pend.append((v,u,'up')); continue
        cur=v; steps=0
        while steps<40:
            steps+=1
            nxt=cur-1
            if nxt<0: stray+=1; break
            k=barr(nxt,u,u)
            if k is not None:
                bv,b0,b1=bars[k]; band=cur
                for nu in (b0-1,b1+1):
                    if nu<0 or nu>umax: off+=1; continue
                    lo,hi=(nu,u) if nu<u else (u,nu)
                    if barr(band,lo,hi,ex=k) is not None or blk(band,lo,hi): stray+=1; continue
                    pend.append((band,nu,'h'))
                break
            if blk(nxt,u,u): stray+=1; break
            if nxt==1:
                if u in gaps: cap[gaps.index(u)]+=1; break
                if any(s0<=u<=s1 for (s0,s1) in spans): stray+=1; break
            cur=nxt
    return cap,stray,off,side

def mpos(ax):
    g=[list(x) for x in ax]; out=[]
    for r_ in range(64):
        for c_ in range(64):
            if g[r_][c_]=='M': out.append(((r_-5)//3,(c_-5)//3))
    # cluster
    return out

def objs6():
    out=[]
    for n in current_frame.segmentation['nodes']:
        b=n['boundary']; r0=min(p[0] for p in b); r1=max(p[0] for p in b); c0=min(p[1] for p in b); c1=max(p[1] for p in b)
        if r1-r0<2 and c1-c0<2 and n['pixels']<=1: continue
        out.append((n['color'],(r0-2)//3,(c0-5)//3,(r1-2)//3,(c1-5)//3,n['pixels']))
    return out
