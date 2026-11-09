try:
    import json
except Exception:
    pass
try:
    import copy
except Exception:
    pass
try:
    import re
except Exception:
    pass
try:
    from collections import Counter
except Exception:
    pass
try:
    from collections import defaultdict
except Exception:
    pass
try:
    import time
except Exception:
    pass
try:
    import random
except Exception:
    pass
try:
    import pickle
except Exception:
    pass

def mask(n):
    b = n['boundary']
    r0 = min(p[0] for p in b); r1 = max(p[0] for p in b)
    c0 = min(p[1] for p in b); c1 = max(p[1] for p in b)
    return r0,c0,r1,c1

def crop(r0,c0,r1,c1,label=''):
    print(label)
    print('    ' + ''.join(str(c%10) for c in range(c0,c1+1)))
    for r in range(r0,r1+1):
        print(r, a[r][c0:c1+1])

def compact(d):
    out = {}
    for k,v in d.items():
        if k=='changed_cell_count': out[k]=v
        else:
            out[k]=[{kk:vv for kk,vv in e.items() if kk in ('color','pixels','at','at_before','at_after','before_color','after_color','bbox','rotated_by')} for e in v]
    return out

def summary(fr):
    out=[]
    for n in fr.segmentation['nodes']:
        b=n['boundary']
        r0=min(p[0] for p in b); r1=max(p[0] for p in b)
        c0=min(p[1] for p in b); c1=max(p[1] for p in b)
        out.append((n['id'],n['color'],n['pixels'],(r0,c0,r1,c1)))
    return out

def st(r):
    res = r[0] if isinstance(r, list) else r
    return {k: res.get(k) for k in ('no_op','gameplay_changed','level_completed','game_over','done')}

def gray_bb(fr=None):
    fr = fr or current_frame
    for x in summary(fr):
        if x[1]=='c': return x[3]

def probe(a):
    try:
        action([a])
    except Exception as e:
        return (a, 'NOOP/stale', gray_bb())
    res = last_action_call_result
    return (a, 'moved' if not res.get('no_op') else 'noop', gray_bb(), res.get('level_completed'))

def state(fr=None):
    fr = fr or current_frame
    seg = fr.segmentation['nodes']
    S = [n for n in seg if n['color']=='S' and n['pixels']>100]
    R = [n for n in seg if n['color']=='B' and n['pixels']>20 and n['boundary'][0][0]>10]
    C = [n for n in seg if n['color']=='c']
    out = {}
    if S:
        b=S[0]['boundary']; out['band']=(min(p[1] for p in b), max(p[1] for p in b))
    if R:
        b=R[0]['boundary']; out['refl']=((min(p[0] for p in b),min(p[1] for p in b),max(p[0] for p in b),max(p[1] for p in b)), R[0]['pixels'])
    cs=[]
    for n in C:
        for p in n['boundary']: pass
        b=n['boundary']; cs.append(((min(p[0] for p in b),min(p[1] for p in b),max(p[0] for p in b),max(p[1] for p in b)), n['pixels']))
    out['c']=cs
    # dots location
    Wd=[(n['boundary'][0][0],n['boundary'][0][1]) for n in seg if n['color']=='W' and n['pixels']==1]
    out['Wdots']= 'band' if (S and Wd and abs(Wd[0][1]-(sum(out['band'])/2))<=1) else ('refl/other' if Wd else 'none')
    out['Wcols']= sorted(set(c for r,c in Wd))
    if S and R:
        m = sum(out['band'])/2
        rb = R[0]['boundary']
        out['piece_cols'] = (2*m-max(p[1] for p in rb), 2*m-min(p[1] for p in rb))
    return out

def show(r0,c0,r1,c1,label):
    print(label)
    for r in range(r0,r1+1):
        print('  ', a[r][c0:c1+1])

def objs(fr=None):
    fr = fr or current_frame
    out=[]
    for n in fr.segmentation['nodes']:
        if n['pixels']<20 or n['pixels']>90: continue
        b=n['boundary']
        out.append((n['color'],n['pixels'],(min(p[0] for p in b),min(p[1] for p in b),max(p[0] for p in b),max(p[1] for p in b))))
    return sorted(out)

def pix(fr, color, r0,c0,r1,c1):
    a = fr.ascii.split('\n')
    return [a[r][c0:c1+1] for r in range(r0,r1+1)]

def st3(fr=None):
    fr = fr or current_frame
    seg = fr.segmentation['nodes']
    Cc=[n for n in seg if n['color']=='c' and n['pixels']>5]
    Bb=[n for n in seg if n['color']=='B' and 30<n['pixels']<62]
    def bb(n):
        p=n['boundary']; return (min(x[0] for x in p),min(x[1] for x in p),max(x[0] for x in p),max(x[1] for x in p))
    A=B=None; ra=rb=None
    for n in Cc:
        b=bb(n)
        if b[2]-b[0]>=9: A=b
        else: B=b
    for n in Bb:
        b=bb(n)
        if b[2]-b[0]>=9: ra=b
        else: rb=b
    S=[n for n in seg if n['color']=='S' and n['pixels']>3]
    bc=None
    if S:
        rows=[bb(n)[0] for n in S]
        bc=sorted(rows)[len(rows)//2]+1
    if bc is None and A and ra: bc=(ra[2]+A[0])//2
    Wd=[(n['boundary'][0][0],n['boundary'][0][1]) for n in seg if n['color']=='W' and n['pixels']==1]
    act='?'
    if Wd:
        rr=max(set(x[0] for x in Wd), key=lambda v: sum(1 for x in Wd if x[0]==v))
        act='band' if (S and any(bb(n)[0]<=rr<=bb(n)[2] for n in S)) else ('reflA' if ra and ra[0]<=rr<=ra[2] else ('reflB' if rb and rb[0]<=rr<=rb[2] else 'other@%d'%rr))
    return {'bc':bc,'A':A,'B':B,'rA':ra,'rB':rb,'act':act}

def pos(fr=None):
    fr = fr or current_frame
    seg=fr.segmentation['nodes']
    def bb(n):
        p=n['boundary']; return (min(x[0] for x in p),min(x[1] for x in p),max(x[0] for x in p),max(x[1] for x in p))
    S=[n for n in seg if n['color']=='S' and n['pixels']>3]
    bc=None
    if S:
        rows=sorted(set(bb(n)[0] for n in S))
        bc=rows[len(rows)//2]+1
    Bb=[bb(n) for n in seg if n['color']=='B' and 30<n['pixels']<62]
    rA=rB=None
    for b in Bb:
        if b[2]-b[0]>=9: rA=b
        else: rB=b
    A=B=None
    if bc is not None:
        if rA: A=(2*bc-rA[2], rA[1])
        if rB: B=(2*bc-rB[2], rB[1])
    Wd=[(n['boundary'][0][0],n['boundary'][0][1]) for n in seg if n['color']=='W' and n['pixels']==1]
    act='?'
    if Wd:
        from collections import Counter
        cnt=Counter(r for r,c in Wd)
        rr,mx=cnt.most_common(1)[0]
        if mx>=8: act='band'
        elif rA and rA[0]<=rr<=rA[2]: act='reflA'
        elif rB and rB[0]<=rr<=rB[2]: act='reflB'
        else: act='other%d'%rr
    return {'bc':bc,'A':A,'B':B,'rA':rA,'rB':rB,'act':act}

def step_model(s,a):
    t=dict(bc=s['bc'],A=s['A'],B=s['B'],act=s['act'])
    if a=='SPACE':
        t['act']={'band':'reflA','reflA':'reflB','reflB':'band'}.get(s['act'],'?')
        return t,False
    if s['act']=='band':
        if a=='UP': t['bc']-=3; t['A']=(s['A'][0]-6,s['A'][1]); t['B']=(s['B'][0]-6,s['B'][1])
        elif a=='DOWN': t['bc']+=3; t['A']=(s['A'][0]+6,s['A'][1]); t['B']=(s['B'][0]+6,s['B'][1])
        else: return t,True
    elif s['act']=='reflA':
        d={'UP':(3,0),'DOWN':(-3,0),'LEFT':(0,-3),'RIGHT':(0,3)}.get(a)
        if d: t['A']=(s['A'][0]+d[0], s['A'][1]+d[1])
        else: return t,True
    elif s['act']=='reflB':
        d={'UP':(3,0),'DOWN':(-3,0),'LEFT':(0,-3),'RIGHT':(0,3)}.get(a)
        if d: t['B']=(s['B'][0]+d[0], s['B'][1]+d[1])
        else: return t,True
    return t,False

def run(KR,KC):
    def orb(S):
        o=set(S); o|={(2*KR-r,c) for r,c in S}; o|={(r,2*KC-c) for r,c in S}; o|={(2*KR-r,2*KC-c) for r,c in S}
        return {p for p in o if -1<=p[0]<=19 and -7<=p[1]<=13}
    def allorbits(B):
        d={}
        for dr in range(-24,25):
            for dc in range(-24,25):
                O=frozenset(orb({(r+dr,c+dc) for r,c in B}))
                d.setdefault(O,[]).append((dr,dc))
        return d
    d1=allorbits(B1); d2=allorbits(B2)
    # best single coverages
    m1=max(d1,key=lambda s:len(s&YF)); m2=max(d2,key=lambda s:len(s&YF))
    best=None
    for U1 in d1:
        c1=len(U1&YF)
        if c1+len(YF)<42: pass
        for U2 in d2:
            U=U1|U2
            miss=42-len(U&YF)
            if miss==0:
                over=len(U-YF)
                if best is None or over<best[0]: best=(over,d1[U1][0],d2[U2][0])
    return (len(d1),len(d2),max(len(s&YF) for s in d1),max(len(s&YF) for s in d2),best)

def big(fr=None):
    fr=fr or current_frame
    out=[]
    for n in fr.segmentation['nodes']:
        if n['pixels']>=8:
            b=n['boundary']; out.append((n['color'],n['pixels'],(min(p[0] for p in b),min(p[1] for p in b),max(p[0] for p in b),max(p[1] for p in b))))
    return out

def cbbox(fr=None):
    fr=fr or current_frame
    a=fr.ascii.split('\n')
    pts=[(r,c) for r,row in enumerate(a) for c,ch in enumerate(row) if ch=='c']
    if not pts: return None
    return (min(p[0] for p in pts),min(p[1] for p in pts),max(p[0] for p in pts),max(p[1] for p in pts))

def wact(fr=None):
    fr=fr or current_frame
    from collections import Counter
    Wd=[(n['boundary'][0][0],n['boundary'][0][1]) for n in fr.segmentation['nodes'] if n['color']=='W' and n['pixels']==1]
    if not Wd: return '?'
    cnt=Counter(r for r,c in Wd); rr,mx=cnt.most_common(1)[0]
    return ('band' if mx>=8 else 'dots@row%d'%rr)

def dots(fr=None):
    fr=fr or current_frame
    Wd=[(n['boundary'][0][0],n['boundary'][0][1]) for n in fr.segmentation['nodes'] if n['color']=='W' and n['pixels']==1]
    from collections import Counter
    return Counter(r for r,c in Wd).most_common(6), Counter(c for r,c in Wd).most_common(6), len(Wd)

def bbcol(ch,fr=None):
    fr=fr or current_frame
    a=fr.ascii.split('\n')
    pts=[(r,c) for r,row in enumerate(a) for c,x in enumerate(row) if x==ch]
    if not pts: return None
    return (min(p[0] for p in pts),min(p[1] for p in pts),max(p[0] for p in pts),max(p[1] for p in pts),len(pts))

def st5(fr=None):
    fr=fr or current_frame
    a=fr.ascii.split('\n')
    def bb(ch):
        pts=[(r,c) for r,row in enumerate(a) for c,x in enumerate(row) if x==ch and not (r==63 or c==63)]
        if not pts: return None
        return (min(p[0] for p in pts),min(p[1] for p in pts),max(p[0] for p in pts),max(p[1] for p in pts))
    S=[(r,c) for r,row in enumerate(a) for c,x in enumerate(row) if x=='S' and not (r==63 or c==63)]
    hrows=sorted(set(r for r,c in S if sum(1 for rr,cc in S if rr==r)>30))
    vcols=sorted(set(c for r,c in S if sum(1 for rr,cc in S if cc==c)>30))
    d,_,n=dots(fr)
    return {'c':bb('c'),'B':bb('B'),'hband':(hrows[0],hrows[-1]) if hrows else None,'vband':(vcols[0],vcols[-1]) if vcols else None,'ndots':n}

def run5(plan):
    for i,a in enumerate(plan):
        try: action([a])
        except Exception as e: print('ERR',i,a,type(e).__name__); return i
        r=last_action_call_result
        if r.get('no_op'): print('NOOP at',i,a, st5()); return i
        if r.get('level_completed') or r.get('game_over'): print('FINISH',i,a,'lvl' if r.get('level_completed') else 'over'); return i
    return len(plan)

def score(p):
    rh,cv,r1,c1,r2,c2=p
    U=orbit(V1,r1,c1,rh,cv)|orbit(V2,r2,c2,rh,cv)
    return (U&YB).bit_count()-(U&~YB).bit_count()

def add(rr0,rr1,cc0,cc1):
    for r in range(rr0,rr1+1):
        for c in range(cc0,cc1+1): P.add((r,c))

def norm(S):
    r0=min(r for r,c in S); c0=min(c for r,c in S)
    return frozenset((r-r0,c-c0) for r,c in S)

def addrect(S,rr0,rr1,cc0,cc1):
    for r in range(rr0,rr1+1):
        for c in range(cc0,cc1+1): S.add((r,c))

def mh(S): return {(r,-c) for r,c in S}

def mv(S): return {(-r,c) for r,c in S}

def orbit(V,dr,dc,rh,cv):
    r1=2*rh-dr-V['h']+1; c1=2*cv-dc-V['w']+1
    return (sh(V['F'],dr,dc)|sh(V['fH'],r1,dc)|sh(V['fV'],dr,c1)|sh(V['fHV'],r1,c1))&((1<<(63*64))-1)

def st6(fr=None):
    fr=fr or current_frame
    a=fr.ascii.split('\n')
    def pts(ch): return [(r,c) for r,row in enumerate(a) for c,x in enumerate(row) if x==ch and c<63 and r<63]
    def bb(S):
        if not S: return None
        return (min(r for r,c in S),min(c for r,c in S),max(r for r,c in S),max(c for r,c in S))
    C=pts('c'); Bl=pts('B')
    S=pts('S')
    hrows=[r for r in range(63) if sum(1 for c in range(63) if a[r][c]=='S')>30]
    vcols=[c for c in range(63) if sum(1 for r in range(63) if a[r][c]=='S')>30]
    d,cc,n=dots(fr)
    return {'c':bb(C),'nc':len(C),'B':bb(Bl),'nB':len(Bl),'hband':(hrows[0],hrows[-1]) if hrows else None,'vband':(vcols[0],vcols[-1]) if vcols else None,'ndots':n,'dot0':(d[:2],cc[:2])}

def shifts(S,T):
    """return list of (dr,dc) with S+(dr,dc) subset of T"""
    out=[]
    tr0=min(r for r,c in T); tc0=min(c for r,c in T)
    sr0=min(r for r,c in S); sc0=min(c for r,c in S)
    for dr in range(-40,41):
        for dc in range(-40,41):
            Sp={(r+dr,c+dc) for r,c in S}
            if Sp<=T: out.append((dr,dc))
    return out

def dotcells(fr=None):
    fr=fr or current_frame
    a=fr.ascii.split('\n')
    # dots are white 'W' cells
    return sorted((r,c) for r,row in enumerate(a) for c,x in enumerate(row) if x=='W' and c<63 and r<63)

def fits(V):
    out=[]
    h=max(r for r,c in V)+1; w=max(c for r,c in V)+1
    for dr in range(0,63-h+1):
        for dc in range(0,63-w+1):
            S={(r+dr,c+dc) for r,c in V}
            if S<=Y: out.append((dr,dc,frozenset(S)))
    return out

def snap():
    a=current_frame.ascii.split('\n')
    def bb(ch):
        S=[(r,c) for r,row in enumerate(a) for c,x in enumerate(row) if x==ch and c<63 and r<63]
        if not S: return None
        return (min(r for r,c in S),min(c for r,c in S),max(r for r,c in S),max(c for r,c in S),len(S))
    S=pts=[(r,c) for r,row in enumerate(a) for c,x in enumerate(row) if x=='S' and c<63 and r<63]
    hrows=[r for r in range(63) if sum(1 for c in range(63) if a[r][c]=='S')>30]
    vcols=[c for c in range(63) if sum(1 for r in range(63) if a[r][c]=='S')>30]
    return {'c':bb('c'),'B':bb('B'),'h':(hrows[0],hrows[-1]) if hrows else None,'v':(vcols[0],vcols[-1]) if vcols else None,'ndot':len(dotcells())}

def blk():
    a=current_frame.ascii.split('\n')
    S=[(r,c) for r,row in enumerate(a) for c,x in enumerate(row) if x=='B' and c<63 and r<63]
    return (min(r for r,c in S),min(c for r,c in S),max(r for r,c in S),max(c for r,c in S),len(S))

def rot(S): return {(2*R-r,2*C-c) for r,c in S}

def pairs(P):
    out=[]
    for dr in range(-60,61):
        for dc in range(-60,61):
            S={(r+dr,c+dc) for r,c in P}
            U=S|rot(S)
            if len(U)==2*len(P) and U<=Y: out.append((dr,dc,frozenset(U)))
    return out

def bb(S): return (min(r for r,c in S),min(c for r,c in S),max(r for r,c in S),max(c for r,c in S))

def rotf(S): return {(2*R-r,2*C-c) for r,c in S}

def cov(P,dr,dc):
    S={(r+dr,c+dc) for r,c in P}
    return S|rotf(S)

def snap7():
    a=current_frame.ascii.split('\n')
    def pts(ch): return [(r,c) for r,row in enumerate(a) for c,x in enumerate(row) if x==ch and c<63 and r<63]
    C=pts('c'); B=pts('B')
    hrows=[r for r in range(63) if sum(1 for c in range(63) if a[r][c]=='S')>30]
    vcols=[c for c in range(63) if sum(1 for r in range(63) if a[r][c]=='S')>30]
    return {'nc':len(C),'cbb':bb(C) if C else None,'nB':len(B),'h':(hrows[0],hrows[-1]) if hrows else None,'v':(vcols[0],vcols[-1]) if vcols else None,'ndot':len(dotcells())}

def yc():
    a=current_frame.ascii.split('\n')
    return sum(1 for r in range(63) for c in range(63) if a[r][c]=='Y')

def orbitf(S,dr,dc,rh,cv):
    base={(r+dr,c+dc) for r,c in S}
    out=set(base); out|={(r,2*cv-c) for r,c in base}; out|={(2*rh-r,c) for r,c in base}; out|={(2*rh-r,2*cv-c) for r,c in base}
    return out&BRD

def cands(P,rh,cv,lo=-55,hi=56,minscore=-30):
    res=[]
    for dr in range(lo,hi):
        for dc in range(lo,hi):
            U=orbitf(P,dr,dc,rh,cv)
            sc=len(U&Y)-(len(U-Y))
            if sc>=minscore: res.append((sc,dr,dc,frozenset(U)))
    res.sort(reverse=True)
    return res

def top(P,rh,cv,n=400):
    res=[]
    for dr in range(-55,56):
        for dc in range(-55,56):
            U=orbitf(P,dr,dc,rh,cv)
            res.append((len(U&Y),dr,dc,frozenset(U)))
    res.sort(key=lambda t:-t[0])
    return res[:n]

def blockfill(S,r0,c0,nr,nc):
    out=[]
    for i in range(nr):
        row=''
        for j in range(nc):
            n=sum(1 for dr in range(3) for dc in range(3) if (r0+3*i+dr,c0+3*j+dc) in S)
            row+='.' if n==0 else ('#' if n==9 else str(n))
        out.append(row)
    return out

def bitset(cells): 
    m=0
    for r,c in cells: m|=1<<(r*64+c)
    return m

def prep(P):
    r0=min(r for r,c in P); c0=min(c for r,c in P)
    F={(r-r0,c-c0) for r,c in P}; h=max(r for r,c in F)+1; w=max(c for r,c in F)+1
    def bs(S):
        m=0
        for r,c in S: m|=1<<(r*64+c)
        return m
    return dict(F=bs(F),fH=bs({(h-1-r,c) for r,c in F}),fV=bs({(r,w-1-c) for r,c in F}),fHV=bs({(h-1-r,w-1-c) for r,c in F}),h=h,w=w)

def shift(bs,dr,dc):
    if dc>=0: bs<<=dc
    else: bs>>=-dc
    if dr>=0: bs<<=dr*64
    else: bs>>=-dr*64
    return bs

def build(P,rh,cv):
    d=prep(P)
    def orbit(dr,dc):
        r1=2*rh-dr-d['h']+1; c1=2*cv-dc-d['w']+1
        return shift(d['F'],dr,dc)|shift(d['fH'],r1,dc)|shift(d['fV'],dr,c1)|shift(d['fHV'],r1,c1)
    return d,orbit

def pblk(P):
    S=set(((r)//3,(c)//3) for r,c in P)   # absolute block coords (rows/3, cols/3)
    return S

def check(P,S): return len(P)==9*len(S)-sum(1 for _ in S)

def relblk(P): return set(((r//3)-1,(c//3)-7) for r,c in P)

def orbits(B,lo=-22,hi=23):
    d={}
    for dr in range(lo,hi):
        for dc in range(lo,hi):
            S={(r+dr,c+dc) for r,c in B}
            O=orbit(S)
            if O<=Yblk:
                d.setdefault(frozenset(O),[]).append((dr,dc))
    return d

def variants(P):
    r0=min(r for r,c in P); c0=min(c for r,c in P)
    F={(r-r0,c-c0) for r,c in P}; h=max(r for r,c in F)+1; w=max(c for r,c in F)+1
    return {'F':F,'fH':{(h-1-r,c) for r,c in F},'fV':{(r,w-1-c) for r,c in F},'fHV':{(h-1-r,w-1-c) for r,c in F},'h':h,'w':w}

def search(KR,KC):
    def orb(S):
        o=set(S); o|={(2*KR-r,c) for r,c in S}; o|={(r,2*KC-c) for r,c in S}; o|={(2*KR-r,2*KC-c) for r,c in S}
        return {p for p in o if -1<=p[0]<=19 and -7<=p[1]<=13}
    def tops(B,n=25):
        res=[]
        for dr in range(-22,23):
            for dc in range(-22,23):
                O=orb({(r+dr,c+dc) for r,c in B})
                res.append((len(O&YF),len(O-YF),dr,dc,frozenset(O)))
        res.sort(key=lambda t:(-(t[0]-t[1])))
        return res[:n]
    t1=tops(B1); t2=tops(B2)
    best=None
    for (c1,o1,d1,e1,U1) in t1:
        for (c2,o2,d2,e2,U2) in t2:
            U=U1|U2
            miss=len(YF-U); over=len(U-YF)
            if best is None or (miss,over)<best[0]: best=((miss,over),(d1,e1,d2,e2))
    return best

def sh(bs,dr,dc):
    if dc>=0: bs<<=dc
    else: bs>>=-dc
    if dr>=0: bs<<=dr*64
    else: bs>>=-dr*64
    return bs

def sweep():
    out=[]
    for KR in range(0,13):
        for KC in range(0,11):
            def orb(S):
                o=set(S); o|={(2*KR-r,c) for r,c in S}; o|={(r,2*KC-c) for r,c in S}; o|={(2*KR-r,2*KC-c) for r,c in S}
                return frozenset(p for p in o if -1<=p[0]<=19 and -7<=p[1]<=13)
            def tops(B,n=12):
                res=[]
                for dr in range(-20,21):
                    for dc in range(-20,21):
                        O=orb({(r+dr,c+dc) for r,c in B})
                        res.append((len(O&YF)-len(O-YF),O))
                res.sort(key=lambda t:-t[0]); return [O for _,O in res[:n]]
            t1=tops(B1); t2=tops(B2)
            best=None
            for U1 in t1:
                for U2 in t2:
                    U=U1|U2; miss=42-len(U&YF); over=len(U-YF)
                    if best is None or (miss,over)<best: best=(miss,over)
            out.append((best[0],best[1],KR,KC))
    out.sort()
    return out

def info():
    a=current_frame.ascii.split('\n')
    Y=sum(1 for r in range(63) for c in range(63) if a[r][c]=='Y')
    B=sum(1 for r in range(63) for c in range(63) if a[r][c]=='B')
    C=sum(1 for r in range(63) for c in range(63) if a[r][c]=='c')
    return dict(Y=Y,B=B,c=C,ndot=len(dotcells()))

def orbs(V,dr,dc,rh,cv):
    r1=2*rh-dr-V['h']+1; c1=2*cv-dc-V['w']+1
    return (sh(V['F'],dr,dc)|sh(V['fH'],r1,dc)|sh(V['fV'],dr,c1)|sh(V['fHV'],r1,c1))&MASK

def scan(rhlist):
    best=None; rows=[d for d in range(-24,73) if d%3==0]
    for rh in rhlist:
        for cv in range(1,62,3):
            t1=[]; t2=[]
            for dr in rows:
                for dc in rows:
                    U=orbs(V1,dr,dc,rh,cv); t1.append(((U&YB).bit_count(),dr,dc,U))
                    U=orbs(V2,dr,dc,rh,cv); t2.append(((U&YB).bit_count(),dr,dc,U))
            t1.sort(key=lambda x:-x[0]); t2.sort(key=lambda x:-x[0])
            for (k1,d1,e1,U1) in t1[:3]:
                for (k2,d2,e2,U2) in t2[:3]:
                    cov=((U1|U2)&YB).bit_count()
                    if best is None or cov>best[0]: best=(cov,(rh,cv),(d1,e1),(d2,e2))
    return best

def mask_of(S):
    m=0
    for p in S:
        if p not in idx: return None
        m|=1<<idx[p]
    return m
