try:
    import copy
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
try:
    import pickle
except Exception:
    pass
try:
    import heapq
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
    import math
except Exception:
    pass
try:
    from collections import defaultdict
except Exception:
    pass

def bars():
    a = current_frame.ascii.split('\n')
    gb = [a[8+i][25:35] for i in range(5)]
    yb = [a[26+i][8:12] for i in range(11)]
    gp = [a[18+i][35:49] for i in range(7)]
    yp = [a[35+i][19:29] for i in range(13)]
    return gb, yb, gp, yp

def probe(pts):
    for (r,c) in pts:
        before = current_frame.ascii
        res = action([{'action':'MOUSE','row':r,'col':c}])
        item = res if isinstance(res, dict) else res
        d = frame_diff()
        summ = []
        for k in ('moved','rotated','appeared','disappeared','changed_color','resized'):
            for e in d.get(k,[]):
                summ.append((k, e.get('color'), e.get('from'), e.get('to'), e.get('at'), e.get('bbox'), e.get('pixels'), e.get('before_pixels'), e.get('after_pixels')))
        st = item.get('state') if isinstance(item, dict) else '?'
        print('CLICK', r, c, 'noop' if (isinstance(item,dict) and item.get('no_op')) else st, 'changed', d['changed_cell_count'])
        for s in summ: print('   ', s)
        if isinstance(item, dict) and (item.get('level_completed') or item.get('game_over') or item.get('done')):
            print('  SPECIAL', item.get('state'), item.get('level_completed'), item.get('game_over'), item.get('done'))
            break

def barlen():
    nodes = current_frame.segmentation['nodes']
    out = {}
    for n in nodes:
        if n['color'] in ('N','Y'):
            b = n['boundary']
            r0 = min(p[0] for p in b); r1 = max(p[0] for p in b)
            c0 = min(p[1] for p in b); c1 = max(p[1] for p in b)
            out[n['color']] = (r0,c0,r1,c1,n['pixels'])
    return out

def greenbar():
    a = current_frame.ascii.split('\n')
    return '\n'.join(a[8+i][25:60] for i in range(5))

def yellowbar():
    a = current_frame.ascii.split('\n')
    return '\n'.join(a[26+i][8:12] for i in range(11))

def small():
    a = current_frame.ascii.split('\n')
    return '\n'.join(a[34+i][7:22] for i in range(11))

def chain():
    nodes = current_frame.segmentation['nodes']
    out = {}
    for n in nodes:
        b = n['boundary']
        r0=min(p[0] for p in b); r1=max(p[0] for p in b); c0=min(p[1] for p in b); c1=max(p[1] for p in b)
        if r0 > 50: continue
        if n['color']=='O': out['Lo']=(n['pixels']//3,(r0,c0,r1,c1))
        if n['color']=='S': out['Ls']=(n['pixels']//3,(r0,c0,r1,c1))
        if n['color']=='Y': out['Ly']=(n['pixels']//3,(r0,c0,r1,c1))
        if n['color']=='N': out['Lg']=((n['pixels']+1)//3,(r0,c0,r1,c1))
        if n['color']=='r' and n['pixels']==1: out.setdefault('dots',[]).append((r0,c0))
    return out

def run(name, wAb=True, wDb=True, purp=True):
    PURPLE_=rect_(27,45,29,47) if purp else 0
    A={}
    for Lb in LB:
        for Lr in LR:
            core=mA(Lb,Lr,False); full=mA(Lb,Lr,True)
            if core is None or full is None: continue
            if core&PURPLE_ or core&~REGION: continue
            A[(Lb,Lr)]=full if wAb else core
    D={}
    for Lo in LO:
        for Lg in LG:
            core=mD(Lo,Lg,False); full=mD(Lo,Lg,True)
            if core is None or full is None: continue
            if core&PURPLE_ or core&~REGION: continue
            D[(Lo,Lg)]=full if wDb else core
    S={};P={}
    for Ls in LS:
        m=maskS_(Ls)
        if m is None or m&PURPLE_ or m&~REGION: continue
        S[Ls]=m
    for Lp in LP:
        m=maskP_(Lp)
        if m is None or m&PURPLE_ or m&~REGION: continue
        P[Lp]=m
    pAD={(a[0],a[1],d[0],d[1]) for a in A for d in D if not (A[a]&D[d])}
    pAS={(a[0],a[1],s) for a in A for s in S if not (A[a]&S[s])}
    pAP={(a[0],a[1],p) for a in A for p in P if not (A[a]&P[p])}
    pDS={(d[0],d[1],s) for d in D for s in S if not (D[d]&S[s])}
    pDP={(d[0],d[1],p) for d in D for p in P if not (D[d]&P[p])}
    pSP={(s,p) for s in S for p in P if not (S[s]&P[p])}
    def ok(st):
        Lb,Lr,Ls,Lo,Lg,Lp=st
        return ((Lb,Lr) in A and (Lo,Lg) in D and Ls in S and Lp in P and (Lb,Lr,Lo,Lg) in pAD
          and (Lb,Lr,Ls) in pAS and (Lb,Lr,Lp) in pAP and (Lo,Lg,Ls) in pDS and (Lo,Lg,Lp) in pDP and (Ls,Lp) in pSP)
    q=collections.deque([start]); seen={start}
    while q:
        st=q.popleft()
        for i in range(6):
            lst=G[i]; j=lst.index(st[i])
            for dj in (1,-1):
                if not (0<=j+dj<len(lst)): continue
                nv=st[:i]+(lst[j+dj],)+st[i+1:]
                if nv in seen or not ok(nv): continue
                seen.add(nv); q.append(nv)
    print(name,'reach',len(seen),'maxLs',max(s[2] for s in seen),'maxLp',max(s[5] for s in seen),
          'Lb',min(s[0] for s in seen),max(s[0] for s in seen),'Lr',min(s[1] for s in seen),max(s[1] for s in seen),
          'Lo',min(s[3] for s in seen),max(s[3] for s in seen),'Lg',min(s[4] for s in seen),max(s[4] for s in seen))
    return seen

def step(col, sign):
    pt = PLUS[col] if sign>0 else MINUS[col]
    before = chain()
    res = action([{'action':'MOUSE','row':pt[0],'col':pt[1]}])
    it = res if isinstance(res,dict) else res
    after = chain()
    k = KEY[col]
    b = before[k][0]; a = after[k][0]
    fl = ''
    if isinstance(it,dict):
        if it.get('no_op'): fl='NOOP'
        if it.get('level_completed'): fl+=' LC'
        if it.get('game_over'): fl+=' GO'
    print(col,'+' if sign>0 else '-',k,b,'->',a,fl,'dot',[d for d in after.get('dots',[]) if d[1]<45 or d[0]<45])
    return (a!=b) and 'LC' not in fl and 'GO' not in fl

def snap():
    out={}
    for n in current_frame.segmentation['nodes']:
        b=n['boundary']
        r0=min(p[0] for p in b); r1=max(p[0] for p in b); c0=min(p[1] for p in b); c1=max(p[1] for p in b)
        if r0>44: continue
        out.setdefault(n['color'],[]).append((n['pixels'],(r0,c0,r1,c1)))
    return out

def probe_plus(name, r, c):
    before = snap()
    res = action([{'action':'MOUSE','row':r,'col':c}])
    it = res if isinstance(res,dict) else res
    d = frame_diff()
    print(name, 'NOOP' if isinstance(it,dict) and it.get('no_op') else (it.get('state') if isinstance(it,dict) else '?'))
    for k in ('moved','resized','appeared','disappeared'):
        for e in d.get(k,[]):
            if e.get('color')=='G' and e.get('pixels')==1: continue
            print('  ',k,e.get('color'),e.get('from'),e.get('to'),e.get('bbox'),e.get('pixels'),e.get('before_pixels'),e.get('after_pixels'))
    if isinstance(it,dict) and (it.get('level_completed') or it.get('game_over')): print('  END', it.get('level_completed'), it.get('game_over'))

def get(col):
    out={}
    for n in current_frame.segmentation['nodes']:
        b=n['boundary']
        r0=min(p[0] for p in b); r1=max(p[0] for p in b); c0=min(p[1] for p in b); c1=max(p[1] for p in b)
        if r0>44: continue
        if n['color']==col: out.setdefault(col,[]).append((n['pixels'],(r0,c0,r1,c1)))
    return out

def bit(r,c): return 1<<(r*C+c)

def rect(r0,c0,r1,c1):
    if r1<r0 or c1<c0 or r0<0 or c0<0 or r1>62 or c1>63: return None
    m=0
    for r in range(r0,r1+1):
        row=0
        for c in range(c0,c1+1): row|=1<<c
        m|=row<<(r*C64)
    return m

def elems(st):
    Lb,Lr,Ls,Lo,Lg,Lp = st
    e={}
    e['sky_cap']=rect(27,6,29,6)
    e['sky']=rect(27,7,29,6+Ls)
    e['pink_cap']=rect(3,48,3,50)
    e['pink']=rect(4,48,3+Lp,50)
    e['blue_cap']=rect(36,32,38,32)
    e['blue']=rect(36,32-Lb,38,31)
    e['red_cap']=rect(38,32-Lb-3,38,32-Lb-1)
    e['red']=rect(38-Lr,32-Lb-3,37,32-Lb-1)
    e['wA']=rect(35-Lr,32-Lb-3,37-Lr,32-Lb-3+23)
    e['green_cap']=rect(26,54,26,56)
    e['green']=rect(26-Lg,54,25,56)
    e['orange_cap']=rect(23-Lg,56,25-Lg,56)
    e['orange']=rect(23-Lg,56-Lo,25-Lg,55)
    e['wD']=rect(23-Lg,56-Lo-3,23-Lg+17,56-Lo-1)
    return e

def dots(st):
    Lb,Lr,Ls,Lo,Lg,Lp=st
    return [(28,5+Ls),(2+Lp,49)]

def valid(st):
    l,g,b,s=st
    a1,a2=gA(l); h=gH(l,g); c=gC(g); bl=gBL(g,b); w=gW(g,b); sk=gS(s)
    ms=[a1,a2,h,c,bl,w,sk]
    for m in ms:
        if m is None or (m&STATIC): return False
    for i in range(len(ms)):
        for j in range(i+1,len(ms)):
            if ms[i]&ms[j]: return False
    return True

def bit_(r,c): return 1<<(r*C64+c)

def rect_(r0,c0,r1,c1):
    if r1<r0 or c1<c0 or r0<0 or c0<0: return None
    m=0
    for r in range(r0,r1+1):
        row=0
        for c in range(c0,c1+1): row|=1<<c
        m|=row<<(r*C64)
    return m

def maskA(Lb,Lr):
    return (rect_(36,32,38,32)|rect_(36,32-Lb,38,31)|rect_(38,32-Lb-3,38,32-Lb-1)|
            rect_(38-Lr,32-Lb-3,37,32-Lb-1)|rect_(35-Lr,32-Lb-3,37-Lr,32-Lb-3+23))

def maskD(Lo,Lg):
    return (rect_(26,54,26,56)|rect_(26-Lg,54,25,56)|rect_(23-Lg,56,25-Lg,56)|
            rect_(23-Lg,56-Lo,25-Lg,55)|rect_(23-Lg,56-Lo-3,23-Lg+17,56-Lo-1))

def maskS(Ls): return rect_(27,6,29,6)|rect_(27,7,29,6+Ls)

def maskP(Lp): return rect_(3,48,3,50)|rect_(4,48,3+Lp,50)

def ok(st,OBS=OBS):
    r=build4(st[0],st[1],(13,49))
    if r is None: return False
    seen=set()
    for g in r[0]+r[1]:
        if g & seen: return False
        seen|=g
    for g in seen:
        if g[0]<0 or g[0]>41 or g[1]<0 or g[1]>62: return False
    return not (seen & OBS)

def dotcross(st): return (5+st[2]==43) or (2+st[5]==28)

goal=lambda st: st[2]==38 and st[5]==26

def build(dirs,lens,pivot=(16,40)):
    d0=d0r=dirs[0]; L0=lens[0]
    cs=set(); p=pivot
    for k in range(L0):
        for c in band(p,LEFT[d0]): cs.add(c)
        p=add(p,d0)
    segs=[cs]; caps=[set(band(add(pivot,d0,-1),LEFT[d0]))]
    E=add(pivot,d0,L0-1); pd=[d0]
    for (dc,Ln) in ((dirs[1],lens[1]),(dirs[2],lens[2])):
        dp=pd[-1]; v=LEFT[dp]
        if dc==dp:
            cap=set(band(add(E,dc,1),v))
            cells=set()
            for k in range(Ln):
                q=add(E,dc,k+2)
                for c in band(q,v): cells.add(c)
            fe=add(E,dc,Ln+1)
        elif dc in (LEFT[dp],RIGHT(dp)):
            if dc==RIGHT(dp): base=add(E,v,1); step=neg(v)
            else: base=add(E,v,-1); step=v
            cells=set(); cap=set()
            for a in (1,2,3):
                for b in range(Ln+1):
                    c=add(add(base,dp,a),step,b)
                    (cap if b==0 else cells).add(c)
            fe=add(add(base,dp,2),step,Ln)
        else:
            return None
        segs.append(cells); caps.append(cap); pd.append(dc); E=fe
    dot=add(E,dirs[2],-1)
    return segs,caps,dot

def maskS_(Ls): return rect_(27,6,29,6)|rect_(27,7,29,6+Ls)

def maskP_(Lp): return rect_(3,48,3,50)|rect_(4,48,3+Lp,50)

def mA(Lo): return join([rect(6,9,6,11),rect(7,9,6+Lo,11)])

def mD(Lo,Lg):
    parts=[rect_(26,54,26,56),rect_(26-Lg,54,25,56),rect_(23-Lg,56,25-Lg,56),rect_(23-Lg,56-Lo,25-Lg,55),rect_(23-Lg,56-Lo-3,23-Lg+17,56-Lo-1)]
    m=0
    for p in parts:
        if p is None: return None
        m|=p
    return m

def search(wblocks, forbid_dot, goalLp=26):
    A={};D={};S={};P={}
    for Lb in LB:
        for Lr in LR:
            core=mA(Lb,Lr,False); full=mA(Lb,Lr,True)
            if core is None or core&PURPLE_ or core&~REGION: continue
            A[(Lb,Lr)]=full if wblocks else core
    for Lo in LO:
        for Lg in LG:
            core=mD(Lo,Lg,False); full=mD(Lo,Lg,True)
            if core is None or core&PURPLE_ or core&~REGION: continue
            D[(Lo,Lg)]=full if wblocks else core
    for Ls in LS:
        m=rect_(27,6,29,6)|rect_(27,7,29,6+Ls)
        if m and not(m&PURPLE_ or m&~REGION): S[Ls]=m
    for Lp in LP:
        m=rect_(3,48,3,50)|rect_(4,48,3+Lp,50)
        if m and not(m&PURPLE_ or m&~REGION): P[Lp]=m
    pAD={(a[0],a[1],d[0],d[1]) for a in A for d in D if not (A[a]&D[d])}
    pAS={(a[0],a[1],s) for a in A for s in S if not (A[a]&S[s])}
    pAP={(a[0],a[1],p) for a in A for p in P if not (A[a]&P[p])}
    pDS={(d[0],d[1],s) for d in D for s in S if not (D[d]&S[s])}
    pDP={(d[0],d[1],p) for d in D for p in P if not (D[d]&P[p])}
    pSP={(s,p) for s in S for p in P if not (S[s]&P[p])}
    def ok(st):
        Lb,Lr,Ls,Lo,Lg,Lp=st
        return ((Lb,Lr) in A and (Lo,Lg) in D and Ls in S and Lp in P and (Lb,Lr,Lo,Lg) in pAD
          and (Lb,Lr,Ls) in pAS and (Lb,Lr,Lp) in pAP and (Lo,Lg,Ls) in pDS and (Lo,Lg,Lp) in pDP and (Ls,Lp) in pSP)
    def dotcross(st): return (5+st[2]==43) or (2+st[5]==28)
    goal=lambda st: st[2]==38 and st[5]==goalLp
    q=collections.deque([start]); par={start:None}; found=None
    while q:
        st=q.popleft()
        if goal(st): found=st; break
        for i in range(6):
            lst=G[i]; j=lst.index(st[i])
            for dj in (1,-1):
                if not (0<=j+dj<len(lst)): continue
                nv=st[:i]+(lst[j+dj],)+st[i+1:]
                if nv in par or not ok(nv): continue
                if forbid_dot and dotcross(nv) and not goal(nv): continue
                par[nv]=st; q.append(nv)
    if not found: return (None,len(par))
    path=[];cur=found
    while cur is not None: path.append(cur); cur=par[cur]
    path.reverse()
    mv=[]
    for a,b in zip(path,path[1:]):
        for i in range(6):
            if a[i]!=b[i]: mv.append('brosap'[i]+('+' if b[i]>a[i] else '-'))
    return (found,len(mv),' '.join(mv))

def readstate():
    d={}
    for n in current_frame.segmentation['nodes']:
        b=n['boundary']
        r0=min(p[0] for p in b); r1=max(p[0] for p in b); c0=min(p[1] for p in b); c1=max(p[1] for p in b)
        if r0>44: continue
        if n['color']=='S': d['s']=c1-6
        elif n['color']=='P': d['p']=r1-3
        elif n['color']=='b': d['b']=32-c0
        elif n['color']=='R': d['r']=38-r0
        elif n['color']=='O': d['o']=56-c0
        elif n['color']=='N': d['g']=26-r0
    return (d['b'],d['r'],d['s'],d['o'],d['g'],d['p'])

def snap4():
    out=[]
    for n in current_frame.segmentation['nodes']:
        b=n['boundary']
        r0=min(p[0] for p in b); r1=max(p[0] for p in b); c0=min(p[1] for p in b); c1=max(p[1] for p in b)
        if r0>44: continue
        out.append((n['color'],n['pixels'],(r0,c0,r1,c1)))
    return sorted(out)

def chainN(Lg,Ly): return join([rect(20,9,20,11),rect(20-Lg,9,19,11),rect(17-Lg,8,19-Lg,8),rect(17-Lg,9,19-Lg,8+Ly)])

def chainB(Lb,Ly): return join([rect(17,51,17,53),rect(17-Lb,51,16,53),rect(14-Lb,53,16-Lb,53),rect(14-Lb,53-Ly,16-Lb,52)])

def chainR(Lr,Ly): return join([rect(23,45,23,47),rect(23-Lr,45,22,47),rect(20-Lr,47,22-Lr,47),rect(20-Lr,47-Ly,22-Lr,46)])

def chainO(Lo,Ly): return join([rect(53,18,53,20),rect(53-Lo,18,52,20),rect(50-Lo,18,52-Lo,18),rect(50-Lo,19,52-Lo,18+Ly)])

def chainY(Ly): return join([rect(44,30,44,32),rect(44-Ly,30,43,32)])

def join(ps):
    m=0
    for p in ps:
        if p is None: return None
        m|=p
    return m

def pair(X,Y):
    s=set()
    for k1,m1 in X.items():
        for k2,m2 in Y.items():
            if not(m1&m2): s.add((k1,k2))
    return s

def colN(Lg): return join([rect(20,9,20,11),rect(20-Lg,9,19,11)])

def yelN(Lg,Ly): return join([rect(17-Lg,8,19-Lg,8),rect(17-Lg,9,19-Lg,8+Ly)])

def colB(Lb): return join([rect(17,51,17,53),rect(17-Lb,51,16,53)])

def yelB(Lb,Ly): return join([rect(14-Lb,53,16-Lb,53),rect(14-Lb,53-Ly,16-Lb,52)])

def colR(Lr): return join([rect(23,45,23,47),rect(23-Lr,45,22,47)])

def yelR(Lr,Ly): return join([rect(20-Lr,47,22-Lr,47),rect(20-Lr,47-Ly,22-Lr,46)])

def colO(Lo): return join([rect(53,18,53,20),rect(53-Lo,18,52,20)])

def yelO(Lo,Ly): return join([rect(50-Lo,18,52-Lo,18),rect(50-Lo,19,52-Lo,18+Ly)])

def yelE(Ly): return join([rect(44,30,44,32),rect(44-Ly,30,43,32)])

def st4():
    d={}
    for n in current_frame.segmentation['nodes']:
        b=n['boundary']
        r0=min(p[0] for p in b); r1=max(p[0] for p in b); c0=min(p[1] for p in b); c1=max(p[1] for p in b)
        if r0<44:
            if n['color']=='N' and n['pixels']>3: d['g']=20-r0
            elif n['color']=='b' and n['pixels']>3: d['b']=17-r0
            elif n['color']=='R' and n['pixels']>3: d['r']=23-r0
            elif n['color']=='O': d['o']=53-r0
            elif n['color']=='r' and n['pixels']==1: d['y']=45-r0
    return (d.get('g'),d.get('b'),d.get('r'),d.get('o'),d.get('y'))

def brief():
    out=[]
    d=frame_diff()
    for key in ('moved','resized','appeared','disappeared','rotated'):
        for e in d.get(key,[]):
            if e.get('color') in ('B','g','c'): continue
            bb=e.get('bbox')
            out.append((key[0],e.get('color'),e.get('from'),e.get('to'),tuple(bb) if bb else None,e.get('pixels')))
    return out

def try_click(pt):
    try:
        res=action([{'action':'MOUSE','row':pt[0],'col':pt[1]}])
    except Exception as e:
        print(pt,'HALTED',type(e).__name__); return False
    it=res if isinstance(res,dict) else {}
    print(pt,'NOOP' if it.get('no_op') else 'ok','LC' if it.get('level_completed') else '')
    for x in brief(): print('   ',x)
    return True

def mB(Lo): return join([rect(6,42,6,44),rect(7,42,6+Lo,44)])

def mHead(Lo,Lg): return join([rect(7+Lo,9,9+Lo,9),rect(7+Lo,10,9+Lo,9+Lg)])

def mGC(Lg): return join([rect(35,54,35,56),rect(35-Lg,54,34,56)])

def mBlue(Lg,Lb): return join([rect(32-Lg,56,34-Lg,56),rect(32-Lg,56-Lb,34-Lg,55)])

def mW(Lg,Lb): return rect(32-Lg,53-Lb,49-Lg,55-Lb)

def mSky(Ls): return join([rect(24,21,26,21),rect(24,22,26,21+Ls)])

def gA(l):
    if l not in cacheA: cacheA[l]=(mA(l),mB(l))
    return cacheA[l]

def gH(l,g):
    k=(l,g)
    if k not in cacheH: cacheH[k]=mHead(l,g)
    return cacheH[k]

def gC(g):
    if g not in cacheC: cacheC[g]=mGC(g)
    return cacheC[g]

def gBL(g,b):
    k=(g,b)
    if k not in cacheBL: cacheBL[k]=mBlue(g,b)
    return cacheBL[k]

def gW(g,b):
    k=(g,b)
    if k not in cacheW: cacheW[k]=mW(g,b)
    return cacheW[k]

def gS(s):
    if s not in cacheS: cacheS[s]=mSky(s)
    return cacheS[s]

def read5():
    nd=current_frame.segmentation['nodes']
    Lo=Lg=Lb=Ls=None
    for n in nd:
        b=n['boundary']; r0=min(p[0] for p in b); r1=max(p[0] for p in b); c0=min(p[1] for p in b); c1=max(p[1] for p in b)
        col=n['color']
        if col=='O' and c0<=9<=c1 and r0<=7: Lo=r1-6
        if col=='N' and r0<50 and c0>=10 and c1<40: Lg=max(Lg or 0, c1-9)
        if col=='b' and r0<50 and n['pixels']>10: Lb=56-c0
        if col=='S' and r0<50: Ls=max(Ls or 0, c1-21)
    return (Lo,Lg,Lb,Ls)

def nodes6(top=True):
    out=[]
    for n in current_frame.segmentation['nodes']:
        b=n['boundary']
        r0=min(p[0] for p in b); r1=max(p[0] for p in b); c0=min(p[1] for p in b); c1=max(p[1] for p in b)
        if top and r0>44: continue
        out.append((n['color'],n['pixels'],(r0,c0,r1,c1)))
    return out

def add(p,v,k=1): return (p[0]+v[0]*k,p[1]+v[1]*k)

def band(p,v): return [add(p,v,-1),p,add(p,v,1)]

def perp(d): return LEFT[d]

def rectcells(p0,dc,Ln):
    # straight segment rect: Ln cells from p0 stepping dc, each banded
    cs=set(); p=p0
    for k in range(Ln):
        for c in band(p,perp(dc)): cs.add(c)
        p=add(p,dc)
    return cs,p

def bb(x):
    rs=[p[0] for p in x]; cs=[p[1] for p in x]
    return (min(rs),min(cs),max(rs),max(cs)) if x else None

def neg(v): return (-v[0],-v[1])

def RIGHT(d): return LEFT[LEFT[LEFT[d]]]

def cells_all(st):
    dirs,lens=st
    r=build(dirs,lens)
    if r is None: return None
    s,c,dot=r
    out=set()
    for x in s: out|=x
    for x in c: out|=x
    return out,dot

def succ(st):
    d,l=st; out=[]
    for i in range(3):
        nd=list(d)
        for j in range(i,4): nd[j]=(nd[j]+1)%4
        out.append((tuple(nd),l))
    for i in range(4):
        k=LEN.index(l[i])
        for dj in (1,-1):
            if 0<=k+dj<len(LEN):
                nl=list(l); nl[i]=LEN[k+dj]; out.append((d,tuple(nl)))
    return out

def build2(dirs,lens,VEC=((-1,0),(0,-1),(1,0),(0,1))):
    def add(p,v,k=1): return (p[0]+v[0]*k,p[1]+v[1]*k)
    def band(p,v): return [add(p,v,-1),p,add(p,v,1)]
    d0=dirs[0]; v0=VEC[d0]; pv=VEC[(d0+1)%4]
    cs=set(); p=(16,40)
    for k in range(lens[0]):
        for c in band(p,pv): cs.add(c)
        p=add(p,v0)
    segs=[cs]; caps=[set(band(add((16,40),v0,-1),pv))]
    E=add((16,40),v0,lens[0]-1); pd=[d0]
    for i in (1,2):
        dc=dirs[i]; Ln=lens[i]; dp=pd[-1]; v=VEC[(dp+1)%4]; vd=VEC[dp]
        if dc==dp:
            cap=set(band(add(E,VEC[dc],1),v)); cells=set()
            for k in range(Ln):
                for c in band(add(E,VEC[dc],k+2),v): cells.add(c)
            fe=add(E,VEC[dc],Ln+1)
        elif dc==(dp+1)%4:
            base=add(E,v,-1); cells=set(); cap=set()
            for a in (1,2,3):
                for b in range(Ln+1):
                    c=add(add(base,vd,a),v,b)
                    (cap if b==0 else cells).add(c)
            fe=add(add(base,vd,2),v,Ln)
        elif dc==(dp+3)%4:
            base=add(E,v,1); nv=(-v[0],-v[1]); cells=set(); cap=set()
            for a in (1,2,3):
                for b in range(Ln+1):
                    c=add(add(base,vd,a),nv,b)
                    (cap if b==0 else cells).add(c)
            fe=add(add(base,vd,2),nv,Ln)
        else:
            return None
        segs.append(cells); caps.append(cap); pd.append(dc); E=fe
    return segs,caps,add(E,VEC[dirs[2]],-1)

def expected(dirs,lens):
    s,c,dot=build2(dirs,lens)
    out=[]
    for col,grp in (('N',s[0]),('Y',s[1])):
        out.append((col,len(grp),bb(grp)))
    bset=s[2]-{dot}
    out.append(('b',len(bset),bb(bset)))
    for i,g in enumerate(c): out.append(('G',len(g),bb(g)))
    out.append(('r',1,dot))
    return sorted(out)

def actual():
    out=[]
    for n in current_frame.segmentation['nodes']:
        b=n['boundary']
        r0=min(p[0] for p in b); r1=max(p[0] for p in b); c0=min(p[1] for p in b); c1=max(p[1] for p in b)
        if r0>44: continue
        if n['color'] in ('N','Y','b','G','r') and n['pixels']<=200:
            out.append((n['color'],n['pixels'],(r0,c0,r1,c1)))
    return sorted(out)

def actual2():
    out=[]
    for n in current_frame.segmentation['nodes']:
        b=n['boundary']
        r0=min(p[0] for p in b); r1=max(p[0] for p in b); c0=min(p[1] for p in b); c1=max(p[1] for p in b)
        if r0>44: continue
        col=n['color']
        if col=='r' and (r0,c0) in CROSS: continue
        if col in ('N','Y','b','G','r') and n['pixels']<=200:
            out.append((col,n['pixels'],(r0,c0,r1,c1)))
    return sorted(out)

def expected2(dirs,lens):
    e=expected(dirs,lens)
    return sorted([(c,p,(b[0],b[1],b[0],b[1]) if len(b)==2 else b) for c,p,b in e])

def build4(dirs,lens,piv,VEC=((-1,0),(0,-1),(1,0),(0,1))):
    def add(p,v,k=1): return (p[0]+v[0]*k,p[1]+v[1]*k)
    def band(p,v): return [add(p,v,-1),p,add(p,v,1)]
    d0=dirs[0]; v0=VEC[d0]; pv=VEC[(d0+1)%4]
    cs=set(); p=piv
    for k in range(lens[0]):
        for c in band(p,pv): cs.add(c)
        p=add(p,v0)
    segs=[cs]; caps=[set(band(add(piv,v0,-1),pv))]
    E=add(piv,v0,lens[0]-1); pd=[d0]
    for i in range(1,len(dirs)):
        dc=dirs[i]; Ln=lens[i]; dp=pd[-1]; v=VEC[(dp+1)%4]; vd=VEC[dp]
        if dc==dp:
            cap=set(band(add(E,VEC[dc],1),v)); cells=set()
            for k in range(Ln):
                for c in band(add(E,VEC[dc],k+2),v): cells.add(c)
            fe=add(E,VEC[dc],Ln+1)
        elif dc==(dp+1)%4:
            base=add(E,v,-1); cells=set(); cap=set()
            for a in (1,2,3):
                for b in range(Ln+1):
                    c=add(add(base,vd,a),v,b)
                    (cap if b==0 else cells).add(c)
            fe=add(add(base,vd,2),v,Ln)
        elif dc==(dp+3)%4:
            base=add(E,v,1); nv=(-v[0],-v[1]); cells=set(); cap=set()
            for a in (1,2,3):
                for b in range(Ln+1):
                    c=add(add(base,vd,a),nv,b)
                    (cap if b==0 else cells).add(c)
            fe=add(add(base,vd,2),nv,Ln)
        else:
            return None
        segs.append(cells); caps.append(cap); pd.append(dc); E=fe
    return segs,caps,add(E,VEC[dirs[-1]],-1)

def free(st):
    r=geo(st)
    if r is None: return False
    seen=set()
    for g in r[0]+r[1]:
        if g & seen: return False
        seen|=g
    return not (seen & OBS)

def geo(st):
    return build4(st[0],st[1],(13,49))

def h(st):
    r=build4(st[0],st[1],(13,49))
    if r is None: return 9999
    d=r[2]
    return abs(d[0]-GD[0])+abs(d[1]-GD[1]) + 2*abs(sum(st[1])-GTOT)//3 + 2*sum(1 for i in range(4) if st[0][i]!=((2,1,0,3)[i]))

def readC():
    out=[]
    for n in current_frame.segmentation['nodes']:
        b=n['boundary']
        r0=min(p[0] for p in b); r1=max(p[0] for p in b); c0=min(p[1] for p in b); c1=max(p[1] for p in b)
        if r0>44: continue
        col=n['color']
        if col=='r' and (r0,c0) in CROSS: continue
        if col in ('Y','N','b','O','G','r') and n['pixels']<=120: out.append((col,n['pixels'],(r0,c0,r1,c1)))
    return sorted(out)

def expC(st):
    s,c,dot=geo(st); out=[]
    for col,grp in (('Y',s[0]),('N',s[1]),('b',s[2])): out.append((col,len(grp),bb(grp)))
    ob=s[3]-{dot}; out.append(('O',len(ob),bb(ob)))
    for g in c: out.append(('G',3,bb(g)))
    out.append(('r',1,(dot[0],dot[1],dot[0],dot[1])))
    return sorted(out)

def sbar(L):
    # S bar: pivot (7,34), dir L(1); rect L cells cols 34-L+1..34 rows6-8 ; cap col35
    cells=set()
    for c in range(34-L+1,35):
        for r in (6,7,8): cells.add((r,c))
    for r in (6,7,8): cells.add((r,35))
    return cells

def rbar(d):
    # red: pivot (7,13), dir d ; rect 14 cells; cap
    VEC=((-1,0),(0,-1),(1,0),(0,1))
    v=VEC[d]; pv=VEC[(d+1)%4]
    cells=set(); p=(7,13)
    for k in range(14):
        for t in (-1,0,1): cells.add((p[0]+pv[0]*t,p[1]+pv[1]*t))
        p=(p[0]+v[0],p[1]+v[1])
    for t in (-1,0,1): cells.add((7-v[0]+pv[0]*t,7-v[1]+pv[1]*t))
    return cells

def redstate():
    for n in current_frame.segmentation['nodes']:
        if n['color']=='R':
            return n['pixels'], bb(set()) if False else None, min(p[0] for p in n['boundary']), max(p[0] for p in n['boundary']), min(p[1] for p in n['boundary']), max(p[1] for p in n['boundary'])

def sstate():
    for n in current_frame.segmentation['nodes']:
        if n['color']=='S':
            b=n['boundary']; return n['pixels'],(min(p[0] for p in b),min(p[1] for p in b),max(p[0] for p in b),max(p[1] for p in b))

def obsset():
    O=set()
    for n in current_frame.segmentation['nodes']:
        if n['color'] in ('p','R','S'):
            b=n['boundary']
            r0=min(p[0] for p in b); r1=max(p[0] for p in b); c0=min(p[1] for p in b); c1=max(p[1] for p in b)
            if r0>44: continue
            for r in range(r0,r1+1):
                for c in range(c0,c1+1): O.add((r,c))
    return O

def bad(st):
    r=build4(st[0],st[1],(13,49))
    if r is None: return ['INVALID']
    seen=set(); out=[]
    for g in r[0]+r[1]:
        if g&seen: out.append('selfoverlap '+str(sorted(g&seen)[:3]))
        seen|=g
    hit=seen&OBS
    if hit: out.append('obs '+str(sorted(hit)[:4])+' n=%d'%len(hit))
    return out

def cells(ch):
    s=set()
    for r in range(0,45):
        row=G[r]
        for c in range(64):
            if row[c]==ch: s.add((r,c))
    return s

def obsnow():
    a=current_frame.ascii.split('\n')
    O=set()
    for r in range(45):
        for c,ch in enumerate(a[r]):
            if ch in ('p','R','S'): O.add((r,c))
    return O

def chk(st):
    r=build4(st[0],st[1],(13,49))
    if r is None: return False
    seen=set()
    for g in r[0]+r[1]:
        if g&seen: return False
        seen|=g
    if seen & OB: return False
    for (x,y) in seen:
        if x<0 or x>44 or y<0 or y>63: return False
    return True

def hfun(st):
    d=dotof(st)
    if d is None: return 9999
    return abs(d[0]-GD[0])+abs(d[1]-GD[1])

def why(st):
    r=build4(st[0],st[1],(13,49))
    if r is None: return 'INVALID(reversal)'
    seen=set()
    for g in r[0]+r[1]:
        if g&seen: return 'selfovl '+str(sorted(g&seen)[:3])
        seen|=g
    hit=seen&OB
    if hit: return 'obs '+str(sorted(hit)[:4])+' n=%d'%len(hit)
    oob=[p for p in seen if p[0]<0 or p[0]>44 or p[1]<0 or p[1]>63]
    if oob: return 'oob '+str(sorted(oob)[:3])
    return 'ok'

def chk2(st):
    r=build4(st[0],st[1],(13,49))
    if r is None: return False
    seen=set()
    for g in r[0]+r[1]:
        if g&seen: return False
        seen|=g
    if seen & OBch: return False
    for (x,y) in seen:
        if x<0 or x>44 or y<0 or y>63: return False
    return True

def dotof(st):
    r=build4(st[0],st[1],(13,49))
    return None if r is None else r[2]

def obset2():
    a=current_frame.ascii.split('\n')
    O=set()
    for r in range(45):
        for c,ch in enumerate(a[r]):
            if ch in ('p','R','S'): O.add((r,c))
    return O

def chkW(st):
    r=build4(st[0],st[1],(13,49))
    if r is None: return False
    seen=set()
    for g in r[0]+r[1]:
        if g&seen: return False
        seen|=g
    if seen & OB: return False
    for (x,y) in seen:
        if x<0 or x>44 or y<0 or y>63: return False
    return True

def dotat(st):
    r=build4(st[0],st[1],(13,49))
    return None if r is None else r[2]

def succW(st):
    d,l=st; out=[]
    for i in range(3):
        nd=list(d)
        for j in range(i,4): nd[j]=(nd[j]+1)%4
        out.append((tuple(nd),l))
    for i in range(4):
        k=LEN.index(l[i])
        for dj in (1,-1):
            if 0<=k+dj<len(LEN):
                nl=list(l); nl[i]=LEN[k+dj]; out.append((d,tuple(nl)))
    return out

def hW(st):
    d=dotat(st)
    return 9999 if d is None else abs(d[0]-GD[0])+abs(d[1]-GD[1])

def panel(r0,r1,c0,c1):
    from collections import Counter
    cc=Counter()
    for r in range(r0,r1+1):
        for c in range(c0,c1+1): cc[a[r][c]]+=1
    return cc.most_common(4)

def chaincells():
    out={}
    for ch in ('Y','N','b','O'):
        out[ch]=[(r,c) for r in range(45) for c in range(64) if a2[r][c]==ch]
    return out
