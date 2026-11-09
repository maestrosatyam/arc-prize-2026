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
try:
    import time
except Exception:
    pass
try:
    import heapq
except Exception:
    pass
try:
    from collections import Counter
except Exception:
    pass
try:
    import time,heapq
except Exception:
    pass
try:
    import builtins
except Exception:
    pass

def crop(fr,r0,r1,c0,c1):
    a = fr.ascii.split('\n')
    return [a[r][c0:c1+1] for r in range(r0,r1+1)]

def knob(st):
    base=(13,49)
    for i in range(4):
        L,d=st[2*i],st[2*i+1]; dr,dc=DIR[d]
        if i==3: return (base[0]+dr*(L-2),base[1]+dc*(L-2))
        base=(base[0]+dr*(L+1),base[1]+dc*(L+1))

def arm_cells(a,b,c,d):
    cells=set()
    # orange cap col9 rows39-41, body cols10..9+a rows39-41
    for r in (39,40,41):
        for col in range(9,10+a): cells.add((r,col))
    # sky cap row41 cols10+a..12+a ; body rows41-b..40
    for col in range(10+a,13+a): cells.add((41,col))
    for r in range(41-b,41):
        for col in range(10+a,13+a): cells.add((r,col))
    # yellow cap col10+a rows38-b..40-b ; body cols11+a..10+a+c rows38-b..40-b
    for r in range(38-b,41-b): cells.add((r,10+a))
    for r in range(38-b,41-b):
        for col in range(11+a,11+a+c): cells.add((r,col))
    # green cap row38-b cols11+a+c..13+a+c ; body rows39-b..38-b+d
    for col in range(11+a+c,14+a+c): cells.add((38-b,col))
    for r in range(39-b,39-b+d):
        for col in range(11+a+c,14+a+c): cells.add((r,col))
    return cells

def knobpos(a,b,c,d): return (37-b+d, 12+a+c)

def valid(st,Pblk,lo=0,hi=44):
    B=blocks(st)
    allc=set()
    for k,v in B.items():
        if v & allc: return False
        allc|=v
        if any(r<lo or r>hi or c<0 or c>63 for (r,c) in v): return False
    if allc & Pblk: return False
    return True

def purple_set():
    a=current_frame.ascii.split('\n')
    return {(r,c) for r in range(64) for c in range(64) if a[r][c]=='p'}

def valid2(a,b,c,d,P=None):
    if P is None: P=PUR
    cells=arm_cells(a,b,c,d)
    if cells & P: return False
    for (r,col) in cells:
        if r<0 or r>62 or col<0 or col>62: return False
        if 54<=r<=60: return False
    return True

def bfs(start, mins, maxv=47, target=(31,52)):
    from collections import deque
    P=purple_set()
    prev={start:None}
    q=deque([start]); found=None
    while q:
        s=q.popleft()
        if knobpos(*s)==target: found=s; break
        for i in range(4):
            ns=list(s); ns[i]=s[i]+3; ns=tuple(ns)
            if ns[i]>maxv: continue
            if ns not in prev and valid2(*ns,P):
                prev[ns]=(s,i,+3); q.append(ns)
            ns=list(s); ns[i]=s[i]-3; ns=tuple(ns)
            if ns[i]<mins[i]: continue
            if ns not in prev and valid2(*ns,P):
                prev[ns]=(s,i,-3); q.append(ns)
    return found, prev

def valid3(a,b,c,d,P):
    cells=arm_cells(a,b,c,d)
    if cells & P: return False
    for (r,col) in cells:
        if r<0 or r>62 or col<0 or col>62: return False
        if 54<=r<=60: return False
    return True

def bfs2(start, mins, P, maxv=47, target=(31,52)):
    from collections import deque
    prev={start:None}; q=deque([start]); found=None
    while q:
        s=q.popleft()
        if knobpos(*s)==target: found=s; break
        for i in range(4):
            for delta in (+3,-3):
                ns=list(s); ns[i]=s[i]+delta; ns=tuple(ns)
                if ns[i]>maxv or ns[i]<mins[i] or ns in prev: continue
                if valid3(*ns,P):
                    prev[ns]=(s,i,delta); q.append(ns)
    return found, prev

def plan(start=(23,5,5,5), mins=(2,2,2,2), target=(31,52), maxv=47):
    P=purple_set()
    f,prev=bfs2(start,list(mins),P,maxv,target)
    if f is None: return None,None
    path=[]; s=f
    while prev[s] is not None:
        p,i,d=prev[s]; path.append((i,d)); s=p
    path.reverse()
    return f,path

def summarize(d):
    out=[]
    for k in ('moved','rotated','appeared','disappeared','changed_color','resized'):
        for e in d.get(k,[]):
            if k=='moved': out.append('mv %s %s->%s'%(e['color'],e.get('at') or e['from'],e['to']))
            elif k in ('appeared','disappeared'): out.append('%s %s %s'%(k[:3],e['color'],e['bbox']))
            elif k=='resized': out.append('rs %s %s %s->%s'%(e['color'],e['at_before'],e['before_pixels'],e['after_pixels']))
            else: out.append('%s %s'%(k,e))
    return out

def blocks(st):
    Lb,Lr,Ln,Lo,Ls,Lp=st
    B={}
    def h(r,c0,c1): return {(r+dr,c) for dr in (-1,0,1) for c in range(c0,c1+1)}
    def v(c,r0,r1): return {(r,c+dc) for dc in (-1,0,1) for r in range(r0,r1+1)}
    B['b']=v(31,37-Lb+1,37)                      # b: base col31 grows left -> cols 32-Lb..31, rows36-38
    B['b']=h(37,32-Lb,31)
    B['R']=v(30-Lb,38-Lr,37)
    B['wh']=h(36-Lr,30-Lb,29-Lb+Lwh-1) if False else h(36-Lr,30-Lb,30-Lb+23-1)
    B['N']=v(55,26-Ln,25)
    B['O']=h(24-Ln,56-Lo,55)
    B['wv']=v(54-Lo,24-Ln,24-Ln+17-1)
    B['S']=h(28,7,6+Ls)
    B['P']=v(49,4,3+Lp)
    return B

def validP(st):
    return valid(st,set(PB))

def blocks2(st):
    Lb,Lr,Ln,Lo,Ls,Lp=st
    def h(r,c0,c1): return frozenset((r+dr,c) for dr in (-1,0,1) for c in range(c0,c1+1))
    def v(c,r0,r1): return frozenset((r,c+dc) for dc in (-1,0,1) for r in range(r0,r1+1))
    B={}
    B['b']=h(37,32-Lb,31); B['bcap']=v(32,36,38)
    B['R']=v(30-Lb,38-Lr,37); B['Rcap']=h(38,27-Lb,29-Lb)
    B['wh']=h(36-Lr,30-Lb,52-Lb)
    B['N']=v(55,26-Ln,25); B['Ncap']=h(26,54,56)
    B['O']=h(24-Ln,56-Lo,55); B['Ocap']=v(56,23-Ln,25-Ln)
    B['wv']=v(54-Lo,24-Ln,40-Ln)
    B['S']=h(28,7,6+Ls); B['Scap']=v(6,27,29)
    B['P']=v(49,4,3+Lp); B['Pcap']=h(3,48,50)
    return B

def validQ(st):
    B=blocks2(st); seen=set()
    for k,vv in B.items():
        if vv & seen: return False
        seen|=vv
        if any(r<0 or r>44 or c<0 or c>63 for (r,c) in vv): return False
    if seen & set(PB): return False
    return True

def blk3(st):
    Lb,Lr,Ln,Lo,Ls,Lp=st
    def H(r,c0,c1): return frozenset((r+dr,c) for dr in (-1,0,1) for c in range(c0,c1+1))
    def V(c,r0,r1): return frozenset((r,c+dc) for dc in (-1,0,1) for r in range(r0,r1+1))
    def HB(r,c0,c1): return frozenset((r,c) for c in range(c0,c1+1))
    def VB(c,r0,r1): return frozenset((r,c) for r in range(r0,r1+1))
    return {'b':H(37,32-Lb,31),'bcap':VB(32,36,38),
            'R':V(30-Lb,38-Lr,37),'Rcap':HB(38,27-Lb,29-Lb),
            'wh':H(36-Lr,30-Lb,52-Lb),
            'N':V(55,26-Ln,25),'Ncap':HB(26,54,56),
            'O':H(24-Ln,56-Lo,55),'Ocap':VB(56,23-Ln,25-Ln),
            'wv':V(54-Lo,24-Ln,40-Ln),
            'S':H(28,7,6+Ls),'Scap':VB(6,27,29),
            'P':V(49,4,3+Lp),'Pcap':HB(3,48,50)}

def validR(st,PB=PB):
    seen=set()
    for k,vv in blk3(st).items():
        if vv & seen: return False
        seen|=vv
        if any(r<0 or r>44 or c<0 or c>63 for (r,c) in vv): return False
    return not (seen & set(PB))

def validS(st):
    seen=set()
    for k,vv in blk3(st).items():
        if vv & seen: return False
        seen|=vv
        if any(r<0 or r>44 or c<0 or c>63 for (r,c) in vv): return False
    return not (seen & PURP)

def plan3(cur=(5,23,5,20), goal=(38,26), small=(5,5)):
    from collections import deque
    vals=list(range(2,45,3))
    C=set()
    for Lb in vals:
        for Lr in vals:
            for Ln in vals:
                for Lo in vals:
                    if validS((Lb,Lr,Ln,Lo,goal[0],goal[1])): C.add((Lb,Lr,Ln,Lo))
    start=cur
    prev={start:None}; q=deque([start]); found=None
    while q:
        s=q.popleft()
        if s in C: found=s; break
        for i in range(4):
            for d in (+3,-3):
                ns=list(s); ns[i]+=d; ns=tuple(ns)
                if ns[i]<2 or ns[i]>44 or ns in prev: continue
                if not validS(ns+(small[0],small[1])): continue
                prev[ns]=(s,i,d); q.append(ns)
    if found is None: return None
    path=[]; s=found
    while prev[s] is not None:
        p,i,d=prev[s]; path.append((i,d)); s=p
    path.reverse()
    return found,path,len(C)

def validT(st):
    PUR=frozenset((r,c) for r in (27,28,29) for c in (45,46,47))
    seen=set()
    for k,vv in blk3(st).items():
        if vv & seen: return False
        seen|=vv
        if any(r<0 or r>44 or c<0 or c>63 for (r,c) in vv): return False
    return not (seen & PUR)

def plan4(cur=(5,23,5,20), goal=(38,26), small=(5,5), vmax=44):
    from collections import deque
    vals=[v for v in range(2,vmax+1,3)]
    C=set()
    for Lb in vals:
        for Lr in vals:
            for Ln in vals:
                for Lo in vals:
                    if validT((Lb,Lr,Ln,Lo,goal[0],goal[1])): C.add((Lb,Lr,Ln,Lo))
    prev={cur:None}; q=deque([cur]); found=None
    while q:
        s=q.popleft()
        if s in C: found=s; break
        for i in range(4):
            for d in (+3,-3):
                ns=list(s); ns[i]+=d; ns=tuple(ns)
                if ns[i]<2 or ns[i]>vmax or ns in prev: continue
                if not validT(ns+(small[0],small[1])): continue
                prev[ns]=(s,i,d); q.append(ns)
    if found is None: return (None,None,len(C))
    path=[]; s=found
    while prev[s] is not None:
        p,i,d=prev[s]; path.append((i,d)); s=p
    path.reverse()
    return (found,path,len(C))

def run(steps, st):
    log=[]
    st=list(st)
    for (i,d) in steps:
        r_,c_=CLICK[(NAMES[i],d)]
        res=action([{'action':'MOUSE','row':r_,'col':c_}])
        if res.get('no_op'):
            log.append((NAMES[i],d,'NOOP')); break
        st[i]+=d
        log.append((NAMES[i],d,'ok'))
        if res.get('level_completed') or res.get('game_over'):
            log.append(('END',res.get('level_completed'),res.get('game_over'))); break
    return tuple(st),log

def blk4(st):
    LG,LB,LR,LO,LY=st
    def H(r,c0,c1): return frozenset((r+dr,c) for dr in (-1,0,1) for c in range(c0,c1+1))
    def V(c,r0,r1): return frozenset((r,c+dc) for dc in (-1,0,1) for r in range(r0,r1+1))
    def HB(r,c0,c1): return frozenset((r,c) for c in range(c0,c1+1))
    def VB(c,r0,r1): return frozenset((r,c) for r in range(r0,r1+1))
    B={}
    B['G']=V(10,20-LG,19); B['Gcap']=HB(20,9,11)
    B['Y9']=H(18-LG,10,9+LY); B['Y9cap']=VB(9,17-LG,19-LG)
    B['B']=V(52,17-LB,16); B['Bcap']=HB(17,51,53)
    B['Y6']=H(15-LB,53-LY,52); B['Y6cap']=VB(53,14-LB,16-LB)
    B['R']=V(46,23-LR,22); B['Rcap']=HB(23,45,47)
    B['Y13']=H(21-LR,47-LY,46); B['Y13cap']=VB(47,20-LR,22-LR)
    B['O']=V(19,53-LO,52); B['Ocap']=HB(53,18,20)
    B['Y19']=H(51-LO,19,18+LY); B['Y19cap']=VB(18,50-LO,52-LO)
    B['Y21']=V(31,44-LY,43); B['Y21cap']=HB(44,30,32)
    return B

def valid4(st):
    B=blk4(st); seen=set()
    for k,vv in B.items():
        if vv & seen: return False
        seen|=vv
        for (r,c) in vv:
            if r<0 or r>62 or c<0 or c>62: return False
            for (pr0,pr1,pc0,pc1) in PANELS:
                if pr0<=r<=pr1 and pc0<=c<=pc1: return False
    return True

def rects(st):
    LG,LB,LR,LO,LY=st
    R=[]
    def V(c,r0,r1): R.append((r0,r1,c-1,c+1))
    def H(r,c0,c1): R.append((r-1,r+1,c0,c1))
    def HB(r,c0,c1): R.append((r,r,c0,c1))
    def VB(c,r0,r1): R.append((r0,r1,c,c))
    V(10,20-LG,19); HB(20,9,11)
    H(18-LG,10,9+LY); VB(9,17-LG,19-LG)
    V(52,17-LB,16); HB(17,51,53)
    H(15-LB,53-LY,52); VB(53,14-LB,16-LB)
    V(46,23-LR,22); HB(23,45,47)
    H(21-LR,47-LY,46); VB(47,20-LR,22-LR)
    V(19,53-LO,52); HB(53,18,20)
    H(51-LO,19,18+LY); VB(18,50-LO,52-LO)
    V(31,44-LY,43); HB(44,30,32)
    return R

def ok4(st, PAN=((45,51,3,15),(45,51,48,60),(51,57,25,37),(54,60,3,15),(54,60,48,60))):
    R=rects(st); n=len(R)
    for i in range(n):
        a=R[i]
        if a[0]<0 or a[2]<0 or a[1]>62 or a[3]>62: return False
        for p in PAN:
            if not (a[1]<p[0] or a[0]>p[1] or a[3]<p[2] or a[2]>p[3]): return False
        for j in range(i+1,n):
            b=R[j]
            if not (a[1]<b[0] or a[0]>b[1] or a[3]<b[2] or a[2]>b[3]): return False
    return True

def ok5(st, PAN=((45,51,3,15),(45,51,48,60),(51,57,25,37),(54,60,3,15),(54,60,48,60))):
    # rects with color labels; allow overlap only between two YELLOW rects
    LG,LB,LR,LO,LY=st
    R=[]
    def V(c,r0,r1,col): R.append(((r0,r1,c-1,c+1),col))
    def H(r,c0,c1,col): R.append(((r-1,r+1,c0,c1),col))
    def HB(r,c0,c1,col): R.append(((r,r,c0,c1),col))
    def VB(c,r0,r1,col): R.append(((r0,r1,c,c),col))
    V(10,20-LG,19,'N'); HB(20,9,11,'G')
    H(18-LG,10,9+LY,'Y'); VB(9,17-LG,19-LG,'G')
    V(52,17-LB,16,'b'); HB(17,51,53,'G')
    H(15-LB,53-LY,52,'Y'); VB(53,14-LB,16-LB,'G')
    V(46,23-LR,22,'R'); HB(23,45,47,'G')
    H(21-LR,47-LY,46,'Y'); VB(47,20-LR,22-LR,'G')
    V(19,53-LO,52,'O'); HB(53,18,20,'G')
    H(51-LO,19,18+LY,'Y'); VB(18,50-LO,52-LO,'G')
    V(31,44-LY,43,'Y'); HB(44,30,32,'G')
    n=len(R)
    for i in range(n):
        a,ca=R[i]
        if a[0]<0 or a[2]<0 or a[1]>62 or a[3]>62: return False
        for p in PAN:
            if not (a[1]<p[0] or a[0]>p[1] or a[3]<p[2] or a[2]>p[3]): return False
        for j in range(i+1,n):
            b,cb=R[j]
            if ca=='Y' and cb=='Y': continue
            if not (a[1]<b[0] or a[0]>b[1] or a[3]<b[2] or a[2]>b[3]): return False
    return True

def maxLY(cfg):
    best=2
    for LY in range(2,47,3):
        if ok4(cfg+(LY,)): best=LY
        else: break
    return best

def okNP(st): return ok4(st, PAN=())

def panelhit(st):
    out=[]
    for a in rects(st):
        for p in ((45,51,3,15),(45,51,48,60),(51,57,25,37),(54,60,3,15),(54,60,48,60)):
            if not (a[1]<p[0] or a[0]>p[1] or a[3]<p[2] or a[2]>p[3]): out.append((a,p))
    return out

def rects5(st):
    LO,LN,Lb=st
    R=[]
    def H(r,c0,c1): R.append((r-1,r+1,c0,c1))
    def HB(r,c0,c1): R.append((r,r,c0,c1))
    def VB(c,r0,r1): R.append((r0,r1,c,c))
    HB(6,9,11); R.append((7,6+LO,9,11))
    HB(6,42,44); R.append((7,6+LO,42,44))
    VB(9,7+LO,9+LO); H(8+LO,10,9+LN)
    HB(35,54,56); R.append((35-LN,34,54,56))
    VB(56,32-LN,34-LN); R.append((32-LN,34-LN,56-Lb,55))
    R.append((32-LN,49-LN,53-Lb,55-Lb))
    VB(21,24,26); H(25,22,38)
    return R

def ok5b(st, PUR=((9,17,15,17),(21,26,15,17),(36,38,36,38))):
    R=rects5(st); n=len(R)
    for i in range(n):
        a=R[i]
        if a[0]<0 or a[2]<0 or a[1]>62 or a[3]>62: return False
        for p in PUR:
            if not (a[1]<p[0] or a[0]>p[1] or a[3]<p[2] or a[2]>p[3]): return False
        for j in range(i+1,n):
            b=R[j]
            if not (a[1]<b[0] or a[0]>b[1] or a[3]<b[2] or a[2]>b[3]): return False
    return True

def rects5b(st):
    LO,LN,Lb,Ls=st
    R=[]
    R.append((6,6,9,11)); R.append((7,6+LO,9,11))
    R.append((6,6,42,44)); R.append((7,6+LO,42,44))
    R.append((7+LO,9+LO,9,9)); R.append((7+LO,9+LO,10,9+LN))
    R.append((35,35,54,56)); R.append((35-LN,34,54,56))
    R.append((32-LN,34-LN,56,56)); R.append((32-LN,34-LN,56-Lb,55))
    R.append((32-LN,49-LN,53-Lb,55-Lb))
    R.append((24,26,21,21)); R.append((24,26,22,21+Ls))
    return R

def ok5c(st):
    R=rects5b(st); n=len(R)
    for i in range(n):
        a=R[i]
        if a[0]<0 or a[2]<0 or a[1]>62 or a[3]>62: return False
        for p in PUR:
            if not (a[1]<p[0] or a[0]>p[1] or a[3]<p[2] or a[2]>p[3]): return False
        for j in range(i+1,n):
            b=R[j]
            if not (a[1]<b[0] or a[0]>b[1] or a[3]<b[2] or a[2]>b[3]): return False
    return True

def blocks6(st, DIR=((-1,0),(0,1),(1,0),(0,-1)), G=(16,40)):
    LN,dN,LY,dY,LB,dB=st
    def bar(base,L,d):
        dr,dc=DIR[d]; r,c=base
        cells=[(r+dr*i,c+dc*i) for i in range(L)]
        return (min(x[0] for x in cells),max(x[0] for x in cells),min(x[1] for x in cells),max(x[1] for x in cells))
    def cap(base,d):
        dr,dc=DIR[d]; return bar((base[0]-dr,base[1]-dc),1,(d+1)%4)
    R=[bar(G,LN,dN),cap(G,dN)]
    P=(G[0]+(LN+1)*DIR[dN][0], G[1]+(LN+1)*DIR[dN][1])
    R+= [bar(P,LY,dY),cap(P,dY)]
    Q=(P[0]+(LY+1)*DIR[dY][0], P[1]+(LY+1)*DIR[dY][1])
    R+=[bar(Q,LB,dB),cap(Q,dB)]
    K=(Q[0]+(LB-2)*DIR[dB][0], Q[1]+(LB-2)*DIR[dB][1])
    return R,K

def ok6(st, PURW=((27,29,3,8),(27,29,18,59),(30,38,3,5),(30,38,57,59),(39,41,3,5),(39,41,45,59))):
    R,K=blocks6(st); n=len(R)
    for i in range(n):
        a=R[i]
        if a[0]<0 or a[2]<0 or a[1]>62 or a[3]>63: return False
        for p in PURW:
            if not (a[1]<p[0] or a[0]>p[1] or a[3]<p[2] or a[2]>p[3]): return False
        for j in range(i+1,n):
            b=R[j]
            if not (a[1]<b[0] or a[0]>b[1] or a[3]<b[2] or a[2]>b[3]): return False
    return True

def ok6b(st, PURW=((27,29,3,8),(27,29,18,59),(30,38,3,5),(30,38,57,59),(39,41,3,5),(39,41,45,59))):
    R,K=blocks6(st); n=len(R)
    for i in range(n):
        a=R[i]
        for p in PURW:
            if not (a[1]<p[0] or a[0]>p[1] or a[3]<p[2] or a[2]>p[3]): return False
        for j in range(i+1,n):
            b=R[j]
            if not (a[1]<b[0] or a[0]>b[1] or a[3]<b[2] or a[2]>b[3]): return False
    return True

def onb(st):
    R,K=blocks6(st)
    return all(a[0]>=0 and a[2]>=0 and a[1]<=62 and a[3]<=63 for a in R)

def sim(st0,plan):
    st=list(st0); out=[]
    for k in plan:
        i=IDX[k[0]]
        if k.endswith('r'): st[i+1]=(st[i+1]+3)%4
        else: st[i]+= 3 if k.endswith('+') else -3
        out.append(tuple(st))
    return out

def sim2(st0,plan):
    IDX={'N':0,'Y':2,'B':4}; st=list(st0); out=[]
    for k in plan:
        i=IDX[k[0]]
        if k.endswith('r'): st[i+1]=(st[i+1]+3)%4
        else: st[i]+= 3 if k.endswith('+') else -3
        out.append(tuple(st))
    return out

def readstate(fr):
    seg=fr.segmentation; d={}
    for n in seg['nodes']:
        if n['color'] in ('N','Y','b','G') and 1<n['pixels']<200:
            b=n['boundary']; r0=min(p[0] for p in b); r1=max(p[0] for p in b); c0=min(p[1] for p in b); c1=max(p[1] for p in b)
            if r0>44: continue
            d.setdefault(n['color'],[]).append((n['pixels'],(r0,c0,r1,c1)))
    return d

def derive(fr):
    A=fr.ascii.split('\n')
    def ch(r,c): return A[r][c] if 0<=r<64 and 0<=c<64 else '?'
    G=set((r,c) for r in range(45) for c in range(64) if ch(r,c)=='G')
    col={}
    for x in 'YNbO': col[x]=set((r,c) for r in range(45) for c in range(64) if ch(r,c)==x)
    col['O']|=set((r,c) for r in range(45) for c in range(64) if ch(r,c)=='r')
    st=[]; base=(13,49)
    for idx,x in enumerate('YNbO'):
        d=None
        for k,(dr,dc) in enumerate(DIR):
            p=(base[0]-dr,base[1]-dc)
            if p in G:
                if dc==0:
                    if (p[0],p[1]-1) in G and (p[0],p[1]+1) in G: d=k;break
                else:
                    if (p[0]-1,p[1]) in G and (p[0]+1,p[1]) in G: d=k;break
        if d is None: return ('ERRcap',idx,base)
        dr,dc=DIR[d]; cs=col[x]; L=0
        while (base[0]+dr*L,base[1]+dc*L) in cs: L+=1
        if L<2: return ('ERRlen',idx,base,d,L)
        st+=[L,d]; base=(base[0]+dr*(L+1),base[1]+dc*(L+1))
    return tuple(st)

def dirdisp(cap,base):
    dr=base[0]-cap[0]; dc=base[1]-cap[1]
    if dr==0 and dc!=0: return DIRN[(0,dc//abs(dc))]
    if dc==0 and dr!=0: return DIRN[(dr//abs(dr),0)]
    return '?'

def cfg_ok(st, require_onb_i=None):
    R,K=blocks6(st); n=len(R)
    for i in range(n):
        a=R[i]
        for p in ((27,29,3,8),(27,29,18,59),(30,38,3,5),(30,38,57,59),(39,41,3,5),(39,41,45,59)):
            if not (a[1]<p[0] or a[0]>p[1] or a[3]<p[2] or a[2]>p[3]): return False
        for j in range(i+1,n):
            b=R[j]
            if not (a[1]<b[0] or a[0]>b[1] or a[3]<b[2] or a[2]>b[3]): return False
    return True

def onb_i(st,i):
    R,_=blocks6(st)
    for a in (R[2*i],R[2*i+1]):
        if a[0]<0 or a[2]<0 or a[1]>62 or a[3]>63: return False
    return True

def grow_ok(st,i,nv):
    t=list(st); t[2*i]=nv; t=tuple(t)
    return cfg_ok(t) and onb_i(t,i)

def rot_target(st,i):
    for k in (1,2,3):
        t=list(st)
        for j in range(i,3): t[2*j+1]=(t[2*j+1]+3*k)%4
        t=tuple(t)
        if cfg_ok(t) and onb_i(t,i): return k,t
    return None,None

def chain(fr):
    seg=fr.segmentation; bars={}; caps=[]; knobs=[]
    for n in seg['nodes']:
        if n['pixels']>2000: continue
        b=n['boundary']; r0=min(p[0] for p in b); r1=max(p[0] for p in b); c0=min(p[1] for p in b); c1=max(p[1] for p in b)
        if r0>44: continue
        if n['color'] in ('Y','N','b','O','S','R'): bars.setdefault(n['color'],[]).append((r0,c0,r1,c1,n['pixels']))
        elif n['color']=='G': caps.append((r0,c0,r1,c1))
        elif n['color']=='r': knobs.append((r0,c0,r1,c1,n['pixels']))
    return bars,caps,knobs

def chain_blocks(st):
    LY,dY,LG,dG,Lb,db,LO,dO=st
    base=(13,49); out=[]; knob=None
    for idx,(L,d) in enumerate(((LY,dY),(LG,dG),(Lb,db),(LO,dO))):
        dr,dc=DIR[d]
        bar=set((base[0]+dr*k, base[1]+dc*k) for k in range(L))
        cap=(base[0]-dr, base[1]-dc)
        out.append((bar,cap))
        if idx==3: knob=(base[0]+dr*(L-2), base[1]+dc*(L-2))
        base=(base[0]+dr*(L+1), base[1]+dc*(L+1))
    return out,knob

def ok(st):
    allc,K=cells3(st); seen=set()
    for cell in allc:
        if cell in OBS or cell in seen: return False
        if not(0<=cell[0]<=44 and 0<=cell[1]<=63): return False
        seen.add(cell)
    return True

def okf(st, onb=True):
    B,K=chain_blocks(st); seen=set()
    for bar,cap in B:
        for cell in bar|{cap}:
            if cell in OBSS: return False
            if onb and not (0<=cell[0]<=44 and 0<=cell[1]<=63): return False
            if cell in seen: return False
            seen.add(cell)
    return True

def rot_t(st,i):
    for k in (1,2,3):
        t=list(st)
        for j in range(i,4): t[2*j+1]=(t[2*j+1]+3*k)%4
        t=tuple(t)
        if okf(t): return k,t
    return None,None

def mkobs(fr):
    A=fr.ascii.split('\n'); s=set()
    for r in range(0,45):
        for c,ch in enumerate(A[r]):
            if ch=='p': s.add((r,c))
    for r in range(7,21):
        for c in range(12,15): s.add((r,c))
    for r in range(6,9):
        for c in range(21,35): s.add((r,c))
    return frozenset(s)

def okc(st,OBS,onb=True):
    B,K=chain_blocks(st); seen=set()
    for bar,cap in B:
        for cell in bar|{cap}:
            if cell in OBS: return False
            if onb and not (0<=cell[0]<=44 and 0<=cell[1]<=63): return False
            if cell in seen: return False
            seen.add(cell)
    return True

def rotc(st,i,OBS):
    for k in (1,2,3):
        t=list(st)
        for j in range(i,4): t[2*j+1]=(t[2*j+1]+3*k)%4
        t=tuple(t)
        if okc(t,OBS): return k,t
    return None,None

def okc2(st): return okc(st,OBS,False)

def rotc2(st,i):
    for k in (1,2,3):
        t=list(st)
        for j in range(i,4): t[2*j+1]=(t[2*j+1]+3*k)%4
        t=tuple(t)
        if okc2(t): return k,t
    return None,None

def rotw(st,i):
    t=list(st)
    for j in range(i,min(i+3,4)): t[2*j+1]=(t[2*j+1]+3)%4
    t=tuple(t)
    return t if okc(t,OBS) else None

def rot1(st,i):
    t=list(st)
    for j in range(i,4): t[2*j+1]=(t[2*j+1]+3)%4
    t=tuple(t)
    return t if okc(t,OBS) else None

def rot1f(st,i):
    t=list(st)
    for j in range(i,4): t[2*j+1]=(t[2*j+1]+3)%4
    t=tuple(t)
    return t if okc(t,OBS) else None

def okfree(st):
    B,K=chain_blocks(st); seen=set()
    for bar,cap in B:
        for cell in bar|{cap}:
            if cell in OBS or cell in seen: return False
            seen.add(cell)
    return True

def rotccw(st,i):
    t=list(st)
    for j in range(i,4): t[2*j+1]=(t[2*j+1]+3)%4
    return tuple(t)

def h(st):
    K=chain_blocks(st)[1]; return abs(K[0]-T[0])+abs(K[1]-T[1])

def cells(st):
    base=(13,49); allc=[]; knob=None
    for idx in range(4):
        L,d=st[2*idx],st[2*idx+1]; dr,dc=DIR[d]
        for k in range(L): allc.append((base[0]+dr*k, base[1]+dc*k))
        allc.append((base[0]-dr,base[1]-dc))
        if idx==3: knob=(base[0]+dr*(L-2),base[1]+dc*(L-2))
        base=(base[0]+dr*(L+1),base[1]+dc*(L+1))
    return allc,knob

def cluster(cs):
    cs=set(cs); out=[]
    while cs:
        s=cs.pop(); comp={s}; st=[s]
        while st:
            r,c=st.pop()
            for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
                p=(r+dr,c+dc)
                if p in cs: cs.remove(p); comp.add(p); st.append(p)
        out.append(comp)
    return out

def cells_of(st):
    B,K=chain_blocks(st); s=set()
    for bar,cap in B: s|=bar|{cap}
    s.add(K); return s

def succ(st):
    out=[]
    for i in range(4):
        v=st[2*i]
        for nv in (v+3,v-3):
            if nv in LS:
                t=list(st); t[2*i]=nv; t=tuple(t)
                if ok(t): out.append((t,('S',i,nv)))
    for i in range(3):
        for k in (1,2):
            t=list(st)
            for j in range(i,4): t[2*j+1]=(t[2*j+1]-k)%4
            t=tuple(t)
            if ok(t): out.append((t,('R',i,k))); break
    return out

def cb(st):
    base=(13,49); out=[]; knob=None
    for idx in range(4):
        L,d=st[2*idx],st[2*idx+1]; dr,dc=DIR[d]
        bar=set((base[0]+dr*k, base[1]+dc*k) for k in range(L))
        out.append((bar,(base[0]-dr,base[1]-dc)))
        if idx==3: knob=(base[0]+dr*(L-2), base[1]+dc*(L-2))
        base=(base[0]+dr*(L+1), base[1]+dc*(L+1))
    return out,knob

def rotR1(st,i):
    t=list(st); changed=[]
    for j in range(i,4):
        t[2*j+1]=(t[2*j+1]+3)%4
        if ok(tuple(t)): changed.append(j)
        else: t[2*j+1]=(st[2*j+1])
    return tuple(t),changed

def derive2(fr):
    A=fr.ascii.split('\n')
    cells={}
    for col in 'YNbOG':
        cells[col]=set((r,c) for r in range(45) for c,ch in enumerate(A[r]) if ch==col)
    st=[]; base=(13,49)
    for idx,col in enumerate('YNbO'):
        cs=cells[col]
        if base not in cs: return ('ERR',idx,base)
        cap=None
        for k,(dr,dc) in enumerate(DIR):
            p=(base[0]-dr,base[1]-dc)
            if p in cells['G']: cap=(k,p); break
        if cap is None: return ('ERRcap',idx,base)
        k,p=cap; d=k; dr,dc=DIR[d]
        L=0
        while (base[0]+dr*L, base[1]+dc*L) in cs: L+=1
        st += [L,d]
        base=(base[0]+dr*(L+1), base[1]+dc*(L+1))
    return tuple(st)

def cbx(st):
    LY,dY,LG,dG,Lb,db,LO,dO=st
    base=(13,49); knob=None
    for idx,(L,d) in enumerate(((LY,dY),(LG,dG),(Lb,db),(LO,dO))):
        dr,dc=DIR[d]
        if idx==3: knob=(base[0]+dr*(L-2), base[1]+dc*(L-2))
        base=(base[0]+dr*(L+1), base[1]+dc*(L+1))
    return knob

def derive3(fr):
    A=fr.ascii.split('\n')
    def cs_for(col): return set((r,c) for r in range(45) for c,ch in enumerate(A[r]) if ch==col)
    O=cs_for('O')|cs_for('r')
    sets={'Y':cs_for('Y'),'N':cs_for('N'),'b':cs_for('b'),'O':O}
    st=[]; base=(13,49); dbg=[]
    for idx,col in enumerate('YNbO'):
        cs=sets[col]; best=None
        for k,(dr,dc) in enumerate(DIR):
            n=0
            while (base[0]+dr*n, base[1]+dc*n) in cs: n+=1
            dbg.append((idx,k,n))
            if best is None or n>best[1]: best=(k,n)
        if best[1]<2: return ('ERR',idx,base,dbg)
        st += [best[1],best[0]]
        dr,dc=DIR[best[0]]; L=best[1]
        base=(base[0]+dr*(L+1), base[1]+dc*(L+1))
    return tuple(st)

def knob_of(st):
    base=(13,49)
    for idx in range(4):
        L,d=st[2*idx],st[2*idx+1]; dr,dc=DIR[d]
        if idx==3: return (base[0]+dr*(L-2), base[1]+dc*(L-2))
        base=(base[0]+dr*(L+1), base[1]+dc*(L+1))

def okP(st):
    allc,K=cells(st); seen=set()
    for cell in allc:
        if cell in PUR or cell in seen: return False
        if not(0<=cell[0]<=44 and 0<=cell[1]<=63): return False
        seen.add(cell)
    return True

def derive4(fr):
    A=fr.ascii.split('\n')
    G=set((r,c) for r in range(45) for c,ch in enumerate(A[r]) if ch=='G')
    col={}
    for c_ in 'YNbO':
        col[c_]=set((r,c) for r in range(45) for c,ch in enumerate(A[r]) if ch==c_)
    col['O']|=set((r,c) for r in range(45) for c,ch in enumerate(A[r]) if ch=='r')
    st=[]; base=(13,49)
    for idx,c_ in enumerate('YNbO'):
        d=None
        for k,(dr,dc) in enumerate(DIR):
            p=(base[0]-dr,base[1]-dc)
            if p in G:
                if dc==0:
                    if (p[0],p[1]-1) in G and (p[0],p[1]+1) in G: d=k;break
                else:
                    if (p[0]-1,p[1]) in G and (p[0]+1,p[1]) in G: d=k;break
        if d is None: return ('ERRcap',idx,base)
        dr,dc=DIR[d]; cs=col[c_]; L=0
        while (base[0]+dr*L,base[1]+dc*L) in cs: L+=1
        if L<2: return ('ERRlen',idx,base,d,L)
        st+=[L,d]; base=(base[0]+dr*(L+1),base[1]+dc*(L+1))
    return tuple(st)

def kn(st):
    base=(13,49)
    for i in range(4):
        L,d=st[2*i],st[2*i+1]; dr,dc=DIR[d]
        if i==3: return (base[0]+dr*(L-2),base[1]+dc*(L-2))
        base=(base[0]+dr*(L+1),base[1]+dc*(L+1))

def okN(st):
    allc,K=cells(st); seen=set()
    for cell in allc:
        if cell in seen: return False
        if not(0<=cell[0]<=44 and 0<=cell[1]<=63): return False
        seen.add(cell)
    return True

def succN(st):
    out=[]
    for i in range(4):
        v=st[2*i]
        for nv in (v+3,v-3):
            if nv in LS:
                t=list(st); t[2*i]=nv; t=tuple(t)
                if okN(t): out.append((t,('SIZE',i,nv)))
    for i in range(3):
        t=list(st)
        for j in range(i,4): t[2*j+1]=(t[2*j+1]+3)%4
        t=tuple(t)
        if okN(t): out.append((t,('ROT',i)))
    return out

def bar(base,L,d):
    dr,dc=DIR[d]; return [(base[0]+dr*k, base[1]+dc*k) for k in range(L)]

def snap(fr):
    A=fr.ascii.split('\n')
    return A

def mkok(caps):
    def ok(st):
        allc,K=cells(st,caps); seen=set()
        for cell in allc:
            if cell in PUR or cell in seen: return False
            if not(0<=cell[0]<=44 and 0<=cell[1]<=63): return False
            seen.add(cell)
        return True
    return ok

def mksucc(ok):
    def succ(st):
        out=[]
        for i in range(4):
            v=st[2*i]
            for nv in (v+3,v-3):
                if nv in LS:
                    t=list(st); t[2*i]=nv; t=tuple(t)
                    if ok(t): out.append(t)
        for i in range(3):
            t=list(st)
            for j in range(i,4): t[2*j+1]=(t[2*j+1]+3)%4
            t=tuple(t)
            if ok(t): out.append(t)
        return out
    return succ

def comp_of(s0,ok):
    seen={s0}; q=deque([s0])
    while q:
        s=q.popleft()
        for t in mksucc(ok)(s):
            if t not in seen: seen.add(t); q.append(t)
    return seen

def all_succ(st):
    out=[]
    for i in range(4):
        v=st[2*i]
        for nv in (v+3,v-3):
            if nv in LS:
                t=list(st); t[2*i]=nv; t=tuple(t); out.append((t,('SIZE',i,nv)))
    for i in range(3):
        t=list(st)
        for j in range(i,4): t[2*j+1]=(t[2*j+1]+3)%4
        out.append((tuple(t),('ROT',i)))
    return out

def bars(a):
    out={}
    for col in 'YNbO':
        cs=[(r,c) for r in range(45) for c in range(64) if a[r][c]==col]
        out[col]=(len(cs),min(r for r,_ in cs),max(r for r,_ in cs),min(c for _,c in cs),max(c for _,c in cs))
    return out

def bars2(a):
    out={}
    for col in 'YNbO':
        cs=[(r,c) for r in range(45) for c in range(64) if a[r][c]==col]
        if cs: out[col]=(len(cs),min(r for r,_ in cs),max(r for r,_ in cs),min(c for _,c in cs),max(c for _,c in cs))
    return out

def rot(st,i,k):
    t=list(st)
    for j in range(i,4): t[2*j+1]=(t[2*j+1]-k)%4
    return tuple(t)

def derive5(fr):
    A=fr.ascii.split('\n')
    def ch(r,c): return A[r][c] if 0<=r<64 and 0<=c<64 else '?'
    G=set((r,c) for r in range(45) for c in range(64) if ch(r,c)=='G')
    col={}
    for x in 'YNbO': col[x]=set((r,c) for r in range(45) for c in range(64) if ch(r,c)==x)
    col['O']|=set((r,c) for r in range(45) for c in range(64) if ch(r,c)=='r')
    st=[]; base=(13,49)
    for idx,x in enumerate('YNbO'):
        d=None
        for k,(dr,dc) in enumerate(DIR):
            p=(base[0]-dr,base[1]-dc)
            if p in G:
                if dc==0:
                    if (p[0],p[1]-1) in G and (p[0],p[1]+1) in G: d=k;break
                else:
                    if (p[0]-1,p[1]) in G and (p[0]+1,p[1]) in G: d=k;break
        if d is None: return ('ERRcap',idx,base)
        dr,dc=DIR[d]; cs=col[x]; L=0
        while (base[0]+dr*L,base[1]+dc*L) in cs: L+=1
        if L<2: return ('ERRlen',idx,base,d,L)
        st+=[L,d]; base=(base[0]+dr*(L+1),base[1]+dc*(L+1))
    return tuple(st)

def knob5(st):
    base=(13,49)
    for i in range(4):
        L,d=st[2*i],st[2*i+1]; dr,dc=DIR[d]
        if i==3: return (base[0]+dr*(L-2),base[1]+dc*(L-2))
        base=(base[0]+dr*(L+1),base[1]+dc*(L+1))

def _noop(): pass

def cells3(st):
    base=(13,49); allc=[]; knobp=None
    for idx in range(4):
        L,d=st[2*idx],st[2*idx+1]; dr,dc=DIR[d]; pr=(-dc,dr)
        for k in range(L):
            c0=(base[0]+dr*k, base[1]+dc*k)
            for o in (-1,0,1): allc.append((c0[0]+pr[0]*o, c0[1]+pr[1]*o))
        cp=(base[0]-dr,base[1]-dc)
        for o in (-1,0,1): allc.append((cp[0]+pr[0]*o, cp[1]+pr[1]*o))
        if idx==3: knobp=(base[0]+dr*(L-2),base[1]+dc*(L-2))
        base=(base[0]+dr*(L+1),base[1]+dc*(L+1))
    return allc,knobp

def clusters(cs):
    cs=set(cs); out=[]
    while cs:
        s=cs.pop(); comp={s}; st=[s]
        while st:
            r,c=st.pop()
            for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
                p=(r+dr,c+dc)
                if p in cs: cs.remove(p); comp.add(p); st.append(p)
        out.append(comp)
    return out

def rcells(a): return sorted((r,c) for r in range(45) for c in range(64) if a[r][c]=='r')
