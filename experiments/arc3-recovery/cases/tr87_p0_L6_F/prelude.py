try:
    import json
except Exception:
    pass
try:
    import copy
except Exception:
    pass
try:
    import itertools
except Exception:
    pass
try:
    from collections import Counter
except Exception:
    pass
try:
    import itertools, json
except Exception:
    pass
try:
    import time
except Exception:
    pass
try:
    import builtins
except Exception:
    pass

def crop(fr, r0,r1,c0,c1):
    lines = fr.ascii.split('\n')
    return '\n'.join(l[c0:c1] for l in lines[r0:r1])

def glyph(fr, r0, c0):
    lines = fr.ascii.split('\n')
    return [lines[r][c0:c0+5] for r in range(r0, r0+5)]

def show(g):
    return '\n'.join(g)

def pink4(fr): return glyph(fr,52,43)

def parse(fr):
    A=fr.ascii.split('\n')
    def cell(r0,c0): return [[1 if A[r0+1+i][c0+1+j]=='B' else 0 for j in range(5)] for i in range(5)]
    tops=[];bots=[]
    for n in fr.segmentation['nodes']:
        if n['color'] in 'SP':
            r0,c0,r1,c1=bb(n)
            if r1-r0==6:
                (tops if r0<34 else bots).append((r0,c0,c1,n['color'],(c1-c0+1)//7))
    tops.sort();bots.sort()
    bands={}
    for (r0,c0,c1,col,n) in tops: bands.setdefault(r0,[]).append((c0,col,n))
    pairs=[]
    for r0 in sorted(bands):
        lst=sorted(bands[r0])
        for i in range(0,len(lst),2):
            (k0,kc,kn)=lst[i]; (v0,vc,vn)=lst[i+1]
            pairs.append(([canon(cell(r0,k0+7*j)) for j in range(kn)],[canon(cell(r0,v0+7*j)) for j in range(vn)]))
    rows={}
    for (r0,c0,c1,col,n) in bots:
        rows[r0]=[canon(cell(r0,c0+7*j)) for j in range(n)]
    return pairs,rows

def tostr(m): return '\n'.join(''.join('#' if x else '.' for x in row) for row in m)

def rot90(m): return [[m[len(m)-1-j][i] for j in range(len(m))] for i in range(len(m[0]))]

def flipH(g): return [r[::-1] for r in g]

def flipV(g): return g[::-1]

def transp(g): return [[g[c][r] for c in range(5)] for r in range(5)]

def comp(g): return [[1-v for v in r] for r in g]

def bits(g):
    v=0
    for r in range(5):
        for c in range(5):
            v=(v<<1)|g[r][c]
    return v

def frombits(v):
    g=[[0]*5 for _ in range(5)]
    for i in range(25): g[i//5][i%5]=(v>>(24-i))&1
    return g

def get_state(fr=None):
    fr = fr or current_frame
    st = {}
    st['topS']=[]; st['topP']=[]
    for r0 in (5,14,23):
        for (cl,cr) in ((13,23),(36,46)):
            st['topS'].append(parse(glyph(fr,r0,cl)))
            st['topP'].append(parse(glyph(fr,r0,cr)))
    st['sky']=[parse(glyph(fr,41,c)) for c in (15,22,29,36,43)]
    st['pink']=[parse(glyph(fr,52,c)) for c in (15,22,29,36,43)]
    return st

def ham(a,b): return sum(a[r][c]!=b[r][c] for r in range(5) for c in range(5))

rot270 = lambda g: rot90(rot90(rot90(g)))

R270 = lambda g: rot90(rot90(rot90(g)))

def dihmap(x, bank):
    out=[]
    for j,t in enumerate(bank):
        for n,f in {'id':lambda g:g,'r90':rot90,'r180':lambda g:rot90(rot90(g)),'r270':lambda g:rot90(rot90(rot90(g))),
                    'flipH':flipH,'flipV':flipV,'T':transp,'AT':lambda g:flipH(transp(g))}.items():
            if f(x)==t: out.append((j,n))
    return out

def bidx(v,bank):
    return [i for i,b in enumerate(bank) if b==v]

def pg(s): return [[1 if ch=='#' else 0 for ch in row] for row in s.split('|')]

def cursor_slot(fr):
    lines=fr.ascii.split('\n')
    for r in (48,59):
        row=lines[r]
        cs=[c for c,ch in enumerate(row) if ch=='W']
        if cs: return min(cs), max(cs)
    return None

def cls(v, topP):
    m=dihmap(v,topP)
    return m[0][1-1] if m else 'X'

def klass(v, topP):
    m=dihmap(v,topP)
    return m[0][0] if m else 'X'

def cur_class(i): return klass(parse(glyph(current_frame,52,cols[i])), topP)

def bbox_of(n):
    pts=n['boundary']
    rs=[p[0] for p in pts]; cs=[p[1] for p in pts]
    return (min(rs),min(cs),max(rs),max(cs))

def cell_glyph(fr, r0, c0):  # 7-wide cell top-left (r0,c0); glyph = inner 5x5
    lines=fr.ascii.split('\n')
    return parse([lines[r][c0+1:c0+6] for r in range(r0+1,r0+6)])

def strip_cells(fr, r0, c0, c1):
    n=(c1-c0+1)//7
    return [cell_glyph(fr,r0,c0+7*i) for i in range(n)]

def allg(fr):
    top=[(4,5,4,15,21),(4,28,4,38,58),(13,5,13,15,28),(13,35,13,45,58),(22,5,22,15,35),(22,42,22,52,58)]
    L=[];R=[]
    for (pr,pc,yr,yc0,yc1) in top:
        L.append(cell_glyph(fr,pr,pc)); R.append(strip_cells(fr,yr,yc0,yc1))
    return L,R,strip_cells(fr,40,18,45),strip_cells(fr,51,7,55)

def row63(fr):
    row=fr.ascii.split('\n')[63]
    return row.count('w')

def lab(v):
    m=dihmap(v,FG)
    if not m: return 'X'
    return 'p%d.%d:%s'%(flat[m[0][0]][0],flat[m[0][0]][1],m[0][1])

def lab2(v,FG,flat):
    m=dihmap(v,FG)
    if not m: return 'X'
    j=flat[m[0][0]][0]; k=flat[m[0][0]][1]
    return 'p%d.%d:%s'%(j,k,m[0][1])

def rotmatch(x,bank):
    out=[]
    for j,t in enumerate(bank):
        for n,f in rots.items():
            if f(x)==t: out.append((j,n))
    return out

def solve2():
    L,R,botP,botY = allg(current_frame)
    flat=[(j,k,g) for j,seq in enumerate(R) for k,g in enumerate(seq)]
    FG=[g for _,_,g in flat]
    r270=lambda g: rot90(rot90(rot90(g))); r180=lambda g: rot90(rot90(g))
    prim={0:r270(FG[0]),1:FG[8],2:FG[9],3:FG[10],4:r180(FG[6]),5:r180(FG[7]),6:r270(FG[11])}
    alts={1:[r180(FG[8])],2:[r180(FG[9])],3:[r180(FG[10])],6:[rot90(FG[11])]}
    log=[]; fails=[]
    def slotval(i): return strip_cells(current_frame,51,7,55)[i]
    def curs():
        c=cursor_slot(current_frame)
        return None if c is None else (c[0]-8)//7
    slot=curs()
    for s in range(7):
        guard=0
        while slot!=s and guard<8:
            r=action(['RIGHT']); slot=(slot+1)%7; guard+=1
            if r.get('level_completed') or r.get('game_over'): return log,fails,('EVENT',r['state'])
        if slotval(s)==prim[s]:
            log.append((s,'ok0')); continue
        found=False
        for p in range(7):
            r=action(['UP'])
            if r.get('level_completed') or r.get('game_over'): return log,fails,('EVENT',r['state'])
            if slotval(s)==prim[s]: found=True; log.append((s,'up%d'%(p+1))); break
        if not found:
            alt_hit=None
            for k,a in enumerate(alts.get(s,[])):
                # do another lap looking for alt
                for p in range(7):
                    r=action(['UP'])
                    if r.get('level_completed') or r.get('game_over'): return log,fails,('EVENT',r['state'])
                    if slotval(s)==a: alt_hit=(k,p+1); break
                if alt_hit: break
            fails.append((s,alt_hit))
            log.append((s,'FAIL',alt_hit))
        if row63(current_frame)<10: return log,fails,('LOWBUDGET',row63(current_frame))
    return log,fails,('DONE',row63(current_frame))

def cyc(s):
    out=[s[0]]
    for v in s[1:]:
        if v!=out[-1]: out.append(v)
    while len(out)>1 and out[-1]==out[0]: out.pop()
    return out

def clsid(v,FG):
    ids=set()
    for f in (lambda g:g, rot90, lambda g:rot90(rot90(g)), lambda g:rot90(rot90(rot90(g)))):
        fv=f(v)
        for idx,t in enumerate(FG):
            if fv==t: ids.add(idx)
    return tuple(sorted(ids))

def unp(s): return [[1 if ch=='#' else 0 for ch in row] for row in s.split('\n')]

def slot_from_bracket(fr):
    c=cursor_slot(fr)
    if c is None: return None
    lo,hi=c
    return (lo-8)/7.0

def fix_run():
    L,R,botP,botY = allg(current_frame)
    FG=[g for j,seq in enumerate(R) for k,g in enumerate(seq)]
    ORDER=[(3,),(9,),(1,4),(0,2),(5,6,7),(8,10),(11,)]
    POS={c:i for i,c in enumerate(ORDER)}
    TARGET={2:(9,),3:(8,10),4:(5,6,7),6:(11,)}
    def cls_of(i): return tuple(clsid(strip_cells(current_frame,51,7,55)[i],FG))
    def slot(): return int(round(slot_from_bracket(current_frame)))
    log=[]; cur=slot()
    for s in [6,4,3,2]:
        g=0
        while cur!=s and g<8:
            act='LEFT' if s<cur else 'RIGHT'
            r=action([act]); cur=cur-1 if act=='LEFT' else cur+1; g+=1
            if r.get('level_completed') or r.get('game_over'): return log,('EVENT',r['state'])
        c0=cls_of(s)
        d=(POS[TARGET[s]]-POS[c0])%7
        if d>3: acts=['DOWN']*(7-d)
        else: acts=['UP']*d
        for a in acts:
            r=action([a])
            if r.get('level_completed') or r.get('game_over'): return log,('EVENT',r['state'])
        ok = 'ok' if cls_of(s)==TARGET[s] else 'MISMATCH'
        log.append((s,c0,'->',cls_of(s),ok))
    return log,('END',row63(current_frame))

def iscell(color,r,c):
    blk=[A[r+i][c:c+5] for i in range(5)]
    vals=set(''.join(blk))
    return (vals <= set('B'+color)) and (color in vals) and ('B' in vals)

def l3(fr):
    A=fr.ascii.split('\n')
    def cell(r0,c0):  # top-left of 5x5 glyph = strip top+1, col+1
        return [[1 if A[r0+1+i][c0+1+j]=='B' else 0 for j in range(5)] for i in range(5)]
    strips=[(4,5,11,'Y',1),(4,15,21,'S',1),(4,28,41,'Y',2),(4,45,58,'S',2),
            (13,5,11,'Y',1),(13,15,28,'S',2),(13,35,48,'Y',2),(13,52,58,'S',1),
            (22,5,25,'Y',3),(22,29,35,'S',1),(22,42,48,'Y',1),(22,52,58,'S',1),
            (40,4,59,'Y',8),(51,7,55,'S',7)]
    out={}
    for (r0,c0,c1,col,n) in strips:
        out.setdefault(r0,[]).append((c0,col,[cell(r0,c0+7*i) for i in range(n)]))
    return out

def rotset(v):
    return set()

def canon(m):
    c=m
    best=tostr(c)
    for _ in range(3):
        c=rot90(c); s=tostr(c)
        if s<best: best=s
    return best

def cid(s):
    k=canon(s)
    if k not in ids: ids[k]=len(ids)
    return ids[k]

def bots_classes(fr):
    return [cid(v) for v in l3(fr)[51][0][2]]

def l3sol():
    ROF=[lambda x:x, rot90, lambda x:rot90(rot90(x)), lambda x:rot90(rot90(rot90(x)))]
    IDS={}
    def canon(v): return min([tostr(f(v)) for f in ROF])
    def cid(v):
        k=canon(v)
        if k not in IDS: IDS[k]=len(IDS)
        return IDS[k]
    g=l3(current_frame); bands={r0:sorted(g[r0]) for r0 in sorted(g)}
    pairs=[]
    for r0 in [4,13,22]:
        lst=bands[r0]
        for i in range(0,len(lst),2): pairs.append(([cid(v) for v in lst[i][2]],[cid(v) for v in lst[i+1][2]]))
    q=[cid(v) for (c,col,gl) in bands[40] for v in gl]
    # group q into consecutive pair-key sequences (order-preserving subset match)
    tgt=None
    import itertools
    for k in range(1,7):
        for combo in itertools.combinations(range(6),k):
            seq=[]; ok=True
            for pi in combo:
                seq+=pairs[pi][0]
            if seq==q: tgt=[p for p in combo]; break
        if tgt: break
    ans=[]
    for pi in tgt: ans+=pairs[pi][1]
    CY=[8,13,4,6,3,12,1]; POS={c:i for i,c in enumerate(CY)}
    cur=[cid(v) for v in l3(current_frame)[51][0][2]]
    print('pairs(Y->S):',pairs)
    print('selected pairs',tgt,'target classes',ans)
    print('current', cur)
    log=[]
    def cls_at(i): return cid(l3(current_frame)[51][0][2][i])
    slot=int(round(slot_from_bracket(current_frame)))
    for s in range(7):
        while slot!=s:
            r=action(['RIGHT']); slot+=1
            if r.get('level_completed') or r.get('game_over'): return log,('EVENT',r['state'])
        c0=cls_at(s); t=ans[s]
        if c0==t: log.append((s,'ok0')); continue
        d=(POS[t]-POS[c0])%7
        acts=['UP']*d if d<=3 else ['DOWN']*(7-d)
        for a in acts:
            r=action([a])
            if r.get('level_completed') or r.get('game_over'): return log,('EVENT',r['state'])
        got=cls_at(s)
        if got!=t:
            for p in range(7):
                r=action(['UP'])
                if r.get('level_completed') or r.get('game_over'): return log,('EVENT',r['state'])
                got=cls_at(s)
                if got==t: break
        log.append((s,c0,'->',cls_at(s),'ok' if cls_at(s)==t else 'FAIL'))
    return log,('END',row63(current_frame))

def l3sol2():
    ROF=[lambda x:x, rot90, lambda x:rot90(rot90(x)), lambda x:rot90(rot90(rot90(x)))]
    IDS={}
    def canon(v): return min([tostr(f(v)) for f in ROF])
    def cid(v):
        k=canon(v)
        if k not in IDS: IDS[k]=len(IDS)
        return IDS[k]
    g=l3(current_frame); bands={r0:sorted(g[r0]) for r0 in sorted(g)}
    pairs=[]
    for r0 in [4,13,22]:
        lst=bands[r0]
        for i in range(0,len(lst),2): pairs.append(([cid(v) for v in lst[i][2]],[cid(v) for v in lst[i+1][2]]))
    q=[cid(v) for (c,col,gl) in bands[40] for v in gl]
    tgt=None
    for k in range(1,7):
        for perm in itertools.permutations(range(6),k):
            seq=[]
            for pi in perm: seq+=pairs[pi][0]
            if seq==q: tgt=list(perm); break
        if tgt: break
    ans=[]
    for pi in tgt: ans+=pairs[pi][1]
    CY=[8,13,4,6,3,12,1]; POS={c:i for i,c in enumerate(CY)}
    cur=[cid(v) for v in l3(current_frame)[51][0][2]]
    print('selected',tgt,'target',ans,'cur',cur)
    log=[]
    def cls_at(i): return cid(l3(current_frame)[51][0][2][i])
    slot=int(round(slot_from_bracket(current_frame)))
    for s in range(7):
        while slot!=s:
            r=action(['RIGHT']); slot+=1
            if r.get('level_completed') or r.get('game_over'): return log,('EVENT',r['state'])
        c0=cls_at(s); t=ans[s]
        if c0==t: log.append((s,'ok0')); continue
        d=(POS[t]-POS[c0])%7
        acts=['UP']*d if d<=3 else ['DOWN']*(7-d)
        for a in acts:
            r=action([a])
            if r.get('level_completed') or r.get('game_over'): return log,('EVENT',r['state'])
        if cls_at(s)!=t:
            for p in range(7):
                r=action(['UP'])
                if r.get('level_completed') or r.get('game_over'): return log,('EVENT',r['state'])
                if cls_at(s)==t: break
        log.append((s,c0,'->',cls_at(s),'ok' if cls_at(s)==t else 'FAIL'))
    return log,('END',row63(current_frame))

def lvl4(fr):
    A=fr.ascii.split('\n')
    def cell(r0,c0): return [[1 if A[r0+1+i][c0+1+j]=='B' else 0 for j in range(5)] for i in range(5)]
    nodes=fr.segmentation['nodes']
    tops=[];bots=[]
    for n in nodes:
        if n['color'] in 'YSPOWwpM':
            b=bbox_of(n); r0,c0,r1,c1=b
            if r1-r0==6:
                rec=(r0,c0,c1,n['color'],(c1-c0+1)//7)
                (tops if r0<34 else bots).append(rec)
    tops.sort(); bots.sort()
    bands={}
    for (r0,c0,c1,col,n) in tops:
        bands.setdefault(r0,[]).append((c0,col,[cell(r0,c0+7*i) for i in range(n)]))
    pairs=[]
    for r0 in sorted(bands):
        lst=sorted(bands[r0])
        for i in range(0,len(lst),2):
            pairs.append((lst[i][1],lst[i][2],lst[i+1][1],lst[i+1][2]))
    rows={}
    for (r0,c0,c1,col,n) in bots:
        rows[r0]=(col,[cell(r0,c0+7*i) for i in range(n)])
    return pairs,rows

def canon4(v): return min([tostr(f(v)) for f in ROF])

def cid4(v):
    k=canon4(v)
    if k not in IDS4: IDS4[k]=len(IDS4)
    return IDS4[k]

def solve_chain():
    ROF=[lambda x:x, rot90, lambda x:rot90(rot90(x)), lambda x:rot90(rot90(rot90(x)))]
    IDS={}
    def canon(v): return min([tostr(f(v)) for f in ROF])
    def cid(v):
        k=canon(v)
        if k not in IDS: IDS[k]=len(IDS)
        return IDS[k]
    pairs,rows=lvl4(current_frame)
    skeys=sorted([r0 for r0 in rows if rows[r0][0]=='S'])
    akeys=sorted([r0 for r0 in rows if rows[r0][0]!='S'])
    q=[cid(v) for v in rows[skeys[-1]][1]]
    ans_row=akeys[0]
    m1={}; m2={}
    for (kc,kg,vc,vg) in pairs:
        if kc=='S': m1[cid(kg[0])]=cid(vg[0])
        elif kc=='P': m2[cid(kg[0])]=cid(vg[0])
    tgt=[m2[m1[c]] for c in q]
    cur=[cid(v) for v in rows[ans_row][1]]
    print('q',q,'-> tgt',tgt,'cur',cur)
    def cls_at(i):
        _,gl=lvl4(current_frame)[1][ans_row]
        return cid(gl[i])
    n=len(cur)
    # learn cycle at current slot
    slot=int(round(slot_from_bracket(current_frame)))
    start=cls_at(slot); CY=[start]
    for i in range(9):
        r=action(['UP'])
        if r.get('level_completed') or r.get('game_over'): return ('EVENT',r['state']),[]
        c=cls_at(slot)
        if c==start: break
        CY.append(c)
    print('cycle',CY,'len',len(CY))
    POS={c:i for i,c in enumerate(CY)}
    log=[]
    for s in range(n):
        while slot!=s:
            r=action(['RIGHT']); slot+=1
            if r.get('level_completed') or r.get('game_over'): return ('EVENT',r['state']),log
        c0=cls_at(s); t=tgt[s]
        if c0==t: log.append((s,'ok0')); continue
        d=(POS[t]-POS[c0])%len(CY)
        acts=['UP']*d if d<=len(CY)//2 else ['DOWN']*(len(CY)-d)
        for a in acts:
            r=action([a])
            if r.get('level_completed') or r.get('game_over'): return ('EVENT',r['state']),log
        if cls_at(s)!=t:
            for p in range(len(CY)+1):
                r=action(['UP'])
                if r.get('level_completed') or r.get('game_over'): return ('EVENT',r['state']),log
                if cls_at(s)==t: break
        log.append((s,c0,'->',cls_at(s),'ok' if cls_at(s)==t else 'FAIL'))
    return ('END',row63(current_frame)),log

def brief(d):
    out={}
    for k,v in d.items():
        if k=='changed_cell_count': out[k]=v
        elif isinstance(v,list): out[k]=len(v)
    return out

def bracket(fr):
    ws=[bbox_of(n) for n in fr.segmentation['nodes'] if n['color']=='W']
    if not ws: return None
    rmin=min(b[0] for b in ws); rmax=max(b[2] for b in ws)
    cmin=min(b[1] for b in ws); cmax=max(b[3] for b in ws)
    return (rmin,rmax,(cmin+cmax)//2)

def snap(fr):
    pairs,rows=lvl4(fr)
    ROF=[lambda x:x, rot90, lambda x:rot90(rot90(x)), lambda x:rot90(rot90(rot90(x)))]
    IDS={}
    def cid(v):
        k=min([tostr(f(v)) for f in ROF])
        if k not in IDS: IDS[k]=len(IDS)
        return IDS[k]
    out=[]
    for (kc,kg,vc,vg) in pairs: out.append((kc,tuple(cid(v) for v in kg),vc,tuple(cid(v) for v in vg)))
    bot=[(rows[r][0],tuple(cid(v) for v in rows[r][1])) for r in sorted(rows)]
    return out,bot

def cid5(v):
    k=min([tostr(f(v)) for f in ROF])
    if k not in IDS: IDS[k]=len(IDS)
    return IDS[k]

def dump(fr):
    pairs,rows=lvl4(fr)
    print('pairs:')
    for (kc,kg,vc,vg) in pairs:
        print('  ',kc,[cid5(v) for v in kg],'->',vc,[cid5(v) for v in vg], ' exact:',[tostr(v).replace('\n','|') for v in kg],[tostr(v).replace('\n','|') for v in vg])
    for r0 in sorted(rows):
        col,gl=rows[r0]
        print('row',r0,col,[cid5(v) for v in gl])
        print('   ',[tostr(v).replace('\n','|') for v in gl])

def cell_classes(fr):
    pairs,rows=lvl4(fr)
    # flatten cells in reading order: (band, strip color, index)
    out=[]
    for pi,(kc,kg,vc,vg) in enumerate(pairs):
        for i,v in enumerate(kg): out.append(('K%d.%d'%(pi,i),kc,v))
        for i,v in enumerate(vg): out.append(('V%d.%d'%(pi,i),vc,v))
    return out

def clsmap():
    ROF=[lambda x:x, rot90, lambda x:rot90(rot90(x)), lambda x:rot90(rot90(rot90(x)))]
    IDS={}
    def cid(v):
        k=min([tostr(f(v)) for f in ROF])
        if k not in IDS: IDS[k]=len(IDS)
        return IDS[k]
    return IDS,cid

def flat(fr):
    out=[]
    for lbl,col,v in cell_classes(fr): out.append((lbl,col,CID(v),tostr(v).replace('\n','|')))
    return out

def cur_cell(fr):
    b=bracket(fr)
    return b

def cls_at_cell(fr):
    # return class of bracketed cell using global IDS
    b=bracket(fr); col=b[2]
    pairs,rows=lvl4(fr)
    cells=cell_classes(fr)
    # find cell whose bbox center col matches: recompute geometry from nodes
    best=None
    for n in fr.segmentation['nodes']:
        if n['color'] in 'SP':
            bb=bbox_of(n); r0,c0,r1,c1=bb
            if r1-r0==6 and c0-1<=col<=c1+1:
                # within this strip; pick the cell whose 7-wide slot contains col
                slot=(col-c0)//7
                cc=c0+7*slot
                if abs(cc+3-col)<=3: best=(bb,slot)
    return best

def P(s): return [[1 if ch=='#' else 0 for ch in row] for row in s.split('|')]

def K(v): return tostr(v).replace('\n','|')

def state5(fr):
    ROF=[lambda x:x, rot90, lambda x:rot90(rot90(x)), lambda x:rot90(rot90(rot90(x)))]
    IDS={}
    def cid(v):
        k=min([tostr(f(v)) for f in ROF])
        if k not in IDS: IDS[k]=len(IDS)
        return IDS[k]
    pairs,rows=lvl4(fr)
    cells=[]
    for pi,(kc,kg,vc,vg) in enumerate(pairs):
        for i,v in enumerate(kg): cells.append(('K%d.%d'%(pi,i),kc,cid(v),tostr(v).replace('\n','|')))
        for i,v in enumerate(vg): cells.append(('V%d.%d'%(pi,i),vc,cid(v),tostr(v).replace('\n','|')))
    bS=[cid(v) for v in rows[43][1]]; bP=[cid(v) for v in rows[52][1]]
    eS=[tostr(v).replace('\n','|') for v in rows[43][1]]; eP=[tostr(v).replace('\n','|') for v in rows[52][1]]
    return cells,bS,bP,eS,eP

def idx5(fr):
    b=bracket(fr)
    if b is None: return None
    rmin,col=b[0],b[2]
    band=0 if rmin<15 else 1
    centers=[11,21,34,44,51] if band==0 else [11,18,28,41,51]
    j=min(range(5),key=lambda k:abs(centers[k]-col))
    return band*5+j

def fix5(need, start=9):
    log=[]
    ev=None
    order=[8,7,6,5,4,3,2,1,0]
    for ti in order:
        guard=0
        while idx5(current_frame)!=ti and guard<11:
            r=action(['LEFT']); guard+=1
            if r.get('level_completed') or r.get('game_over'): return log,('EVENT',r['state'])
        if ti not in need: continue
        presses=0; found=False
        for p in range(4):
            cells,bS,bP,eS,eP=state5(current_frame)
            if cells[ti][2]==need[ti]: found=True; break
            r=action(['UP']); presses+=1
            if r.get('level_completed') or r.get('game_over'): return log,('EVENT',r['state'])
        if not found:
            cells,bS,bP,eS,eP=state5(current_frame)
            if cells[ti][2]==need[ti]: found=True
        if not found:
            for p in range(4):
                cells,bS,bP,eS,eP=state5(current_frame)
                if cells[ti][2]==need[ti]: found=True; break
                r=action(['DOWN']); presses+=1
                if r.get('level_completed') or r.get('game_over'): return log,('EVENT',r['state'])
        cells,bS,bP,eS,eP=state5(current_frame)
        log.append((ti,need[ti],cells[ti][2],'ok' if cells[ti][2]==need[ti] else 'MISS',presses))
        if found is False: ev='MISS@%d'%ti
    return log,ev

def bb(n):
    rs=[p[0] for p in n['boundary']]; cs=[p[1] for p in n['boundary']]
    return (min(rs),min(cs),max(rs),max(cs))

def sh(s): return s.replace('\n','|')

def wbr(fr):
    ws=[bb(n) for n in fr.segmentation['nodes'] if n['color']=='W']
    if not ws: return None
    return (min(b[0] for b in ws), max(b[2] for b in ws), min(b[1] for b in ws), max(b[3] for b in ws))

def unit_of(fr):
    w=wbr(fr)
    band=0 if w[0]<15 else 1
    cmins=[9,19,32,42] if band==0 else [9,26,39,49]
    j=min(range(4),key=lambda k:abs(cmins[k]-w[2]))
    return band*4+j

def curval(u,fr=None):
    p,_=parse(fr or current_frame)
    m={0:p[0][0][0],1:p[0][1][0],2:p[1][0][0],3:p[1][1][0],4:p[2][0][0],5:p[2][1][0],6:p[3][0][0],7:p[3][1][0]}
    return m[u]

def parse6(fr):
    A=fr.ascii.split('\n')
    def cell(r0,c0): return [[1 if A[r0+1+i][c0+1+j]=='B' else 0 for j in range(5)] for i in range(5)]
    tops=[];bots=[]
    for n in fr.segmentation['nodes']:
        if n['color'] in 'SPY':
            r0,c0,r1,c1=bb(n)
            if r1-r0==6: (tops if r0<34 else bots).append((r0,c0,c1,n['color'],(c1-c0+1)//7))
    tops.sort();bots.sort()
    bands={}
    for t in tops: bands.setdefault(t[0],[]).append(t)
    out=[]
    for r0 in sorted(bands):
        row=[]
        for (rr,c0,c1,col,n) in sorted(bands[r0]):
            row.append((col,[canon(cell(rr,c0+7*j)) for j in range(n)]))
        out.append(row)
    rows={}
    for (r0,c0,c1,col,n) in bots: rows[r0]=(col,[canon(cell(r0,c0+7*j)) for j in range(n)])
    return out,rows

def planfor(pi,cells,Srow,groups,vals):
    best=None
    for sg in itertools.permutations(range(3)):
        cost=0; plan={}
        for j in range(3):
            t=vals[sg[j]]; plan[('Y',j,0)]=t
            if cells[('Y',j,0)]!=t: cost+=1
        slot={v:[] for v in vals}
        for i in range(3):
            for k in range(2): slot[groups[pi[i]][k]].append(('Q',i,k))
        for j in range(3): slot[vals[sg[j]]].append(('U',j,0))
        for v in vals:
            cc=[cells[c] for c in slot[v]]; cnt=Counter(cc); mc=cnt.most_common(1)[0]
            cost+=len(slot[v])-mc[1]
            for c in slot[v]: plan[c]=mc[0]
        for i in range(3):
            t=Srow[pi[i]]; plan[('S',i,0)]=t
            if cells[('S',i,0)]!=t: cost+=1
        if best is None or cost<best[0]: best=(cost,sg,plan)
    return best

def unit_now(fr=None):
    b,_=parse6(fr or current_frame)
    out=[]
    for i in range(3):
        for k in range(4): out.append(b[i][k][1])
    return out

def cur_unit(fr=None):
    w=wbr(fr or current_frame)
    band=int(round((w[0]-2)/12.0))
    cc=(w[2]+w[3])/2.0
    j=min(range(4),key=lambda k:abs([12,25.5,41,51][k]-cc))
    return band*4+j

def fix(u,maxp=7):
    tg=TGT[u]
    for p in range(maxp):
        if unit_now()[u]==tg: return (p,'ok')
        r=action(['UP'])
        if r.get('level_completed'): return (p,'WIN')
        if r.get('game_over'): return (p,'DEAD')
    return (maxp,'ok' if unit_now()[u]==tg else 'MISS')

def sig(v): return tuple((sum(sum(r) for r in g), g[0],g[2],g[4]) for g in v)

def goto(u):
    g=0
    while cur_unit()!=u and g<13:
        rr=action(['RIGHT'])
        if rr.get('level_completed') or rr.get('game_over'): return ('EVENT',rr['state'])
        g+=1
    return None

def fixd(u,maxp=8):
    tg=TGT[u]; cyc=[]
    for p in range(maxp):
        v=unit_now()[u]
        cyc.append(sig(v))
        if v==tg: return (p,'ok',len(set(cyc)))
        rr=action(['DOWN'])
        if rr.get('level_completed'): return (p,'WIN',0)
        if rr.get('game_over'): return (p,'DEAD',0)
    return (maxp,'MISS',len(set(cyc)))
