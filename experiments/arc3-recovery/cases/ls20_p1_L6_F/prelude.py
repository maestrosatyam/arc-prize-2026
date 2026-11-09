try:
    import re
except Exception:
    pass
try:
    from collections import defaultdict
except Exception:
    pass
try:
    from collections import deque
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
    import collections
except Exception:
    pass

def ch(r,c,a=a): return a[r][c] if 0<=r<64 and 0<=c<64 else 'X'

def ppos(fr):
    seg=fr.segmentation
    pos={}
    for n in seg['nodes']:
        if n['color'] in ('O','b') and n['pixels']>=10:
            pos[n['color']]=n['boundary'][0]
    return pos

def legend(fr):
    a=fr.ascii.split('\n')
    units=[]
    for rr in range(3):
        row=''
        for cc in range(3):
            r0,c0=55+2*rr,3+2*cc
            s={a[r0+i][c0+j] for i in range(2) for j in range(2)}
            row+= 'X' if 'b' in s else '.'
        units.append(row)
    return units

def chamber(fr):
    a=fr.ascii.split('\n')
    return [a[r][33:40] for r in range(9,16)]

def plus(fr):
    a=fr.ascii.split('\n')
    return [a[r][19:24] for r in range(30,35)]

def small(fr,label):
    print(label, fr.step)
    for n in fr.segmentation['nodes']:
        if n['pixels']<=30:
            print('  ',n['id'],n['color'],n['pixels'],n['hash'][:8],'pose',n['pose_hash'][:8] if n.get('pose_hash') else None,'rot',n['rotation'],'sym',n['rotational_symmetry'],n['boundary'][0])

def pat(fr, r0,c0,h,w, scale):
    a=fr.ascii.split('\n')
    out=[]
    for rr in range(h//scale):
        row=''
        for cc in range(w//scale):
            s={a[r0+rr*scale+i][c0+cc*scale+j] for i in range(scale) for j in range(scale)}
            row+='X' if 'b' in s else '.'
        out.append(row)
    return out

def build(fr):
    a=fr.ascii.split('\n')
    P=ppos(fr)
    br,bc=P['O']
    ro,bco=br%5,bc%5
    K=(64-ro)//5; M=(64-bco)//5
    grid=[]
    for k in range(K):
        row=[]
        for m in range(M):
            r0,c0=ro+5*k,bco+5*m
            vals={a[r0+i][c0+j] for i in range(5) for j in range(5)}
            if 'c' in vals: row.append('#')
            elif (k,m)==((br-ro)//5,(bc-bco)//5): row.append('@')
            elif 'Y' in vals: row.append('Y')
            elif 'W' in vals or 'w' in vals: row.append('+')
            elif 'b' in vals and 'B' in vals: row.append('H')
            elif 'B' in vals: row.append('B')
            elif 'O' in vals: row.append('@')
            else: row.append('.')
        grid.append(row)
    return grid,(br-ro)//5,(bc-bco)//5

def bfs(grid,s,blocked='#'):
    K=len(grid); M=len(grid[0])
    prev={s:None}; q=deque([s])
    while q:
        u=q.popleft()
        for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
            v=(u[0]+dr,u[1]+dc)
            if 0<=v[0]<K and 0<=v[1]<M and v not in prev and grid[v[0]][v[1]] not in blocked:
                prev[v]=u; q.append(v)
    return prev

def path_to(prev,t):
    if t not in prev: return None
    p=[]; u=t
    while u: p.append(u); u=prev[u]
    return p[::-1]

def moves(grid,fr,targets):
    g,s,e=build(fr)
    prev=bfs(g,s)
    out={}
    for name,t in targets.items():
        p=path_to(prev,t)
        out[name]=None if p is None else len(p)-1
    return out

def dists(fr,targets):
    g,k,m=build(fr)
    prev=bfs(g,(k,m))
    out={}
    for name,t in targets.items():
        p=path_to(prev,t)
        if p is None: out[name]=None; continue
        seq=''.join(_mv((b[0]-a[0],b[1]-a[1])) for a,b in zip(p,p[1:]))
        out[name]=(len(seq),seq)
    return out

def _mv(d):
    return {(0,0):'', (-1,0):'U',(1,0):'D',(0,-1):'L',(0,1):'R'}[d]

def ybar(fr):
    A=fr.ascii.split('\n'); r=61
    c=[i for i,ch in enumerate(A[r]) if ch=='Y']
    return (len(c), c[0] if c else -1, c[-1] if c else -1)

def mseq(p): return ''.join(_mv((b[0]-a[0],b[1]-a[1])) for a,b in zip(p,p[1:]))

def exec_seq(s, label=''):
    tgt=pat(current_frame,41,15,3,3,1)
    for c in s:
        r=action([D[c]])
        print(label,c,'pos',ppos(current_frame)['O'],'lg',legend(current_frame),'match',legend(current_frame)==tgt,
              'lc',r['level_completed'],'go',r['game_over'],'noop',not r['board_changed'],'bar',ybar(current_frame))
        if r['level_completed'] or r['game_over'] or r.get('done'): return 'END'
    return 'OK'

def step(c):
    r=action([Dm[c]])
    if not r['board_changed']: print('NOOP',c)
    if r.get('game_over') or r.get('level_completed') or r.get('done'): print('END',c,r); return False
    return True

def crop(r0,r1,c0,c1):
    for r in range(r0,r1+1): print(r,a[r][c0:c1+1])

def bgrid(fr):
    a=fr.ascii.split('\n'); R,C=fr.shape
    nr=(C-4)//5; nc=(C-4)//5
    rows=[]
    for r in range(13):
        line=''
        for c in range(12):
            cells=set()
            for dr in range(5):
                for dc in range(5):
                    rr,cc=5+5*r+dr,4+5*c+dc
                    if rr<R and cc<C: cells.add(a[rr][cc])
            line+='#' if 'c' in cells else ('.' if 'G' in cells else '?')
        rows.append(line)
    return rows,a

def px(r,c): return 5+5*r,4+5*c

def bgrid2(fr):
    a=fr.ascii.split('\n'); R,C=fr.shape
    rows=[]
    for r in range(12):
        line=''
        for c in range(12):
            cells=set()
            for dr in range(5):
                for dc in range(5):
                    rr,cc=5*r+dr,4+5*c+dc
                    if rr<R and cc<C: cells.add(a[rr][cc])
            line+='#' if 'c' in cells else ('.' if ('G' in cells or 'W' in cells or 'Y' in cells or 'b' in cells or 'O' in cells or 'N' in cells or 'w' in cells) else '?')
        rows.append(line)
    return rows,a

def cellchars(r,c): 
    s=set()
    for dr in range(5):
        for dc in range(5): s.add(a[5*r+dr][4+5*c+dc])
    return ''.join(sorted(s-{'G'}))

def leg3(fr):
    A=fr.ascii.split('\n')
    return [''.join('X' if A[r][c]!='B' else '.' for c in (3,5,7)) for r in (55,57,59)]

def cellpix(r,c,tag):
    print(tag)
    for rr in range(5*r,5*r+5): print(rr, A[rr][4+5*c:9+5*c])

def go(seq):
    for c in seq:
        r=action([Dm[c]])
        if r.get('game_over'): return 'DEAD'
        if r.get('level_completed') or r.get('done'): return 'WON'
    return None

def bcell(fr):
    for n in fr.segmentation['nodes']:
        if n['color']=='b' and n['pixels']==15:
            r,c=n['boundary'][0]; return ((r+2)//5, (c+2-4)//5)
    return None

def legcolor(fr):
    A=fr.ascii.split('\n')
    s=set(A[55][3]+A[59][3])
    return 'b' if 'b' in s else ('O' if 'O' in s else ''.join(s))

def walk(fr0,G,CONV,goal,maxm=12):
    Dm={'U':'UP','D':'DOWN','L':'LEFT','R':'RIGHT'}; log=[]
    for i in range(maxm):
        cur=bcell(fr0)
        if cur==goal: return ('AT',log)
        p=sv(G,CONV,cur)(goal)
        if not p: return ('NOPATH',log)
        d=p[0]; r=action([Dm[d]]); fr0=current_frame; log.append((d,bcell(fr0)))
        if r.get('game_over'): return ('DEAD',log)
        if r.get('level_completed') or r.get('done'): return ('WON',log)
    return ('LIMIT',log)

def pat_box(fr,r0,c0,h,w,scale):
    A=fr.ascii.split('\n')
    cols=set()
    for r in range(r0,r0+3*scale):
        for c in range(c0,c0+3*scale):
            if r<len(A) and c<len(A[r]) and A[r][c] not in 'B': cols.add(A[r][c])
    pat=[]
    for i in range(3):
        row=''
        for j in range(3):
            v=set()
            for dr in range(scale):
                for dc in range(scale):
                    rr,cc=r0+3*0+i*scale+dr,c0+j*scale+dc
                    if rr<len(A) and cc<len(A[rr]): v.add(A[rr][cc])
            row+='X' if (v-{'B'}) else '.'
        pat.append(row)
    return pat,sorted(cols)

def dial(fr): return pat_box(fr,55,3,3,3,2)

def cham(fr): return pat_box(fr,6,10,3,3,1)

def land(cell,dr,dc):
    r,c=cell
    if not (0<=r<12 and 0<=c<12) or G[r][c]=='#': return None
    seen=set()
    while cell in CONV and cell not in seen:
        seen.add(cell); d=CONV[cell]; dr,dc=D[d]
        nr,nc=cell[0]+dr,cell[1]+dc
        if not(0<=nr<12 and 0<=nc<12) or G[nr][nc]=='#': break
        cell=(nr,nc)
    return cell

def nbrs(cell):
    out=[]
    for d,(dr,dc) in D.items():
        t=(cell[0]+dr,cell[1]+dc)
        if 0<=t[0]<12 and 0<=t[1]<12 and G[t[0]][t[1]]=='.':
            l=land(t,dr,dc)
            if l: out.append((d,l))
    return out

def bfs2(s):
    from collections import deque
    prev={s:None}; q=deque([s])
    while q:
        u=q.popleft()
        for d,v in nbrs(u):
            if v not in prev: prev[v]=(u,d); q.append(v)
    return prev

def path2(s,t):
    prev=bfs2(s)
    if t not in prev: return None
    seq=[]; cur=t
    while prev[cur] is not None:
        p,d=prev[cur]; seq.append(d); cur=p
    return ''.join(reversed(seq))

def cell(r,c):
    s=set()
    for rr in range(5*r,5*r+5):
        for cc in range(4+5*c,9+5*c):
            if rr<64 and cc<64: s.add(A[rr][cc])
    return ''.join(sorted(s))

def make_model(G,CONV):
    D={'U':(-1,0),'D':(1,0),'L':(0,-1),'R':(0,1)}
    free=lambda r,c: 0<=r<12 and 0<=c<12 and G[r][c]=='.'
    def slide(cell,d):
        dr,dc=D[d]; r,c=cell
        while free(r+dr,c+dc): r,c=r+dr,c+dc
        return (r,c)
    def land(cell,d):
        # cell = entered cell; apply its conveyor if any
        if cell in CONV:
            return slide(cell,CONV[cell])
        return cell
    def nbrs(cell):
        out=[]
        for d,(dr,dc) in D.items():
            t=(cell[0]+dr,cell[1]+dc)
            if free(*t):
                l=land(t,d)
                if l!=cell: out.append((d,l))
        return out
    return nbrs

def solve(G,CONV,s):
    from collections import deque
    nbrs=make_model(G,CONV); prev={s:None}; q=deque([s])
    while q:
        u=q.popleft()
        for d,v in nbrs(u):
            if v not in prev: prev[v]=(u,d); q.append(v)
    def path(t):
        if t not in prev: return None
        seq=[]; cur=t
        while prev[cur] is not None:
            p,d=prev[cur]; seq.append(d); cur=p
        return ''.join(reversed(seq))
    return prev,path

def drain(maxn=10):
    for i in range(maxn):
        for d in order:
            r=action([Dm[d]])
            if r.get('game_over'): print('DIED after',i+1,'moves at',bcell(current_frame)); return
            if r.get('level_completed') or r.get('done'): print('WON?!',r); return
            if r.get('gameplay_changed'): break
        print(i,bcell(current_frame),ybar(current_frame))
    print('bar',ybar(current_frame))

def dial2(fr):
    A=fr.ascii.split('\n'); p=[]; col=set()
    for i in range(3):
        row=''
        for j in range(3):
            s={A[55+i*2][3+j*2],A[55+i*2][3+j*2+1],A[56+i*2][3+j*2],A[56+i*2][3+j*2+1]}
            v=s-{'B'}
            row+='X' if v else '.'
            col|=v
        p.append(row)
    return p,sorted(col)

def gb(fr):
    A=fr.ascii.split('\n'); g=[]
    for r in range(12):
        row=''
        for c in range(12):
            s=set()
            for dr in range(5):
                for dc in range(5):
                    rr,cc=5*r+dr,4+5*c+dc
                    if rr<64 and cc<64: s.add(A[rr][cc])
            row+='.' if (s&{'G','W','Y','b','O','N','R','w'}) and not (s&{'c'}) else '#'
        g.append(list(row))
    return g

def sv(G,CONV,s):
    D={'U':(-1,0),'D':(1,0),'L':(0,-1),'R':(0,1)}
    free=lambda r,c: 0<=r<12 and 0<=c<12 and G[r][c]=='.'
    def land(cell):
        if cell in CONV:
            dr,dc=D[CONV[cell]]; r,c=cell
            while free(r+dr,c+dc): r,c=r+dr,c+dc
            return (r,c)
        return cell
    prev={s:None}; q=collections.deque([s])
    while q:
        u=q.popleft()
        for d,(dr,dc) in D.items():
            t=(u[0]+dr,u[1]+dc)
            if free(*t):
                v=land(t)
                if v!=u and v not in prev: prev[v]=(u,d); q.append(v)
    def path(t):
        if t not in prev: return None
        out=[]; cur=t
        while prev[cur] is not None:
            p,d=prev[cur]; out.append(d); cur=p
        return ''.join(reversed(out))
    return path

def lvl5(fr):
    G=gb(fr)
    for r in G:
        for i,ch in enumerate(r):
            if ch!='.': r[i]='#'
    G[1][10]='.'
    CONV={(1,6):'D',(4,6):'R',(5,5):'L',(6,9):'D',(5,7):'U',(6,10):'L',(8,8):'R',(10,10):'U',(5,2):'L'}
    return G,CONV

def L5():
    G,CONV=lvl5(current_frame); CONV[(5,2)]='L'; return G,CONV

def rot90(p):
    s={(r,c) for r in range(3) for c in range(3) if p[r][c]=='X'}
    t={(c,2-r) for r,c in s}
    return [''.join('X' if (r,c) in t else '.' for c in range(3)) for r in range(3)]

def orbit(p):
    o=[p]
    for _ in range(3): o.append(rot90(o[-1]))
    return o

def panel(r0,c0,cols=12,rows=10):
    for r in range(r0,r0+rows): print(r,A[r][c0:c0+cols])

def lvl6(fr):
    G=gb(fr)
    for r in G:
        for i,ch in enumerate(r):
            if ch!='.': r[i]='#'
    G[7][10]='.'; G[10][10]='.'
    CONV={(1,9):'D',(4,9):'L'}
    return G,CONV

def objs(fr):
    out={}
    for n in fr.segmentation['nodes']:
        if n['color'] in ('W','Y','N','b','O','R','w') and n['pixels']<=60:
            r,c=n['boundary'][0]
            out.setdefault(n['color'],[]).append(((r+2)//5,(c-2)//5,n['pixels']))
    return out

def find_plus(fr):
    # plus = W node with 3 px (the 3 white arms) near w nodes
    Ws=[n for n in fr.segmentation['nodes'] if n['color']=='W' and n['pixels']==3]
    for n in Ws:
        r,c=n['boundary'][0]
        cell=((r+2)//5,(c-2)//5)
        # read 3x3 pattern rows r-? : plus occupies 3x3 px at rows r..r+2, cols c-1..c+1 approx
        return n,cell
    return None,None

def plus_pat(fr):
    A=fr.ascii.split('\n')
    for n in fr.segmentation['nodes']:
        if n['color']=='W' and n['pixels']==3:
            r,c=n['boundary'][0]
            # find the 3x3 plus bounding: use node bbox
            rs=[p[0] for p in n['boundary']]; cs=[p[1] for p in n['boundary']]
            r0,r1=min(rs),max(rs); c0,c1=min(cs),max(cs)
            return r0,c0,r1,c1
    return None

def cell_of(n):
    r,c=n['boundary'][0]; return ((r+2)//5,(c-2)//5)

def locate(fr):
    plus=mixed=None
    for n in fr.segmentation['nodes']:
        if n['color']=='W' and n['pixels']==3: plus=cell_of(n)
        if n['color']=='N' and n['pixels']==2: mixed=cell_of(n)
    return plus,mixed

def locate2(fr):
    cells={}
    for n in fr.segmentation['nodes']:
        if n['color'] in ('W','w','N','b','R','O') and n['pixels']<=20:
            cells.setdefault(cell_of(n),[]).append((n['color'],n['pixels']))
    zig=plus=mixed=None
    for c,v in cells.items():
        d={}
        for col,p in v: d[col]=d.get(col,0)+p
        if d.get('W',0)==4 and 'w' not in d: zig=c
        elif d.get('W',0)==3 and d.get('w',0)==2: plus=c
        elif d.get('N',0)==2 and d.get('b',0)==2 and d.get('O',0)==2: mixed=c
    return zig,plus,mixed

def loc3(fr):
    W3=[]; wsmall=[]; N2=None
    for n in fr.segmentation['nodes']:
        r,c=n['boundary'][0]; p=n['pixels']
        if n['color']=='W' and p==3: W3.append((r,c))
        elif n['color']=='w' and p<=2: wsmall.append((r,c))
        elif n['color']=='N' and p==2: N2=(r,c)
    near=lambda a,b: abs(a[0]-b[0])<=3 and abs(a[1]-b[1])<=3
    plus=zig=None
    for t in W3:
        if any(near(t,x) for x in wsmall): plus=cc(*t)
        else: zig=cc(*t)
    return zig,plus,(cc(*N2) if N2 else None)

def cc(r,c): return ((r+2)//5,(c-2)//5)

def step_toward(fr,goal,avoid):
    Gx,CONVx=lvl6(fr)
    for c in avoid:
        if c and 0<=c[0]<12 and 0<=c[1]<12: Gx[c[0]][c[1]]='#'
    return sv(Gx,CONVx,bcell(fr))(goal)
