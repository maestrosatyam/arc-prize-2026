try:
    import json
except Exception:
    pass
try:
    from collections import Counter
except Exception:
    pass
try:
    from collections import Counter, deque
except Exception:
    pass
try:
    from collections import deque
except Exception:
    pass

def tilestate(r0,c0):
    return ''.join(a[r][c0:c0+4] for r in range(r0+1,r0+5))

def ts(a,r0,c0):
    return ''.join(a[r][c0:c0+4] for r in range(r0,r0+4))

def board(a):
    out={}
    for ri,r0 in enumerate([12,18,24]):
        for ci,c0 in enumerate([11,17,23,29,35,41,47]):
            out[(ri,ci)]=ts(a,r0,c0)
    for ri,r0 in enumerate([30,36,42,48]):
        for ci,c0 in enumerate([35,41,47]):
            out[(ri+3,ci+4)]=ts(a,r0,c0)
    return out

def clk(ri,ci): return {'action':'MOUSE','row':12+6*ri+1,'col':11+6*ci+1}

def pegs(a):
    b = board(a)
    return sorted(k for k,v in b.items() if 'N' in v)

def cell(a,r0,c0):
    from collections import Counter
    cnt = Counter()
    for r in range(r0,r0+4):
        for c in range(c0,c0+4):
            cnt[a[r][c]]+=1
    return cnt.most_common(1)[0][0]

def scan(a):
    out=[]
    for ri in range(0,9):
        for ci in range(0,9):
            if 9+6*ri+3<64 and 7+6*ci+3<64:
                v = ''.join(a[9+6*ri+t][7+6*ci:7+6*ci+4] for t in range(4))
                if v!='w'*16 and any(ch in v for ch in 'N gG'):
                    if any(ch in v for ch in 'NgG'): out.append(((ri,ci),v))
    return out

def sock(a):
    pts = [(r,c) for r in range(64) for c in range(64) if a[r][c]=='O']
    if not pts: return None
    return (min(p[0] for p in pts), max(p[0] for p in pts), min(p[1] for p in pts), max(p[1] for p in pts))

def step(act):
    res = action([{'action':act}])
    rr = res[0] if isinstance(res,list) else res
    a = current_frame.ascii.split('\n')
    from collections import Counter
    return rr.get('gameplay_changed'), sock(a), Counter(a[0]).get('W'), rr.get('level_completed')

def ysock(a):
    pts = [(r,c) for r in range(64) for c in range(64) if a[r][c]=='Y']
    if not pts: return None
    return (min(p[0] for p in pts), max(p[0] for p in pts), min(p[1] for p in pts), max(p[1] for p in pts))

def run(acts):
    out=[]
    for act in acts:
        try:
            res = action([{'action':act}]); rr = res[0] if isinstance(res,list) else res
        except Exception as e:
            out.append((act,'HALT',type(e).__name__)); break
        a = current_frame.ascii.split('\n')
        out.append((act, ysock(a), rr.get('gameplay_changed'), rr.get('level_completed'), rr.get('game_over')))
        if rr.get('level_completed') or rr.get('game_over'): break
        if not rr.get('gameplay_changed'): break
    return out

def gridmap(a, r0, c0, nr, nc):
    out=[]
    for i in range(nr):
        row=''
        for j in range(nc):
            ch = a[r0+6*i+1][c0+6*j+1]
            row += ch
        out.append(row)
    return out

def ysocks(a):
    pts=[(r,c) for r in range(64) for c in range(64) if a[r][c]=='Y']
    # cluster by row
    top=[p for p in pts if p[0]<32]; bot=[p for p in pts if p[0]>=32]
    def bb(ps):
        return None if not ps else (min(p[0] for p in ps),max(p[0] for p in ps),min(p[1] for p in ps),max(p[1] for p in ps))
    return bb(top), bb(bot)

def boards(a):
    from collections import deque
    seen=[[False]*64 for _ in range(64)]; out=[]
    for r in range(64):
        for c in range(64):
            if a[r][c]=='W' and not seen[r][c]:
                q=deque([(r,c)]); seen[r][c]=True; pts=[]
                while q:
                    x,y=q.popleft(); pts.append((x,y))
                    for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
                        nx,ny=x+dx,y+dy
                        if 0<=nx<64 and 0<=ny<64 and not seen[nx][ny] and a[nx][ny]=='W':
                            seen[nx][ny]=True; q.append((nx,ny))
                rs=[p[0] for p in pts]; cs=[p[1] for p in pts]
                out.append((len(p if False else pts),min(rs),max(rs),min(cs),max(cs)))
    return out

def snap(a): return ysocks(a)

def bb(ps): return None if not ps else (min(p[0] for p in ps),max(p[0] for p in ps),min(p[1] for p in ps),max(p[1] for p in ps))

def center(i,j): return (7+6*i, 7+6*j)

def preview(a):
    pts=[(r,c) for r in range(2,64) for c in range(64) if a[r][c]=='g']
    if not pts: return None
    b=bb(pts); return ((b[0]+b[2])//2 if False else (b[0]+b[1])//2, (b[2]+b[3])//2)

def jump(src,dst):
    a=current_frame.ascii.split('\n')
    r0,c0=center(*src)
    res=action([{'action':'MOUSE','row':r0,'col':c0}]); rr=res[0] if isinstance(res,list) else res
    a=current_frame.ascii.split('\n')
    p=preview(a)
    er,ec=center(*dst)
    if p is None or abs(p[0]-er)>2 or abs(p[1]-ec)>2:
        return ('no-preview',src,dst,p)
    res=action([{'action':'MOUSE','row':er,'col':ec}]); rr=res[0] if isinstance(res,list) else res
    return ('ok',src,dst,rr.get('gameplay_changed'),rr.get('level_completed'),rr.get('game_over'))

def pegset(a):
    out=[]
    for R in range(0,10):
        for C in range(0,10):
            r,c=center(R,C)
            if 0<=r<64 and 0<=c<64 and a[r][c]=='N': out.append((R,C))
    return out

def gsel(a):
    pts=[(r,c) for r in range(2,64) for c in range(64) if a[r][c]=='G']
    if not pts: return None
    b=bb(pts); return ((b[0]+b[1])//2,(b[2]+b[3])//2)

def jump2(src,dst):
    a=current_frame.ascii.split('\n')
    sr,sc=center(*src); dr,dc=center(*dst)
    if gsel(a) is None:
        res=action([{'action':'MOUSE','row':sr,'col':sc}])
        a=current_frame.ascii.split('\n')
    p=preview(a)
    if p is None or abs(p[0]-dr)>2 or abs(p[1]-dc)>2:
        return ('no-preview',src,dst,p)
    res=action([{'action':'MOUSE','row':dr,'col':dc}]); rr=res[0] if isinstance(res,list) else res
    return ('ok',src,dst,'gp',rr.get('gameplay_changed'),'win',rr.get('level_completed'),'over',rr.get('game_over'))

def cellch(a,r,c): return a[r][c]

def scroll(a):
    cols=[c for c in range(30,64) if a[6][c] in 'wN' and a[7][c] in 'wN']
    return 66-min(cols) if cols else None

def st():
    a=current_frame.ascii.split('\n'); s=scroll(a)
    from collections import deque
    seen=[[False]*64 for _ in range(64)]; P=set()
    for r in range(64):
        for c in range(64):
            if a[r][c]=='N' and not seen[r][c]:
                q=deque([(r,c)]); seen[r][c]=True; pts=[]
                while q:
                    x,y=q.popleft(); pts.append((x,y))
                    for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
                        nx,ny=x+dx,y+dy
                        if 0<=nx<64 and 0<=ny<64 and not seen[nx][ny] and a[nx][ny]=='N':
                            seen[nx][ny]=True; q.append((nx,ny))
                b=bb(pts)
                if b[1]-b[0]>=3: P.add(((b[0]-6)//6,(b[2]+s-6)//6))
    ys=ysocks(a); S=[]
    for (r0,r1,c0,c1) in ys: S.append(((r0-5)//6,(c0+s-5)//6))
    return s,sorted(P),S

def vcell(R,C,s): return (6+6*R, 6+6*C-s)

def gjump(src,dst,verbose=True):
    s,P,S=st()
    a=current_frame.ascii.split('\n')
    sr,sc=7+6*src[0],7+6*src[1]-s; dr,dc=7+6*dst[0],7+6*dst[1]-s
    if not (0<=sc<64 and 0<=sr<64): return ('src offscreen',src,s)
    if not (0<=dc<64 and 0<=dr<64): return ('dst offscreen',dst,s)
    if gsel(a) is None:
        res=action([{'action':'MOUSE','row':sr,'col':sc}]); a=current_frame.ascii.split('\n')
    p=preview(a)
    if p is None: return ('no preview',src,dst,s,P,S)
    if abs(p[0]-dr)>2 or abs(p[1]-dc)>2: return ('preview elsewhere',src,dst,p,(dr,dc))
    res=action([{'action':'MOUSE','row':dr,'col':dc}]); rr=res[0] if isinstance(res,list) else res
    s2,P2,S2=st()
    return ('ok',src,dst,'npegs',len(P2),'win',rr.get('level_completed'),'over',rr.get('game_over'),P2 if verbose else None)

def gjump2(src,dst,s):
    a=current_frame.ascii.split('\n')
    sr,sc=7+6*src[0],7+6*src[1]-s; dr,dc=7+6*dst[0],7+6*dst[1]-s
    if gsel(a) is None:
        action([{'action':'MOUSE','row':sr,'col':sc}]); a=current_frame.ascii.split('\n')
    p=preview(a)
    if p is None: return ('no preview',src,dst)
    if abs(p[0]-dr)>2 or abs(p[1]-dc)>2: return ('preview elsewhere',src,dst,p,(dr,dc))
    res=action([{'action':'MOUSE','row':dr,'col':dc}]); rr=res[0] if isinstance(res,list) else res
    return ('ok',src,dst,'win',rr.get('level_completed'),'over',rr.get('game_over'),'done',rr.get('run_complete') or rr.get('done'))

def shiftof(a,b,state={'s':0}):
    best=(0,-10**9)
    for k in range(-56,57):
        m=0
        for r in range(0,64,2):
            ra=a[r]; rb=b[r]
            for c in range(0,64,2):
                ca=c+k
                if 0<=ca<64 and ra[c]==rb[ca]: m+=1
        if m>best[1]: best=(k,m)
    return best

def bump(state={'s':0},d=0):
    state['s']+=d; return state['s']

def ys2(a):
    ys=ysocks(a); return ys

def sockcells():
    a=current_frame.ascii.split('\n'); s=bump()
    ys=[y for y in ysocks(a) if y]
    return [((r0-5)//6,(c0+s-5)//6) for (r0,r1,c0,c1) in ys]

def yb():
    a=current_frame.ascii.split('\n')
    yl=[(r,c) for r in range(64) for c in range(64) if a[r][c] in 'YO']
    return bb(yl) if yl else None

def toggle():
    # click peg at (3,3) to select, then click it again to deselect -> both real changes
    res=action([{'action':'MOUSE','row':25,'col':25}]); r=res[0] if isinstance(res,list) else res
    return (r.get('no_op'), yb())
