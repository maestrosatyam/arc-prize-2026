try:
    import json
except Exception:
    pass
try:
    from collections import deque
except Exception:
    pass
try:
    from collections import Counter
except Exception:
    pass

def cellgrid(frame):
    a = frame.ascii.split('\n')
    rowidx = [12,18,24,30,36,42,48]
    colidx = [11,17,23,29,35,41,47]
    out=[]
    for ri,r in enumerate(rowidx):
        line=''
        for ci,c in enumerate(colidx):
            block=[a[r+i][c+j] for i in range(4) for j in range(4)]
            # summarize
            from collections import Counter
            cnt = Counter(block)
            line += max(cnt, key=lambda k:(cnt[k],k)) + ' '
        out.append((r,line))
    for r,l in out: print(r, l)

def classify(frame):
    a = frame.ascii.split('\n')
    rowidx = [12,18,24,30,36,42,48]
    colidx = [11,17,23,29,35,41,47]
    g=[]
    for r in rowidx:
        row=[]
        for c in colidx:
            block=[a[r+i][c+j] for i in range(4) for j in range(4)]
            s=set(block)
            if 'S' in s and len(s)==1: row.append(' .')
            elif 'N' in s: row.append('G' + ('*' if 'G' in s else ''))
            elif 'g' in s: row.append('h')
            elif 'w' in s: row.append('#')
            else: row.append('.')
        g.append(row)
    return g

def moves(state):
    out=[]
    for p in state:
        for dr,dc in DIRS:
            m=(p[0]+dr,p[1]+dc); t=(p[0]+2*dr,p[1]+2*dc)
            if m in state and t in CELLS and t not in state:
                out.append((p,m,t))
    return out

def apply(state, mv):
    p,m,t = mv
    s=set(state); s.discard(p); s.discard(m); s.add(t); return frozenset(s)

def bfs(start, target_count=1):
    start=frozenset(start); seen={start}; q=deque([(start,[])])
    while q:
        s,path = q.popleft()
        if len(s)==target_count: return path
        for mv in moves(s):
            ns=apply(s,mv)
            if ns not in seen:
                seen.add(ns); q.append((ns,path+[mv]))
    return None

def minpegs(start):
    start=frozenset(start); seen={start}; best=len(start); q=deque([start])
    while q:
        s=q.popleft(); best=min(best,len(s))
        for mv in moves(s):
            ns=apply(s,mv)
            if ns not in seen: seen.add(ns); q.append(ns)
    return best, len(seen)

def pegs(frame):
    out=[]
    for i,row in enumerate(classify(frame)):
        for j,cell in enumerate(row):
            if cell.startswith('G'): out.append((i,j))
    return out

def cellpx(r,c): return (10+6*r, 9+6*c)

def pegs2(frame):
    a=frame.ascii.split('\n'); out=[]
    for r in range(8):
        for c in range(8):
            rr,cc = 9+6*r, 7+6*c
            if rr+3>63 or cc+3>63: continue
            blk=[a[rr+i][cc+j] for i in range(4) for j in range(4)]
            s=set(blk)
            if 'N' in s: out.append((r,c))
            elif 'O' in s: out.append(('O',r,c))
    return out

def st(frame):
    a=frame.ascii.split('\n'); peg=[]; other=[]
    for r in range(9):
        for c in range(8):
            rr,cc = 9+6*r, 7+6*c
            if rr+3>63 or cc+3>63: continue
            blk=[a[rr+i][cc+j] for i in range(4) for j in range(4)]
            s=set(blk)
            if 'N' in s: peg.append((r,c))
            elif 'O' in s: other.append(('O',(r,c)))
    return sorted(peg), other

def mvs(s, cells=CELLS):
    out=[]
    for p in s:
        for dr,dc in DIRS:
            m=(p[0]+dr,p[1]+dc); t=(p[0]+2*dr,p[1]+2*dc)
            if m in s and t in cells and t not in s: out.append((p,m,t))
    return out

def ap(s,mv):
    n=set(s)-{mv[0],mv[1]}; n.add(mv[2]); return frozenset(n)

def barlen(frame):
    a=frame.ascii.split('\n')[0]
    n=0
    for ch in a:
        if ch!='W': n+=1
        else: break
    return n

def clusters(a, color):
    cells=[(r,c) for r in range(64) for c in range(64) if a[r][c]==color]
    if not cells: return []
    seen=set(); out=[]
    for cell in cells:
        if cell in seen: continue
        stack=[cell]; comp=[]
        while stack:
            r,c=stack.pop()
            if (r,c) in seen or (r,c) not in set(cells): continue
            seen.add((r,c)); comp.append((r,c))
            for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
                stack.append((r+dr,c+dc))
        rs=[p[0] for p in comp]; cs=[p[1] for p in comp]
        out.append((len(comp),min(rs),min(cs),max(rs),max(cs)))
    return out
