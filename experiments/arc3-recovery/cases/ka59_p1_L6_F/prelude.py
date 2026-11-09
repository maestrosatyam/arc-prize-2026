try:
    import json
except Exception:
    pass
try:
    from collections import deque
except Exception:
    pass
try:
    import frame_diff as _fd
except Exception:
    pass

def crop(f, r0,r1,c0,c1):
    a=f.ascii.split('\n')
    return [f"{r_} "+a[r_][c0:c1] for r_ in range(r0,r1)]

def bar(f):
    for n in f.segmentation['nodes']:
        if n['color']=='c' and n['pixels']<=70 and n['boundary'][0][0]==63:
            return n['pixels']
    return None

def passable(r_,c_):
    return G[r_][c_] not in ('g','p')

def ok(r_,c_):
    if r_<0 or c_<0 or r_+2>=R or c_+2>=C: return False
    return all(passable(r_+i,c_+j) for i in range(3) for j in range(3))

def okf(r_,c_,G=G,R=R,C=C):
    if r_<0 or c_<0 or r_+2>=R or c_+2>=C: return False
    return all(G[r_+i][c_+j] not in ('g','p') for i in range(3) for j in range(3))

def runs_of(rr, ch='w'):
    row=a[rr]; out=[]; st=None
    for i in range(64):
        if row[i]==ch:
            if st is None: st=i
        else:
            if st is not None: out.append((st,i-1)); st=None
    if st is not None: out.append((st,63))
    return out

def pos(f,col):
    out=[]
    for n in f.segmentation['nodes']:
        if n['color']==col:
            b=n['boundary']; rs=[p[0] for p in b]; cs=[p[1] for p in b]
            out.append((n['pixels'],min(rs),max(rs),min(cs),max(cs)))
    return out

def runs(rr,ch):
    row=a[rr]; out=[]; st=None
    for i in range(64):
        if row[i]==ch:
            if st is None: st=i
        else:
            if st is not None: out.append((st,i-1)); st=None
    if st is not None: out.append((st,63))
    return out
