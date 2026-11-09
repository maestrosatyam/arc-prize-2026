try:
    import json
except Exception:
    pass
try:
    import copy
except Exception:
    pass

def crop(r0,r1,c0,c1):
    for r in range(r0,r1+1):
        print(f'{r:2d}', a[r][c0:c1+1])

def head():
    out=[]
    for n in current_frame.segmentation['nodes']:
        if n['color'] in ('b','S') and n['pixels']<=20 and n['boundary'][0][0]<30 and n['boundary'][0][1]>20:
            out.append((n['color'], n['pixels'], tuple(n['boundary'][0]), tuple(n['boundary'][2])))
    return out

def headinfo(fr=None):
    fr = fr or current_frame
    a = fr.ascii.split('\n')
    cells=[]
    for r in range(16,25):
        row = a[r]
        for c in range(17,46):
            if row[c] in 'bS':
                cells.append((r,c,row[c]))
    return cells

def summarize(fr=None):
    cs = headinfo(fr)
    if not cs: return 'no head'
    rs=[p[0] for p in cs]; cl=[p[1] for p in cs]
    return f'n={len(cs)} rows {min(rs)}-{max(rs)} cols {min(cl)}-{max(cl)}'

def wire_right(fr=None):
    fr = fr or current_frame
    a = fr.ascii.split('\n')
    mx = -1
    for r in range(17,24):
        for c in range(17,63):
            if a[r][c]=='g': mx = max(mx,c)
    return mx

def plugparts(fr=None):
    fr = fr or current_frame
    a = fr.ascii.split('\n')
    head=[]; hand=[]
    for r in range(5,45):
        for c in range(25,42):
            ch = a[r][c]
            if ch in 'bS':
                (hand if r<15 else head).append((r,c))
    return head, hand

def headpos(fr=None):
    fr = fr or current_frame
    a = fr.ascii.split('\n')
    cells=[(r,c) for r in range(5,45) for c in range(20,60) if a[r][c] in 'bS']
    # exclude handle region rows>=37 cols<=28
    hd=[p for p in cells if not (p[0]>=36 and p[1]<=29)]
    if not hd: return None
    rs=[p[0] for p in hd]; cs=[p[1] for p in hd]
    return (len(hd), min(rs), max(rs), min(cs), max(cs))

def h2(fr=None):
    fr = fr or current_frame
    a = fr.ascii.split('\n')
    cells=[(r,c) for r in range(11,40) for c in range(27,50) if a[r][c] in 'bS']
    if not cells: return None
    rs=[p[0] for p in cells]; cs=[p[1] for p in cells]
    return (len(cells), min(rs), max(rs), min(cs), max(cs), ''.join(sorted(set(a[r][c] for r,c in cells))))

def hrows():
    a = current_frame.ascii.split('\n')
    cells = sorted(set((r_,c_) for r_ in range(5,45) for c_ in range(48,58) if a[r_][c_] in 'bS'))
    return cells

def bar():
    for n in current_frame.segmentation['nodes']:
        if n['color']=='N' and n['pixels']>10: return n['pixels']

def hd():
    a = current_frame.ascii.split('\n')
    c = sorted(set((r_,c_) for r_ in range(5,50) for c_ in range(0,60) if a[r_][c_] in 'bS' and not (r_<16 and c_>45)))
    return (c[0], c[-1], len(c)) if c else None
