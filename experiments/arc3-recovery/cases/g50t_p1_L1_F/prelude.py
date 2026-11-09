try:
    import json
except Exception:
    pass
try:
    from collections import Counter
except Exception:
    pass

def compact(d):
    return {k:(v if not isinstance(v,list) else [x for x in v]) for k,v in d.items()}

def cell_area(i,j):
    r0, c0 = 8+6*i, 14+6*j
    return [a[r0+dr][c0:c0+5] for dr in range(5)]

def sep(i,j,di,dj):
    # separator between (i,j) and neighbor
    r0, c0 = 8+6*i, 14+6*j
    if di==1: return a[r0+5][c0:c0+5]
    if di==-1: return a[r0-1][c0:c0+5]
    if dj==1: return ''.join(a[r][c0+5] for r in range(r0,r0+5))
    return ''.join(a[r][c0-1] for r in range(r0,r0+5))

def sig(f):
    a = f.ascii.split('\n')
    # red blob col range at row 38
    r38 = a[38]
    redcols = [c for c in range(64) if r38[c]=='R']
    gcols = [c for c in range(64) if a[10][c]=='g']
    bcols = [c for c in range(64) if a[10][c]=='b' and c<60]
    return {'red38': (min(redcols),max(redcols)) if redcols else None, 'gray_row10': (min(gcols),max(gcols)) if gcols else None, 'blue_row10': (min(bcols),max(bcols)) if bcols else None, 'bar': a[63].count('b'), 'hud1': a[1][:10]}
