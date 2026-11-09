try:
    from collections import deque
except Exception:
    pass

def player_pos(fr):
    seg = fr.segmentation
    # player = N object containing a W child
    for n in seg['nodes']:
        if n['color']=='N' and n['pixels']>=5:
            ch = [seg['nodes'][c] for c in n['children']]
            kind = ''.join(sorted(c['color'] for c in ch))
            b = n['boundary']
            r0,c0 = min(p[0] for p in b), min(p[1] for p in b)
            r1,c1 = max(p[0] for p in b), max(p[1] for p in b)
            yield (n['color'], kind, (r0,c0,r1,c1), n['pixels'])

def objs(fr):
    out=[]
    for n in fr.segmentation['nodes']:
        if n['color'] in ('N',):
            b=n['boundary']; bb=(min(p[0] for p in b),min(p[1] for p in b),max(p[0] for p in b),max(p[1] for p in b))
            ch=[fr.segmentation['nodes'][c]['color'] for c in n['children']]
            out.append((bb,n['pixels'],''.join(ch)))
    return out

def do(act):
    try:
        return action([act])
    except Exception as e:
        if 'Stale' in type(e).__name__:
            return action([act])
        raise

def wpos(fr):
    for o in objs(fr):
        if o[2]=='W': return o[0]

def bbox(n):
    b=n['boundary']; rs=[p[0] for p in b]; cs=[p[1] for p in b]
    return (min(rs),min(cs),max(rs),max(cs))

def grid(fr): return [list(r) for r in fr.ascii.split('\n')]

def ok(r,c): return 5<=r<=58 and 5<=c<=58

def free(top,left,avoid_small=True):
    for dr in range(3):
        for dc in range(3):
            r,c=top+dr,left+dc
            if not ok(r,c): return False
            if (r,c) in Y: return False
            if avoid_small and (r,c) in small: return False
    return True

def freeblk(top,left,blk):
    for dr in range(3):
        for dc in range(3):
            r,c=top+dr,left+dc
            if not ok(r,c): return False
            if (r,c) in blk: return False
    return True

def greens(fr):
    out=[]
    for n in fr.segmentation['nodes']:
        if n['color']=='N': out.append(bbox(n))
    return out

def st(fr):
    d={}
    for n in fr.segmentation['nodes']:
        if n['color'] in 'NY': d.setdefault(n['color'],[]).append(bbox(n))
    return d

def snap(fr):
    d={}
    for n in fr.segmentation['nodes']:
        if n['color']=='Y': d['Y']=bbox(n)
        elif n['color'] in 'NW': d.setdefault('N',[]).append(bbox(n))
    return d

def ss(fr): return sorted([(n['color'],bbox(n),n['pixels']) for n in fr.segmentation['nodes'] if n['pixels']<300 and n['color'] in 'NOrpW'])
