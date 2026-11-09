try:
    import json
except Exception:
    pass
try:
    from collections import defaultdict
except Exception:
    pass
try:
    import re
except Exception:
    pass
try:
    import collections
except Exception:
    pass
try:
    import math
except Exception:
    pass

def crop(fr_a,r0,r1,c0,c1,tag):
    print(tag)
    for r in range(r0,r1+1): print('  ',r, fr_a[r][c0:c1+1])

def probe(r,c):
    res = action([{'action':'MOUSE','row':r,'col':c}])
    anim = res.get('animation')
    out = {'rc':(r,c),'noop':res.get('no_op'),'gp':res.get('gameplay_changed'),
           'anim_frames': anim.get('frames') if anim else None,
           'anim_tb': anim.get('transient_bbox') if anim else None}
    return out

def summary(fr):
    seg = fr.segmentation
    out=[]
    for n in seg['nodes']:
        if n['color'] in ('Y','b','S','p','R','O','N','P','M'):
            b=n['boundary']
            rs=[p[0] for p in b]; cs=[p[1] for p in b]
            out.append((n['color'], n['pixels'], (min(rs),min(cs),max(rs),max(cs))))
    return out

def one(r,c,lab):
    try:
        res = action([{'action':'MOUSE','row':r,'col':c}])
    except Exception as e:
        print(lab,'ERR',type(e).__name__); return
    anim = res.get('animation')
    d = frame_diff()
    ch=[]
    for k in ('appeared','disappeared','changed_color','moved','rotated'):
        for e in d.get(k,[]):
            at = e.get('at', e.get('bbox',[9,9]))
            if at[0]>2: ch.append((k,e))
    print(lab,'noop=',res.get('no_op'),'gp=',res.get('gameplay_changed'),'anim=',(anim or {}).get('frames'),'cells',d['changed_cell'])
    for e in ch[:4]: print('    ',e[0], e[1])

def notchpos(fr=None):
    fr = fr or current_frame
    out=[]
    for n in fr.segmentation['nodes']:
        if n['color']=='c' and n['pixels']==14:
            b=n['boundary']; rs=[p[0] for p in b]; cs=[p[1] for p in b]
            if min(cs)<32: out.append((min(rs),min(cs),max(rs),max(cs)))
    return out

def marks(fr):
    out=[]
    for n in fr.segmentation['nodes']:
        if n['pixels']==3 and n['color']=='B':
            b=n['boundary']; rs=[p[0] for p in b]; cs=[p[1] for p in b]
            out.append((min(rs),min(cs)))
    return sorted(out)

def yellowbits(fr):
    out=[]
    for n in fr.segmentation['nodes']:
        if n['color']=='Y' and n['pixels']>=4:
            b=n['boundary']; rs=[p[0] for p in b]; cs=[p[1] for p in b]
            out.append((n['pixels'],(min(rs),min(cs),max(rs),max(cs))))
    return sorted(out,key=lambda t:(t[1][0],t[1][1]))

def notchL(fr):
    for n in fr.segmentation['nodes']:
        b=n['boundary']; rs=[p[0] for p in b]; cs=[p[1] for p in b]
        if n['color']=='c' and 10<=n['pixels']<=20 and max(cs)<31: return (n['pixels'],(min(rs),min(cs),max(rs),max(cs)))

def cellset(fr, color, r0,r1,c0,c1):
    a=fr.ascii.split('\n')
    s=set()
    for r in range(r0,r1+1):
        row=a[r]
        for c in range(c0,c1+1):
            if row[c]==color: s.add((r,c))
    return s

def leftnotch(fr):
    n=[x for x in fr.segmentation['nodes'] if x['color']=='c' and x['pixels']>100 and x['boundary'][0][0]<32]
    return [(x['pixels'],tuple(x['boundary'])) for x in n]

def bits(fr):
    out=[]
    for n in fr.segmentation['nodes']:
        if n['color'] in ('Y','p') and n['pixels']<=200 and n['boundary'][0][0]<40:
            b=n['boundary']; out.append((n['color'],n['pixels'],(b[0][0],b[0][1],b[-1][0],b[-1][1])))
    return out

def leftstate(fr):
    m=sorted([x for x in marks(fr) if x[1]<32])
    nt=[(n['color'],n['pixels'],(n['boundary'][0][0],n['boundary'][0][1])) for n in fr.segmentation['nodes'] if n['pixels']>100 and n['boundary'][0][1]<32 and n['boundary'][0][0]>5 and n['boundary'][0][0]<32]
    return m,nt

def ok(pos,shape):
    r0,c0=pos
    for dr,dc in shape:
        r,c=r0+dr,c0+dc
        if r<R0 or r>R1 or c<C0 or c>C1: return False
        if (r,c) in wall: return False
    return True

def okf(pos,shape,wall,R0,R1,C0,C1):
    r0,c0=pos
    for dr,dc in shape:
        r,c=r0+dr,c0+dc
        if r<R0 or r>R1 or c<C0 or c>C1 or (r,c) in wall: return False
    return True
