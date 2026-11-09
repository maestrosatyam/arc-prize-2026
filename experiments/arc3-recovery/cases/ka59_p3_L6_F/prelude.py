try:
    import json
except Exception:
    pass
try:
    from collections import deque
except Exception:
    pass

def compact(d):
    return {k:v for k,v in d.items()}

def brief(d):
    out={}
    for k in ('moved','appeared','disappeared','changed_color','rotated'):
        if d.get(k):
            out[k]=[{kk:v[kk] for kk in ('color','pixels','from','to','at','bbox') if kk in v} for v in d[k]]
    out['changed']=d.get('changed_cell_count')
    return out

def nboxes(fr):
    out=[]
    for n in fr.segmentation['nodes']:
        if n['color']=='N':
            b=n['boundary']
            out.append((b[0],b[2]))
        if n['color'] in ('B','W') and n['pixels']==1:
            out.append((n['color'],b[0])) if False else None
    return out

def dots(fr):
    res=[]
    for n in fr.segmentation['nodes']:
        if n['pixels']==1 and n['color'] in ('B','W'):
            res.append((n['color'], n['boundary'][0]))
    return res

def bb(n):
    pts=n['boundary']
    rs=[p[0] for p in pts]; cs=[p[1] for p in pts]
    return (min(rs),min(cs),max(rs),max(cs))

def pos_of_player():
    seg=current_frame.segmentation
    out=[]
    for n in seg['nodes']:
        if n['color']=='N':
            out.append(('N',bb(n),n['children']))
    return out

def run(seq, verbose=True):
    out=[]
    for item in seq:
        if isinstance(item, tuple):
            act = {'action':'MOUSE','row':item[0],'col':item[1]}
            label=f'CLICK{item}'
        else:
            act=item; label=item
        r=action([act])
        d=frame_diff()
        rec=(label, r.get('no_op'), r.get('level_completed'), r.get('game_over'), d.get('changed_cell_count'))
        out.append(rec)
        if r.get('level_completed') or r.get('game_over') or r.get('done'):
            break
    for x in out: print(x)
    return out

def boxes(fr=current_frame):
    res=[]
    for n in fr.segmentation['nodes']:
        if n['color']=='N':
            res.append(bb(n))
    return res

def boxes2(fr):
    res=[]
    for n in fr.segmentation['nodes']:
        if n['color']=='N':
            res.append(bb(n))
    return res

def objs(fr):
    out=[]
    for n in fr.segmentation['nodes']:
        if n['color'] in ('N','Y'):
            out.append((n['color'], bb(n)))
    return out

def summ(fr):
    out=[]
    for n in fr.segmentation['nodes']:
        if n['color'] in ('N','O','r'): out.append((n['color'],bb(n)))
    return out

def pl():
    for n in current_frame.segmentation['nodes']:
        if n['color']=='N' and n['pixels']<12 and bb(n)[0]<60: return bb(n)
    return None
