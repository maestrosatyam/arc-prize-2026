try:
    import json
except Exception:
    pass

def compact(d):
    return {k:v for k,v in d.items() if k!='changed_cell_count'}

def state():
    seg = current_frame.segmentation
    out = []
    for n in seg['nodes']:
        if n['color'] in ('c','G','N','W','b','g'):
            b = n['boundary']
            out.append((n['color'], n['pixels'], b[0][0], b[0][1]))
    return out

def crop(fr,r0,c0,r1,c1):
    a=fr.ascii.split('\n')
    for r in range(r0*4,r1*4+4):
        print(r, a[r][c0*4:c1*4+4])

def objs(fr):
    out=[]
    for n in fr.segmentation['nodes']:
        b=n['boundary']
        r0,c0=min(p[0] for p in b)//4,min(p[1] for p in b)//4
        r1,c1=max(p[0] for p in b)//4,max(p[1] for p in b)//4
        out.append((n['color'],n['pixels'],(r0,c0),(r1,c1)))
    return out

def find(fr, col):
    out=[]
    for n in fr.segmentation['nodes']:
        if n['color']==col:
            b=n['boundary']
            out.append(((min(p[0] for p in b)//4, min(p[1] for p in b)//4), n['pixels']))
    return out

def snap(fr):
    d={}
    for n in fr.segmentation['nodes']:
        b=n['boundary']; p=(min(x[0] for x in b)//4,min(x[1] for x in b)//4)
        d.setdefault(n['color'],[]).append((p,n['pixels']))
    return {k:sorted(v) for k,v in d.items()}

def pos(fr,col):
    for n in fr.segmentation['nodes']:
        if n['color']==col:
            b=n['boundary']; return (min(p[0] for p in b)//4,min(p[1] for p in b)//4)

def bar(fr):
    row=fr.ascii.split('\n')[63]; return row.count('P'),row.count('c')

def run(seq, verbose=True):
    out=[]; i=0
    while i < len(seq):
        a=seq[i]
        try:
            r=action([a])
            st=r.get('state') or r.get('stop_reason')
            out.append((a,st))
            if r.get('game_over') or r.get('level_completed') or r.get('done'):
                out.append(('END',st)); break
            i+=1
        except Exception as e:
            out.append((a,'stale:'+type(e).__name__))
            try:
                r=action([a]); out.append((a,'retry_ok')); i+=1
            except Exception as e2:
                out.append((a,'retry_fail:'+type(e2).__name__)); break
    if verbose:
        s=snap(current_frame)
        print(out)
        print('N',s.get('N'),'W',s.get('W'),'O',s.get('O'),'B',s.get('B'),'c',s.get('c'),'P',bar(current_frame)[0])
    return out

def grid(fr):
    rows=fr.ascii.split('\n')
    return [[rows[r*4+2][c*4+2] for c in range(16)] for r in range(16)]
