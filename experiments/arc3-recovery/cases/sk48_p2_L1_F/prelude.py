try:
    import json
except Exception:
    pass

def crop(r0,r1,c0,c1):
    for r in range(r0,r1+1):
        print(r, a[r][c0:c1+1])

def barlen():
    for n in current_frame.segmentation['nodes']:
        if n['boundary'][0][0]==53: return n['pixels'], n['boundary'][0], n['boundary'][1]

def wireinfo():
    a=current_frame.ascii.split('\n')
    seg=current_frame.segmentation
    head=[n for n in seg['nodes'] if n['color']=='M' and n['pixels']==18 and n['boundary'][0][0]<50]
    h=head[0]['boundary'][0] if head else None
    # find row of wire: row with 'w' chars right of head
    return h

def snapshot():
    seg=current_frame.segmentation
    out={}
    for n in seg['nodes']:
        c=n['color']; b=n['boundary'][0]; p=n['pixels']
        if c in ('R','b','N') and p==16 and b[0]<50: out[c]=b[1]
        if c=='M' and p==18 and b[0]<50: out['head']=(b[0],b[1])
        if c in ('R','b','N') and p==16 and b[0]>=50: out['leg'+c]=p
    # legend rings: white inside legend squares
    for n in seg['nodes']:
        if n['color']=='W' and n['boundary'][0][0]>=50: out.setdefault('legW',[]).append(n['boundary'][0])
    out['bar']=barlen()[0]
    return out

def wirerow():
    a=current_frame.ascii.split('\n')
    seg=current_frame.segmentation
    head=[n for n in seg['nodes'] if n['color']=='M' and n['pixels']==18 and n['boundary'][0][0]<50][0]
    r0=head['boundary'][0][0]
    mid=r0+2
    s=a[mid]
    # find last col >=16 that is w or g before first R/b/N
    end=None
    for c in range(16,47):
        if s[c] in 'wg': end=c
    return r0, mid, end
