try:
    import json
except Exception:
    pass

def crop(r0,r1,c0,c1):
    for r in range(r0,r1+1):
        print(r, a[r][c0:c1+1])

def rows_runs(rows, cols):
    out=[]
    for r in rows:
        line=a[r]
        runs=[]
        c=0
        while c<64:
            ch=line[c]
            if ch in 'SNbp':
                c2=c
                while c2<64 and line[c2]==ch: c2+=1
                runs.append((ch,c,c2-1))
                c=c2
            else: c+=1
        out.append((r,runs))
    return out

def summarize(tag):
    d = frame_diff()
    print(tag,'changed',d['changed_cell_count'])
    for key in ('moved','appeared','disappeared','changed_color','resized','rotated'):
        if d.get(key):
            for e in d[key][:6]: print('  ',key,e)

def player_pos(fr):
    seg = fr.segmentation
    for n in seg['nodes']:
        if n['color']=='b':
            b=n['boundary']
            rs=[p[0] for p in b]; cs=[p[1] for p in b]
            return (min(rs),max(rs),min(cs),max(cs))
    return None

def grid(fr):
    return [list(fr.ascii.split('\n')[r]) for r in range(64)]

def best_shift(gA, gB, lo=0, hi=48):
    # find s>=0 s.t. gA[r+s][c] == gB[r][c] for most cells (world moved down by s in view)
    best=(0,-1)
    for s in range(lo,hi+1):
        m=0; tot=0
        for r in range(0,64-s,2):
            rowA=gA[r+s]; rowB=gB[r]
            for c in range(0,64,2):
                tot+=1
                if rowA[c]==rowB[c]: m+=1
        if m/tot>best[1]: best=(s,m/tot)
    return best

def best_shift2(gA,gB,lo=0,hi=48):
    best=(0,-1)
    for s in range(lo,hi+1):
        m=0;tot=0
        for r in range(0,64-s,1):
            rowA=gA[r+s]; rowB=gB[r]
            for c in range(0,64,1):
                if rowB[c] in 'SNP':
                    tot+=1
                    if rowA[c] in 'SNP': m+=1
        if tot and m/tot>best[1]: best=(s,round(m/tot,3))
    return best

def binmap(fr):
    out=[]
    for line in fr.ascii.split('\n'):
        out.append([1 if ch in 'SNPbY' else 0 for ch in line])
    return out

def rel_shift(A,B):
    # returns s such that B[r] ~ A[r+s]  (content moved DOWN by s)
    best=(0,10**9)
    for s in range(-60,61):
        mism=0; tot=0
        r0=max(0,-s); r1=min(64,64-s)
        for r in range(r0,r1):
            ra=A[r+s]; rb=B[r]
            for c in range(64):
                tot+=1
                if ra[c]!=rb[c]: mism+=1
        if tot<500: continue
        if mism<best[1]: best=(s,mism)
    return best

def ncells(fr):
    s=set()
    for r,line in enumerate(fr.ascii.split('\n')):
        for c,ch in enumerate(line):
            if ch in 'NPG': s.add((r,c))
    return s

def shift_by_blocks(A,B):
    a=ncells(A); b=ncells(B)
    res=[]
    for s in range(-60,61):
        m=sum(1 for (r,c) in b if (r+s,c) in a)
        res.append((m/len(b), s, m, len(b)))
    res.sort(reverse=True)
    return res[:4]

def halo(fr):
    p=player_pos(fr)
    if not p: return None
    r0,r1,c0,c1=p
    return (r0-1,r1+1,c0-2,c1+2)

def build_world():
    frames=[history[i].frame for i in range(len(history))]
    offs=[0]
    for i in range(1,len(frames)):
        res=shift_by_blocks(frames[i-1],frames[i])
        s=res[0][1]
        if res[0][0]<0.5: s=0
        offs.append(offs[-1]-s)   # camera shift = -s (content moved down by -s)
    W={}
    for i,fr in enumerate(frames):
        cam=offs[i]; al=fr.ascii.split('\n'); h=halo(fr)
        for r in range(63):
            wr=r-cam
            for c in range(64):
                ch=al[r][c]
                if h and (h[0]<=r<=h[1]) and (h[2]<=c<=h[3]) and ch=='B': continue
                W[(wr,c)]=ch
    return offs,W

def world_rows(W): return sorted(set(k[0] for k in W))

def wline(W,wr,c0=10,c1=56): return ''.join(W.get((wr,c),'.') for c in range(c0,c1))

def build_world_lv(lv):
    idx=[i for i in range(len(history)) if history[i].frame.level==lv]
    frames=[history[i].frame for i in idx]
    offs=[0]
    for i in range(1,len(frames)):
        res=shift_by_blocks(frames[i-1],frames[i])
        s=res[0][1]
        if res[0][0]<0.5: s=0
        offs.append(offs[-1]-s)
    W={}
    for i,fr in enumerate(frames):
        cam=offs[i]; al=fr.ascii.split('\n'); h=halo(fr)
        for r in range(63):
            wr=r-cam
            for c in range(64):
                ch=al[r][c]
                if h and (h[0]<=r<=h[1]) and (h[2]<=c<=h[3]) and ch=='B': continue
                W[(wr,c)]=ch
    return idx,offs,W

def bar(fr):
    line=fr.ascii.split('\n')[63]
    from collections import Counter
    return Counter(ch for ch in line if ch!='B')

def scells(fr,chars='S'):
    st=set()
    for r,line in enumerate(fr.ascii.split('\n')):
        if r==63: continue
        for c,ch in enumerate(line):
            if ch in chars: st.add((r,c))
    return st

def shift_by_water(A,B):
    a=scells(A); b=scells(B)
    res=[]
    for s in range(-60,61):
        inter=sum(1 for (r,c) in b if (r+s,c) in a)
        rec=inter/len(b)
        aregion=set((r-s,c) for (r,c) in a if 0<=r-s<63)
        prec=inter/max(1,len(aregion))
        res.append((round(rec,3),round(prec,3),s))
    res.sort(reverse=True)
    return res[:5]

def shift_by_purple(A,B):
    a=scells(A,'p'); b=scells(B,'p')
    if not b: return []
    res=[]
    for s in range(-60,61):
        inter=sum(1 for (r,c) in b if (r+s,c) in a)
        res.append((inter/len(b),s,inter,len(b)))
    res.sort(reverse=True); return res[:5]

def build_world2():
    cams={20:0,21:0,22:0,23:0,24:0,25:0,26:0,27:48,28:66}
    W={}
    for i,cam in cams.items():
        fr=history[i].frame; al=fr.ascii.split('\n'); h=halo(fr)
        for r in range(63):
            wr=r-cam
            for c in range(64):
                ch=al[r][c]
                if h and (h[0]<=r<=h[1]) and (h[2]<=c<=h[3]) and ch=='B': continue
                W[(wr,c)]=ch
    return W

def runs_row(W,wr,chars):
    out=[]
    c=0
    while c<64:
        ch=W.get((wr,c),'.')
        if ch in chars:
            c2=c
            while c2<64 and W.get((wr,c2),'.')==ch: c2+=1
            out.append((ch,c,c2-1)); c=c2
        else: c+=1
    return out

def wruns(fr,r0,r1,c0=12,c1=56,chars='SN'):
    al=fr.ascii.split('\n')
    for r in range(r0,r1+1):
        row=al[r]
        out=[]
        for ch in chars:
            for run in runs_row({(c,1):row[c] for c in range(64)},r,ch):
                if run[1]<=c1 and run[2]>=c0: out.append(run[1:])
        print(r,out)
