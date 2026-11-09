try:
    import json
except Exception:
    pass
try:
    import itertools, json
except Exception:
    pass
try:
    import pickle
except Exception:
    pass
try:
    import itertools
except Exception:
    pass
try:
    from collections import Counter
except Exception:
    pass
try:
    import inspect
except Exception:
    pass
try:
    from collections import Counter as C2
except Exception:
    pass
try:
    import time
except Exception:
    pass
try:
    from collections import Counter as C4
except Exception:
    pass
try:
    from collections import Counter as CC
except Exception:
    pass

def crop(r0,r1,c0,c1,lbl):
    print('---',lbl)
    print('    '+''.join(str((12+c)%10) for c in range(c0,c1+1)))
    for rr in range(r0,r1+1):
        print(rr, a[rr][c0:c1+1])

def compact(d):
    out={}
    for k,v in d.items():
        if isinstance(v,list):
            out[k]=[{kk:vv for kk,vv in e.items() if kk in ('color','pixels','at','at_before','at_after','bbox','before_color','after_color','rotated_by')} for e in v]
        else:
            out[k]=v
    return out

def dump(r0,r1,c0,c1):
    print('    '+''.join(('%d'%c)[1] if c>=10 else ' ' for c in range(c0,c1+1)))
    for rr in range(r0,r1+1):
        print('%2d '%rr + ''.join(a[rr][c] for c in range(c0,c1+1)))

def comp(V,pp,cpar,ccol):
    out=[]
    bp=[q for q in absv(V[cpar],pp[0],pp[1])[1]]
    for k in range(4):
        cq=absv(V[ccol],k,(0,0))[1]
        for px in bp:
            for py in cq:
                off=(min(r for r,c in px)-min(r for r,c in py), min(c for r,c in px)-min(c for r,c in py))
                if off in [o for _,o in out]: continue
                r,nn=okp(V,pp,cpar,(k,off),ccol)
                if r: out.append(((k,off),nn))
    return out

def comps(pred):
    seen=[[False]*64 for _ in range(64)]; out=[]
    for r in range(64):
        for c in range(64):
            if pred(r,c,G[r][c]) and not seen[r][c]:
                st=[(r,c)]; comp=[]; col=G[r][c]
                while st:
                    y,x=st.pop()
                    if seen[y][x]: continue
                    seen[y][x]=True; comp.append((y,x))
                    for dy,dx in((1,0),(-1,0),(0,1),(0,-1)):
                        ny,nx=y+dy,x+dx
                        if 0<=ny<64 and 0<=nx<64 and not seen[ny][nx] and G[ny][nx]==col: st.append((ny,nx))
                out.append((col,comp))
    return out

def bbox(cs):
    rs=[p[0] for p in cs];c2=[p[1] for p in cs]
    return (min(rs),max(rs),min(c2),max(c2))

def norm(cells):
    r0=min(p[0] for p in cells); c0=min(p[1] for p in cells)
    return frozenset((r-r0,c-c0) for r,c in cells)

def rot90(cells):  # 90 CW about origin: (r,c)->(c,-r)
    return [(c,-r) for r,c in cells]

def owner_of(comp):
    for (x,y) in comp:
        for dx,dy in((-1,0),(1,0),(0,-1),(0,1)):
            nx,ny=x+dx,y+dy
            if 0<=nx<64 and 0<=ny<64 and G[nx][ny] in 'WNYb': return G[nx][ny]
    return None

def largest(ch): return max(comps(ch), key=len)

def getdata():
    a=current_frame.ascii.split('\n')
    G=[[a[r][c] for c in range(64)] for r in range(64)]
    def comps(ch):
        seen=set(); out=[]
        for r in range(64):
            for c in range(64):
                if G[r][c]==ch and (r,c) not in seen:
                    st=[(r,c)]; comp=[]
                    while st:
                        x,y=st.pop()
                        if (x,y) in seen: continue
                        seen.add((x,y)); comp.append((x,y))
                        for dx,dy in((1,0),(-1,0),(0,1),(0,-1)):
                            nx,ny=x+dx,y+dy
                            if 0<=nx<64 and 0<=ny<64 and grid_ok(G,nx,ny) and G[nx][ny]==ch and (nx,ny) not in seen: st.append((nx,ny))
                    out.append(comp)
        return out
    body={}
    for ch in 'WNYb':
        body[ch]=max(comps(ch),key=len)
        if ch=='W' and len(body['W'])<10:  # ignore hud dot
            cs=[c for c in comps(ch) if len(c)>10]
            body[ch]=max(cs,key=len) if cs else []
    prongs={ch:[] for ch in 'WNYb'}
    for c in comps('R'):
        if min(x for x,y in c)==0: continue
        own=None
        for (x,y) in c:
            for dx,dy in((-1,0),(1,0),(0,-1),(0,1)):
                if 0<=x+dx<64 and 0<=y+dy<64 and G[x+dx][y+dy] in 'WNYb': own=G[x+dx][y+dy]
        if own: prongs[own].append(c)
    return {'body':body,'prongs':prongs,'G':G}

def grid_ok(G,r,c): return 0<=r<64 and 0<=c<64

def search(names,root,need_all=True,maxres=4,cap=200000):
    import itertools
    res=[];nodes=[0]
    def rec(placed,occ_b,occ_p,prcnt):
        if len(res)>=maxres or nodes[0]>cap: return
        nodes[0]+=1
        if len(placed)==len(names):
            if need_all and any(v!=2 for v in prcnt.values()): return
            res.append({n:(placed[n][0],placed[n][1]) for n in names});return
        rest=[n for n in names if n not in placed]
        for n in rest:
            for k in range(4):
                b0,p0=pose(n,k,(0,0))
                tried=set()
                for q in p0:
                    for x in occ_p:
                        off=(x[0]-q[0],x[1]-q[1])
                        if off in tried: continue
                        tried.add(off)
                        b,p=pose(n,k,off)
                        if b&occ_b or (b&occ_p) or (p&occ_b): continue
                        pc=dict(prcnt);bad=False
                        for y in p:
                            pc[y]=pc.get(y,0)+1
                            if pc[y]>2: bad=True
                        if bad: continue
                        placed[n]=(k,off,b,p)
                        rec(placed,occ_b|b,occ_p|p,pc)
                        placed.pop(n,None)
                        if len(res)>=maxres: return
    for kr in range(4):
        b,p=pose(root,kr,(0,0))
        rec({root:(kr,(0,0),b,p)},b,p,{q:1 for q in p})
        if len(res)>=maxres: break
    return res

def placements(d):
    Wb=d['body']['W']; Wp=d['prongs']['W']
    unit=set(map(tuple,Wb))|set(tuple(v) for pr in Wp for v in pr)
    ru=norm(unit); rp=[norm(set(map(tuple,pr))) for pr in Wp]
    others={c:set(map(tuple,d['body'][c])) for c in 'WNYb'}
    othersP={c:set().union(*[set(map(tuple,p)) for p in d['prongs'][c]]) for c in 'WNYb'}
    out=[]; seen=set()
    for k in range(4):
        if k>0:
            ru=norm({(c,-r) for r,c in ru}); rp=[norm({(c,-r) for r,c in p}) for p in rp]
        for S in 'NYb':
            Sb=set(map(tuple,d['body'][S])); Sp=[set(map(tuple,p)) for p in d['prongs'][S]]
            SpU=set().union(*Sp)
            otherB=set().union(*[others[c] for c in 'WNYb' if c!=S])
            otherP=set().union(*[othersP[c] for c in 'WNYb' if c!=S])
            for i in range(2):
                for j in range(len(Sp)):
                    src=min(rp[i]); tgt=min(Sp[j]); t=(tgt[0]-src[0],tgt[1]-src[1])
                    placed={(r+t[0],c+t[1]) for r,c in ru}
                    prset=[{(r+t[0],c+t[1]) for r,c in p} for p in rp]
                    Wbody=placed-set().union(*prset)
                    ok=True; matched=0
                    if Wbody & Sb or Wbody & SpU: ok=False
                    for p in prset:
                        if p & Sb: ok=False
                        for q in Sp:
                            if p&q:
                                if p==q: matched+=1
                                else: ok=False
                    if min(r for r,c in placed)<1 or max(r for r,c in placed)>63: ok=False
                    if min(c for r,c in placed)<0 or max(c for r,c in placed)>63: ok=False
                    if Wbody & otherB or placed & otherP: ok=False
                    key=(k,S,bbox(placed))
                    if key in seen: continue
                    seen.add(key); out.append((k,S,matched,ok,bbox(placed)))
    return out

def getdata2():
    a=current_frame.ascii.split('\n')
    G=[[a[r][c] for c in range(64)] for r in range(64)]
    seen=[[False]*64 for _ in range(64)]
    out=[]
    for r in range(64):
        for c in range(64):
            if G[r][c] in 'WpNb' and not seen[r][c]:
                st=[(r,c)]; comp=[]
                while st:
                    x,y=st.pop()
                    if seen[x][y]: continue
                    seen[x][y]=True; comp.append((x,y))
                    for dx,dy in((1,0),(-1,0),(0,1),(0,-1)):
                        nx,ny=x+dx,y+dy
                        if 0<=nx<64 and 0<=ny<64 and not seen[nx][ny] and G[nx][ny]==G[x][y]: st.append((nx,ny))
                if len(comp)>20: out.append({'color':G[r][c],'body':sorted(comp)})
    # red prongs
    seenr=[[False]*64 for _ in range(64)]
    prs=[]
    for r in range(64):
        for c in range(64):
            if G[r][c]=='R' and not seenr[r][c]:
                st=[(r,c)]; comp=[]
                while st:
                    x,y=st.pop()
                    if seenr[x][y]: continue
                    seenr[x][y]=True; comp.append((x,y))
                    for dx,dy in((1,0),(-1,0),(0,1),(0,-1)):
                        nx,ny=x+dx,y+dy
                        if 0<=nx<64 and 0<=ny<64 and G[nx][ny]=='R' and not seenr[nx][ny]: st.append((nx,ny))
                if min(x for x,y in comp)>0: prs.append(sorted(comp))
    for p in prs:
        own=None
        for (x,y) in p:
            for dx,dy in((-1,0),(1,0),(0,-1),(0,1)):
                if 0<=x+dx<64 and 0<=y+dy<64 and G[x+dx][y+dy] in 'WpNb': own=G[x+dx][y+dy]
        p.append('->'+str(own))
    return out, prs

def build():
    objs,prs=getdata2()
    data={}
    for o in objs:
        pr=[p[:-1] for p in prs if p[-1]=='->'+o['color']]
        data[o['color']]={'body':[tuple(x) for x in o['body']],'prongs':[[tuple(x) for x in q] for q in pr]}
    return data

def canon(d):
    # returns per color: list over k of (body_set, prong_sets, anchor) normalized so bbox of whole unit at (0,0)
    res={}
    for ch,v in d.items():
        unit=[tuple(x) for x in v['body']]+[x for p in v['prongs'] for x in p]
        base=norm(set(unit))
        cur_b=norm(set(v['body'])); cur_p=[norm(set(p)) for p in v['prongs']]
        variants=[]
        for k in range(4):
            if k>0:
                base=norm({(c,-r) for r,c in base})
                cur_b=norm({(c,-r) for r,c in cur_b}); cur_p=[norm({(c,-r) for r,c in p}) for p in cur_p]
            variants.append((set(cur_b),[set(p) for p in cur_p],set(base)))
        res[ch]=variants
    return res

def pose_cells(var, off):
    b,p,u=var
    return {('B',r+off[0],c+off[1]) for r,c in b}, {('P',r+off[0],c+off[1]) for r,c in p}

def compat(sp,a,b,pa,offb,kb):
    A=pose2(sp,a,pa,(0,0));B=pose2(sp,b,kb,offb)
    if A[0]&B[0]: return False
    if (A[0]&B[1]) or (A[1]&B[0]): return False
    inter=A[1]&B[1]
    if not inter: return False
    same=[p for p in inter if any(q in A[1] for q in [p])]
    # forbid 3 prongs at one location: check no 3rd object needed; here only 2 objects so fine
    return True

def poses_comp(C, poseA, Acol, Bcol):
    kA,offA=poseA
    bA,pA,uA=C[Acol][kA]
    Ab={(r+offA[0],c+offA[1]) for r,c in bA}
    Ap=[{(r+offA[0],c+offA[1]) for r,c in q} for q in pA]
    ApU=set().union(*Ap)
    res=[]; seen=set()
    for k in range(4):
        bB,pB,uB=C[Bcol][k]
        for q in Ap:
            qt=min(q)
            for pb in pB:
                pt=min(pb)
                off=(qt[0]-pt[0], qt[1]-pt[1])
                if (k,off) in seen: continue
                seen.add((k,off))
                Bb={(r+off[0],c+off[1]) for r,c in bB}
                Bp=[{(r+off[0],c+off[1]) for r,c in p} for p in pB]
                BpU=set().union(*Bp)
                if Bb & Ab: continue
                if Bb & ApU: continue
                if BpU & Ab: continue
                bad=False
                for p in Bp:
                    for qq in Ap:
                        if p & qq and p!=qq: bad=True
                    if bad: break
                if bad: continue
                res.append((k,off))
    return res

def cells_of(SH,cfg,names):
    out=set()
    for n in names:
        k,off=cfg[n]
        b,p=poseS(SH,n,k,off)
        out|=b|p
    return out

def ok_pair(C,c1,pose1,c2,pose2):
    b1,p1=cells_of(C,c1,pose1); b2,p2=cells_of(C,c2,pose2)
    if b1&b2: return False
    p1u=set().union(*p1); p2u=set().union(*p2)
    if b1&p2u or b2&p1u: return False
    for a in p1:
        for b in p2:
            if a&b and a!=b: return False
    return True

def trees4():
    import itertools
    edges=[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    out=[]
    for sub in itertools.combinations(edges,3):
        par={0:0,1:1,2:2,3:3}
        def find(x):
            while par[x]!=x: x=par[x]
            return x
        for a,b in sub:
            ra,rb=find(a),find(b)
            if ra!=rb: par[ra]=rb
        if len({find(x) for x in range(4)})==1: out.append(list(sub))
    return out

def solve(C, root='W'):
    cols=['W','p','N','b']
    # root pose fixed at canonical k=0 with anchor = current position
    d=build()
    unit=[tuple(x) for x in d[root]['body']]+[x for p in d[root]['prongs'] for x in p]
    anchor=(min(r for r,c in unit), min(c for r,c in unit))
    rootpose=(0,anchor)
    sols=[]
    for T in trees4():
        # map node idx -> color
        order=[cols.index(root)]
        adj={i:[] for i in range(4)}
        for a,b in T: adj[a].append(b); adj[b].append(a)
        # BFS order
        seq=[]; frontier=[cols.index(root)]
        while frontier:
            nxt=[]
            for n in frontier:
                seq.append(n)
                for m in adj[n]:
                    if m not in seq and m not in nxt: nxt.append(m)
            frontier=nxt
        def dfs(i, poses):
            if i>=len(seq):
                # verify all pairs ok
                for a,b in [(x,y) for x in range(4) for y in range(4) if x<y]:
                    if not ok_pair(C,cols[a],poses[a],cols[b],poses[b]): return
                # connectivity via coincident prongs
                conn=[[False]*4 for _ in range(4)]
                matched=0
                for a in range(4):
                    for b in range(a+1,4):
                        _,pa=cells_of(C,cols[a],poses[a]); _,pb=cells_of(C,cols[b],poses[b])
                        for x in pa:
                            for y in pb:
                                if x==y: matched+=1
                for a in range(4):
                    for b in range(4):
                        if a!=b:
                            _,pa=cells_of(C,cols[a],poses[a]); _,pb=cells_of(C,cols[b],poses[b])
                            if any(x==y for x in pa for y in pb): conn[a][b]=True
                # BFS connectivity
                seen={seq[0]}; st=[seq[0]]
                while st:
                    n=st.pop()
                    for m in range(4):
                        if conn[n][m] and m not in seen: seen.add(m); st.append(m)
                if len(seen)<4: return
                sols.append((matched, dict(zip(cols,poses))))
                return
            node=seq[i]; col=cols[node]
            # parent already placed: pick first placed neighbor
            parents=[m for m in adj[node] if m in poses]
            cand=[]
            for pnode in parents[:1]:
                cand=poses_comp(C,poses[pnode],cols[pnode],col)
            for pose in cand:
                good=True
                for m,pp in poses.items():
                    if not ok_pair(C,col,pose,cols[m],pp): good=False; break
                if good:
                    poses[node]=pose
                    dfs(i+1,poses)
                    del poses[node]
        dfs(1,{cols.index(root):rootpose})
    return sols

def curpose(col):
    unit=[tuple(x) for x in d[col]['body']]+[x for p in d[col]['prongs'] for x in p]
    return (0,(min(r for r,c in unit), min(c for r,c in unit)))

def curpose2(d,col):
    unit=[tuple(x) for x in d[col]['body']]+[x for p in d[col]['prongs'] for x in p]
    return (0,(min(r for r,c in unit), min(c for r,c in unit)))

def dbg(C,poseA,Acol,Bcol):
    kA,offA=poseA
    bA,pA,uA=C[Acol][kA]
    Ab={(r+offA[0],c+offA[1]) for r,c in bA}
    Ap=[{(r+offA[0],c+offA[1]) for r,c in q} for q in pA]
    ApU=set().union(*Ap)
    stats={'total':0,'bodybody':0,'bodyprong':0,'prongbody':0,'prongmis':0,'ok':0}
    seen=set()
    for k in range(4):
        bB,pB,uB=C[Bcol][k]
        for q in Ap:
            qt=min(q)
            for pb in pB:
                pt=min(pb); off=(qt[0]-pt[0],qt[1]-pt[1])
                if (k,off) in seen: continue
                seen.add((k,off)); stats['total']+=1
                Bb={(r+off[0],c+off[1]) for r,c in bB}
                Bp=[{(r+off[0],c+off[1]) for r,c in p} for p in pB]
                BpU=set().union(*Bp)
                if Bb&Ab: stats['bodybody']+=1; continue
                if Bb&ApU: stats['bodyprong']+=1; continue
                if BpU&Ab: stats['prongbody']+=1; continue
                bad=False
                for p in Bp:
                    for qq in Ap:
                        if p&qq and p!=qq: bad=True
                if bad: stats['prongmis']+=1; continue
                stats['ok']+=1
    return stats

def canon2(d):
    res={}
    for ch,v in d.items():
        unit=[tuple(x) for x in v['body']]+[tuple(x) for p in v['prongs'] for x in p]
        B=[tuple(x) for x in v['body']]; P=[[tuple(x) for x in q] for q in v['prongs']]
        variants=[]
        for k in range(4):
            if k>0:
                unit=[(c,-r) for r,c in unit]; B=[(c,-r) for r,c in B]
                P=[[(c,-r) for r,c in q] for q in P]
            dr=min(r for r,c in unit); dc=min(c for r,c in unit)
            us={(r-dr,c-dc) for r,c in unit}
            bs={(r-dr,c-dc) for r,c in B}
            ps=[{(r-dr,c-dc) for r,c in q} for q in P]
            variants.append((bs,ps,us))
        res[ch]=variants
    return res

def curpose3(dd,col):
    unit=[tuple(x) for x in dd[col]['body']]+[x for p in dd[col]['prongs'] for x in p]
    return (0,(min(r for r,c in unit), min(c for r,c in unit)))

def solve2(sp,names,root,dmin=-14,dmax=14,need_all=True):
    import itertools
    res=[]
    others=[n for n in names if n!=root]
    def rec(order,i,placed,occ_b,occ_p):
        if i==len(order):
            # check all prong locations have exactly 2 prongs
            if need_all:
                cnt={}
                for n in names:
                    for p in placed[n][1]: cnt[p]=cnt.get(p,0)+1
                if any(v!=2 for v in cnt.values()): return
            res.append(dict(placed));return
        n=order[i]
        for k in range(4):
            for dr in range(dmin,dmax+1):
                for dc in range(dmin,dmax+1):
                    b,p=pose2(sp,n,k,(dr,dc))
                    if b&occ_b: continue
                    if (b&occ_p) or (p&occ_b): continue
                    shared=p&occ_p
                    if not shared: continue
                    # ensure no location gets >2 prongs
                    ok=True
                    for q in p:
                        c=occ_p.count(q) if isinstance(occ_p,list) else None
                    cnt={}
                    for q in p: cnt[q]=1
                    for q in occ_p: 
                        if q in cnt: cnt[q]+=1
                    if any(v>2 for v in cnt.values()): ok=False
                    if not ok: continue
                    placed[n]=(k,(dr,dc),b,p)
                    rec(order,i+1,placed,occ_b|b,occ_p|p)
                    del placed[n]
    for order in itertools.permutations(others):
        for kr in range(4):
            b,p=pose2(sp,root,kr,(0,0))
            rec(list(order),1,{root:(kr,(0,0),b,p)},b,p)
            if res: return res
    return res

def solve3(sp,names,root,dmin=-14,dmax=14,need_all=True,maxres=1):
    import itertools
    res=[]
    others=[n for n in names if n!=root]
    def rec(order,i,placed,occ_b,occ_p):
        if len(res)>=maxres: return
        if i==len(order):
            if need_all:
                cnt={}
                for n in names:
                    for p in placed[n][3]: cnt[p]=cnt.get(p,0)+1
                if any(v!=2 for v in cnt.values()): return
            res.append({n:(placed[n][0],placed[n][1]) for n in names});return
        n=order[i]
        for k in range(4):
            for dr in range(dmin,dmax+1):
                for dc in range(dmin,dmax+1):
                    b,p=pose2(sp,n,k,(dr,dc))
                    if b&occ_b or (b&occ_p) or (p&occ_b): continue
                    if not (p&occ_p): continue
                    cnt={}
                    for q in p: cnt[q]=cnt.get(q,0)+1
                    for q in occ_p: cnt[q]=cnt.get(q,0)+1
                    if any(v>2 for v in cnt.values()): continue
                    placed[n]=(k,(dr,dc),b,p)
                    rec(order,i+1,placed,occ_b|b,occ_p|p)
                    placed.pop(n,None)
                    if len(res)>=maxres: return
    for order in itertools.permutations(others):
        for kr in range(4):
            b,p=pose2(sp,root,kr,(0,0))
            rec(list(order),0,{root:(kr,(0,0),b,p)},b,p)
            if len(res)>=maxres: return res
    return res

def cost(sh,P0,active,names):
    tot=0
    for n in names:
        dr=abs(sh[n][1][0]-P0[n][1][0]);dc=abs(sh[n][1][1]-P0[n][1][1])
        if dr%3 or dc%3: return None
        tot+=(dr+dc)//3+(sh[n][0]-P0[n][0])%4
        if n!=active: tot+=1
    return tot

def costf(ps):
    tot=0; detail=[]
    for c in 'WpNb':
        k,off=ps[c]; ck,coff=cur[c]
        dr=off[0]-coff[0]; dc=off[1]-coff[1]
        if dr%3 or dc%3: return None
        mv=(abs(dr)+abs(dc))//3
        tot+= k%4 + mv + (0 if c=='W' else 1)
        detail.append((c,k,dr,dc))
    return tot,detail

def costf2(ps,cur):
    tot=0
    for c in 'WpNb':
        k,off=ps[c]; ck,coff=cur[c]
        dr=off[0]-coff[0]; dc=off[1]-coff[1]
        if dr%3 or dc%3: return None
        tot+= k%4+(abs(dr)+abs(dc))//3+(0 if c=='W' else 1)
    return tot

def cells_at(col,pose):
    return cells_of(C,col,pose)

def ok_board(ps):
    for c in 'WpNb':
        b,p=cells_at(c,ps[c])
        allc=b|set().union(*p)
        if min(r for r,cc in allc)<1 or max(r for r,cc in allc)>63: return False
        if min(cc for r,cc in allc)<0 or max(cc for r,cc in allc)>63: return False
    return True

def solve4(C, root, rootpose):
    cols=['W','p','N','b']
    sols=[]
    for T in trees4():
        adj={i:[] for i in range(4)}
        for a,b in T: adj[a].append(b); adj[b].append(a)
        ri=cols.index(root); seq=[ri]; seen={ri}; frontier=[ri]
        while frontier:
            nxt=[]
            for n in frontier:
                for m in adj[n]:
                    if m not in seen: seen.add(m); nxt.append(m)
            seq+=nxt; frontier=nxt
        def dfs(i,poses):
            if i==4:
                conn=[[False]*4 for _ in range(4)]; matched=0
                for a in range(4):
                    for b in range(a+1,4):
                        if not ok_pair(C,cols[a],poses[a],cols[b],poses[b]): return
                        _,pa=cells_of(C,cols[a],poses[a]); _,pb=cells_of(C,cols[b],poses[b])
                        for x in pa:
                            for y in pb:
                                if x==y: matched+=1; conn[a][b]=True
                vis={seq[0]}; st=[seq[0]]
                while st:
                    n=st.pop()
                    for m in range(4):
                        if conn[n][m] and m not in vis: vis.add(m); st.append(m)
                if len(vis)<4: return
                sols.append((matched,{cols[i]:poses[i] for i in range(4)}))
                return
            node=seq[i]; col=cols[node]
            pn=[m for m in adj[node] if m in poses][0]
            for pose in poses_comp(C,poses[pn],cols[pn],col):
                if all(ok_pair(C,col,pose,cols[m],pp) for m,pp in poses.items()):
                    poses[node]=pose; dfs(i+1,poses); del poses[node]
        dfs(1,{ri:rootpose})
    return sols

def board_ok(sh):
    for c in 'WpNb':
        b,p=cells_of(C,c,sh[c]); allc=b|set().union(*p)
        if min(r for r,_ in allc)<1 or max(r for r,_ in allc)>63: return False
        if min(x for _,x in allc)<0 or max(x for _,x in allc)>63: return False
    return True

def m(x,y): return sum(1 for i in x for j in y if i==j)

def extract(fr):
    a=fr.ascii.split('\n'); G=[[a[r][c] for c in range(64)] for r in range(64)]
    seen=[[False]*64 for _ in range(64)]; objs=[]
    for r in range(64):
        for c in range(64):
            if G[r][c] in 'WpNbNY' and not seen[r][c]:
                st=[(r,c)]; comp=[]; col=G[r][c]
                while st:
                    x,y=st.pop()
                    if seen[x][y]: continue
                    seen[x][y]=True; comp.append((x,y))
                    for dx,dy in((1,0),(-1,0),(0,1),(0,-1)):
                        nx,ny=x+dx,y+dy
                        if 0<=nx<64 and 0<=ny<64 and not seen[nx][ny] and G[nx][ny]==col: st.append((nx,ny))
                if len(comp)>20: objs.append({'color':col,'body':sorted(comp)})
    sr=[[False]*64 for _ in range(64)]; prs=[]
    for r in range(64):
        for c in range(64):
            if G[r][c]=='R' and not sr[r][c]:
                st=[(r,c)]; comp=[]
                while st:
                    x,y=st.pop()
                    if sr[x][y]: continue
                    sr[x][y]=True; comp.append((x,y))
                    for dx,dy in((1,0),(-1,0),(0,1),(0,-1)):
                        nx,ny=x+dx,y+dy
                        if 0<=nx<64 and 0<=ny<64 and G[nx][ny]=='R' and not sr[nx][ny]: st.append((nx,ny))
                if min(x for x,y in comp)>0:
                    own=None
                    for (x,y) in comp:
                        for dx,dy in((-1,0),(1,0),(0,-1),(0,1)):
                            if 0<=x+dx<64 and 0<=y+dy<64 and G[x+dx][y+dy] in 'WpNbNY': own=G[x+dx][y+dy]
                    prs.append((sorted(comp),own))
    for o in objs:
        o['prongs']=[p for p,own in prs if own==o['color']]
    return objs

def mkref(fr):
    ro=extract(fr); names=['A','B','C','D']
    ref={}
    for nm,o in zip(names,ro):
        unit=[tuple(x) for x in o['body']]+[tuple(x) for p in o['prongs'] for x in p]
        variants=[]; u=[tuple(x) for x in unit]
        for k in range(4):
            if k>0: u=[(c,-r) for r,c in u]
            dr=min(r for r,c in u); dc=min(c for r,c in u)
            variants.append(frozenset((r-dr,c-dc) for r,c in u))
        seenv=[]; k=0
        for v in variants:
            if v in seenv: break
            seenv.append(v); k+=1
        ref[nm]={'variants':variantset for variantset in variants}
    return ref

def identify(fr):
    objs=extract(fr); out={}
    for o in objs:
        unit=[tuple(x) for x in o['body']]+[tuple(x) for p in o['prongs'] for x in p]
        u=norm(set(unit))
        found=None
        for nm,v in REF.items():
            for k in range(4):
                if u==v['variants'][k]: found=(nm,k); break
            if found: break
        if found is None: print('UNMATCHED', o['color'], len(unit)); continue
        nm,k=found
        out[nm]={'color':o['color'],'k':k,'anchor':min(u),'body':o['body'],'prongs':o['prongs']}
    return out

def refframe():
    for h in history:
        if h.frame.level==2: return h.frame
    return None

def mkref2(fr):
    ro=extract(fr); names=['A','B','C','D']; out={}
    for nm,o in zip(names,ro):
        u=[tuple(x) for x in o['body']]+[tuple(x) for p in o['prongs'] for x in p]
        variants=[]
        for k in range(4):
            if k>0: u=[(c,-r) for r,c in u]
            dr=min(r for r,c in u); dc=min(c for r,c in u)
            variants.append(frozenset((r-dr,c-dc) for r,c in u))
        out[nm]=variants
    return out

def identify2(fr,REF):
    objs=extract(fr); out={}
    for o in objs:
        u=norm(set([tuple(x) for x in o['body']]+[tuple(x) for p in o['prongs'] for x in p]))
        found=None
        for nm,variants in REF.items():
            for k in range(4):
                if u==variants[k]: found=(nm,k); break
            if found: break
        if not found: continue
        nm,k=found
        out[nm]={'color':o['color'],'k':k,'anchor':min(u),'body':o['body'],'prongs':o['prongs']}
    return out

def identify3(fr,REF):
    objs=extract(fr); out={}
    for o in objs:
        raw=[tuple(x) for x in o['body']]+[tuple(x) for p in o['prongs'] for x in p]
        a=(min(r for r,c in raw), min(c for r,c in raw))
        u=norm(set(raw))
        found=None
        for nm,variants in REF.items():
            for k in range(4):
                if u==variants[k]: found=(nm,k); break
            if found: break
        if not found: continue
        nm,k=found
        out[nm]={'color':o['color'],'k':k,'anchor':a,'body':o['body'],'prongs':o['prongs']}
    return out

def extract2(fr):
    x=a2=fr.ascii.split('\n'); G=[[x[r][c] for c in range(64)] for r in range(64)]
    seen=[[False]*64 for _ in range(64)]; objs=[]
    for r in range(64):
        for c in range(64):
            if G[r][c] in 'WpNbNY' and not seen[r][c]:
                st=[(r,c)]; comp=[]; col=G[r][c]
                while st:
                    yy,xx=st.pop()
                    if seen[yy][xx]: continue
                    seen[yy][xx]=True; comp.append((yy,xx))
                    for dy,dx in((1,0),(-1,0),(0,1),(0,-1)):
                        ny,nx=yy+dy,xx+dx
                        if 0<=ny<64 and 0<=nx<64 and not seen[ny][nx] and G[ny][nx]==col: st.append((ny,nx))
                if len(comp)>20: objs.append({'color':col,'body':sorted(comp)})
    sr=[[False]*64 for _ in range(64)]; prs=[]
    for r in range(1,64):
        for c in range(64):
            if G[r][c] in 'RG' and not sr[r][c]:
                st=[(r,c)]; comp=[]
                while st:
                    yy,xx=st.pop()
                    if sr[yy][xx]: continue
                    sr[yy][xx]=True; comp.append((yy,xx))
                    for dy,dx in((1,0),(-1,0),(0,1),(0,-1)):
                        ny,nx=yy+dy,xx+dx
                        if 0<=ny<64 and 0<=nx<64 and G[ny][nx]==G[r][c] and not sr[ny][nx]: st.append((ny,nx))
                if min(y for y,xx in comp)>0: prs.append((sorted(comp),G[r][c]))
    for o in objs:
        mine=[]
        for comp,col in prs:
            for (y,xx) in comp:
                for dy,dx in((-1,0),(1,0),(0,-1),(0,1)):
                    if 0<=y+dy<64 and 0<=xx+dx<64 and (y+dy,xx+dx) in o['body']: mine.append(tuple(comp)); break
                else: continue
                break
        o['prongs']=mine
    return objs

def identify4(fr,REF):
    objs=extract2(fr); out={}
    for o in objs:
        raw=[tuple(x) for x in o['body']]+[tuple(x) for p in o['prongs'] for x in p]
        anc=(min(r for r,c in raw), min(c for r,c in raw))
        u=norm(set(raw)); found=None
        for nm,variants in REF.items():
            for k in range(4):
                if u==variants[k]: found=(nm,k); break
            if found: break
        if not found: continue
        nm,k=found; out[nm]={'color':o['color'],'k':k,'anchor':anc,'body':o['body']}
    return out

def canonref(fr):
    ro=extract(fr); names=['A','B','C','D']; res={}
    for nm,o in zip(names,ro):
        B=[tuple(x) for x in o['body']]; P=[[tuple(x) for x in q] for q in o['prongs']]
        U=B+[x for q in P for x in q]
        variants=[]
        for k in range(4):
            if k>0:
                B=[(c,-r) for r,c in B]; P=[[(c,-r) for r,c in q] for q in P]; U=[(c,-r) for r,c in U]
            dr=min(r for r,c in U); dc=min(c for r,c in U)
            variants.append(({(r-dr,c-dc) for r,c in B},[{(r-dr,c-dc) for r,c in q} for q in P],{(r-dr,c-dc) for r,c in U}))
        res[nm]=variants
    return res

def curposes(fr,REFV):
    objs=extract2(fr); out={}
    for o in objs:
        raw=[tuple(x) for x in o['body']]+[tuple(x) for p in o['prongs'] for x in p]
        anc=(min(r for r,c in raw),min(c for r,c in raw)); u=norm(set(raw)); f=None
        for nm,vs in REFV.items():
            for k in range(4):
                if u==vs[2]: f=(nm,k); break
            if f: break
        if not f: continue
        out[f[0]]=(f[1],anc)
    return out

def curposes2(fr,REFV):
    out={}
    for o in extract2(fr):
        raw=[tuple(x) for x in o['body']]+[tuple(x) for p in o['prongs'] for x in p]
        anc=(min(x for x,y in raw),min(y for x,y in raw)); u=norm(set(raw))
        for nm,vs in REFV.items():
            hit=None
            for k in range(4):
                if u==vs[2]: hit=k; break
            if hit is not None:
                out[nm]=(hit,anc); break
    return out

def curposes3(fr,REFV):
    out={}
    for o in extract2(fr):
        raw=[tuple(x) for x in o['body']]+[tuple(x) for p in o['prongs'] for x in p]
        anc=(min(x for x,y in raw),min(y for x,y in raw)); u=norm(set(raw))
        for nm,vs in REFV.items():
            hit=None
            for k in range(4):
                if u==vs[k][2]: hit=k; break
            if hit is not None:
                out[nm]=(hit,anc); break
    return out

def board_ok2(sh,CV):
    for c in sh:
        b,p=cells_of(CV,c,sh[c]); allc=b|set().union(*p)
        if min(x for x,y in allc)<1 or max(x for x,y in allc)>63: return False
        if min(y for x,y in allc)<0 or max(y for x,y in allc)>63: return False
    return True

def cost3(sh,P0,active):
    tot=0
    for c in P0:
        k,off=sh[c]; ck,coff=P0[c]
        dr=off[0]-coff[0]; dc=off[1]-coff[1]
        if dr%3 or dc%3: return None
        mv=(abs(dr)+abs(dc))//3; sp=(k-ck)%4
        if mv==0 and sp==0: continue
        tot+=mv+sp+(0 if c==active else 1)
    return tot

def solve5(CV, root, rootpose, names=('A','B','C','D')):
    sols=[]
    for T in trees4():
        adj={i:[] for i in range(4)}
        for a,b in T: adj[a].append(b); adj[b].append(a)
        ri=names.index(root); seq=[ri]; seen={ri}; frontier=[ri]
        while frontier:
            nxt=[]
            for n in frontier:
                for m in adj[n]:
                    if m not in seen: seen.add(m); nxt.append(m)
            seq+=nxt; frontier=nxt
        def dfs(i,poses):
            if i==4:
                conn=[[False]*4 for _ in range(4)]; matched=0
                for a in range(4):
                    for b in range(a+1,4):
                        if not ok_pair(CV,names[a],poses[a],names[b],poses[b]): return
                        _,pa=cells_of(CV,names[a],poses[a]); _,pb=cells_of(CV,names[b],poses[b])
                        for x in pa:
                            for y in pb:
                                if x==y: matched+=1; conn[a][b]=True
                vis={seq[0]}; st=[seq[0]]
                while st:
                    n=st.pop()
                    for mm in range(4):
                        if conn[n][mm] and mm not in vis: vis.add(mm); st.append(mm)
                if len(vis)<4: return
                sols.append((matched,{names[i]:poses[i] for i in range(4)}))
                return
            node=seq[i]; col=names[node]
            pn=[mm for mm in adj[node] if mm in poses][0]
            for pose in poses_comp(CV,poses[pn],names[pn],col):
                if all(ok_pair(CV,col,pose,names[mm],pp) for mm,pp in poses.items()):
                    poses[node]=pose; dfs(i+1,poses); del poses[node]
        dfs(1,{ri:rootpose})
    return sols

def try_tree(T):
    idx={n:i for i,n in enumerate(names)}
    adj={i:[] for i in range(4)}
    for a,b in T: adj[idx[a]].append(idx[b]); adj[idx[b]].append(idx[a])
    ri=idx['C']; seq=[ri]; seen={ri}; frontier=[ri]
    while frontier:
        nxt=[]
        for n in frontier:
            for m in adj[n]:
                if m not in seen: seen.add(m); nxt.append(m)
        seq+=nxt; frontier=nxt
    cnt={'leaves':0,'ok':0}
    poses={ri:(0,P0['C'][1])}
    def dfs(i):
        if i==4:
            cnt['leaves']+=1
            matched=0; conn=[[False]*4 for _ in range(4)]
            for a in range(4):
                for b in range(a+1,4):
                    if not ok_pair(CV,names[a],poses[a],names[b],poses[b]): return
                    _,pa=cells_of(CV,names[a],poses[a]); _,pb=cells_of(CV,names[b],poses[b])
                    for x in pa:
                        for y in pb:
                            if x==y: matched+=1; conn[a][b]=True
            vis={ri}; st=[ri]
            while st:
                n=st.pop()
                for m in range(4):
                    if conn[n][m] and m not in vis: vis.add(m); st.append(m)
            if len(vis)==4: cnt['ok']+=1; print('SOL matched',matched,{names[i]:poses[i] for i in range(4)})
            return
        node=seq[i]; col=names[node]
        pn=[m for m in adj[node] if m in poses][0]
        for pose in poses_comp(CV,poses[pn],names[pn],col):
            if all(ok_pair(CV,col,pose,names[m],pp) for m,pp in poses.items()):
                poses[node]=pose; dfs(i+1); del poses[node]
    dfs(1)
    print('seq',[names[i] for i in seq],'leaves',cnt['leaves'],'ok',cnt['ok'])

def dfs(i):
    if i==4:
        stat['leaf']+=1
        if all(poses[k]==target[k] for k in range(4)):
            stat['reached_target']+=1
            for a in range(4):
                for b in range(a+1,4):
                    if not ok_pair(CV,names[a],poses[a],names[b],poses[b]): print('failpair',names[a],names[b])
        return
    node=seq[i]; col=names[node]
    pn=[m for m in adj[node] if m in poses][0]
    for pose in poses_comp(CV,poses[pn],names[pn],col):
        if all(ok_pair(CV,col,pose,names[m],pp) for m,pp in poses.items()):
            poses[node]=pose; dfs(i+1); del poses[node]

def solve6(CV, root, rootpose, names=('A','B','C','D')):
    sols=[]
    for T in trees4():
        adj={i:[] for i in range(4)}
        for a,b in T: adj[a].append(b); adj[b].append(a)
        ri=names.index(root); seq=[ri]; seen={ri}; frontier=[ri]
        while frontier:
            nxt=[]
            for n in frontier:
                for m in adj[n]:
                    if m not in seen: seen.add(m); nxt.append(m)
            seq+=nxt; frontier=nxt
        poses={ri:rootpose}
        def dfs(i):
            if i==4:
                matched=0; conn=[[False]*4 for _ in range(4)]
                for a in range(4):
                    for b in range(a+1,4):
                        if not ok_pair(CV,names[a],poses[a],names[b],poses[b]): return
                        _,pa=cells_of(CV,names[a],poses[a]); _,pb=cells_of(CV,names[b],poses[b])
                        for x in pa:
                            for y in pb:
                                if x==y: matched+=1; conn[a][b]=conn[b][a]=True
                vis={ri}; st=[ri]
                while st:
                    n=st.pop()
                    for m in range(4):
                        if conn[n][m] and m not in vis: vis.add(m); st.append(m)
                if len(vis)<4: return
                sols.append((matched,{names[i]:poses[i] for i in range(4)}))
                return
            node=seq[i]; col=names[node]
            pn=[m for m in adj[node] if m in poses][0]
            for pose in poses_comp(CV,poses[pn],names[pn],col):
                if all(ok_pair(CV,col,pose,names[m],pp) for m,pp in poses.items()):
                    poses[node]=pose; dfs(i+1); del poses[node]
        dfs(1)
    return sols

def curposes4(fr,CV):
    out={}
    bodies={}
    for nm,vs in CV.items():
        offs=[]
        for k in range(4):
            b,p,u=vs[k]
            offs.append((min(r for r,c in b)-min(r for r,c in u), min(c for r,c in b)-min(c for r,c in u), frozenset(b)))
        bodies[nm]=offs
    for o in extract2(fr):
        raw=[tuple(x) for x in o['body']]
        anc=(min(r for r,c in raw), min(c for r,c in raw))
        ub=norm(set(raw)); f=None
        for nm,offs in bodies.items():
            for k,(dr,dc,fb) in enumerate(offs):
                if ub==fb: f=(nm,k,dr,dc); break
            if f: break
        if not f: continue
        nm,k,dr,dc=f
        out[nm]=(k,(anc[0]-dr,anc[1]-dc))
    return out

def show(r0,r1,c0,c1):
    for r in range(r0,r1+1):
        print(r3 if False else r, ''.join(G[r][c0:c1+1]).replace('O','.'))

def normb(cs):
    r0=min(r for r,c in cs); c0=min(c for r,c in cs)
    return frozenset((r-r0,c-c0) for r,c in cs)

def rots(cs):
    cur=set(cs); out=[]
    for k in range(4):
        out.append(normb(cur))
        cur={(c,-r) for r,c in cur}
    seen=[]; 
    for o in out:
        if o not in seen: seen.append(o)
    return seen

def rect_tilings(pieces, R, Cc):
    cells={(r,c) for r in range(R) for c in range(Cc)}
    variants={i:rots(p) for i,p in enumerate(pieces)}
    sols=[]
    def place(i, avail):
        if i==len(pieces):
            sols.append([]); return
        if not avail: return
        tgt=min(avail)
        for vi,v in enumerate(variants[i]):
            for anchor in sorted(v):
                # place so that tgt covered: shift
                sh=(tgt[0]-anchor[0], tgt[1]-anchor[1])
                placed={(r+sh[0],c+sh[1]) for r,c in v}
                if placed<=avail:
                    rest=avail-placed
                    place(i+1, rest)
                    if sols: return
        return
    place(0,set(cells))
    return sols

def comps_of(fr,pred):
    G=G_of(fr);seen=[[False]*64 for _ in range(64)];out=[]
    for r in range(64):
        for c in range(64):
            if seen[r][c] or not pred(r,c,G[r][c]): continue
            st=[(r,c)];comp=[]
            while st:
                x,y=st.pop()
                if x<0 or y<0 or x>63 or y>63 or seen[x][y] or not pred(x,y,G[x][y]): continue
                seen[x][y]=True;comp.append((x,y))
                st+=[(x+1,y),(x-1,y),(x,y+1),(x,y-1)]
            out.append((G[r][c],comp))
    return out

def build_cv(specs):
    V={}
    for n,(body,prongs) in specs.items():
        bs=rotsv(body);ps=[rotsv(p) for p in prongs]
        npr=len(prongs)
        vars=[]
        for k in range(4):
            body_k=bs[k];pr_k=[ps[j][k] for j in range(npr)]
            allc=set(body_k)
            for p in pr_k: allc|=p
            r0=min(r for r,_ in allc);c0=min(c for _,c in allc)
            sh=(max(r for r,_ in allc)-r0, max(c for _,c in allc)-c0)
            body_n=frozenset((r-r0,c-c0) for r,c in body_k)
            pr_n=[frozenset((r-r0,c-c0) for r,c in p) for p in pr_k]
            vars.append((sh,body_n,pr_n))
        V[n]=vars
    return V

def rect(rs):
    r0,r1,c0,c1=rs; return [(r,c) for r in range(r0,r1+1) for c in range(c0,c1+1)]

def rect2(rs):
    r0,r1,c0,c1=rs; return [(r,c) for r in range(r0,r1+1) for c in range(c0,c1+1)]

def build_cv2(specs):
    out={}
    for nm in specs:
        body=specs[nm][0]; prongs=specs[nm][1]
        B=set(map(tuple,body)); P=[set(map(tuple,q)) for q in prongs]
        U=set(B)
        for q in P: U|=q
        variants=[]
        for k in range(4):
            if k>0:
                B={(c,-r) for (r,c) in B}; P=[{(c,-r) for (r,c) in q} for q in P]; U={(c,-r) for (r,c) in U}
            dr=min([q[0] for q in U]); dc=min([q[1] for q in U])
            variants.append((set((r-dr,c-dc) for (r,c) in B),[set((r-dr,c-dc) for (r,c) in q) for q in P],set((r-dr,c-dc) for (r,c) in U)))
        out[nm]=variants
    return out

def rct(rs):
    r0,r1,c0,c1=rs; return [(r,c) for r in range(r0,r1+1) for c in range(c0,c1+1)]

def L3():
    def rc(rs):
        r0,r1,c0,c1=rs; return [(r,c) for r in range(r0,r1+1) for c in range(c0,c1+1)]
    Zbody=rc((14,16,14,19))+rc((17,19,14,16))+rc((20,22,8,16))+rc((23,25,8,10))+rc((26,28,8,16))+rc((29,31,14,16))+rc((32,34,14,19))
    Bbody=rc((14,16,35,49))
    Sbody=rc((32,34,29,37))+rc((35,43,29,31))+rc((35,40,35,37))+rc((41,43,35,46))+rc((44,46,29,31))+rc((44,46,44,46))+rc((47,49,29,46))
    specs={'Z':(Zbody,[rc((14,16,20,22)),rc((32,34,20,22))]),
           'B':(Bbody,[rc((11,13,47,49)),rc((14,16,32,34))]),
           'S':(Sbody,[rc((29,31,32,34)),rc((44,46,47,49))])}
    return build_cv2(specs), {'Z':(0,(14,8)),'B':(0,(11,32)),'S':(0,(29,29))}

def absv(V,n,k,off):
    sh,b,ps=V[n][k]
    return (frozenset((r+off[0],c+off[1]) for r,c in b), [frozenset((r+off[0],c+off[1]) for r,c in p) for p in ps])

def okp(V,p1,c1,p2,c2):
    b1,q1=absv(V,c1,p1[0],p1[1]);b2,q2=absv(V,c2,p2[0],p2[1])
    if b1&b2 or b1&set().union(*q2) or b2&set().union(*q1): return False,None
    pairs=[]
    for i,a in enumerate(q1):
        for j,bb in enumerate(q2):
            if min(a)==min(bb):
                if a!=bb: return False,None
                pairs.append((i,j))
    return True,pairs

def solve3p(V,names,root,rootpose,need):
    res=[]
    idx={n:i for i,n in enumerate(names)}
    import itertools
    trees=[[(0,1),(1,2)],[(0,2),(2,1)],[(1,0),(0,2)]]
    for T in trees:
        adj={i:[] for i in range(3)}
        for a,b in T: adj[a].append(b); adj[b].append(a)
        ri=idx[root]; seq=[ri]; seen={ri}; fr=[ri]
        while fr:
            nx=[]
            for n in fr:
                for m in adj[n]:
                    if m not in seen: seen.add(m); nx.append(m)
            seq+=nx; fr=nx
        poses={ri:rootpose}
        def dfs(i):
            if i==3:
                tot=0; conn={a:{} for a in range(3)}; good=True
                for a in range(3):
                    for b in range(a+1,3):
                        r,nn=okp(V,poses[a],names[a],poses[b],names[b])
                        if not r: good=False
                        elif nn: conn[a][b]=conn[b][a]=True; tot+=nn
                if not good or tot<need: return
                vis={ri}; st=[ri]
                while st:
                    n=st.pop()
                    for m in range(3):
                        if conn[n].get(m) and m not in vis: vis.add(m); st.append(m)
                if len(vis)==3: res.append((tot,{names[i]:poses[i] for i in range(3)}))
                return
            node=seq[i]; col=names[node]
            pn=[m for m in adj[node] if m in poses][0]
            for pose,nn in comp(V,poses[pn],names[pn],col):
                good=True
                for m,pp in poses.items():
                    r,_=okp(V,pose,col,pp,names[m])
                    if not r: good=False; break
                if good:
                    poses[node]=pose; dfs(i+1); del poses[node]
        dfs(1)
    return res

def boardok(sh,V):
    names=list(sh)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            ok,_=okp(V,sh[a],a,sh[b],b)
            if not ok: return False
    return True

def blk(fr,k0,k1,m0,m1,mp=None):
    G=[list(x) for x in fr.ascii.split('\n')]
    if mp is None: mp={'b':'.','p':'.','W':'W','R':'P','c':'#','Y':'Y','S':'S','O':'O','N':'N','G':'G','M':'M','P':'P','w':'w','g':'g','W':'W'}
    return [''.join(mp.get(G[3*k+2][3*m+2],'?') for m in range(m0,m1+1)) for k in range(k0,k1+1)]

def pairs_info(V,sh,names):
    info={}
    used={}
    for i in range(len(names)):
        for j in range(i+1,len(names)):
            a,b=names[i],names[j]
            _,qa=absv(V[a],sh[a][0],sh[a][1]); _,qb=absv(V[b],sh[b][0],sh[b][1])
            locs=[]
            for x in qa:
                for y in qb:
                    if x==y: locs.append((min(x),len(x)))
            info[(a,b)]=locs
    return info

def perfect(V,sh,names):
    inf=pairs_info(V,sh,names)
    prong_use={}
    tot=0
    for k,locs in inf.items():
        if len(locs)!=1: return False
        tot+=1
    # each prong used exactly once
    for n in names:
        _,q=absv(V[n],sh[n][0],sh[n][1])
        cnt=[0]*len(q)
        for (a,b),locs in inf.items():
            if a==n or b==n:
                o=b if a==n else a
                _,qo=absv(V[o],sh[o][0],sh[o][1])
                for i,pi in enumerate(q):
                    for po in qo:
                        if pi==po: cnt[i]+=1
        if sorted(cnt)!=[1]*len(q): return False
    return tot==3

def bbstr(cs): 
    return (min(r for r,c in cs),max(r for r,c in cs),min(c for r,c in cs),max(c for r,c in cs))

def cost_norot(sh):
    tot=0
    for c in 'ZBS':
        dr=sh[c][1][0]-PC[c][1][0]; dc=sh[c][1][1]-PC[c][1][1]
        if dr%3 or dc%3: return None
        mv=(abs(dr)+abs(dc))//3
        if mv: tot+=mv+1
    return tot

def G_of(fr): return [list(r) for r in fr.ascii.split('\n')]

def sel_obj(fr):
    a=fr.ascii.split('\n'); G=[[a[r][c] for c in range(64)] for r in range(64)]
    cols=Counter(''.join(a))
    body=[c for c in cols if c not in 'bRWcO ']
    out={}
    for col in body:
        cs=[(r,c) for r in range(64) for c in range(64) if G[r][c]==col]
        out['body']=(col,sorted(cs))
    pr=[(col,comp) for col,comp in comps_of(fr, lambda r,c,ch: ch=='R')]
    out['prongs']=[comp for _,comp in pr]
    return out

def blkcells(blocks):
    out=[]
    for k,m in blocks:
        r=3*k+2;c=3*m+2
        out+=[(r+i,c+j) for i in range(3) for j in range(3)]
    return out

def L4():
    Ab=[(3,2),(3,3),(3,4),(3,5),(3,6),(3,7),(4,2),(4,7),(5,2),(5,4),(5,5),(5,7),(6,2),(6,4),(6,7),(7,2),(7,4),(7,5),(7,6),(7,7),(8,2),(8,4),(9,2)]
    Ap=[(8,7),(10,2)]
    Ub=[(6,12),(6,14),(7,12),(7,14),(8,12),(8,14),(9,12),(9,13),(9,14)]
    Up=[(6,11),(6,15)]
    Bb=[(15,6)]; Bp=[(15,5),(15,7)]
    Cb=[(13,15),(13,16),(13,17),(14,15),(14,17),(15,15),(15,17),(16,15),(16,17),(17,15),(17,17),(18,15),(18,16),(18,17)]
    Cp=[(13,14),(18,14)]
    specs={'A':(blkcells(Ab),[blkcells([p]) for p in Ap]),
           'U':(blkcells(Ub),[blkcells([p]) for p in Up]),
           'B':(blkcells(Bb),[blkcells([p]) for p in Bp]),
           'C':(blkcells(Cb),[blkcells([p]) for p in Cp])}
    V=build_cv2(specs)
    P0={'A':(0,(11,8)),'U':(0,(20,35)),'B':(0,(47,17)),'C':(0,(41,44))}
    return V,P0

def gen_poses(V,placed,names,ncol):
    out={}
    for k in range(4):
        cq=absv(V,ncol,k,(0,0))[1]
        for pn,pp in placed.items():
            pq=absv(V,pn,pp[0],pp[1])[1]
            for px in pq:
                for py in cq:
                    off=(min(r for r,_ in px)-min(r for r,_ in py),min(c for _,c in px)-min(c for _,c in py))
                    if off in out: continue
                    ok=True
                    for pm,pv in placed.items():
                        r,_=okp(V,(k,off),ncol,pv,pm)
                        if not r: ok=False;break
                    if ok: out[off]=k
    return [((k,o),) for o,k in out.items()]

def solveN(V,names,root,rootpose):
    res=[]
    others=[n for n in names if n!=root]
    import itertools
    seen=set()
    for order in itertools.permutations(others):
        poses={root:rootpose}
        def dfs(i):
            if i==len(order):
                if perfect(V,poses,names):
                    key=tuple(sorted((poses[n][0],poses[n][1]) for n in names))
                    if key not in seen:
                        seen.add(key); res.append(dict(poses))
                return
            n=order[i]
            for (pose,) in gen_poses(V,poses,names,n):
                poses[n]=pose; dfs(i+1); del poses[n]
        dfs(0)
    return res

def solveN2(V,names,root,rootpose,need=None):
    res=[]; seen=set()
    import itertools
    others=[n for n in names if n!=root]
    for order in itertools.permutations(others):
        poses={root:rootpose}
        def dfs(i):
            if i==len(order):
                inf=pairs_info(V,poses,names)
                tot=sum(len(v) for v in inf.values())
                deg={n:0 for n in names}
                for (a,b),v in inf.items():
                    deg[a]+=len(v); deg[b]+=len(v)
                key=tuple(sorted((poses[n][0],poses[n][1]) for n in names))
                if key not in seen:
                    seen.add(key); res.append((tot,dict(poses),deg))
                return
            n=order[i]
            for (pose,) in gen_poses(V,poses,names,n):
                poses[n]=pose; dfs(i+1); del poses[n]
        dfs(0)
    return res

def analyze(V,poses,names):
    locs={}
    bodies={}
    for n in names:
        b,q=absv(V[n],poses[n][0],poses[n][1])
        bodies[n]=b
        for i,pr in enumerate(q):
            key=min(pr)
            locs.setdefault(key,[]).append((n,i,pr))
    for n in names:
        for m in names:
            if n>=m: continue
            if bodies[n]&bodies[m]: return None
    bad=False; pairs=[]
    for key,v in locs.items():
        if len(v)>2: bad=True
        if len(v)==2:
            (n,i,pn),(m,j,pm)=v
            if pn!=pm: bad=True
            else: pairs.append((n,m))
    if bad: return None
    adj={n:set() for n in names}
    for n,m in pairs: adj[n].add(m); adj[m].add(n)
    vis={names[0]}; st=[names[0]]
    while st:
        x=st.pop()
        for y in adj[x]:
            if y not in vis: vis.add(y); st.append(y)
    return (len(pairs),len(vis)==len(names))

def body_at(V,n,pose): return absv(V[n],pose[0],pose[1])[0]

def onb(cs):
    return min(r for r,_ in cs)>=1 and max(r for r,_ in cs)<=63 and min(c for _,c in cs)>=0 and max(c for _,c in cs)<=63

def plan(V,cur,tgt,active,names):
    import itertools
    best=None
    for order in itertools.permutations(names):
        for rotfirst in (True,False):
            for pathmode in (0,1):
                poses=dict(cur); sel=active; acts=[]; ok=True
                for n in order:
                    if poses[n]==tgt[n]: continue
                    if sel!=n: acts.append(('click',n)); sel=n
                    def occ(exc):
                        s=set()
                        for m in names:
                            if m!=exc: s|=body_at(V,m,poses[m])
                        return s
                    if rotfirst:
                        k=poses[n][0]
                        while k!=tgt[n][0]:
                            nk=(k+1)%4
                            if not onb(body_at(V,n,(nk,poses[n][1]))) or (body_at(V,n,(nk,poses[n][1]))&occ(n)): ok=False;break
                            acts.append(('rot',n)); k=nk; poses[n]=(k,poses[n][1])
                        if not ok: break
                    r0,c0=poses[n][1]; r1,c1=tgt[n][1]
                    steps=[]
                    if pathmode==0: steps+= [('r',r1)]+[('c',c1)]
                    else: steps+= [('c',c1)]+[('r',r1)]
                    cr,cc=r0,c0
                    for axis,tv in steps:
                        while (cr if axis=='r' else cc)!=tv:
                            d=1 if (cr if axis=='r' else cc)<tv else -1
                            nr,nc=(cr+d,cc) if axis=='r' else (cr,cc+d)
                            b=body_at(V,n,(poses[n][0],(nr,nc)))
                            if not onb(b) or (b&occ(n)): ok=False;break
                            acts.append(('move',n,('DOWN' if d>0 else 'UP') if axis=='r' else ('RIGHT' if d>0 else 'LEFT')))
                            cr,cc=nr,nc; poses[n]=(poses[n][0],(nr,nc))
                        if not ok: break
                    if not ok: break
                    if not rotfirst:
                        k=poses[n][0]
                        while k!=tgt[n][0]:
                            nk=(k+1)%4
                            if not onb(body_at(V,n,(nk,poses[n][1]))) or (body_at(V,n,(nk,poses[n][1]))&occ(n)): ok=False;break
                            acts.append(('rot',n)); k=nk; poses[n]=(k,poses[n][1])
                        if not ok: break
                if ok and all(poses[n]==tgt[n] for n in names):
                    if best is None or len(acts)<len(best): best=acts
    return best

def plan3(V,cur,tgt,active,names,step=3):
    import itertools
    best=None
    for order in itertools.permutations(names):
        for rotfirst in (True,False):
            for pathmode in (0,1):
                poses=dict(cur); sel=active; acts=[]; ok=True
                def occ(exc):
                    s=set()
                    for m in names:
                        if m!=exc: s|=body_at(V,m,poses[m])
                    return s
                for n in order:
                    if poses[n]==tgt[n]: continue
                    if sel!=n: acts.append(('click',n)); sel=n
                    def do_rot():
                        nonlocal ok,poses
                        k=poses[n][0]
                        while k!=tgt[n][0]:
                            nk=(k+1)%4
                            b=body_at(V,n,(nk,poses[n][1]))
                            if not onb(b) or (b&occ(n)): ok=False; return
                            acts.append(('rot',n)); k=nk; poses[n]=(k,poses[n][1])
                    if rotfirst:
                        do_rot()
                        if not ok: break
                    r0,c0=poses[n][1]; r1,c1=tgt[n][1]
                    steps=([('r',r1),('c',c1)] if pathmode==0 else [('c',c1),('r',r1)])
                    cr,cc=r0,c0
                    for axis,tv in steps:
                        while (cr if axis=='r' else cc)!=tv:
                            d=step if (cr if axis=='r' else cc)<tv else -step
                            nr,nc=(cr+d,cc) if axis=='r' else (cr,cc+d)
                            b=body_at(V,n,(poses[n][0],(nr,nc)))
                            if not onb(b) or (b&occ(n)): ok=False;break
                            acts.append(('move',n,('DOWN' if d>0 else 'UP') if axis=='r' else ('RIGHT' if d>0 else 'LEFT')))
                            cr,cc=nr,nc; poses[n]=(poses[n][0],(nr,nc))
                        if not ok: break
                    if not ok: break
                    if not rotfirst:
                        do_rot()
                        if not ok: break
                if ok and all(poses[n]==tgt[n] for n in names):
                    if best is None or len(acts)<len(best): best=acts
    return best

def blocks_of(comp):
    bl=set()
    for r,c in comp:
        k=(r-2)//3;m=(c-2)//3
        if 3*k+2<=r<=3*k+4 and 3*m+2<=c<=3*m+4: bl.add((k,m))
    return sorted(bl)

def L5():
    Yb=[(5,8),(6,8),(7,8),(7,9),(8,8),(9,8),(9,9),(9,10),(9,11),(10,8),(11,8),(11,9),(11,10)]
    Yp=[(5,9),(6,7),(8,7),(10,7),(7,10),(9,12),(11,11)]
    Ub=[(2,17),(3,15),(3,17),(4,15),(4,17),(5,15),(5,16),(5,17)]
    Up=[(1,17),(2,15)]
    Eb=[(12,1),(12,2),(13,1),(14,1),(14,2),(15,1),(16,1),(16,2)]
    Ep=[(12,3),(14,3),(16,3)]
    Lb=[(16,15),(17,15),(17,16),(17,17)]
    Lp=[(15,15),(16,17)]
    specs={'Y':(blkcells(Yb),[blkcells([p]) for p in Yp]),
           'U':(blkcells(Ub),[blkcells([p]) for p in Up]),
           'E':(blkcells(Eb),[blkcells([p]) for p in Ep]),
           'L':(blkcells(Lb),[blkcells([p]) for p in Lp])}
    P0={'Y':(0,(14,23)),'U':(0,(5,44)),'E':(0,(35,2)),'L':(0,(44,44))}
    return build_cv2(specs),P0

def analyze5(V,poses,names):
    locs={}
    bodies={}
    for n in names:
        b,q=absv(V[n],poses[n][0],poses[n][1])
        bodies[n]=b
        for i,pr in enumerate(q): locs.setdefault(min(pr),[]).append((n,i,pr))
    for n in names:
        for m in names:
            if n>=m: continue
            if bodies[n]&bodies[m]: return None
    pairs=[]
    for key,v in locs.items():
        if len(v)!=2: return None
        (n,i,pn),(m,j,pm)=v
        if pn!=pm or n==m: return None
        pairs.append((n,m))
    adj={n:set() for n in names}
    for n,m in pairs: adj[n].add(m); adj[m].add(n)
    vis={names[0]}; st=[names[0]]
    while st:
        x=st.pop()
        for y in adj[x]:
            if y not in vis: vis.add(y); st.append(y)
    return (len(pairs), len(vis)==len(names))

def solve5p(V,names,root):
    out=[];seen=set()
    for kr in range(4):
        for order in itertools.permutations([n for n in names if n!=root]):
            poses={root:(kr,(0,0))}
            def dfs(i):
                if i==len(order):
                    r=analyze5(V,poses,names)
                    if r:
                        key=tuple(sorted(((poses[n][0],poses[n][1][0]-poses[root][1][0],poses[n][1][1]-poses[root][1][1]) for n in names)))
                        if key not in seen: seen.add(key); out.append({n:(poses[n][0],(poses[n][1][0]-poses[root][1][0],poses[n][1][1]-poses[root][1][1])) for n in names})
                    return
                n=order[i]
                for (pose,) in gen_poses(V,poses,names,n):
                    poses[n]=pose; dfs(i+1); del poses[n]
            dfs(0)
    return out

def S5get():
    V5,P5=L5()
    P5={'Y':(0,(17,23)),'U':(0,(5,47)),'E':(0,(38,5)),'L':(0,(47,47))}
    return V5,P5,solve5p(V5,'YUEL','Y')

def region(G,k0,k1,m0,m1):
    mp={'b':'.','c':'#','R':'P','W':'W','Y':'Y','S':'S','N':'N','p':'p','O':'O'}
    out=[]
    for k in range(k0,k1+1):
        out.append(''.join(mp.get(G[3*k+2][3*m+2],'?') for m in range(m0,m1+1)))
    return out

def normc(cs):
    r0=min(r for r,_ in cs);c0=min(c for _,c in cs);return frozenset((r-r0,c-c0) for r,c in cs)

def rotsv(cs):
    cur=normc(cs);out=[cur]
    for _ in range(3):
        cur=normc([(c,-r) for r,c in cur]);out.append(cur)
    return out

def allpr(V,poses,names):
    locs={};bodies={}
    for n in names:
        b,q=absv(V,n,poses[n][0],poses[n][1]);bodies[n]=b
        for i,p in enumerate(q): locs.setdefault(min(p),[]).append((n,i,p))
    for i,a in enumerate(names):
        for b_ in names[i+1:]:
            if bodies[a]&bodies[b_]: return None
    pairs=[]
    for key,v in locs.items():
        if len(v)!=2: return None
        (n,i,pn),(m,j,pm)=v
        if pn!=pm or n==m: return None
        pairs.append((n,m))
    adj={n:set() for n in names}
    for n,m in pairs: adj[n].add(m);adj[m].add(n)
    vis={names[0]};st=[names[0]]
    while st:
        x=st.pop()
        for y in adj[x]:
            if y not in vis: vis.add(y);st.append(y)
    return (len(pairs),len(vis)==len(names))

def solvep(V,names,root):
    import itertools
    out=[];seen=set()
    for kr in range(4):
        for order in itertools.permutations([n for n in names if n!=root]):
            poses={root:(kr,(0,0))}
            def dfs(i):
                if i==len(order):
                    r=allpr(V,poses,names)
                    if r:
                        key=tuple(sorted((poses[n][0],poses[n][1][0]-poses[root][1][0],poses[n][1][1]-poses[root][1][1]) for n in names))
                        if key not in seen:
                            seen.add(key);out.append({n:(poses[n][0],(poses[n][1][0]-poses[root][1][0],poses[n][1][1]-poses[root][1][1])) for n in names})
                    return
                n=order[i]
                for (pose,) in gen_poses(V,poses,names,n):
                    poses[n]=pose;dfs(i+1);del poses[n]
            dfs(0)
    return out

def L6():
    specs={
    'G':([(1,1),(1,2),(1,3),(1,4),(1,5),(2,1),(2,5),(3,1),(4,1),(5,1),(5,2)],[[(3,5)],[(5,3)],[(6,1)]]),
    'B':([(1,15),(1,16),(1,17),(1,18),(2,15),(2,18),(3,18),(4,18)],[[(3,15)],[(4,17)]]),
    'M':([(7,7),(7,8),(8,7),(9,7),(10,7)],[[(6,8)],[(8,6)]]),
    'L':([(13,1),(14,1),(15,1),(16,1),(17,1)],[[(12,1)],[(17,2)]]),
    'U':([(15,11),(15,13),(16,10),(16,11),(16,13),(16,14),(17,10),(17,14),(18,10),(18,11),(18,13),(18,14),(18,12)],[[(14,11)],[(14,13)],[(17,9)]])}
    specs={n:(blkcells(b),[blkcells(p) for p in pr]) for n,(b,pr) in specs.items()}
    P0={'G':(0,(5,5)),'B':(0,(5,47)),'M':(0,(23,23)),'L':(0,(41,5)),'U':(0,(47,32))}
    return build_cv(specs),P0

def build_cvB(specs):
    V={}
    for n,(body,prongs) in specs.items():
        npr=len(prongs);vars=[]
        cur_b=body;cur_p=[list(p) for p in prongs]
        for k in range(4):
            allc=list(cur_b)
            for p in cur_p: allc+=p
            r0=min(r for r,_ in allc);c0=min(c for _,c in allc)
            sh=(max(r for r,_ in allc)-r0,max(c for _,c in allc)-c0)
            bn=frozenset((r-r0,c-c0) for r,c in cur_b)
            pn=[frozenset((r-r0,c-c0) for r,c in p) for p in cur_p]
            vars.append((sh,bn,pn))
            cur_b=[(c,-r) for r,c in cur_b];cur_p=[[(c,-r) for r,c in p] for p in cur_p]
        V[n]=vars
    return V

def L6B():
    specs={
    'G':([(1,1),(1,2),(1,3),(1,4),(1,5),(2,1),(2,5),(3,1),(4,1),(5,1),(5,2)],[[(3,5)],[(5,3)],[(6,1)]]),
    'B':([(1,15),(1,16),(1,17),(1,18),(2,15),(2,18),(3,18),(4,18)],[[(3,15)],[(4,17)]]),
    'M':([(7,7),(7,8),(8,7),(9,7),(10,7)],[[(6,8)],[(8,6)]]),
    'L':([(13,1),(14,1),(15,1),(16,1),(17,1)],[[(12,1)],[(17,2)]]),
    'U':([(15,11),(15,13),(16,10),(16,11),(16,13),(16,14),(17,10),(17,14),(18,10),(18,11),(18,13),(18,14),(18,12)],[[(14,11)],[(14,13)],[(17,9)]])}
    specs={n:(blkcells(b),[blkcells(p) for p in pr]) for n,(b,pr) in specs.items()}
    P0={'G':(0,(5,5)),'B':(0,(5,44)),'M':(0,(23,23)),'L':(0,(41,5)),'U':(0,(47,32))}
    return build_cvB(specs),P0

def P6fix(V):
    P0={}
    for n in V:
        sh,b,ps=V[n][0]
        allc=set(b)|set().union(*ps)
        P0[n]=(0,(min(r for r,_ in allc),min(c for _,c in allc)))
    return P0

def L6C():
    specs={
    'G':([(1,1),(1,2),(1,3),(1,4),(1,5),(2,1),(2,5),(3,1),(4,1),(5,1),(5,2)],[[(3,5)],[(5,3)],[(6,1)]]),
    'B':([(1,15),(1,16),(1,17),(1,18),(2,15),(2,18),(3,18),(4,18)],[[(3,15)],[(4,17)]]),
    'M':([(7,7),(7,8),(8,7),(9,7),(10,7)],[[(6,8)],[(8,6)]]),
    'L':([(13,1),(14,1),(15,1),(16,1),(17,1)],[[(12,1)],[(17,2)]]),
    'U':([(15,11),(15,13),(16,10),(16,11),(16,13),(16,14),(17,10),(17,14),(18,10),(18,11),(18,13),(18,14),(18,12)],[[(14,11)],[(14,13)],[(17,9)]])}
    P0={}
    for n,(b,pr) in specs.items():
        allb=b+[x for p in pr for x in p]
        P0[n]=(0,(3*min(k for k,m in allb)+2-1,3*min(m for k,m in allb)+2-1))
    specs={n:(blkcells(b),[blkcells(p) for p in pr]) for n,(b,pr) in specs.items()}
    return build_cvB(specs),P0

def nrm(bs):
    r0=min(r for r,_ in bs);c0=min(c for _,c in bs);return frozenset((r-r0,c-c0) for r,c in bs)

def buildB(specs):
    V={}
    for n,(body,prongs) in specs.items():
        vars=[];b=list(body);p=[list(x) for x in prongs]
        for k in range(4):
            allc=b+p
            r0=min(r for r,_ in allc);c0=min(c for _,c in allc)
            sh=(max(r for r,_ in allc)-r0,max(c for _,c in allc)-c0)
            bn=frozenset((r-r0,c-c0) for r,c in b)
            pn=[(r-r0,c-c0) for r,c in p]
            vars.append((sh,bn,pn))
            b=[(c,-r) for r,c in b];p=[(c,-r) for r,c in p]
        V[n]=vars
    return V

def ab(V,n,k,off):
    sh,b,ps=V[n][k]
    return (frozenset((r+off[0],c+off[1]) for r,c in b),[(r+off[0],c+off[1]) for r,c in ps])

def okb(V,p1,c1,p2,c2):
    b1,q1=ab(V,c1,p1[0],p1[1]);b2,q2=ab(V,c2,p2[0],p2[1])
    if b1&b2: return False
    s2=set(q2);s1=set(q1)
    if b1&s2 or b2&s1: return False
    for a in q1:
        for bb in q2:
            if a==bb: return True
    return False

def bok(sh,V,names):
    for i,a in enumerate(names):
        for b in names[i+1:]:
            if not okb(V,sh[a],a,sh[b],b): return False
    return True

def allprB(V,poses,names):
    locs={};bodies={}
    for n in names:
        b,q=ab(V,n,poses[n][0],poses[n][1]);bodies[n]=b
        for i,p in enumerate(q): locs.setdefault(p,[]).append((n,i))
    for i,a in enumerate(names):
        for b in names[i+1:]:
            if bodies[a]&bodies[b]: return None
    pairs=[]
    for key,v in locs.items():
        if len(v)!=2: return None
        if v[0][0]==v[1][0]: return None
        pairs.append((v[0][0],v[1][0]))
    adj={n:set() for n in names}
    for n,m in pairs: adj[n].add(m);adj[m].add(n)
    vis={names[0]};st=[names[0]]
    while st:
        x=st.pop()
        for y in adj[x]:
            if y not in vis: vis.add(y);st.append(y)
    return (len(pairs),len(vis)==len(names))

def genB(V,placed,names,ncol):
    out={}
    for k in range(4):
        cq=ab(V,ncol,k,(0,0))[1]
        for pn,pp in placed.items():
            pq=ab(V,pn,pp[0],pp[1])[1]
            for px in pq:
                for py in cq:
                    off=(px[0]-py[0],px[1]-py[1])
                    if off in out: continue
                    if all(okb(V,(k,off),ncol,pv,pm) for pm,pv in placed.items()): out[off]=k
    return [((k,o),) for o,k in out.items()]

def solveB(V,names,root):
    import itertools
    out=[];seen=set()
    for kr in range(4):
        for order in itertools.permutations([n for n in names if n!=root]):
            poses={root:(kr,(0,0))}
            def dfs(i):
                if i==len(order):
                    if allprB(V,poses,names):
                        key=tuple(sorted((poses[n][0],poses[n][1][0]-poses[root][1][0],poses[n][1][1]-poses[root][1][1]) for n in names))
                        if key not in seen:
                            seen.add(key);out.append({n:(poses[n][0],(poses[n][1][0]-poses[root][1][0],poses[n][1][1]-poses[root][1][1])) for n in names})
                    return
                n=order[i]
                for (pose,) in genB(V,poses,names,n):
                    poses[n]=pose;dfs(i+1);del poses[n]
            dfs(0)
    return out

def SPEC6():
    return {
    'G':([(1,1),(1,2),(1,3),(1,4),(1,5),(2,1),(2,5),(3,1),(4,1),(5,1),(5,2)],[(3,5),(5,3),(6,1)]),
    'B':([(1,15),(1,16),(1,17),(1,18),(2,15),(2,18),(3,18),(4,18)],[(3,15),(4,17)]),
    'M':([(7,7),(7,8),(8,7),(9,7),(10,7)],[(6,8),(8,6)]),
    'L':([(13,1),(14,1),(15,1),(16,1),(17,1)],[(12,1),(17,2)]),
    'U':([(15,11),(15,13),(16,10),(16,11),(16,13),(16,14),(17,10),(17,14),(18,10),(18,11),(18,13),(18,14),(18,12)],[(14,11),(14,13),(17,9)])}

def P6of(sp):
    P0={}
    for n,(b,pr) in sp.items():
        allb=b+list(pr)
        P0[n]=(0,(min(k for k,m in allb),min(m for k,m in allb)))
    return P0

def blkmap(G,k0,k1,m0,m1):
    mp={'b':'.','c':'#','R':'P','W':'W','Y':'Y','S':'S','N':'N','p':'p','O':'o','g':'g'}
    return [''.join(mp.get(G[3*k+2][3*m+2],'?') for m in range(m0,m1+1)) for k in range(k0,k1+1)]

def Gf(): return [list(r) for r in current_frame.ascii.split('\n')]

def reg(fr,k0,k1,m0,m1):
    G=[list(x) for x in fr.ascii.split('\n')]
    mp={'b':'.','c':'#','R':'P','W':'W','Y':'Y','p':'p','S':'S','N':'N','O':'o'}
    return [''.join(mp.get(G[3*k+2][3*m+2],'?') for m in range(m0,m1+1)) for k in range(k0,k1+1)]

def SP6B():
    return {
    'G':([(1,1),(1,2),(1,3),(1,4),(1,5),(2,1),(2,5),(3,1),(4,1),(5,1),(5,2)],[(3,5),(5,3),(6,1)]),
    'B':([(1,15),(1,16),(1,17),(1,18),(2,15),(2,18),(3,18),(4,18)],[(3,15),(4,17)]),
    'M':([(7,7),(7,8),(8,7),(9,7),(10,7),(10,8),(10,9),(10,10),(10,11),(10,12),(8,13),(9,13),(7,13)],[(6,8),(8,6),(11,7),(11,13)]),
    'L':([(13,1),(14,1),(15,1),(16,1),(17,1)],[(12,1),(17,2)]),
    'U':([(15,11),(15,16),(16,10),(16,11),(16,16),(16,17),(17,10),(17,17),(18,10),(18,11),(18,12),(18,13),(18,14),(18,15),(18,16),(18,17)],[(14,11),(14,16),(17,9)])}

def SP6C():
    return {
    'G':([(1,1),(1,2),(1,3),(1,4),(1,5),(2,1),(2,5),(3,1),(4,1),(5,1),(5,2)],[(3,5),(5,3),(6,1)]),
    'B':([(1,15),(1,16),(1,17),(1,18),(2,15),(2,18),(3,18),(4,18)],[(3,15),(4,17)]),
    'M':([(7,7),(7,8),(8,7),(9,7),(10,7),(10,8),(10,9),(10,10),(10,11),(10,12),(8,12),(9,12),(7,12)],[(6,8),(8,6),(11,7),(11,12)]),
    'L':([(13,1),(14,1),(15,1),(16,1),(17,1)],[(12,1),(17,2)]),
    'U':([(15,11),(15,16),(16,10),(16,11),(16,16),(16,17),(17,10),(17,17),(18,10),(18,11),(18,12),(18,13),(18,14),(18,15),(18,16),(18,17)],[(14,11),(14,16),(17,9)])}

def extract6(fr):
    G=[list(x) for x in fr.ascii.split('\n')]
    cen={}
    for k in range(0,21):
        for m in range(0,21):
            r,c=3*k+2,3*m+2
            if r>63 or c>63: continue
            ch=G[r][c]
            if ch!='b' and r>1: cen[(k,m)]=ch
    body={p:ch for p,ch in cen.items() if ch not in 'RW'}
    seen=set();comps=[]
    for p in body:
        if p in seen: continue
        st=[p];comp=[]
        while st:
            x=st.pop()
            if x in seen or x not in body: continue
            seen.add(x);comp.append(x)
            st+=[(x[0]+dx,x[1]+dy) for dx,dy in ((1,0),(-1,0),(0,1),(0,-1))]
        comps.append({'body':set(comp),'prong':[],'white':[]})
    def owner(p):
        for i,cp in enumerate(comps):
            for q in cp['body']:
                if abs(q[0]-p[0])+abs(q[1]-p[1])==1: return i
        return None
    for p,ch in cen.items():
        if ch in 'RW':
            i=owner(p)
            if i is None: print('UNASSIGNED',p,ch)
            else: comps[i]['prong' if ch=='R' else 'white'].append(p)
    return comps

def spec_from(comps):
    out={}
    for i,cp in enumerate(comps):
        out[i]=(sorted(cp['body']|set(cp['white'])), sorted(cp['prong']))
    return out

def mspec(fr):
    for c in extract6(fr):
        if c['white']:
            return (sorted(c['body']|set(c['white'])),sorted(c['prong']))
    return None

def try_search(fr):
    ms=mspec(fr)
    if ms is None: return None,None
    sp=dict(FIX);sp['M']=ms
    V=buildB(sp);S=solveB(V,'GBMLU','G')
    return sp,S

def FIXD():
    return {'G':([(1,1),(1,2),(1,3),(1,4),(1,5),(2,1),(2,5),(3,1),(4,1),(5,1),(5,2)],[(3,5),(5,3),(6,1)]),
     'B':([(1,15),(1,16),(1,17),(1,18),(2,15),(2,18),(3,18),(4,18)],[(3,15),(4,17)]),
     'L':([(13,1),(14,1),(15,1),(16,1),(17,1)],[(12,1),(17,2)]),
     'U':([(15,11),(15,16),(16,10),(16,11),(16,16),(16,17),(17,10),(17,17),(18,10),(18,11),(18,12),(18,13),(18,14),(18,15),(18,16),(18,17)],[(14,11),(14,16),(17,9)])}

def mfinal(fr):
    for c in extract6(fr):
        if len(c['prong'])>=4 and 7<=bbox(c['body'])[0]<=12:
            return (sorted(c['body']),sorted(c['prong']))

def evalcfg(V,poses,names):
    locs={};bodies={}
    for n in names:
        b,q=ab(V,n,poses[n][0],poses[n][1]);bodies[n]=b
        for i,p in enumerate(q): locs.setdefault(p,[]).append((n,i))
    for i,a in enumerate(names):
        for b in names[i+1:]:
            if bodies[a]&bodies[b]: return None
            if set(ab(V,a,poses[a][0],poses[a][1])[1]) & set(ab(V,b,poses[b][0],poses[b][1])[1]): pass
    pairs=[]
    for key,v in locs.items():
        if len(v)>2: return None
        if len(v)==2:
            if v[0][0]==v[1][0]: return None
            pairs.append((v[0][0],v[1][0]))
    adj={n:set() for n in names}
    for n,m in pairs: adj[n].add(m);adj[m].add(n)
    vis={names[0]};st=[names[0]]
    while st:
        x=st.pop()
        for y in adj[x]:
            if y not in vis: vis.add(y);st.append(y)
    if len(vis)!=len(names): return None
    return (len(pairs),sorted(pairs))

def searchmax(V,names,root):
    import itertools
    best={};seen=set()
    for kr in range(4):
        for order in itertools.permutations([n for n in names if n!=root]):
            poses={root:(kr,(0,0))}
            def dfs(i):
                if i==len(order):
                    r=evalcfg(V,poses,names)
                    if r:
                        key=tuple(sorted((poses[n][0],poses[n][1][0]-poses[root][1][0],poses[n][1][1]-poses[root][1][1]) for n in names))
                        if key not in seen:
                            seen.add(key);best.setdefault(r[0],{n:(poses[n][0],(poses[n][1][0]-poses[root][1][0],poses[n][1][1]-poses[root][1][1])) for n in names})
                    return
                n=order[i]
                for (pose,) in genB(V,poses,names,n):
                    poses[n]=pose;dfs(i+1);del poses[n]
            dfs(0)
    return best

def rot(cs,k):
    return [(r,-c) for (r,c) in cs] if k==1 else [(-r,-c) for (r,c) in cs] if k==2 else [(-r,c) for (r,c) in cs] if k==3 else list(cs)

def nrm2(cs):
    mr=min(r for r,c in cs);mc=min(c for r,c in cs)
    return frozenset((r-mr,c-mc) for r,c in cs)

def pose2(sp,n,k,off):
    b,pr=sp[n]
    br=nrm2(rot(b,k));pr2=nrm2(rot(pr,k))
    mr=min(r for r,c in br);mc=min(c for r,c in br)
    return (frozenset((r-mr+off[0],c-mc+off[1]) for r,c in br),frozenset((r-mr+off[0],c-mc+off[1]) for r,c in pr2))

def rotc(cs,k):
    out=list(cs)
    for _ in range(k): out=[(r,-c) for (r,c) in out]
    return out

def nrmc(cs):
    mr=min(r for r,c in cs);mc=min(c for r,c in cs)
    return frozenset((r-mr,c-mc) for r,c in cs)

def pose(n,k,off):
    b,pr=SH[n]
    br=nrmc(rotc(b,k));pr2=nrmc(rotc(pr,k))
    mr=min(r for r,c in br);mc=min(c for r,c in br)
    return (frozenset((r-mr+off[0],c-mc+off[1]) for r,c in br),
            frozenset((r-mr+off[0],c-mc+off[1]) for r,c in pr2))

def P(n): return pose(n,0,(0,0))[1]

def search2(names,root,best=[0],res=[],cap=300000):
    nodes=[0];res.clear();best[0]=-1
    def score(prcnt): return sum(2 for v in prcnt.values() if v==2)
    def rec(placed,occ_b,occ_p,prcnt):
        nodes[0]+=1
        if nodes[0]>cap: return
        if len(placed)==len(names):
            s=score(prcnt)
            if s>best[0]:
                best[0]=s;res.clear();res.append({n:(placed[n][0],placed[n][1]) for n in names})
            elif s==best[0]: res.append({n:(placed[n][0],placed[n][1]) for n in names})
            return
        for n in [x for x in names if x not in placed]:
            for k in range(4):
                b0,p0=pose(n,k,(0,0))
                tried=set()
                for q in p0:
                    for x in occ_p:
                        off=(x[0]-q[0],x[1]-q[1])
                        if off in tried: continue
                        tried.add(off)
                        b,p=pose(n,k,off)
                        if b&occ_b or (b&occ_p) or (p&occ_b): continue
                        pc=dict(prcnt);bad=False
                        for y in p:
                            pc[y]=pc.get(y,0)+1
                            if pc[y]>2: bad=True
                        if bad: continue
                        placed[n]=(k,off,b,p)
                        rec(placed,occ_b|b,occ_p|p,pc)
                        placed.pop(n,None)
    for kr in range(4):
        b,p=pose(root,kr,(0,0))
        rec({root:(kr,(0,0),b,p)},b,p,{q:1 for q in p})
    return best[0],res

def SHD():
    return {'G':([(1,1),(1,2),(1,3),(1,4),(1,5),(2,1),(2,5),(3,1),(4,1),(5,1),(5,2)],[(3,5),(5,3),(6,1)]),
 'B':([(1,15),(1,16),(1,17),(1,18),(2,15),(2,18),(3,18),(4,18)],[(3,15),(4,17)]),
 'M':([(6,12),(7,7),(7,8),(7,12),(8,7),(8,12),(9,7),(9,12),(10,7),(10,8),(10,9),(10,10),(10,11),(10,12)],[(6,8),(6,11),(7,13),(8,6),(11,7),(11,12)]),
 'L':([(13,1),(14,1),(15,1),(16,1),(17,1)],[(12,1),(17,2)]),
 'U':([(15,11),(15,16),(16,10),(16,11),(16,16),(16,17),(17,10),(17,17),(18,10),(18,11),(18,12),(18,13),(18,14),(18,15),(18,16),(18,17)],[(14,11),(14,16),(17,9)])}

def poseS(SH,n,k,off):
    b,pr=SH[n]
    br=nrmc(rotc(b,k));pr2=nrmc(rotc(pr,k))
    mr=min(r for r,c in br);mc=min(c for r,c in br)
    return (frozenset((r-mr+off[0],c-mc+off[1]) for r,c in br),
            frozenset((r-mr+off[0],c-mc+off[1]) for r,c in pr2))

def search4(SH,names,root,cap=400000):
    best=[-1];res=[];nodes=[0]
    def rec(placed,occ_b,occ_p,prcnt):
        nodes[0]+=1
        if nodes[0]>cap: return
        if len(placed)==len(names):
            s=sum(2 for v in prcnt.values() if v==2)
            if s>best[0]: best[0]=s;res.clear()
            if s==best[0]: res.append({n:(placed[n][0],placed[n][1]) for n in names})
            return
        for n in [x for x in names if x not in placed]:
            for k in range(4):
                b0,p0=poseS(SH,n,k,(0,0))
                tried=set()
                for q in p0:
                    for x in occ_p:
                        off=(x[0]-q[0],x[1]-q[1])
                        if off in tried: continue
                        tried.add(off)
                        b,p=poseS(SH,n,k,off)
                        if b&occ_b or (b&occ_p) or (p&occ_b): continue
                        pc=dict(prcnt);bad=False
                        for y in p:
                            pc[y]=pc.get(y,0)+1
                            if pc[y]>2: bad=True
                        if bad: continue
                        placed[n]=(k,off,b,p)
                        rec(placed,occ_b|b,occ_p|p,pc)
                        placed.pop(n,None)
    for kr in range(4):
        b,p=poseS(SH,root,kr,(0,0))
        rec({root:(kr,(0,0),b,p)},b,p,{q:1 for q in p})
    return best[0],res

def blk2(fr,k0,k1,m0,m1):
    G=[list(x) for x in fr.ascii.split('\n')]
    mp={'b':'.','p':'.','W':'W','R':'P','c':'#','Y':'Y','S':'S','O':'O','N':'N','G':'G','M':'M','P':'P','O':'O'}
    out=[]
    for k in range(k0,k1+1):
        s=''
        for m in range(m0,m1+1):
            r,c=3*k+2,3*m+2
            s+= mp.get(G[r][c],'?') if r<64 and c<64 else ' '
        out.append(s)
    return out

def rectcheck(cs):
    rs=[x[0] for x in cs];cs2=[x[1] for x in cs]
    r0,r1,c0,c1=min(rs),max(rs),min(cs2),max(cs2)
    area=(r1-r0+1)*(c1-c0+1)
    return (r1-r0+1,c1-c0+1,len(cs),area)

def enclosed(fr):
    G=[list(x) for x in fr.ascii.split('\n')]
    cnt=Counter(''.join(''.join(r) for r in G))
    bg=cnt.most_common(1)[0][0]
    R=len(G);C=len(G[0])
    occ=[[G[r][c]!=bg for c in range(C)] for r in range(R)]
    for c in range(C): occ[0][c]=True   # treat HUD row as blocked? better: start flood from row1
    seen=[[False]*C for _ in range(R)]
    st=[]
    for c in range(C):
        for r in (1,):
            if not occ[r][c] and not seen[r][c]: st.append((r,c));seen[r][c]=True
    for r in range(R):
        for cc in (0,C-1):
            if not occ[r][cc] and not seen[r][cc]: st.append((r,cc));seen[r][cc]=True
    for r in (R-1,):
        for cc in range(C):
            if not occ[r][cc] and not seen[r][cc]: st.append((r,cc));seen[r][cc]=True
    while st:
        r,c=st.pop()
        for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
            nr,nc=r+dr,c+dc
            if 1<=nr<R and 0<=nc<C and not occ[nr][nc] and not seen[nr][nc]:
                seen[nr][nc]=True;st.append((nr,nc))
    tot=sum(1 for r in range(1,R) for c in range(C) if not occ[r][c] and not seen[r][c])
    return tot

def enclosed_blocks(fr):
    G=[list(x) for x in fr.ascii.split('\n')]
    cnt=Counter(''.join(''.join(r) for r in G))
    bg=cnt.most_common(1)[0][0]
    K=21;M=21
    occ=[[False]*M for _ in range(K)]
    for k in range(K):
        for m in range(M):
            r,c=3*k+1,3*m+1   # sample top-left of block to catch blocks even if center differs
            r2,c2=3*k+2,3*m+2
            if r2<64 and c2<64 and G[r2][c2]!=bg: occ[k][m]=True
    seen=[[False]*M for _ in range(K)]
    st=[]
    for k in range(K):
        for m in (0,M-1):
            if not occ[k][m] and not seen[k][m]: st.append((k,m));seen[k][m]=True
    for m in range(M):
        for k in (0,K-1):
            if not occ[k][m] and not seen[k][m]: st.append((k,m));seen[k][m]=True
    while st:
        k,m=st.pop()
        for dk,dm in ((1,0),(-1,0),(0,1),(0,-1)):
            nk,nm=k+dk,m+dm
            if 0<=nk<K and 0<=nm<M and not occ[nk][nm] and not seen[nk][nm]:
                seen[nk][nm]=True;st.append((nk,nm))
    return sum(1 for k in range(K) for m in range(M) if not occ[k][m] and not seen[k][m])
