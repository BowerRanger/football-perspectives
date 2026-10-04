import sys, json, numpy as np, cv2
O='/Users/joebower/workplace/football-perspectives/'
def blob(fr,uv,R=34):
    u,v=int(round(uv[0])),int(round(uv[1])); h,w=fr.shape[:2]
    x0,x1,y0,y1=max(0,u-R),min(w,u+R+1),max(0,v-R),min(h,v+R+1)
    p=fr[y0:y1,x0:x1].astype(np.int32)
    mn=p.min(2); mx=p.max(2)
    m=((p[:,:,0]>105)&(p[:,:,2]>185)).astype(np.uint8)
    n,lab,st,cen=cv2.connectedComponentsWithStats(m,connectivity=8)
    best=None
    for i in range(1,n):
        a=st[i,4]; bw,bh=st[i,2],st[i,3]
        if a<30 or a>700 or max(bw,bh)>4*min(bw,bh)+4 or max(bw,bh)>40: continue
        c=(cen[i][0]+x0,cen[i][1]+y0); d=np.hypot(c[0]-uv[0],c[1]-uv[1])
        if best is None or d<best[0]: best=(d,c,a)
    return best
def run(clip,anchors,gt=None):
    cap=cv2.VideoCapture(clip); frs=[]
    while True:
        ok,f=cap.read()
        if not ok: break
        frs.append(f)
    print(clip,len(frs))
    dup=lambda i: i>0 and np.array_equal(frs[i],frs[i-1])
    out=[]
    for fr_n,uv in anchors:
        if fr_n<2 or fr_n+2>=len(frs): continue
        r={k:blob(frs[fr_n+k],uv) for k in (-1,0,1)}
        # ball position per frame via blob nearest to anchor: need discriminability -> blobs must differ
        out.append((fr_n,uv,r,dup(fr_n),dup(fr_n+1),dup(fr_n-1)))
    return out,frs
if __name__=='__main__':
    # validation on GT
    GT={414:(717.5,488.5),416:(648.5,483.5),417:(617.5,480),430:(577.5,491.5),444:(555,507.5),446:(490,500),449:(427.5,487.5),450:(401,486.5),452:(342.5,485),453:(320,475),454:(317.5,467.5)}
    cap=cv2.VideoCapture(O+'output-origi-shorts/shots/origi01.mp4'); frs=[]
    while True:
        ok,f=cap.read()
        if not ok:break
        frs.append(f)
    for g,uv in GT.items():
        b=blob(frs[g],uv,20); print(g,uv,None if b is None else (round(b[0],1),tuple(round(float(x),1) for x in b[1]),b[2]))
