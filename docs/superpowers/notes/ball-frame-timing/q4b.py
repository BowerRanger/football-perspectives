import sys, json, numpy as np, cv2
sys.path.insert(0,'/Users/joebower/.claude/jobs/88133eb1/tmp/lag')
from q4 import blob
O='/Users/joebower/workplace/football-perspectives/'
for out,clip in [('output-origi-shorts','origi01'),('output-shorts','gberch'),('output-kroupi-shorts','kroupi01')]:
    try: A=json.load(open(f'{O}{out}/ball/{clip}_ball_anchors.json'))['anchors']
    except Exception as e: print(clip,'no anchors',e); continue
    cap=cv2.VideoCapture(f'{O}{out}/shots/{clip}.mp4'); frs=[]
    while True:
        ok,f=cap.read()
        if not ok:break
        frs.append(f)
    A=[a for a in A if a.get('image_xy') and a['state']!='off_screen_flight']
    cnt={-2:0,-1:0,0:0,1:0,2:0}; tot=0; amb=0; nob=0; rows=[]
    for a in A:
        N=a['frame']; uv=a['image_xy']
        if N<3 or N+3>=len(frs): continue
        d={}
        for k in range(-2,3):
            b=blob(frs[N+k],uv,60)
            d[k]=b[0] if b else 999
        # require ball moving: positions differ between frames -> best vs second best margin
        ks=sorted(d,key=d.get)
        # skip near-duplicate frames: pixel-identical neighbours break discrimination
        if d[ks[0]]>9: nob+=1; continue
        if d[ks[1]]-d[ks[0]]<4: amb+=1; continue
        tot+=1; cnt[ks[0]]+=1; rows.append((N,ks[0],round(d[ks[0]],1)))
    print(clip,'anchors',len(A),'discriminable',tot,'ambiguous(static/dup)',amb,'no blob/too far',nob,'best-frame offset counts (frame=N+k)',cnt)
    json.dump(rows,open(f'/Users/joebower/.claude/jobs/88133eb1/tmp/lag/q4_{clip}.json','w'))
