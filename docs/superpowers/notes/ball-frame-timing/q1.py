import sys, json, numpy as np, cv2
sys.path.insert(0,'.')
from src.utils.wasb_ball_detector import WASBBallDetector, heatmap_candidates, _affine_apply
GT={414:(717.5,488.5),416:(648.5,483.5),417:(617.5,480),430:(577.5,491.5),444:(555,507.5),446:(490,500),449:(427.5,487.5),450:(401,486.5),452:(342.5,485),453:(320,475),454:(317.5,467.5)}
clip='/Users/joebower/workplace/football-perspectives/output-origi-shorts/shots/origi01.mp4'
W='third_party/wasb_sbdt/pretrained_weights/'
res={}
for name in ['wasb_soccer_best','wasb_soccer_finetuned_v1','wasb_soccer_finetuned_v2']:
    d=WASBBallDetector(W+name+'.pth.tar',device='cpu')
    cap=cv2.VideoCapture(clip)
    # warm-up from 400
    cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    for i in range(400): cap.read()
    per={}
    for f in range(400,460):
        ok,fr=cap.read()
        d._forward  # noqa
        # replicate _forward to get all 3 heatmaps
        if not d._buffer: d._buffer=[fr.copy()]*3
        else:
            d._buffer.append(fr.copy()); d._buffer=d._buffer[-3:]
        h,w=fr.shape[:2]
        st,ti=d._preprocess_buffer((h,w))
        inp=d._torch.from_numpy(st).unsqueeze(0)
        with d._torch.no_grad(): out=d._model(inp)
        hms=d._torch.sigmoid(out[0]).numpy()[0]
        r=[]
        for k in range(3):
            c=heatmap_candidates(hms[k],0.1,1)
            if c:
                uv=_affine_apply(np.array([c[0][0],c[0][1]],dtype=np.float32),ti); r.append((float(uv[0]),float(uv[1]),float(c[0][2])))
            else: r.append(None)
        per[f]=r
    res[name]=per
    print(name,'done',flush=True)
json.dump(res,open('/Users/joebower/.claude/jobs/88133eb1/tmp/lag/q1.json','w'))
