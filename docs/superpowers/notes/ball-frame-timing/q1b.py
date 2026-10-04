import json, numpy as np
GT={414:(717.5,488.5),416:(648.5,483.5),417:(617.5,480),430:(577.5,491.5),444:(555,507.5),446:(490,500),449:(427.5,487.5),450:(401,486.5),452:(342.5,485),453:(320,475),454:(317.5,467.5)}
R=json.load(open('/Users/joebower/.claude/jobs/88133eb1/tmp/lag/q1.json'))
for ck,per in R.items():
    for k in range(3):
        print(ck,'hm',k)
        row=[]
        for lag in (-2,-1,0,1,2):   # det frame = g+lag
            errs=[]
            for g,(u,v) in GT.items():
                d=per.get(str(g+lag))
                if d and d[k]: errs.append(np.hypot(d[k][0]-u,d[k][1]-v))
                else: errs.append(np.nan)
            row.append(errs)
        a=np.array(row) # lag x gt
        best=np.nanargmin(np.where(np.isnan(a),1e9,a),axis=0)
        print('  median err by detframe=g+lag:',{l:round(float(np.nanmedian(a[i])),1) for i,l in enumerate((-2,-1,0,1,2))},' best-lag counts',{l:int((best==i).sum()) for i,l in enumerate((-2,-1,0,1,2))})
# per frame detail for v1 hm2 & stock hm2
for ck in R:
    print(ck,'hm2 per-gt-frame err at lag -1/0/+1')
    for g in GT:
        e=[]
        for lag in (-1,0,1):
            d=R[ck].get(str(g+lag)); e.append(round(float(np.hypot(d[2][0]-GT[g][0],d[2][1]-GT[g][1])),1) if d and d[2] else None)
        print('  ',g,e)
