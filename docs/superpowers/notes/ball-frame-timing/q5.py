import json,numpy as np
O='/Users/joebower/workplace/football-perspectives/'
for out,c in [('output-origi-shorts','origi01'),('output-shorts','gberch'),('output-kroupi-shorts','kroupi01')]:
    t=json.load(open(f'{O}{out}/ball/{c}_ball_track.json'))
    xyz={f['frame']:np.array(f['world_xyz']) for f in t['frames'] if f.get('world_xyz')}
    sp=np.array([np.linalg.norm(xyz[k+1]-v)*30 for k,v in xyz.items() if k+1 in xyz])
    print(c,'n',len(sp),'speed m/s p25/50/75/90/max',np.percentile(sp,[25,50,75,90,100]).round(1),'1 frame@30fps m (p50,p75,p90):',(np.percentile(sp,[50,75,90])/30).round(2))
n=np.arange(0,60); m=np.floor(n*25/30+1e-9); st=n/30-m/25
print('staleness s mean',st.mean().round(4),'max',st.max().round(4),'frac>0.02',(st>0.02).mean())
