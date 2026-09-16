import sys,os,json,hashlib
os.environ['OMP_NUM_THREADS']='2'
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0,ROOT)
from pathlib import Path
import pandas as pd
from pdm.suites import DEFAULT
import copy
R=Path(ROOT); OUT=Path(sys.argv[1] if len(sys.argv)>1 else os.path.join(ROOT,'output')); DATA=R/'data'
jobs=[]
for seed in [11,22,33]:
    for m in ['fedavg','gossip']:
        for sc in ['packet_loss','node_failure','server_outage']: jobs.append(dict(method=m,seed=seed,scenario=sc))
        jobs.append(dict(method=m,seed=seed,partition='lifetime'))
        jobs.append(dict(method=m,seed=seed,use_cycle=False))
        for n in [5,20,40]: jobs.append(dict(method=m,seed=seed,nodes=n))
    jobs.append(dict(method='local',seed=seed,partition='lifetime'))
sha=hashlib.sha256(b''.join(p.read_bytes() for p in [R/'run_pdm.py']+sorted((R/'pdm').glob('*.py')))).hexdigest()
dh=hashlib.sha256(b''.join((DATA/n).read_bytes() for n in ['PM_train.txt','PM_test.txt','PM_truth.txt'])).hexdigest()
from pdm.engine import run_job
from pdm.report import summary
for j in jobs:
    c=copy.deepcopy(DEFAULT);c.update(j);c['threads']=2;c['dataset']='FD001';c['software_sha256']=sha;c['data_content_sha256']=dh
    digest=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()[:12]
    p=OUT/f"{c['dataset']}_{c['method']}_s{c['seed']}_n{c['nodes']}_{digest}"
    if (p/'DONE.json').exists(): continue
    run_job(c,DATA,p); summary(OUT)
print('ALLDONE',flush=True)
