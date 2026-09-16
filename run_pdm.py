"""Command-line interface for the gossip-LSTM C-MAPSS experiments. Default: short CPU pilot; use plan before a large suite."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent

def audit(folder,out):
    import numpy as np
    import pandas as pd
    from pdm.data import load_raw,EXPECTED
    from pdm.engine import save_json
    from pdm.report import dataset_plots
    tr,te,truth,hashes=load_raw(folder)
    all_windows=sum(max(0,len(te[te[:,0]==i])-49) for i in np.unique(te[:,0]))
    terminal=sum(len(te[te[:,0]==i])>=50 for i in np.unique(te[:,0]))
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    data=dict(hashes=hashes,identical_to_supplied_data=hashes==EXPECTED,train_rows=len(tr),test_rows=len(te),
              test_engines=len(truth),all_valid_windows=all_windows,terminal_windows=terminal,
              parameter_count=80651,
              ideal_ring_and_fedavg_50_rounds_bytes=80651*4*20*50,
              ideal_ring_and_fedavg_50_rounds_MiB=80651*4*20*50/2**20)
    save_json(out/'audit.json',data);dataset_plots(folder,out)
    print(json.dumps(data,indent=2),flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',nargs='?',default='run',choices=['run','plan','audit','report','verify','doctor'])
    parser.add_argument('--suite',default='pilot',choices=['smoke','pilot','core','novelty','robustness','scalability','all'])
    parser.add_argument('--data',type=Path,default=ROOT/'data',help='Folder containing PM_train/test/truth.txt')
    parser.add_argument('--dataset-name',default='FD001',help='Label for supplied triplet; no download performed')
    parser.add_argument('--output',type=Path,default=ROOT/'output')
    parser.add_argument('--seeds',nargs='+',type=int)
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--methods',nargs='+',choices=['local','centralized','fedavg','gossip','fedavg_weighted','gossip_weighted','event_gossip','event_gossip_weighted','mass_gossip','event_mass_gossip'])
    parser.add_argument('--bootstrap',type=int)
    parser.add_argument('--max-jobs',type=int,help='Limit NEW/unfinished jobs this session; same command resumes remaining jobs')
    args=parser.parse_args()
    if args.threads<1 or (args.epochs is not None and args.epochs<1) or (args.bootstrap is not None and args.bootstrap<1):parser.error('Positive counts required')
    # Must set before importing torch/numerical libraries.
    os.environ['OMP_NUM_THREADS']=str(args.threads);os.environ['MKL_NUM_THREADS']=str(args.threads)
    try:
        import torch
        import numpy as np
        import pandas as pd
        import scipy,sklearn,matplotlib
    except ImportError as e:
        print('Missing dependency:',e,'\nInstall the requirements as explained in README.md.');return 2
    if args.command=='doctor':
        import platform
        print(json.dumps(dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,pandas=pd.__version__,
                    scipy=scipy.__version__,sklearn=sklearn.__version__,matplotlib=matplotlib.__version__,
                    cpu=platform.processor(),logical_cpus=os.cpu_count(),threads=args.threads,device='CPU'),indent=2));return 0
    if args.command=='audit':audit(args.data,args.output/'audit');return 0
    if args.command=='report':
        from pdm.report import summary
        summary(args.output);print('Report:',args.output/'REPORT.html');return 0
    if args.command=='verify':
        from pdm.report import verify_job
        paths=list(args.output.glob('*/DONE.json'))
        if not paths:print('No completed jobs.');return 2
        failed=[]
        for p in paths:
            check=verify_job(p.parent);print(p.parent.name,check['passed'])
            if not check['passed']:failed.append(str(p.parent))
        return 1 if failed else 0
    from pdm.suites import suite
    configs=suite(args.suite,args.seeds)
    if args.methods:configs=[c for c in configs if c['method'] in args.methods]
    for c in configs:
        c['threads']=args.threads;c['dataset']=args.dataset_name
        c['software_sha256']=hashlib.sha256(b''.join(p.read_bytes() for p in [ROOT/'run_pdm.py']+sorted((ROOT/'pdm').glob('*.py')))).hexdigest()
        if args.epochs is not None:c['epochs']=args.epochs
        if args.bootstrap is not None:c['bootstrap']=args.bootstrap
    args.output.mkdir(parents=True,exist_ok=True)
    tasks=[]
    # Dataset hash is part of identity so a different triplet cannot reuse a previous result.
    dh=hashlib.sha256(b''.join((args.data/n).read_bytes() for n in ['PM_train.txt','PM_test.txt','PM_truth.txt'])).hexdigest()
    for c in configs:
        c['data_content_sha256']=dh
        digest=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()[:12]
        name=f"{c['dataset']}_{c['method']}_s{c['seed']}_n{c['nodes']}_{digest}"
        tasks.append((c,args.output/name))
    plan=pd.DataFrame([dict(job=p.name,complete=(p/'DONE.json').exists(),**c) for c,p in tasks])
    plan.to_csv(args.output/f'plan_{args.suite}.csv',index=False)
    print(f'Suite {args.suite}: {len(tasks)} jobs; {sum(not(p/"DONE.json").exists() for _,p in tasks)} unfinished.',flush=True)
    print('Large suites can take many CPU hours/days. --max-jobs bounds each session. No measured runtime estimate until pilot runs.',flush=True)
    if args.command=='plan':
        print(plan[['method','seed','nodes','partition','scenario','topology','period','epochs']].to_string(index=False));return 0
    from pdm.engine import run_job
    from pdm.report import summary
    count=0
    try:
        for c,p in tasks:
            if (p/'DONE.json').exists():continue
            if args.max_jobs is not None and count>=args.max_jobs:break
            run_job(c,args.data,p);count+=1
            summary(args.output)
    except KeyboardInterrupt:
        print('\nStopped. Completed epochs are retained. Repeat identical command to resume.',flush=True)
        return 130
    summary(args.output)
    print('Report:',args.output/'REPORT.html',flush=True)
    print('Next: run_pdm.py verify --output "'+str(args.output)+'"',flush=True)
    return 0

if __name__=='__main__':sys.exit(main())
