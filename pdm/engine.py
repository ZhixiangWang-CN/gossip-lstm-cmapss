import copy
import json
import os
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .data import prepare
from .model import LSTM, seed_all, vector, put_vector, train_epoch, predict
from .metrics import metrics, select_threshold, cluster_ci, alarm_metrics, membership_diagnostic
from .network import communicate, active_nodes, modeled_network_time

def clean_json(x):
    if isinstance(x,dict):return {str(k):clean_json(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean_json(v) for v in x]
    if isinstance(x,np.generic):return clean_json(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x

def save_json(path,data):
    path=Path(path);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(clean_json(data),indent=2,ensure_ascii=False),encoding='utf-8')
    os.replace(tmp,path)

def assigned_predictions(models,w,cfg):
    x=torch.from_numpy(w.x);out=np.empty(len(w.meta))
    if cfg['method']=='centralized':return predict(models[0],x)
    for i,m in enumerate(models):
        ix=np.flatnonzero(w.meta.node.to_numpy()==i)
        if len(ix):out[ix]=predict(m,x[ix])
    return out

def evaluate_nodes(models,w,cfg):
    frame=w.meta.copy();frame['prob']=assigned_predictions(models,w,cfg)
    return frame

def run_job(cfg,folder,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'DONE.json').exists():
        print('SKIP complete:',out.name,flush=True);return
    cfgfile=out/'config.json'
    if cfgfile.exists() and json.loads(cfgfile.read_text())!=cfg:
        raise ValueError(f'Configuration conflict: {out}. Use a different output directory.')
    save_json(cfgfile,cfg)
    torch.set_num_threads(cfg['threads'])
    torch.use_deterministic_algorithms(True)
    train,val,test,audit=prepare(folder,cfg)
    save_json(out/'data_audit.json',audit)
    save_json(out/'environment.json',dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
                 platform=platform.platform(),processor=platform.processor(),threads=cfg['threads'],device='cpu'))
    node_rows=[]
    for label,w in [('train',train),('validation',val),('test',test)]:
        for i,g in w.meta.groupby('node'):
            node_rows.append(dict(split=label,node=i,engines=g.engine.nunique(),windows=len(g),positive=int(g.y.sum()),positive_fraction=g.y.mean()))
    pd.DataFrame(node_rows).to_csv(out/'node_data.csv',index=False)
    seed_all(cfg['seed']);base=LSTM(tuple(cfg['hidden']))
    models=[copy.deepcopy(base) for _ in range(cfg['nodes'])]
    initial=vector(base);payload=initial.numel()*initial.element_size()
    xt=torch.from_numpy(train.x);yt=torch.tensor(train.meta.y.to_numpy(),dtype=torch.float32)
    slices=[np.flatnonzero(train.meta.node.to_numpy()==i) for i in range(cfg['nodes'])]
    train_sizes=[len(ix) for ix in slices]
    weighted=cfg['method'].endswith('_weighted')
    pw=min(20.,float((len(yt)-yt.sum())/max(1.,float(yt.sum())))) if weighted else 1.
    server=initial.clone();state=None;history=[];node_history=[];ledger=[];begin=0
    ckpt=out/'checkpoint.pt'
    if ckpt.exists():
        data=torch.load(ckpt,map_location='cpu',weights_only=False)
        for m,v in zip(models,data['vectors']):put_vector(m,v)
        server=data['server'];state=data['state'];history=data['history'];node_history=data.get('node_history',[]);ledger=data['ledger'];begin=data['epoch']
        print('RESUME epoch',begin,'/',cfg['epochs'],flush=True)
    # A common engine-balanced validation probe, not test data, used only for diagnostics.
    probe_ix=[]
    for _,g in val.meta.groupby('engine'):
        probe_ix.extend(g.iloc[np.linspace(0,len(g)-1,min(8,len(g))).astype(int)].index.tolist())
    probe=torch.from_numpy(val.x[probe_ix])
    for epoch in range(begin,cfg['epochs']):
        start=time.perf_counter();active=active_nodes(cfg,epoch);times=[];losses=[];processed=0
        train_count=1 if cfg['method']=='centralized' else cfg['nodes']
        for i in range(train_count):
            ix=np.arange(len(xt)) if cfg['method']=='centralized' else slices[i]
            if not active[i]:times.append(0.);continue
            t=time.perf_counter()
            losses.append(train_epoch(models[i],xt[ix],yt[ix],cfg,cfg['seed']*100003+epoch*100+i,pw))
            times.append(time.perf_counter()-t);processed+=len(ix)
        before=[vector(m) for m in models]
        if cfg['method']=='centralized':
            for m in models[1:]:put_vector(m,before[0])
            before=[before[0].clone() for _ in models]
        t=time.perf_counter()
        vs,server,state,records,net=communicate(before,server,train_sizes,cfg,epoch,state)
        for m,v in zip(models,vs):put_vector(m,v)
        mix_seconds=time.perf_counter()-t
        train_wall=time.perf_counter()-start
        # No checkpoint reload here: mixed weights become next round's starting weights.
        vstack=torch.stack(vs);mean=vstack.mean(0)
        param_rms=float(torch.sqrt(torch.mean((vstack-mean)**2)))
        parameter_relative=float(torch.linalg.vector_norm(vstack-mean)/(torch.linalg.vector_norm(mean)*np.sqrt(len(models))+1e-12))
        if (epoch+1)%cfg['eval_every']==0 or epoch+1==cfg['epochs']:
            vp=assigned_predictions(models,val,cfg);vm=metrics(val.meta.y,vp)
            pred=np.stack([predict(m,probe) for m in models])
            disagreement=float(np.mean([(pred[i]>=.5)!=(pred[j]>=.5) for i in range(len(models)) for j in range(i+1,len(models))]))
            val_f1=vm['f1'];val_loss=vm['log_loss'];val_acc=vm['accuracy']
            for split,window in [('train',train),('validation',val)]:
                frame=evaluate_nodes(models,window,cfg)
                for node,g in frame.groupby('node'):
                    node_history.append(dict(epoch=epoch+1,split=split,node=int(node),**metrics(g.y,g.prob)))
        else:val_f1=val_loss=val_acc=disagreement=None
        row=dict(epoch=epoch+1,train_loss=float(np.mean(losses)),val_loss=val_loss,val_accuracy=val_acc,val_f1=val_f1,
                 parameter_rms=param_rms,parameter_relative=parameter_relative,probe_disagreement=disagreement,
                 mix_change_l2=float(torch.linalg.vector_norm(torch.stack(vs)-torch.stack(before))),
                 bytes_attempted=sum(r['bytes'] for r in records),messages=len(records),
                 bytes_delivered=sum(r['bytes'] for r in records if r['delivered']),
                 model_bytes=sum(r['bytes'] for r in records if r['kind']=='model'),
                 control_bytes=sum(r['bytes'] for r in records if r['kind']!='model'),
                 training_wall_s=train_wall,local_compute_sum_s=sum(times),local_compute_max_s=max(times),mix_cpu_s=mix_seconds,
                 modeled_network_s=modeled_network_time(records,cfg,cfg['method'].startswith('fedavg')),
                 samples_processed=processed,**net)
        history.append(row);ledger.extend(records)
        pd.DataFrame(history).to_csv(out/'history.csv',index=False)
        pd.DataFrame(node_history).to_csv(out/'history_node.csv',index=False)
        tmp=out/'checkpoint.tmp'
        torch.save(dict(epoch=epoch+1,vectors=vs,server=server,state=state,history=history,node_history=node_history,ledger=ledger),tmp)
        os.replace(tmp,ckpt)
        print(f"{out.name}: epoch {epoch+1}/{cfg['epochs']} train={row['train_loss']:.4f} val_F1={val_f1} bytes={row['bytes_attempted']} time={train_wall:.1f}s",flush=True)
    pd.DataFrame(ledger,columns=['epoch','sender','receiver','kind','bytes','delivered','latency_s','applied']).to_csv(out/'messages.csv.gz',index=False)
    train_frame=evaluate_nodes(models,train,cfg);val_frame=evaluate_nodes(models,val,cfg);test_frame=evaluate_nodes(models,test,cfg)
    for name,f in [('train',train_frame),('validation',val_frame),('test',test_frame)]:f.to_csv(out/f'predictions_{name}.csv.gz',index=False)
    thresholds=dict(fixed=.5,val_f1=select_threshold(val_frame.y,val_frame.prob,'val_f1'),
                    val_fpr02=select_threshold(val_frame.y,val_frame.prob,'val_fpr02'))
    result=dict(config=cfg,trainable_parameters=initial.numel(),immutable_zero_bias_parameters=sum(p.numel() for p in base.parameters() if not p.requires_grad),
                payload_bytes=payload,positive_weight=pw,thresholds=thresholds,
                bytes_total=sum(r['bytes'] for r in ledger),messages_total=len(ledger),
                training_wall_s=sum(h['training_wall_s'] for h in history),
                modeled_parallel_s=sum(h['local_compute_max_s']+h['modeled_network_s'] for h in history),
                samples_processed=sum(h['samples_processed'] for h in history),
                final_available_node_fraction=float(active_nodes(cfg,cfg['epochs']-1,True).mean()),metrics={})
    tables=[]
    for split,f in [('train',train_frame),('validation',val_frame),('test_all',test_frame),('test_terminal',test_frame[test_frame.terminal])]:
        for policy,threshold in thresholds.items():
            m=metrics(f.y,f.prob,threshold)
            result['metrics'][f'{split}/{policy}']=m
            if split.startswith('test'):
                m['cluster_ci95']=cluster_ci(f,threshold,cfg['bootstrap'],cfg['seed'])
            for i,g in f.groupby('node'):
                tables.append(dict(split=split,policy=policy,node=int(i),**metrics(g.y,g.prob,threshold)))
    node_table=pd.DataFrame(tables)
    node_table.to_csv(out/'node_metrics.csv',index=False)
    # Equal-weight mean across nonempty nodes, NOT macro over the two classes.
    node_table.groupby(['split','policy']).agg(
        n_nodes=('node','nunique'),accuracy=('accuracy','mean'),precision=('precision','mean'),
        recall=('recall','mean'),f1=('f1','mean'),fpr=('fpr','mean'),log_loss=('log_loss','mean')
    ).reset_index().to_csv(out/'node_macro_metrics.csv',index=False)
    for policy,t in thresholds.items():alarm_metrics(test_frame,t).to_csv(out/f'alarms_{policy}.csv',index=False)
    # Common-probe distances directly test consensus; local accuracy similarity does not.
    v=torch.stack([vector(m) for m in models])
    distances=torch.cdist(v,v)/np.sqrt(v.shape[1])
    pd.DataFrame(distances.numpy()).to_csv(out/'pairwise_parameter_rms.csv',index=False)
    probabilities=np.stack([predict(m,probe) for m in models])
    pd.DataFrame(probabilities.T).to_csv(out/'common_probe_predictions.csv',index=False)
    val.meta.iloc[probe_ix].to_csv(out/'common_probe_metadata.csv',index=False)
    pd.DataFrame([dict(node=i,**metrics(val.meta.iloc[probe_ix].y,p)) for i,p in enumerate(probabilities)]).to_csv(out/'common_probe_node_metrics.csv',index=False)
    attack,diag=membership_diagnostic(train_frame,val_frame)
    attack.to_csv(out/'membership_engine_scores.csv',index=False);result['privacy_diagnostic']=diag
    # Recalculate deployed metrics only for nodes still serviceable at the last round.
    available=active_nodes(cfg,cfg['epochs']-1,True)
    served=test_frame[test_frame.node.map(lambda i:available[i])]
    result['served_window_fraction']=len(served)/len(test_frame)
    result['served_metrics']=metrics(served.y,served.prob) if len(served) else None
    save_json(out/'results.json',result)
    from .report import job_plots, verify_job
    job_plots(out)
    checks=verify_job(out)
    save_json(out/'checks.json',checks)
    if not checks['passed']:raise AssertionError(f'Output consistency failed: {out}')
    save_json(out/'DONE.json',dict(complete=True,checks_passed=True))
    # Matplotlib figures contain reference cycles; bound memory across long suites.
    import gc
    gc.collect()
