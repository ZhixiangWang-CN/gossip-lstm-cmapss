"""Synchronous message-level simulator. No wall-clock/network-speedup claim."""
import numpy as np
import torch

def edges_for(n,topology):
    edges=set()
    offsets = [1] if topology=='ring' else [1,2]
    if topology=='complete':
        return [(i,j) for i in range(n) for j in range(i+1,n)]
    for i in range(n):
        for d in offsets:
            j=(i+d)%n
            if j!=i:edges.add(tuple(sorted((i,j))))
    return sorted(edges)

def mixing_matrix(n,edges,masses=None):
    deg=np.zeros(n,int)
    for i,j in edges:deg[i]+=1;deg[j]+=1
    w=np.eye(n)
    for i,j in edges:
        if masses is None:
            a=b=1/(1+max(deg[i],deg[j]))
        else:
            a=min(1/(1+deg[i]),masses[j]/(masses[i]*(1+deg[j])))
            b=min(1/(1+deg[j]),masses[i]/(masses[j]*(1+deg[i])))
        w[i,j]=a;w[j,i]=b;w[i,i]-=a;w[j,j]-=b
    return w

def active_nodes(cfg,epoch,communication=False):
    active=np.ones(cfg['nodes'],bool)
    victim=min(7,cfg['nodes']-1)
    if cfg['scenario']=='node_failure' and epoch>=cfg['epochs']//2:
        active[victim]=False
    if cfg['scenario']=='intermittent' and communication:
        rng=np.random.default_rng(cfg['seed']*100003+epoch*500+91)
        active=rng.random(cfg['nodes'])>=.3
    return active

def communicate(vectors, server, train_sizes, cfg, epoch, state):
    n=len(vectors); payload=vectors[0].numel()*vectors[0].element_size()
    method=cfg['method'];records=[]
    active=active_nodes(cfg,epoch,communication=True)
    isfed=method.startswith('fedavg')
    no_comm=method in ['local','centralized'] or (epoch+1)%cfg['period']!=0
    if no_comm:
        return vectors,server,state,records,dict(applied_edges=0,active_nodes=int(active.sum()),slem=1.)
    if isfed and cfg['scenario']=='server_outage' and cfg['epochs']*.4<=epoch<cfg['epochs']*.7:
        return vectors,server,state,records,dict(applied_edges=0,active_nodes=int(active.sum()),slem=1.)

    def message(i,j,kind='model'):
        # Stable per-round sender/receiver stream, shared between matched jobs.
        rng=np.random.default_rng(cfg['seed']*1000003+epoch*10007+(i+2)*211+(j+2)*37)
        latency=float(rng.uniform(.02,.3)) if cfg['scenario']=='delay' else cfg['latency_ms']/1000
        lost=cfg['scenario']=='packet_loss' and rng.random()<.2
        delivered=(not lost) and latency<=cfg['deadline_ms']/1000
        records.append(dict(epoch=epoch+1,sender=i,receiver=j,kind=kind,
                            bytes=payload if kind=='model' else 16,delivered=bool(delivered),
                            latency_s=latency,applied=False))
        return delivered

    if isfed:
        good=[]
        for i in range(n):
            if active[i] and message(i,-1):good.append(i)
        if good:
            weight=np.array([train_sizes[i] for i in good],float);weight/=weight.sum()
            server=sum(float(a)*vectors[i] for a,i in zip(weight,good))
            for r in records:
                if r['receiver']==-1 and r['delivered']:r['applied']=True
            for i in range(n):
                if active[i] and message(-1,i):
                    vectors[i]=server.clone();records[-1]['applied']=True
        return vectors,server,state,records,dict(applied_edges=len(good),active_nodes=int(active.sum()),slem=None)

    event=method.startswith('event')
    if state is None:
        state=dict(last=[v.clone() for v in vectors],last_round=[-cfg['max_silence']]*n)
    triggered=[float(torch.linalg.vector_norm(v-state['last'][i])/(torch.linalg.vector_norm(state['last'][i])+1e-12))>=cfg['event_threshold']
               or epoch-state['last_round'][i]>=cfg['max_silence'] for i,v in enumerate(vectors)]
    successful=[]
    for i,j in edges_for(n,cfg['topology']):
        if not (active[i] and active[j]):continue
        if event:
            ok1=message(i,j,'trigger');ok2=message(j,i,'trigger')
            if not(ok1 and ok2) or not(triggered[i] or triggered[j]):continue
        ok1=message(i,j);idx1=len(records)-1
        ok2=message(j,i);idx2=len(records)-1
        if ok1 and ok2:
            successful.append((i,j));records[idx1]['applied']=records[idx2]['applied']=True
    w=mixing_matrix(n,successful,train_sizes if 'mass' in method else None)
    new=list(torch.as_tensor(w,dtype=vectors[0].dtype)@torch.stack(vectors))
    # Atomic pair exchange is a simulation assumption; coordination/ACK protocol not implemented.
    for i in set(k for e in successful for k in e):
        state['last'][i]=vectors[i].clone();state['last_round'][i]=epoch
    eig=np.sort(np.abs(np.linalg.eigvals(w)))
    return new,server,state,records,dict(applied_edges=len(successful),active_nodes=int(active.sum()),slem=float(eig[-2]) if n>1 else 0.)

def modeled_network_time(records,cfg,isfed):
    if not records:return 0.
    # Per-node shared full-duplex link budget; sum upload/download phases for star.
    phases=[records] if not isfed else [[r for r in records if r['receiver']==-1],[r for r in records if r['sender']==-1]]
    total=0.
    for phase in phases:
        if not phase:continue
        tx={};rx={}
        for r in phase:
            tx[r['sender']]=tx.get(r['sender'],0)+r['bytes']
            rx[r['receiver']]=rx.get(r['receiver'],0)+r['bytes']
        total+=max(list(tx.values())+list(rx.values()))*8/(cfg['bandwidth_mbps']*1e6)
        total+=max(min(r['latency_s'],cfg['deadline_ms']/1000) for r in phase)
    return total
