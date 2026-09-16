import copy

DEFAULT=dict(dataset='FD001',method='gossip',seed=11,nodes=10,epochs=50,hidden=[100,50],length=50,stride=1,horizon=30,
             use_cycle=True,partition='iid',topology='ring',scenario='none',period=1,lr=.001,batch=200,threads=4,
             event_threshold=.02,max_silence=5,bandwidth_mbps=10.,latency_ms=10.,deadline_ms=100.,
             eval_every=5,bootstrap=500,smoke=False)
CORE=['local','centralized','fedavg','gossip']
FACTORS=['fedavg','fedavg_weighted','gossip','gossip_weighted','event_gossip','event_gossip_weighted']

def suite(name,seeds=None):
    jobs=[]
    def add(method,**kwargs):
        c=copy.deepcopy(DEFAULT);c.update(method=method,**kwargs)
        if c not in jobs:jobs.append(c)
    if name=='smoke':
        for method in CORE+['fedavg_weighted','gossip_weighted','event_gossip','event_gossip_weighted','mass_gossip','event_mass_gossip']:
            add(method,nodes=3,epochs=2,hidden=[8,4],stride=10,eval_every=1,bootstrap=30,smoke=True,threads=2)
        for scenario in ['packet_loss','delay','intermittent','node_failure','server_outage']:
            for method in ['fedavg','event_gossip']:
                add(method,nodes=3,epochs=2,hidden=[8,4],stride=10,eval_every=1,bootstrap=30,smoke=True,threads=2,scenario=scenario)
    elif name=='pilot':
        for method in CORE:
            add(method,epochs=5,eval_every=1,bootstrap=100)
    else:
        seeds=seeds or [11,22,33,44,55]
        for seed in seeds:
            if name in ['core','all']:
                for method in CORE+['fedavg_weighted','gossip_weighted','event_gossip','event_gossip_weighted']:
                    add(method,seed=seed)
            if name in ['novelty','all']:
                for part in ['iid','lifetime']:
                    for method in FACTORS+['mass_gossip','event_mass_gossip']:add(method,seed=seed,partition=part)
                # Periodic communication is the essential low-frequency comparator.
                for period in [2,5]:
                    for method in ['gossip','gossip_weighted']:add(method,seed=seed,period=period)
                for topology in ['ring','chord2']:
                    add('gossip',seed=seed,partition='lifetime',topology=topology)
                    add('event_gossip_weighted',seed=seed,partition='lifetime',topology=topology)
                for method in ['centralized','fedavg','gossip']:add(method,seed=seed,use_cycle=False)
                # Threshold sensitivity fixed in advance; do not choose using test scores.
                for threshold in [.01,.05]:add('event_gossip',seed=seed,event_threshold=threshold)
            if name in ['robustness','all']:
                for scenario in ['none','packet_loss','delay','intermittent','node_failure','server_outage']:
                    for method in ['fedavg','gossip','event_gossip_weighted']:add(method,seed=seed,scenario=scenario)
            if name in ['scalability','all']:
                for nodes in [5,10,20,40]:
                    for method in ['fedavg','gossip','event_gossip']:add(method,seed=seed,nodes=nodes)
    return jobs
