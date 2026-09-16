import base64
import html
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t as student_t
from sklearn.metrics import roc_curve, precision_recall_curve
from .metrics import metrics, signflip_p

def savefig(fig,path):
    fig.tight_layout()
    fig.savefig(str(path)+'.png',dpi=600,bbox_inches='tight')
    fig.savefig(str(path)+'.svg',bbox_inches='tight')
    plt.close(fig)

def job_plots(out):
    out=Path(out);f=out/'figures';f.mkdir(exist_ok=True)
    h=pd.read_csv(out/'history.csv');a=pd.read_csv(out/'predictions_test.csv.gz')
    fig,ax=plt.subplots(2,2,figsize=(9,6))
    ax[0,0].plot(h.epoch,h.train_loss,label='Train objective');ax[0,0].plot(h.epoch,h.val_loss,label='Validation BCE');ax[0,0].legend(fontsize=8)
    ax[0,1].plot(h.epoch,h.val_f1,label='Validation F1');ax[0,1].plot(h.epoch,h.val_accuracy,label='Validation accuracy');ax[0,1].legend(fontsize=8)
    ax[1,0].plot(h.epoch,h.parameter_relative,label='Relative parameter divergence');ax[1,0].plot(h.epoch,h.probe_disagreement,label='Prediction disagreement');ax[1,0].legend(fontsize=8)
    ax[1,1].plot(h.epoch,h.bytes_attempted.cumsum()/2**20,label='Attempted payload + control');ax[1,1].set_ylabel('MiB');ax[1,1].legend(fontsize=8)
    for x in ax.flat:x.set_xlabel('Epoch');x.grid(alpha=.2)
    savefig(fig,f/'learning_consensus_communication')
    nh=pd.read_csv(out/'history_node.csv')
    fig,axes=plt.subplots(2,2,figsize=(9,6))
    for row,split in enumerate(['train','validation']):
        for node,g in nh[nh.split==split].groupby('node'):
            axes[row,0].plot(g.epoch,g.accuracy,label=str(node+1),alpha=.8)
            axes[row,1].plot(g.epoch,g.log_loss,label=str(node+1),alpha=.8)
        axes[row,0].set(ylabel=split+' accuracy',xlabel='Epoch')
        axes[row,1].set(ylabel=split+' BCE',xlabel='Epoch')
    if nh.node.nunique()<=10:axes[0,0].legend(title='Node',fontsize=6,ncol=2)
    for x in axes.flat:x.grid(alpha=.2)
    savefig(fig,f/'per_node_learning')
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    for label,b in [('All windows',a),('Terminal windows',a[a.terminal])]:
        if b.y.nunique()==2:
            fp,tp,_=roc_curve(b.y,b.prob);pr,re,_=precision_recall_curve(b.y,b.prob)
            axes[0].plot(fp,tp,label=label);axes[1].plot(re,pr,label=label)
    axes[0].set(xlabel='False positive rate',ylabel='Recall');axes[1].set(xlabel='Recall',ylabel='Precision')
    for x in axes:
        if x.get_legend_handles_labels()[0]:x.legend()
        x.grid(alpha=.2)
    savefig(fig,f/'roc_pr')
    nodes=pd.read_csv(out/'node_metrics.csv');nodes=nodes[(nodes.split=='test_all')&(nodes.policy=='fixed')]
    counts=nodes[['tn','fp','fn','tp']].to_numpy()
    fig,ax=plt.subplots(figsize=(6,max(3,len(nodes)*.23)))
    ax.imshow(np.log1p(counts),cmap='Blues',aspect='auto')
    for i in range(len(counts)):
        for j in range(4):ax.text(j,i,str(counts[i,j]),ha='center',va='center',fontsize=8,color='black')
    ax.set_xticks(range(4),['TN','FP','FN','TP']);ax.set_yticks(range(len(nodes)),[f'Node {i+1}' for i in nodes.node]);ax.set_title('Local inference: confusion counts (color = log1p)')
    savefig(fig,f/'node_confusion')
    dist=pd.read_csv(out/'pairwise_parameter_rms.csv').to_numpy()
    fig,ax=plt.subplots(figsize=(5,4));im=ax.imshow(dist,cmap='viridis');fig.colorbar(im,ax=ax,label='Parameter RMS distance');ax.set(xlabel='Node (zero-based)',ylabel='Node (zero-based)',title='Final common-coordinate parameter distances')
    savefig(fig,f/'consensus_distances')

def verify_job(out):
    out=Path(out);r=json.loads((out/'results.json').read_text());a=json.loads((out/'data_audit.json').read_text())
    pred=pd.read_csv(out/'predictions_test.csv.gz');ledger=pd.read_csv(out/'messages.csv.gz');h=pd.read_csv(out/'history.csv')
    checks={}
    checks['engine_disjoint_train_validation']=not bool(set(a['train_engines'])&set(a['validation_engines']))
    checks['probabilities_finite_and_bounded']=bool(np.isfinite(pred.prob).all() and pred.prob.between(0,1).all())
    checks['unique_engine_cycle']=not pred.duplicated(['engine','cycle']).any()
    checks['label_rule']=bool((pred.y==(pred.rul<=r['config']['horizon']).astype(int)).all())
    checks['ledger_total_bytes']=int(ledger.bytes.sum())==r['bytes_total']==int(h.bytes_attempted.sum())
    checks['ledger_message_count']=len(ledger)==r['messages_total']==int(h.messages.sum())
    for split,b in [('test_all',pred),('test_terminal',pred[pred.terminal])]:
        for policy,threshold in r['thresholds'].items():
            m=metrics(b.y,b.prob,threshold);stored=r['metrics'][split+'/'+policy]
            checks[f'{split}_{policy}_metrics']=all(abs(m[k]-stored[k])<1e-9 for k in ['accuracy','precision','recall','f1','tn','fp','fn','tp'])
    checks['epochs_complete']=len(h)==r['config']['epochs']
    return {'passed':bool(all(checks.values())),'checks':{k:bool(v) for k,v in checks.items()}}

def summary(root):
    root=Path(root);results=[]
    for path in sorted(root.glob('*/results.json')):
        if not (path.parent/'DONE.json').exists():continue
        r=json.loads(path.read_text());c=r['config']
        protocol_id=hashlib.sha256(json.dumps({k:v for k,v in c.items() if k not in ['method','seed']},sort_keys=True).encode()).hexdigest()[:12]
        for key,m in r['metrics'].items():
            if not key.startswith('test'):continue
            split,policy=key.split('/')
            results.append(dict(job=path.parent.name,method=c['method'],seed=c['seed'],nodes=c['nodes'],partition=c['partition'],
                 scenario=c['scenario'],topology=c['topology'],period=c['period'],epochs=c['epochs'],dataset=c['dataset'],
                 use_cycle=c['use_cycle'],event_threshold=c['event_threshold'],hidden=str(c['hidden']),stride=c['stride'],
                 protocol_id=protocol_id,split=split,policy=policy,accuracy=m['accuracy'],precision=m['precision'],recall=m['recall'],f1=m['f1'],
                 average_precision=m['average_precision'],roc_auc=m['roc_auc'],fpr=m['fpr'],
                 bytes_total=r['bytes_total'],training_wall_s=r['training_wall_s'],modeled_parallel_s=r['modeled_parallel_s']))
    if not results:return
    df=pd.DataFrame(results);df.to_csv(root/'all_results.csv',index=False)
    groups=['dataset','nodes','partition','scenario','topology','period','epochs','use_cycle','event_threshold','hidden','stride','protocol_id','split','policy']
    stats=[]
    for keys,g in df.groupby(groups+['method'],dropna=False):
        row=dict(zip(groups+['method'],keys));row['runs']=len(g)
        for metric in ['accuracy','precision','recall','f1','average_precision','bytes_total','training_wall_s','modeled_parallel_s']:
            v=g[metric].dropna().to_numpy();mean=float(v.mean()) if len(v) else np.nan
            sd=float(v.std(ddof=1)) if len(v)>1 else np.nan
            half=float(student_t.ppf(.975,len(v)-1)*sd/np.sqrt(len(v))) if len(v)>1 else np.nan
            row.update({metric+'_mean':mean,metric+'_sd':sd,metric+'_ci_low':mean-half,metric+'_ci_high':mean+half})
        stats.append(row)
    sd=pd.DataFrame(stats);sd.to_csv(root/'seed_summary.csv',index=False)
    # Predeclared primary contrasts. Exclude policy/metric multiplicity by choosing terminal fixed F1.
    tests=[];primary=df[(df.split=='test_terminal')&(df.policy=='fixed')]
    comparisons=[('gossip','local'),('gossip','fedavg'),('gossip_weighted','gossip'),('fedavg_weighted','fedavg'),
                 ('event_gossip','gossip'),('event_gossip_weighted','event_gossip'),('event_gossip_weighted','gossip_weighted'),
                 ('mass_gossip','gossip'),('mass_gossip','fedavg'),('event_mass_gossip','mass_gossip')]
    condition=groups[:-2]
    for keys,g in primary.groupby(condition,dropna=False):
        for treatment,control in comparisons:
            x=g[g.method==treatment].set_index('seed');y=g[g.method==control].set_index('seed');common=x.index.intersection(y.index)
            if not len(common):continue
            d=x.loc[common,'f1'].to_numpy()-y.loc[common,'f1'].to_numpy()
            tests.append(dict(zip(condition,keys),treatment=treatment,control=control,n_pairs=len(d),mean_delta_f1=float(d.mean()),
                              exact_signflip_p=signflip_p(d) if len(d)>=2 else None))
    td=pd.DataFrame(tests)
    if len(td):
        td['holm_p']=np.nan;valid=td.exact_signflip_p.dropna().sort_values();running=0.
        for rank,(ix,p) in enumerate(valid.items()):
            running=max(running,min(1.,p*(len(valid)-rank)));td.loc[ix,'holm_p']=running
    td.to_csv(root/'paired_tests.csv',index=False)
    # Export communication-performance tradeoff without connecting unrelated scenarios.
    base=primary[(primary.scenario=='none')&(primary.nodes==10)&(primary.partition=='iid')&(primary.topology=='ring')&(primary.use_cycle)&(primary.period==1)&(primary.event_threshold==.02)]
    for protocol_id,base_group in base.groupby('protocol_id'):
        fig,ax=plt.subplots(figsize=(7,4))
        for name,g in base_group.groupby('method'):
            ax.scatter(g.bytes_total.mean()/2**20,g.f1.mean(),label=name)
        ax.set(xlabel='Attempted communication (MiB)',ylabel='Terminal-window F1',title='Seed means; inspect conditions in CSV')
        ax.legend(fontsize=7,bbox_to_anchor=(1.01,1));ax.grid(alpha=.2);savefig(fig,root/f'communication_f1_{protocol_id}')
    show=sd[(sd.split=='test_terminal')&(sd.policy=='fixed')]
    cols=['method','protocol_id','nodes','partition','scenario','topology','period','runs','f1_mean','f1_sd','recall_mean','bytes_total_mean']
    text='''<!doctype html><html><meta charset="utf-8"><title>Gossip-LSTM C-MAPSS experiment report</title>
<style>body{font:15px system-ui;max-width:1250px;margin:40px auto;color:#183044}table{border-collapse:collapse;font-size:12px}td,th{padding:7px;border:1px solid #cbd5df}th{background:#e8f1f7}h1,h2{color:#123950}</style>
<h1>Gossip-LSTM C-MAPSS experiments</h1><p>Primary endpoint: terminal-window positive-class F1 at threshold 0.5. All-window results and engine-cluster confidence intervals are secondary.</p>
<p>Wall time is sequential CPU training. Modeled parallel time is an assumption-based estimate, not a measured distributed speedup. A single seed is exploratory. Five pairs cannot yield a two-sided exact sign-flip p below 0.0625; absence of significance is not equivalence.</p>'''
    text+='<h2>Terminal-window results</h2>'+show[cols].to_html(index=False,float_format=lambda x:f'{x:.4f}')
    text+='<h2>Paired primary contrasts</h2>'+td.to_html(index=False,float_format=lambda x:f'{x:.4f}')
    text+='<h2>Scope</h2><p>FD001 is a simulation benchmark. Additional C-MAPSS subsets can be supplied separately. No industrial deployment, differential privacy, asynchronous implementation, or unrestricted scalability claim is supported. Membership loss diagnostics do not prove privacy.</p>'
    text+='<h2>Completed runs</h2><ul>'
    for path in sorted(root.glob('*/DONE.json')):text+='<li>'+html.escape(path.parent.name)+'</li>'
    text+='</ul></html>'
    (root/'REPORT.html').write_text(text,encoding='utf-8')

def dataset_plots(folder,out):
    from .data import load_raw,features
    tr,te,truth,_=load_raw(folder);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(1,2,figsize=(9,4))
    ax[0].hist([np.sum(tr[:,0]==i) for i in np.unique(tr[:,0])],bins=20);ax[0].set(xlabel='Training engine lifetime (cycles)',ylabel='Engines')
    ax[1].hist(truth,bins=20);ax[1].axvline(30,color='red',linestyle='--');ax[1].set(xlabel='Terminal test RUL',ylabel='Engines')
    savefig(fig,out/'dataset_lifetimes')
    corr=pd.DataFrame(features(tr)).corr();corr.to_csv(out/'feature_correlation.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,6));im=ax.imshow(corr,vmin=-1,vmax=1,cmap='coolwarm');fig.colorbar(im,ax=ax);ax.set(title='Training raw feature correlations; constants undefined',xlabel='Feature index',ylabel='Feature index');savefig(fig,out/'feature_correlation')
