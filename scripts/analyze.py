import json,glob,os,itertools
import numpy as np,pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats
import argparse
_p=argparse.ArgumentParser(description='Aggregate completed runs into the tables and figures of the paper.')
_p.add_argument('--runs',default='output',help='folder with completed run directories')
_p.add_argument('--out',default='analysis',help='folder for tables and figures')
_a=_p.parse_args(); OUT=_a.runs; A=_a.out; os.makedirs(A,exist_ok=True)
rows=[]
for f in glob.glob(f'{OUT}/*/results.json'):
    d=os.path.dirname(f); r=json.load(open(d+'/results.json')); c=r['config']
    h=pd.read_csv(d+'/history.csv'); last=h.iloc[-1]; ev=h.dropna(subset=['probe_disagreement']).iloc[-1]
    base=dict(dir=d,method=c['method'],seed=c['seed'],nodes=c['nodes'],partition=c['partition'],scenario=c['scenario'],use_cycle=c['use_cycle'],
              bytes=r['bytes_total'],messages=r['messages_total'],wall=r['training_wall_s'],modeled=r['modeled_parallel_s'],
              param_rms=last.parameter_rms,param_rel=last.parameter_relative,probe_dis=ev.probe_disagreement,
              mia_auc=r.get('privacy_diagnostic',{}).get('engine_loss_attack_auc'),served_frac=r.get('served_window_fraction'),
              served_f1=r.get('served_metrics',{}).get('f1'))
    for split in ['test_terminal','test_all']:
        m=r['metrics'][f'{split}/fixed']
        for k in ['accuracy','precision','recall','f1','average_precision','roc_auc','fpr','tn','fp','fn','tp','n','positive']:
            base[f'{split}.{k}']=m.get(k)
        ci=m.get('cluster_ci95',{}); base[f'{split}.f1_ci']=ci.get('f1')
    rows.append(base)
D=pd.DataFrame(rows); D.to_pickle(A+'/runs.pkl')
def cond(df,**kw):
    q=df
    defaults=dict(nodes=10,partition='iid',scenario='none',use_cycle=True); defaults.update(kw)
    for k,v in defaults.items(): q=q[q[k]==v]
    return q
def ms(x): x=np.asarray(x,float); return (x.mean(), x.std(ddof=1) if len(x)>1 else np.nan, len(x))
S={}
main=cond(D)
order=['centralized','fedavg','fedavg_weighted','gossip','gossip_weighted','event_gossip','event_gossip_weighted','local']
tab=[]
for m in order:
    q=main[main.method==m]
    if not len(q): continue
    row=dict(method=m,runs=len(q))
    for split in ['test_terminal','test_all']:
        for k in ['accuracy','precision','recall','f1','average_precision']:
            mu,sd,n=ms(q[f'{split}.{k}']); row[f'{split}.{k}']=(mu,sd)
    for k in ['bytes','param_rms','probe_dis','mia_auc','wall','modeled']:
        row[k]=ms(q[k])[:2]
    tab.append(row)
S['main']=tab
# paired
def paired(t,c,split):
    a=main[main.method==t].set_index('seed')[f'{split}.f1']; b=main[main.method==c].set_index('seed')[f'{split}.f1']
    s=a.index.intersection(b.index); d=(a[s]-b[s]).values
    if len(d)==0: return None
    signs=np.array(list(itertools.product([-1,1],repeat=len(d))))
    obs=abs(d.mean()); p=float(np.mean(np.abs((signs*np.abs(d)).mean(1))>=obs-1e-12))
    return dict(n=len(d),mean=float(d.mean()),sd=float(d.std(ddof=1)) if len(d)>1 else None,p=p,wins=int((d>0).sum()))
S['paired']={f'{t}-vs-{c}/{sp}':paired(t,c,sp) for t,c in [('gossip','fedavg'),('gossip','local'),('gossip','centralized'),('fedavg','local'),('event_gossip','gossip')] for sp in ['test_terminal','test_all']}
# extended
ext=[]
for label,kw in [('IID, no faults',{}),('Packet loss 20%',dict(scenario='packet_loss')),('Node failure (epoch 25)',dict(scenario='node_failure')),('Server outage (epochs 20-35)',dict(scenario='server_outage')),
                 ('Lifetime non-IID',dict(partition='lifetime')),('Cycle input removed',dict(use_cycle=False)),('N = 5',dict(nodes=5)),('N = 20',dict(nodes=20)),('N = 40',dict(nodes=40))]:
    q=cond(D,**kw); q=q[q.seed.isin([11,22,33])]
    for m in ['fedavg','gossip','local']:
        z=q[q.method==m]
        if len(z): ext.append(dict(setting=label,method=m,runs=len(z),term_f1=ms(z['test_terminal.f1'])[:2],all_f1=ms(z['test_all.f1'])[:2],
                                   all_ap=ms(z['test_all.average_precision'])[:2],bytes=ms(z['bytes'])[:2],probe=ms(z['probe_dis'])[:2]))
S['ext']=ext
json.dump(S,open(A+'/summary.json','w'),indent=1,default=float)
# ---------- figures (gossip seed 11 as exemplar) ----------
def rundir(m,seed=11,**kw):
    q=cond(D,**kw); q=q[(q.method==m)&(q.seed==seed)]; return q.dir.iloc[0] if len(q) else None
plt.rcParams.update({'font.size':9,'font.family':'DejaVu Sans'})
g=rundir('gossip')
if g and os.path.exists(g+'/history_node.csv'):  # per-node histories are produced by fresh runs
    hn=pd.read_csv(g+'/history_node.csv')
    fig,ax=plt.subplots(1,2,figsize=(7.2,2.8))
    for i,(split,metric,t) in enumerate([('train','log_loss','Training loss (BCE)'),('validation','f1','Validation F1')]):
        for node,z in hn[hn.split==split].groupby('node'):
            ax[i].plot(z.epoch,z[metric],lw=1,label=f'Node {node+1}')
        ax[i].set_xlabel('Epoch (communication round)'); ax[i].set_title(t); ax[i].grid(alpha=.3)
    ax[1].legend(fontsize=6,ncol=2,frameon=False); fig.tight_layout(); fig.savefig(A+'/fig4_learning_curves.png',dpi=300); plt.close(fig)
# consensus
fig,ax=plt.subplots(1,2,figsize=(7.2,2.8))
cols={'local':'#888888','gossip':'#1f77b4','event_gossip':'#2ca02c','fedavg':'#d62728'}
for m,c in cols.items():
    q=main[main.method==m]
    if not len(q): continue
    H=[pd.read_csv(d+'/history.csv') for d in q.dir]
    ep=H[0].epoch
    r=np.array([h.parameter_rms for h in H]); ax[0].plot(ep,r.mean(0),color=c,label=m.replace('_',' ')); 
    if len(H)>1: ax[0].fill_between(ep,r.min(0),r.max(0),color=c,alpha=.15)
    P=[h.dropna(subset=['probe_disagreement']) for h in H]
    p=np.array([h.probe_disagreement for h in P]); ax[1].plot(P[0].epoch,p.mean(0),'o-',ms=3,color=c,label=m.replace('_',' '))
ax[0].set_yscale('symlog',linthresh=1e-4); ax[0].set_title('Parameter RMS distance to network mean'); ax[1].set_title('Pairwise disagreement on common probe')
for a in ax: a.set_xlabel('Epoch (communication round)'); a.grid(alpha=.3)
ax[1].legend(frameon=False,fontsize=7,loc='center right'); fig.tight_layout(); fig.savefig(A+'/fig5_consensus.png',dpi=300); plt.close(fig)
# bytes vs F1
fig,ax=plt.subplots(figsize=(7.2,3.0))
pal=plt.cm.tab10.colors; k=0
for m in order:
    q=main[main.method==m]
    if not len(q): continue
    x=q.bytes.mean()/2**20; y=q['test_all.f1'].mean(); sd=q['test_all.f1'].std(ddof=1)
    off={'fedavg':-18,'fedavg_weighted':-10,'gossip':-2,'gossip_weighted':6,'event_gossip':14,'event_gossip_weighted':22}.get(m,0)
    ax.errorbar(x+off,y,yerr=sd,fmt='o',capsize=3,color=pal[k],label=m.replace('_',' ')); k+=1
ax.set_xlabel('Total model payload over 50 rounds (MiB; points near 307.7 MiB offset horizontally for visibility)'); ax.set_ylabel('All-window F1 (mean ± SD)')
ax.grid(alpha=.3); ax.legend(fontsize=7,frameon=False,loc='center left',bbox_to_anchor=(1.01,.5))
fig.tight_layout(); fig.savefig(A+'/fig6_bytes_f1.png',dpi=300); plt.close(fig)
# per-node confusion gossip seed 11
if g:
    nm=pd.read_csv(g+'/node_metrics.csv'); z=nm[(nm.split=='test_all')&(nm.policy=='fixed')].sort_values('node')
    fig,ax=plt.subplots(figsize=(7.2,2.8)); x=np.arange(len(z)); w=.2
    for k,(col,c) in enumerate([('tn','#9ecae1'),('fp','#fdae6b'),('fn','#e6550d'),('tp','#31a354')]):
        b=ax.bar(x+(k-1.5)*w,z[col],w,color=c,label=col.upper()); ax.bar_label(b,fontsize=5)
    ax.set_yscale('symlog',linthresh=10); ax.set_xticks(x,[f'Node {i+1}' for i in z.node]); ax.legend(ncol=4,frameon=False,fontsize=7,loc='lower center',bbox_to_anchor=(.5,1.0)); ax.set_ylim(0,3000); ax.set_ylabel('Windows (symlog)')
    fig.tight_layout(); fig.savefig(A+'/fig7_node_confusion.png',dpi=300); plt.close(fig)
    z.to_csv(A+'/node_table_gossip_s11.csv',index=False)
    pd.read_csv(g+'/node_data.csv').to_csv(A+'/node_data_gossip_s11.csv',index=False)
# robustness fig
E=pd.DataFrame(S['ext'])
if len(E):
    fig,ax=plt.subplots(figsize=(7.2,2.8)); sets=list(dict.fromkeys(E.setting)); x=np.arange(len(sets))
    for k,(m,c) in enumerate([('fedavg','#d62728'),('gossip','#1f77b4')]):
        z=E[E.method==m].set_index('setting').reindex(sets)
        mu=[v[0] if isinstance(v,(list,tuple)) else np.nan for v in z.all_f1]; sd=[v[1] if isinstance(v,(list,tuple)) else np.nan for v in z.all_f1]
        ax.bar(x+(k-.5)*.38,mu,.38,yerr=sd,capsize=2,color=c,label=m)
    ax.set_xticks(x,sets,rotation=25,ha='right',fontsize=7); ax.set_ylabel('All-window F1'); ax.set_ylim(.5,1); ax.legend(frameon=False); ax.grid(axis='y',alpha=.3)
    fig.tight_layout(); fig.savefig(A+'/fig8_robustness_scaling.png',dpi=300); plt.close(fig)
print(json.dumps(S['main'],default=float)[:1500]); print(S['paired'])
