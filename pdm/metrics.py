import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

def from_counts(tn, fp, fn, tp):
    def div(a,b): return float(a/b) if b else 0.
    return dict(tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),
                accuracy=div(tp+tn,tp+tn+fp+fn),precision=div(tp,tp+fp), recall=div(tp,tp+fn),
                f1=div(2*tp,2*tp+fp+fn), fpr=div(fp,fp+tn),
                balanced_accuracy=(div(tp,tp+fn)+div(tn,tn+fp))/2)

def metrics(y,p,threshold=.5):
    y=np.asarray(y,int);p=np.asarray(p,float);hat=p>=threshold
    result=from_counts(np.sum((y==0)&~hat),np.sum((y==0)&hat),np.sum((y==1)&~hat),np.sum((y==1)&hat))
    result.update(n=len(y),positive=int(y.sum()),threshold=float(threshold),
                  roc_auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
                  average_precision=float(average_precision_score(y,p)) if y.sum() else None,
                  brier=float(np.mean((p-y)**2)),
                  log_loss=float(np.mean(-y*np.log(np.clip(p,1e-7,1-1e-7))-(1-y)*np.log(np.clip(1-p,1e-7,1)))))
    return result

def select_threshold(y,p,mode):
    # Predeclared finite grid; validation only. Equal F1 -> larger threshold.
    candidates = np.linspace(.01,.99,99)
    scores=[metrics(y,p,t) for t in candidates]
    if mode=='val_f1':
        return max(scores,key=lambda d:(d['f1'],d['threshold']))['threshold']
    valid=[d for d in scores if d['fpr']<=.02]
    return max(valid,key=lambda d:(d['recall'],d['threshold']))['threshold'] if valid else 1.000001

def cluster_ci(frame, threshold, reps=500, seed=0):
    counts=[]
    for _,g in frame.groupby('engine'):
        m=metrics(g.y,g.prob,threshold)
        counts.append([m[k] for k in ['tn','fp','fn','tp']])
    a=np.array(counts)
    rng=np.random.default_rng(seed)
    vals=[];degenerate=0
    for _ in range(reps):
        c=a[rng.integers(0,len(a),len(a))].sum(0)
        if c[2]+c[3]==0:
            degenerate+=1
            continue
        vals.append(from_counts(*c))
    result={'resampling_unit':'test engine','replicates':reps,'valid_replicates':len(vals),'no_positive_replicates':degenerate}
    for k in ['accuracy','precision','recall','f1','fpr']:
        result[k]=list(map(float,np.percentile([v[k] for v in vals],[2.5,97.5]))) if vals else [None,None]
    return result

def alarm_metrics(frame, threshold):
    rows=[]
    for engine,g in frame.groupby('engine'):
        g=g.sort_values('cycle'); alarm=g.prob>=threshold;healthy=g.y==0;critical=g.y==1
        hit=g[alarm&critical]
        rows.append(dict(engine=int(engine),observable_positive=bool(critical.any()),
                         detected=bool(len(hit)),first_true_alarm_rul=float(hit.iloc[0].rul) if len(hit) else None,
                         any_early_false_alarm=bool((alarm&healthy).any()),
                         false_alarm_windows=int((alarm&healthy).sum()),observed_windows=len(g)))
    return pd.DataFrame(rows)

def membership_diagnostic(member,nonmember):
    # Whole-engine mean loss; not a formal privacy guarantee or calibrated attack.
    rows=[]
    for label, frame in [(1,member),(0,nonmember)]:
        for engine,g in frame.groupby('engine'):
            # Match observation budget by using last 50 available windows per engine.
            g=g.sort_values('cycle').tail(50)
            rows.append(dict(member=label,engine=int(engine),score=-metrics(g.y,g.prob)['log_loss']))
    a=pd.DataFrame(rows)
    auc=float(roc_auc_score(a.member,a.score))
    fpr,tpr,_=roc_curve(a.member,a.score)
    return a,dict(engine_loss_attack_auc=auc,tpr_at_fpr_10pct=float(max(tpr[fpr<=.1],default=0)),
                  member_engines=int(a.member.sum()),nonmember_engines=int((a.member==0).sum()),
                  limitation='Exploratory final-output loss attack; unequal trajectory/class distributions remain; no DP or privacy guarantee.')

def signflip_p(d):
    d=np.asarray(d,float)
    if len(d)==0:return None
    rng=np.random.default_rng(443)
    signs=np.array(list(itertools.product([-1,1],repeat=len(d)))) if len(d)<=16 else rng.choice([-1,1],(10000,len(d)))
    return float(np.mean(np.abs((signs*d).mean(1))>=abs(d.mean())-1e-12))
