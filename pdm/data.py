"""Engine-isolated preprocessing; terminal labels never enter features."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

FEATURES = ['setting1', 'setting2', 'setting3', 'cycle_norm'] + [f's{i}' for i in range(1, 22)]
EXPECTED = {'PM_train.txt': '963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8',
            'PM_test.txt': '3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851',
            'PM_truth.txt': 'a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca'}

def load_raw(folder):
    folder = Path(folder)
    arrays = [np.loadtxt(folder / name) for name in EXPECTED]
    train, test, truth = arrays
    truth = np.atleast_1d(truth).reshape(-1)
    for a in [train, test]:
        if a.ndim != 2 or a.shape[1] != 26 or not np.isfinite(a).all():
            raise ValueError('Expected finite 26-column C-MAPSS data.')
        if not np.all(a[:, :2] == a[:, :2].astype(int)):
            raise ValueError('Engine IDs and cycles must be integers.')
        for eid in np.unique(a[:, 0]):
            cyc = a[a[:, 0] == eid, 1]
            if not np.array_equal(cyc, np.arange(1, len(cyc) + 1)):
                raise ValueError(f'Engine {eid}: cycles must be contiguous and ordered from 1.')
    if not np.array_equal(np.unique(test[:, 0]), np.arange(1, len(truth) + 1)):
        raise ValueError('Terminal RUL rows must match contiguous test engine IDs starting at 1.')
    if not np.isfinite(truth).all() or (truth < 0).any():
        raise ValueError('Invalid terminal RUL.')
    hashes = {name: hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in EXPECTED}
    return train, test, truth, hashes

@dataclass
class Windows:
    x: np.ndarray
    meta: pd.DataFrame

    def subset(self, mask):
        ix = np.flatnonzero(np.asarray(mask))
        return Windows(self.x[ix], self.meta.iloc[ix].reset_index(drop=True))

def features(a):
    return np.column_stack([a[:, 2:5], a[:, 1], a[:, 5:26]])

def windows(raw, ids, low, span, length=50, stride=1, horizon=30, truth=None, use_cycle=True):
    xs, rows = [], []
    for eid in sorted(ids):
        a = raw[raw[:, 0] == eid]
        end_of_life = a[-1, 1] + (0 if truth is None else truth[int(eid)-1])
        f = ((features(a) - low) / span).astype('float32')
        if not use_cycle:
            f[:, 3] = 0  # dimension held fixed for controlled ablation
        # Always include the terminal window even when training stride > 1.
        ends = list(range(length-1, len(a), stride))
        if len(a) >= length and (not ends or ends[-1] != len(a)-1):
            ends.append(len(a)-1)
        for end in ends:
            rul = end_of_life - a[end, 1]
            xs.append(f[end-length+1:end+1])
            rows.append((int(eid), int(a[end, 1]), float(rul), int(rul <= horizon), end == len(a)-1))
    if not xs:
        return Windows(np.empty((0, length, 25), dtype='float32'),
                       pd.DataFrame(columns=['engine','cycle','rul','y','terminal']))
    return Windows(np.stack(xs), pd.DataFrame(rows, columns=['engine','cycle','rul','y','terminal']))

def partition(ids, raw, nodes, seed, mode):
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(ids), dtype=int)
    if mode == 'iid':
        rng.shuffle(ids)
        groups = np.array_split(ids, nodes)
    elif mode == 'lifetime':
        # Train-only lifetime shards; induced heterogeneity, not real site identities.
        ids = np.array(sorted(ids, key=lambda e: np.sum(raw[:, 0] == e)))
        groups = np.array_split(ids, nodes)
        rng.shuffle(groups)
    else:
        raise ValueError(mode)
    return {int(e): i for i, group in enumerate(groups) for e in group}

def prepare(folder, cfg):
    tr, te, truth, hashes = load_raw(folder)
    rng = np.random.default_rng(cfg['seed'])
    ids = np.unique(tr[:, 0]).astype(int)
    rng.shuffle(ids)
    if cfg.get('smoke'):
        ids = ids[:16]
    nv = max(2, int(round(len(ids)*0.2)))
    val_ids, train_ids = ids[:nv], ids[nv:]
    if len(train_ids) < cfg['nodes']:
        raise ValueError('Fewer training engines than nodes. Reduce --nodes.')
    assert not set(val_ids) & set(train_ids)
    ft = features(tr[np.isin(tr[:, 0], train_ids)])
    low, high = ft.min(0), ft.max(0)
    span = np.where(high > low, high-low, 1)
    kw = dict(low=low, span=span, length=cfg['length'], horizon=cfg['horizon'], use_cycle=cfg['use_cycle'])
    train = windows(tr, train_ids, stride=cfg['stride'], **kw)
    val = windows(tr, val_ids, **kw)
    test_ids = np.unique(te[:, 0]).astype(int)
    if cfg.get('smoke'):
        eligible=[e for e in test_ids if np.sum(te[:,0]==e)>=cfg['length']]
        test_ids=np.array([e for e in eligible if truth[e-1]<=cfg['horizon']][:6]+
                          [e for e in eligible if truth[e-1]>cfg['horizon']][:6])
    test = windows(te, test_ids, truth=truth, **kw)
    assignment = partition(train_ids, tr, cfg['nodes'], cfg['seed']+100, cfg['partition'])
    train.meta['node'] = train.meta.engine.map(assignment).astype(int)
    # Test IDs refer to different physical engines than train IDs. Independent assignment.
    for w, offset in [(val,200), (test,300)]:
        unique = w.meta.engine.unique().copy()
        np.random.default_rng(cfg['seed']+offset).shuffle(unique)
        mapping = {int(e): i % cfg['nodes'] for i,e in enumerate(unique)}
        w.meta['node'] = w.meta.engine.map(mapping).astype(int)
    audit = dict(hashes=hashes, provided_dataset_identical=hashes == EXPECTED,
                 train_engines=sorted(map(int,train_ids)), validation_engines=sorted(map(int,val_ids)),
                 test_engines=sorted(map(int,test_ids)), excluded_short_test_engines=sorted(set(map(int,test_ids))-set(test.meta.engine)),
                 train_windows=len(train.meta), validation_windows=len(val.meta), test_windows=len(test.meta),
                 terminal_test_windows=int(test.meta.terminal.sum()), test_positive_windows=int(test.meta.y.sum()),
                 features=FEATURES, scaler_min=low.tolist(), scaler_span=span.tolist(),
                 scaler_fit='training engines only; common feature-wise min/max preprocessing reference',
                 train_assignment=assignment, label_alignment='RUL at final INPUT cycle, no next-cycle shift')
    return train, val, test, audit
