import random
import numpy as np
import torch
from torch import nn

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class LSTM(nn.Module):
    def __init__(self, hidden=(100,50)):
        super().__init__()
        self.l1 = nn.LSTM(25, hidden[0], batch_first=True)
        self.l2 = nn.LSTM(hidden[0], hidden[1], batch_first=True)
        self.drop = nn.Dropout(0.2)
        self.out = nn.Linear(hidden[1], 1)
        # One trainable bias per gate (Keras-style LSTM parameterization).
        # PyTorch's second bias stays zero, fixed and is not transmitted.
        for layer in [self.l1, self.l2]:
            for name, p in layer.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(p)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(p.T)
                else:
                    nn.init.zeros_(p)
                    if 'bias_hh' in name:
                        p.requires_grad_(False)
                    else:
                        h = p.numel()//4
                        with torch.no_grad(): p[h:2*h].fill_(1)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        x, _ = self.l1(x)
        x, _ = self.l2(self.drop(x))
        return self.out(self.drop(x[:, -1])).squeeze(-1)

def params(model):
    return [p for p in model.parameters() if p.requires_grad]

def vector(model):
    return torch.cat([p.detach().reshape(-1) for p in params(model)]).clone()

def put_vector(model, v):
    offset = 0
    with torch.no_grad():
        for p in params(model):
            p.copy_(v[offset:offset+p.numel()].reshape_as(p))
            offset += p.numel()
    assert offset == v.numel()

def train_epoch(model, x, y, cfg, seed, positive_weight=1.):
    seed_all(seed)
    model.train()
    # Explicit round-local Adam policy, identical across methods, no stale moments after mixing.
    opt = torch.optim.Adam(params(model), lr=cfg['lr'])
    order = torch.randperm(len(x))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(positive_weight)))
    loss_sum = 0.
    for ix in order.split(cfg['batch']):
        opt.zero_grad(set_to_none=True)
        logits = model(x[ix])
        loss = loss_fn(logits, y[ix])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training loss')
        loss.backward()
        nn.utils.clip_grad_norm_(params(model), 5.)
        opt.step()
        loss_sum += float(loss.detach())*len(ix)
    return loss_sum/max(1,len(x))

def predict(model, x, batch=512):
    model.eval()
    with torch.inference_mode():
        return torch.cat([model(b).sigmoid() for b in x.split(batch)]).numpy() if len(x) else np.empty(0)
