"""Training-only aggregation of supplied targets for byte-identical images.

Soft CE preserves every weighted record contribution. Candidate marginal loss
retains candidate membership and group mass, but discards within-set proportions.
It is an experimental assumption, not a correction of labels or a diagnosis.
"""
import numpy as np

def aggregate_targets(keys, labels, weights, role='fit', classes=4):
    if role != 'fit':
        raise ValueError('Candidate targets may be constructed from FIT records only')
    keys=np.asarray(keys); labels=np.asarray(labels); weights=np.asarray(weights,dtype=np.float64)
    if keys.ndim != 1 or labels.shape != keys.shape or weights.shape != keys.shape or not len(keys):
        raise ValueError('Expected nonempty, aligned one-dimensional record arrays')
    if not np.issubdtype(labels.dtype,np.integer) or np.any(labels<0) or np.any(labels>=classes):
        raise ValueError('Invalid supplied class index')
    if not np.all(np.isfinite(weights)) or np.any(weights<=0):
        raise ValueError('Expected finite positive record weights')
    unique,inverse=np.unique(keys,return_inverse=True)
    mass=np.zeros((len(unique),classes),np.float64)
    np.add.at(mass,(inverse,labels),weights)
    total=mass.sum(axis=1)
    return {'keys':unique,'record_to_group':inverse,'soft_targets':mass/total[:,None],
            'candidate_mask':mass>0,'group_weights':total*len(unique)/len(keys),
            'records_represented':len(keys)}

def objective_gradient(logits,targets,weights,kind):
    """Return mean group-weighted loss and exact gradient with respect to logits."""
    from scipy.special import logsumexp
    z=np.asarray(logits,dtype=np.float64); w=np.asarray(weights,dtype=np.float64)
    if z.ndim!=2 or np.shape(targets)!=z.shape or w.shape!=(len(z),):
        raise ValueError('Misaligned objective inputs')
    logp=z-logsumexp(z,axis=1,keepdims=True); p=np.exp(logp)
    if kind=='soft_ce':
        q=np.asarray(targets,dtype=np.float64)
        if np.any(q<0) or not np.allclose(q.sum(axis=1),1):raise ValueError('Invalid soft targets')
        losses=-np.sum(q*logp,axis=1); grad=p-q
    elif kind=='candidate_marginal':
        allowed=np.asarray(targets,dtype=bool)
        if not np.all(allowed.any(axis=1)):raise ValueError('Every group needs a candidate')
        inside=np.where(allowed,z,-np.inf)
        candidate_logsum=logsumexp(inside,axis=1,keepdims=True)
        losses=logsumexp(z,axis=1)-candidate_logsum[:,0]
        grad=p-np.exp(inside-candidate_logsum)
    else:raise ValueError('Unknown training objective')
    return float(np.mean(w*losses)),grad*w[:,None]/len(z)
