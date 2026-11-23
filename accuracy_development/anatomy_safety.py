"""Experimental crop fallbacks; these heuristics do not verify anatomy."""
import numpy as np
from continuation.morphology import crop_bounds

def crop_decision(mask,margin=.25,min_crop_fraction=.15,min_mask_fraction=.01):
    mask=np.asarray(mask,dtype=bool)
    if mask.ndim!=2 or not mask.size:raise ValueError('Expected nonempty 2D mask')
    bounds=tuple(int(v) for v in crop_bounds(mask,margin=margin));y0,y1,x0,x1=bounds
    area=float(mask.mean());crop_area=(y1-y0)*(x1-x0)/mask.size
    # Judge the original proposal before expansion; adding context must not
    # turn a failed small-region check into a pass.
    gy0,gy1,gx0,gx1=crop_bounds(mask,margin=.1)
    guard_area=(gy1-gy0)*(gx1-gx0)/mask.size
    reasons=[]
    if not mask.any():reasons.append('empty_prediction')
    if area<min_mask_fraction:reasons.append('small_predicted_region')
    if guard_area<min_crop_fraction:reasons.append('small_proposed_crop')
    if reasons:bounds=(0,mask.shape[0],0,mask.shape[1])
    return {'bounds':bounds,'fallback':bool(reasons),'reasons':reasons,'predicted_mask_fraction':area,'proposed_crop_fraction':crop_area,'original_proposal_fraction':guard_area,'scope':'Plausibility heuristic, not verified anatomical correctness'}

def guarded_postprocess(image,prediction,config):
    from continuation.features import postprocess
    crop,vector,ovary,instances=postprocess(image,prediction,config)
    decision=crop_decision(ovary,**config.get('crop_safety',{}))
    if decision['fallback']:
        return image.copy(),np.full_like(vector,np.nan),ovary,instances,decision
    size=config['seg_size'];scale=min(size/image.width,size/image.height)
    width=max(1,round(image.width*scale));height=max(1,round(image.height*scale));left=(size-width)//2;top=(size-height)//2
    y0,y1,x0,x1=decision['bounds'];bounds=(max(0,int((x0-left)/scale)),max(0,int((y0-top)/scale)),min(image.width,int(np.ceil((x1-left)/scale))),min(image.height,int(np.ceil((y1-top)/scale))))
    if bounds[2]<=bounds[0] or bounds[3]<=bounds[1]:
        decision.update(fallback=True,reasons=decision['reasons']+['invalid_original_grid_crop'])
        return image.copy(),np.full_like(vector,np.nan),ovary,instances,decision
    return image.crop(bounds),vector,ovary,instances,decision
