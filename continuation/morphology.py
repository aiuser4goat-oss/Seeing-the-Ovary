"""2D relative morphology. No millimetres or whole-ovary count inferred."""
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy.optimize import linear_sum_assignment

FEATURES = ['ovary_fraction','ovary_aspect','ovary_circularity','visible_count',
    'follicle_area_fraction','mean_relative_diameter','std_relative_diameter',
    'max_relative_diameter','peripheral_fraction','mean_radial_position',
    'nearest_neighbor_mean','nearest_neighbor_std','centroid_dispersion']

def instance_matches(truth,prediction,threshold=.5):
    gt,gi=np.unique(truth,return_inverse=True);pr,pi=np.unique(prediction,return_inverse=True)
    intersections=np.bincount(gi.ravel()*len(pr)+pi.ravel(),minlength=len(gt)*len(pr)).reshape(len(gt),len(pr))
    ga=np.bincount(gi.ravel());pa=np.bincount(pi.ravel())
    iou=intersections/np.maximum(ga[:,None]+pa[None,:]-intersections,1)
    iou=iou[np.ix_(gt>0,pr>0)]
    tp=0
    if iou.size:
        a,b=linear_sum_assignment(-1000*(iou>=threshold)-iou)
        tp=int(np.count_nonzero(iou[a,b]>=threshold))
    return {'instance_tp':tp,'instance_fp':int(np.count_nonzero(pr))-tp,
            'instance_fn':int(np.count_nonzero(gt))-tp}

def instances_from_predictions(ovary_probability, follicle_probability, distance=None,
                               ovary_threshold=.5, follicle_threshold=.5,
                               min_area=5, min_peak_distance=3, method='watershed'):
    if method not in ['watershed','connected_components']:
        raise ValueError('Unknown follicle instance method: '+str(method))
    ovary = ovary_probability > ovary_threshold
    components,n = ndi.label(ovary)
    if n:
        sizes=np.bincount(components.ravel());sizes[0]=0
        ovary=ndi.binary_fill_holes(components==np.argmax(sizes))
    binary = (follicle_probability>follicle_threshold)&ovary
    objects,n=ndi.label(binary)
    if n:
        sizes=np.bincount(objects.ravel());binary&=sizes[objects]>=min_area
    if method=='connected_components':
        # One instance per connected semantic region; touching follicles can merge.
        return ovary,ndi.label(binary)[0].astype('uint16')
    elevation=ndi.distance_transform_edt(binary)
    if distance is not None: elevation=elevation*(.25+np.clip(distance,0,1))
    peaks=peak_local_max(elevation,min_distance=min_peak_distance,labels=binary.astype('uint8'),exclude_border=False)
    markers=np.zeros(binary.shape,'int32')
    for i,(y,x) in enumerate(peaks,1):markers[y,x]=i
    # Every foreground component receives at least one seed.
    objects,n=ndi.label(binary)
    for i in range(1,n+1):
        region=objects==i
        if not np.any(markers[region]):
            index=np.argmax(np.where(region,elevation,-1));markers.flat[index]=int(markers.max())+1
    labels=watershed(-elevation,markers,mask=binary)
    clean=np.zeros_like(labels); next_id=1
    for i in np.unique(labels):
        if i and np.count_nonzero(labels==i)>=min_area:clean[labels==i]=next_id;next_id+=1
    return ovary,clean.astype('uint16')

def measurements(ovary, instances):
    ovary=np.asarray(ovary,bool);instances=np.asarray(instances)
    area=int(ovary.sum())
    if area<9: return np.full(len(FEATURES),np.nan,'float32')
    yy,xx=np.nonzero(ovary); height=yy.max()-yy.min()+1;width=xx.max()-xx.min()+1
    perimeter=np.count_nonzero(ovary ^ ndi.binary_erosion(ovary))
    cy,cx=yy.mean(),xx.mean(); radius=np.sqrt(area/np.pi)
    sizes=[];centers=[]
    for label in np.unique(instances):
        if not label:continue
        y,x=np.nonzero((instances==label)&ovary)
        if len(y):sizes.append(len(y));centers.append([y.mean(),x.mean()])
    sizes=np.array(sizes);centers=np.array(centers).reshape(-1,2)
    diam=2*np.sqrt(sizes/np.pi)/(2*radius)
    edge=ndi.distance_transform_edt(ovary)
    if len(centers):
        ys,xs=np.rint(centers).astype(int).T
        peripheral=np.mean(edge[ys,xs]<.25*max(edge.max(),1))
        radial=np.linalg.norm(centers-[cy,cx],axis=1)/radius
        dispersion=float(np.sqrt(np.var(centers,axis=0).sum())/radius)
    else:peripheral=np.nan;radial=np.array([]);dispersion=np.nan
    if len(centers)>1:
        dist=np.linalg.norm(centers[:,None,:]-centers[None,:,:],axis=-1)/radius
        np.fill_diagonal(dist,np.inf);nearest=dist.min(axis=1)
    else:nearest=np.array([])
    return np.array([area/ovary.size,width/height,4*np.pi*area/max(perimeter**2,1),
        len(sizes),sizes.sum()/area,diam.mean() if len(diam) else np.nan,
        diam.std() if len(diam) else np.nan,diam.max() if len(diam) else np.nan,
        peripheral,radial.mean() if len(radial) else np.nan,
        nearest.mean() if len(nearest) else np.nan,nearest.std() if len(nearest) else np.nan,dispersion],dtype='float32')

def crop_bounds(mask, margin=.1):
    y,x=np.nonzero(mask)
    if not len(y):return (0,mask.shape[0],0,mask.shape[1])
    dy=max(2,int((y.max()-y.min()+1)*margin));dx=max(2,int((x.max()-x.min()+1)*margin))
    return (max(0,y.min()-dy),min(mask.shape[0],y.max()+dy+1),
            max(0,x.min()-dx),min(mask.shape[1],x.max()+dx+1))
