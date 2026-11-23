"""Exercise augmentation on real fitting records, without modifying the dataset."""
from pathlib import Path
from collections import Counter
import random
import numpy as np
from PIL import Image,ImageOps,ImageDraw,ImageFont
from .common import CLASSES,config,runtime,role_rows,dump,file_hash,write_csv
from .augmentation import make_augmentation

def bulk_check(root,out,c,views_per_image=4,batch_size=32):
    """Stream all fitting files through augmentation without duplicating storage."""
    if views_per_image<1 or batch_size<1:raise ValueError('Views and batch size must be positive')
    runtime(c);root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rows=role_rows(root,'fit');random.Random(c['seed']).shuffle(rows)
    before=file_hash(root/'metadata/dataset_manifest.csv')
    layer=make_augmentation(c);audit=[];changed=0;generated=0
    for start in range(0,len(rows),batch_size):
        batch=rows[start:start+batch_size];images=[]
        for row in batch:
            path=root/row['relative_path']
            if file_hash(path)!=row['sha256']:raise ValueError('Source image changed: '+row['relative_path'])
            with Image.open(path) as im:
                images.append(np.asarray(ImageOps.pad(im.convert('RGB'),(c['image_size'],c['image_size']),method=Image.Resampling.BILINEAR),dtype='float32'))
        x=np.array(images);means=[];changes=[]
        for draw in range(views_per_image):
            y=layer(x,training=True).numpy()
            if not np.isfinite(y).all() or y.min()<0 or y.max()>255:raise ValueError('Invalid augmentation values')
            means.append(y.mean(axis=(1,2,3)));changes.append(np.abs(y-x).mean(axis=(1,2,3)))
            changed+=int(np.count_nonzero(changes[-1]>0));generated+=len(batch)
        means=np.array(means);changes=np.array(changes)
        state=layer.generator.state.numpy().copy()
        np.testing.assert_array_equal(layer(x,training=False).numpy(),x)
        np.testing.assert_array_equal(layer.generator.state.numpy(),state)
        for i,row in enumerate(batch):
            audit.append({'record_id':row['record_id'],'relative_path':row['relative_path'],
                'class_name':row['class_name'],'source_sha256':row['sha256'],'views_generated':views_per_image,
                'original_mean':float(x[i].mean()),'darker_view_mean':float(means[:,i].min()),
                'brighter_view_mean':float(means[:,i].max()),'mean_absolute_change':float(changes[:,i].mean())})
        if start%1024==0 or start+batch_size>=len(rows):
            print('Augmented fitting files',min(start+batch_size,len(rows)),'/',len(rows),'views',generated,flush=True)
    if file_hash(root/'metadata/dataset_manifest.csv')!=before:raise ValueError('Manifest unexpectedly changed')
    write_csv(out/'all_fitting_images_augmentation.csv',audit)
    epochs=c['head_epochs']+c['fine_epochs']+c['full_epochs']
    report={'status':'PASS','fitting_image_files_processed':len(rows),'views_per_file':views_per_image,
        'augmented_views_generated':generated,'views_with_changed_pixels':changed,
        'per_class_original_files':dict(Counter(r['class_name'] for r in rows)),
        'source_file_checksums':'PASS','validation_identity_and_rng_checks':'PASS','dataset_manifest_unchanged':True,
        'stored_augmented_training_copies':0,'training_uses':'Fresh random fitting views on the fly; the streamed audit arrays are discarded after checking.',
        'configured_training_epoch_ceiling':epochs,'maximum_fitting_views_at_epoch_ceiling':len(rows)*epochs,
        'scope':'This command generated/checks image variants. It does not train model weights or create new independent ultrasound cases. Actual training may stop early.'}
    dump(out/'bulk_augmentation_report.json',report);return report

def preview(root,out,c,samples=500):
    tf=runtime(c);root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rows=role_rows(root,'fit');random.Random(c['seed']).shuffle(rows)
    selected=[];seen=set();quota=max(1,samples//4)
    for name in CLASSES:
        count=0
        for row in rows:
            if row['class_name']!=name or row['pixel_sha256'] in seen:continue
            selected.append(row);seen.add(row['pixel_sha256']);count+=1
            if count>=quota:break
    for row in rows:
        if len(selected)>=samples:break
        if row['pixel_sha256'] not in seen:selected.append(row);seen.add(row['pixel_sha256'])
    selected=selected[:samples]
    def image(row):
        with Image.open(root/row['relative_path']) as im:
            return np.asarray(ImageOps.pad(im.convert('RGB'),(c['image_size'],c['image_size']),method=Image.Resampling.BILINEAR),dtype='float32')
    # No flips in comparison pictures so the appearance difference is visible.
    layer=make_augmentation(c,flips=False);results=[]
    for start in range(0,len(selected),16):
        batch=selected[start:start+16];x=np.array([image(row) for row in batch])
        y=layer(x,training=True).numpy();identity=layer(x,training=False).numpy()
        np.testing.assert_array_equal(identity,x)
        for i,row in enumerate(batch):
            assert file_hash(root/row['relative_path'])==row['sha256']
            results.append({'relative_path':row['relative_path'],'class_name':row['class_name'],
                'original_mean':float(x[i].mean()),'augmented_mean':float(y[i].mean()),
                'mean_absolute_change':float(np.abs(y[i]-x[i]).mean()),
                'new_saturated_fraction':float(np.mean((y[i]>=254.99)&(x[i]<254.99)))})
    size=224;header=42;row_height=size+34
    canvas=Image.new('RGB',(4*size,header+4*row_height),'white');draw=ImageDraw.Draw(canvas)
    try:font=ImageFont.load_default(size=18)
    except TypeError:font=ImageFont.load_default()
    for col,title in enumerate(['Original','Darker draw','Brighter draw','Another draw']):draw.text((col*size+8,10),title,fill='black',font=font)
    for row_index,name in enumerate(CLASSES):
        source=next((r for r in selected if r['class_name']==name),None)
        if source is None:continue
        x=image(source);draws=layer(np.repeat(x[None],24,axis=0),training=True).numpy()
        means=draws.mean(axis=(1,2,3));views=[x,draws[means.argmin()],draws[means.argmax()],draws[len(draws)//2]]
        for col,view in enumerate(views):
            im=Image.fromarray(np.clip(view,0,255).astype('uint8')).resize((size,size))
            im.save(out/f'{name}_{col}.png')
            canvas.paste(im,(col*size,header+row_index*row_height+28))
        draw.text((8,header+row_index*row_height+4),name+' (supplied label)',fill='black',font=font)
    canvas.save(out/'augmentation_preview.png')
    write_csv(out/'augmentation_audit.csv',results)
    report={'real_fitting_images_processed':len(results),'per_class':dict(Counter(r['class_name'] for r in results)),
        'validation_identity_check':'PASS','original_file_checksums':'PASS',
        'brightness_gain_range':[1-c['augment_gain'],1+c['augment_gain']],
        'contrast_factor_range':[1-c['augment_contrast'],1+c['augment_contrast']],
        'gamma_range':[1-c['augment_gamma'],1+c['augment_gamma']],
        'maximum_relative_speckle_std':c['augment_speckle'],
        'strategy':'On-the-fly fitting augmentation. Preview PNGs are demonstrations, not extra training cases.',
        'scope':'Engineering perturbations, not a validated simulation of scanner physics. Bright/dark preview columns show the extremes of 24 random draws for illustration.'}
    dump(out/'augmentation_report.json',report);return report
