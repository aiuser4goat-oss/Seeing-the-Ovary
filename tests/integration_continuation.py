"""Small synthetic end-to-end run; produces no clinical performance claim."""
from pathlib import Path
import hashlib,json,sys,tempfile
import numpy as np
from PIL import Image
from continuation.common import CLASSES,config,dump,write_csv,runtime
from continuation.anatomy import train as train_anatomy, evaluate as eval_anatomy
from continuation.features import extract
from continuation.training import train
from continuation.evaluation import calibrate,evaluate

def main():
    with tempfile.TemporaryDirectory(prefix='ovary_integration_') as temporary:
        root=Path(temporary);anatomy=root/'anatomy';anatomy.mkdir();rows=[]
        y,x=np.ogrid[:64,:64];ov=(x-32)**2+(y-32)**2<25**2
        ins=np.zeros((64,64),'uint16');ins[(x-24)**2+(y-30)**2<6**2]=1;ins[(x-40)**2+(y-35)**2<5**2]=2
        from continuation.anatomy import instance_distance
        rng=np.random.default_rng(2627)
        for i in range(12):
            image=np.clip(rng.normal(120,25,(64,64)),0,255).astype('uint8');image[ins>0]=20
            np.savez_compressed(anatomy/f'{i}.npz',image=image,ovary=ov.astype('uint8'),follicle=(ins>0).astype('uint8'),instances=ins,distance=instance_distance(ins))
            rows.append({'file':f'{i}.npz','group':f'case{i}','split':'train' if i<6 else 'val' if i<10 else 'test','instance_valid':1,'source':'synthetic'})
        write_csv(anatomy/'manifest.csv',rows)
        seg_config=config(overrides={'seg_size':64,'seg_filters':4,'batch_size':2,'seg_epochs':1,'data_threads':1})
        seg_run=root/'seg';train_anatomy(anatomy,seg_run,seg_config);eval_anatomy(anatomy,seg_run,root/'seg_eval')
        classification=root/'classification';classification.mkdir();records=[]
        for role in ['fit','select','calibrate','test']:
            for label,name in enumerate(CLASSES):
                for repeat in range(2 if role=='fit' else 1):
                    key=f'{role}_{label}_{repeat}';group=key
                    if role in ['select','calibrate']:
                        n=0
                        while True:
                            group=key+'_'+str(n)
                            is_calib=int(hashlib.sha256(group.encode()).hexdigest()[:8],16)%3==0
                            if is_calib==(role=='calibrate'):break
                            n+=1
                    image=np.clip(rng.normal(100+label*10,30,(64,64,3)),0,255).astype('uint8')
                    file=classification/(key+'.png');Image.fromarray(image).save(file)
                    records.append({'record_id':key,'relative_path':file.name,'class_name':name,
                        'pixel_sha256':hashlib.sha256(image.tobytes()).hexdigest(),'image_group':group,
                        'training_role':'val' if role in ['select','calibrate'] else role,'source':'synthetic',
                        'sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
        write_csv(classification/'metadata/dataset_manifest.csv',records)
        features=root/'features';extract(classification,seg_run,features,batch_size=4)
        c=config(overrides={'mode':'fusion','weights':None,'image_size':64,'batch_size':2,
            'head_epochs':1,'fine_epochs':1,'full_epochs':1,'data_threads':1})
        run=root/'classifier';train(classification,run,c,features)
        train(classification,run,c,features,resume=True)
        calibration=calibrate(classification,[run],root/'calibration',features)
        report=evaluate(classification,[run],root/'evaluation',root/'calibration/calibration.json',features,bootstrap=5)
        assert len(report['confusion_matrix'])==4
        from continuation.inference import predict
        prediction=predict(classification/records[-1]['relative_path'],[run],root/'calibration/calibration.json',
                           root/'prediction',seg_run,explain=True)
        assert prediction['predicted_class'] in CLASSES
        # Check provenance rejects a changed checkpoint/calibration pairing.
        altered=json.loads((root/'calibration/calibration.json').read_text());altered['checkpoints']=['wrong']
        dump(root/'wrong_calibration.json',altered)
        # Validate stale-artifact rejection without rerunning the test protocol.
        try:predict(classification/records[-1]['relative_path'],[run],root/'wrong_calibration.json',root/'invalid',seg_run)
        except ValueError:pass
        else:raise AssertionError('Checkpoint mismatch was not rejected')
        print('SYNTHETIC END-TO-END PASS: segmentation training/evaluation, morphology export, fusion head and fine-tuning, calibration, test reporting.',flush=True)

if __name__=='__main__':main()
