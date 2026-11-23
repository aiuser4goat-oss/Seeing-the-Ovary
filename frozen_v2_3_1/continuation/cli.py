import argparse,json
from pathlib import Path
from .common import config,dump,validate_manifest

def main(argv=None):
    p=argparse.ArgumentParser(description='Anatomy-guided ovarian ultrasound continuation')
    sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('audit');a.add_argument('--classification',required=True)
    a=sub.add_parser('preview-augmentation');a.add_argument('--classification',required=True);a.add_argument('--out',required=True)
    a.add_argument('--config');a.add_argument('--samples',type=int,default=500)
    a=sub.add_parser('bulk-augmentation-check');a.add_argument('--classification',required=True);a.add_argument('--out',required=True)
    a.add_argument('--config');a.add_argument('--views-per-image',type=int,default=4);a.add_argument('--batch-size',type=int,default=32)
    a=sub.add_parser('prepare-anatomy');a.add_argument('--anatomy',required=True);a.add_argument('--out',required=True)
    a.add_argument('--config');a.add_argument('--sources',nargs='+',choices=['usova','folliclefinder'],default=['usova','folliclefinder'])
    a.add_argument('--max-volumes',type=int);a.add_argument('--slice-step',type=int,default=4)
    a.add_argument('--classification')
    for name in ['train-anatomy','eval-anatomy']:
        a=sub.add_parser(name);a.add_argument('--prepared',required=True);a.add_argument('--out',required=True)
        if name=='train-anatomy':
            a.add_argument('--config');a.add_argument('--resume',action='store_true')
        else:a.add_argument('--run',required=True);a.add_argument('--role',choices=['val','test'],default='test')
    a=sub.add_parser('extract-anatomy');a.add_argument('--classification',required=True);a.add_argument('--seg-run',required=True);a.add_argument('--out',required=True);a.add_argument('--batch-size',type=int,default=16)
    for name in ['train','search']:
        a=sub.add_parser(name);a.add_argument('--classification',required=True);a.add_argument('--out',required=True)
        a.add_argument('--config');a.add_argument('--features')
        if name=='search':a.add_argument('--trials',type=int,default=4)
        else:
            a.add_argument('--smoke',action='store_true');a.add_argument('--resume',action='store_true')
    for name in ['calibrate','evaluate']:
        a=sub.add_parser(name);a.add_argument('--classification',required=True);a.add_argument('--runs',nargs='+',required=True)
        a.add_argument('--out',required=True);a.add_argument('--features');a.add_argument('--tta',action='store_true')
        if name=='calibrate':a.add_argument('--target-accuracy',type=float,default=.95)
        else:a.add_argument('--calibration',required=True);a.add_argument('--bootstrap',type=int,default=300)
    a=sub.add_parser('predict');a.add_argument('--image',required=True);a.add_argument('--runs',nargs='+',required=True)
    a.add_argument('--calibration',required=True);a.add_argument('--out',required=True);a.add_argument('--seg-run');a.add_argument('--explain',action='store_true')
    args=p.parse_args(argv)
    if args.command=='audit':print(json.dumps(validate_manifest(args.classification),indent=2))
    elif args.command=='preview-augmentation':
        from .augmentation_preview import preview
        print(json.dumps(preview(args.classification,args.out,config(args.config),args.samples),indent=2))
    elif args.command=='bulk-augmentation-check':
        from .augmentation_preview import bulk_check
        print(json.dumps(bulk_check(args.classification,args.out,config(args.config),args.views_per_image,args.batch_size),indent=2))
    elif args.command=='prepare-anatomy':
        from .anatomy import prepare
        prepare(args.anatomy,args.out,config(args.config),args.sources,args.max_volumes,args.slice_step,args.classification)
    elif args.command=='train-anatomy':
        from .anatomy import train
        train(args.prepared,args.out,config(args.config),args.resume)
    elif args.command=='eval-anatomy':
        from .anatomy import evaluate
        evaluate(args.prepared,args.run,args.out,args.role)
    elif args.command=='extract-anatomy':
        from .features import extract
        extract(args.classification,args.seg_run,args.out,args.batch_size)
    elif args.command in ['train','search']:
        from .training import train,search
        c=config(args.config)
        if args.command=='train':train(args.classification,args.out,c,args.features,args.smoke,args.resume)
        else:search(args.classification,args.out,c,args.features,args.trials)
    elif args.command=='calibrate':
        from .evaluation import calibrate
        calibrate(args.classification,args.runs,args.out,args.features,args.tta,args.target_accuracy)
    elif args.command=='evaluate':
        from .evaluation import evaluate
        evaluate(args.classification,args.runs,args.out,args.calibration,args.features,args.tta,args.bootstrap)
    elif args.command=='predict':
        from .inference import predict
        print(json.dumps(predict(args.image,args.runs,args.calibration,args.out,args.seg_run,args.explain),indent=2))
