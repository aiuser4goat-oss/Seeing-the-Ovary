import argparse
from pathlib import Path
def parse_args(argv=None):
 p=argparse.ArgumentParser()
 for name in ['classification','model_source','warm_run','freeze','out']:
  p.add_argument('--'+name.replace('_','-'),dest=name,type=Path,required=True)
 p.add_argument('--smoke',action='store_true');p.add_argument('--max-minutes',type=float,default=90);p.add_argument('--epochs',type=int,default=40)
 return p.parse_args(argv)
