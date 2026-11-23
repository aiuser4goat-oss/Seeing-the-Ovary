"""Check deterministic image-classification constraints without judging label correctness."""
from pathlib import Path
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv,hashlib,json,datetime
B=Path('/Users/ik-mac/2627 Science Fair Data/Model_Datasets');root=B/'Classification';manifest=root/'metadata/dataset_manifest.csv'
def role(r):
 if r['training_role']!='val':return r['training_role']
 return 'calibrate' if int(hashlib.sha256(r['image_group'].encode()).hexdigest()[:8],16)%3==0 else 'select'
with manifest.open(newline='') as f:rows=[r for r in csv.DictReader(f) if role(r) in ['fit','select']]
def verify(r):
 h=hashlib.sha256()
 with (root/r['relative_path']).open('rb') as f:
  for chunk in iter(lambda:f.read(262144),b''):h.update(chunk)
 if h.hexdigest()!=r['sha256']:raise ValueError('Input differs from manifest: '+r['record_id'])
 return r['record_id']
with ThreadPoolExecutor(max_workers=4) as pool:verified=list(pool.map(verify,rows))
result={'verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'verified_files':len(verified),'test_and_calibration_files_read':False,'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'roles':{}}
for split in ['fit','select']:
 rs=[r for r in rows if role(r)==split];support=Counter(r['class_name'] for r in rs);groups=defaultdict(Counter)
 for r in rs:groups[r['sha256']][r['class_name']]+=1
 ceiling=sum(max(count/(4*support[c]) for c,count in counts.items()) for counts in groups.values())
 result['roles'][split]={'records':len(rs),'class_support':dict(support),'byte_identical_groups_with_multiple_supplied_classes':sum(len(v)>1 for v in groups.values()),'deterministic_image_only_balanced_accuracy_upper_bound':ceiling,'ordinary_accuracy_upper_bound':sum(max(v.values()) for v in groups.values())/len(rs)}
result['proof']='For each byte-identical image group g, a deterministic classifier must choose one class. Its maximum contribution to balanced accuracy is max_c n(g,c)/(4*N(c)). Summing over groups gives an upper bound; visually different groups may introduce additional difficulty.'
result['interpretation']='This is a constraint of the current combined four-class record protocol, not an assertion that any particular supplied label or prior-year study is incorrect. It does not establish a test-set ceiling. Labels, files, roles and primary scores are unchanged.'
p=B/'Accuracy_Development_2026_11_22/boundary_study/verified_input_constraints.json';p.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
