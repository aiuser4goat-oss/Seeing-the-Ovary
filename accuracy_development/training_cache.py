"""Verify the unchanged classification archive and stage it on cloud-local storage."""
from pathlib import Path
import csv,hashlib,json,stat,tempfile,time,zipfile
ARCHIVE_SHA='604df1cc7c9ce8b6ace8e72f31c0a76bcce90dd661a5aa30f78674a46fa2d998'
MANIFEST_SHA='d0112fb807fe61accdd0fac1a2a606ff74f51558a650e5767e35d22bde62c190'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def prepare(input_root,destination):
 start=time.monotonic();input_root=Path(input_root);destination=Path(destination)
 manifests=list(input_root.rglob('cache_manifest.json'))
 if len(manifests)!=1:raise ValueError('Expected one explicit private cache manifest')
 m=json.loads(manifests[0].read_text());archive=manifests[0].parent/'Classification_Cache.bin'
 assert m['archive']=='Classification_Cache.bin' and m['archive_sha256']==ARCHIVE_SHA and m['classification_manifest_sha256']==MANIFEST_SHA
 assert archive.stat().st_size==m['archive_bytes']
 if destination.exists():raise ValueError('Use a fresh local staging directory; existing data is never overwritten')
 assert sha(archive)==ARCHIVE_SHA,'Classification archive changed'
 destination.parent.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix='classification-staging-',dir=destination.parent) as temp:
  temp=Path(temp)
  with zipfile.ZipFile(archive) as z:
   entries=z.infolist();assert len(entries)<=30000 and sum(i.file_size for i in entries)==m['uncompressed_bytes']<=4_000_000_000
   for i in entries:
    relative=Path(i.filename)
    if relative.is_absolute() or '..' in relative.parts or not relative.parts or relative.parts[0]!='Classification':raise ValueError('Unsafe archive path')
    if stat.S_IFMT(i.external_attr>>16)==stat.S_IFLNK:raise ValueError('Archive symlinks are not accepted')
   assert hashlib.sha256(z.read('Classification/metadata/dataset_manifest.csv')).hexdigest()==MANIFEST_SHA
   z.extractall(temp)
  staged=temp/'Classification';assert sha(staged/'metadata/dataset_manifest.csv')==MANIFEST_SHA
  with (staged/'metadata/dataset_manifest.csv').open() as f:rows=list(csv.DictReader(f))
  assert len(rows)==19942 and sum(r['old_file']=='True' for r in rows)==16729
  for r in rows:
   relative=Path(r['relative_path']);assert not relative.is_absolute() and '..' not in relative.parts and (staged/relative).is_file()
  staged.rename(destination)
 receipt={'archive_sha256':ARCHIVE_SHA,'manifest_sha256':MANIFEST_SHA,'records':19942,'previous_year_records':16729,'elapsed_staging_seconds':time.monotonic()-start,'destination':str(destination),'images_labels_and_splits_changed':False}
 (destination.parent/(destination.name+'_cache_receipt.json')).write_text(json.dumps(receipt,indent=2));return destination,receipt
