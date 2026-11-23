"""Exercise archive-integrity and extraction boundaries without touching real data."""
import csv
import hashlib
import io
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile
from accuracy_development import training_cache as cache

class TrainingCacheTests(unittest.TestCase):
    def fixture(self, root, unsafe=None, symlink=False):
        text = io.StringIO()
        writer = csv.DictWriter(text, fieldnames=['relative_path', 'old_file'])
        writer.writeheader()
        for index in range(19942):
            writer.writerow({'relative_path': 'images/example.bin', 'old_file': index < 16729})
        manifest = text.getvalue().encode()
        archive = root / 'Classification_Cache.bin'
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
            z.writestr('Classification/metadata/dataset_manifest.csv', manifest)
            z.writestr('Classification/images/example.bin', b'unchanged image bytes')
            if unsafe:
                entry = zipfile.ZipInfo(unsafe)
                if symlink:
                    entry.create_system = 3
                    entry.external_attr = (stat.S_IFLNK | 0o777) << 16
                z.writestr(entry, b'outside')
        with zipfile.ZipFile(archive) as z:
            size = sum(i.file_size for i in z.infolist())
        archive_sha = cache.sha(archive)
        manifest_sha = hashlib.sha256(manifest).hexdigest()
        (root / 'cache_manifest.json').write_text(json.dumps({
            'archive': archive.name, 'archive_sha256': archive_sha,
            'classification_manifest_sha256': manifest_sha,
            'archive_bytes': archive.stat().st_size, 'uncompressed_bytes': size}))
        return archive_sha, manifest_sha

    def run_fixture(self, unsafe=None, symlink=False, changed=False, existing=False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'source'
            source.mkdir()
            archive_sha, manifest_sha = self.fixture(source, unsafe, symlink)
            destination = root / 'destination'
            if existing:
                destination.mkdir()
                (destination / 'preserve.txt').write_text('keep')
            if changed:
                archive = source / 'Classification_Cache.bin'
                data = bytearray(archive.read_bytes())
                data[30] ^= 1
                archive.write_bytes(data)
            with patch.object(cache, 'ARCHIVE_SHA', archive_sha), patch.object(cache, 'MANIFEST_SHA', manifest_sha):
                if unsafe or changed or existing:
                    with self.assertRaises((ValueError, AssertionError)):
                        cache.prepare(source, destination)
                    self.assertFalse((root / 'escape').exists())
                    if existing:
                        self.assertEqual((destination / 'preserve.txt').read_text(), 'keep')
                    else:
                        self.assertFalse(destination.exists())
                else:
                    staged, receipt = cache.prepare(source, destination)
                    self.assertEqual((staged / 'images/example.bin').read_bytes(), b'unchanged image bytes')
                    self.assertEqual(receipt['records'], 19942)
                    self.assertEqual(receipt['previous_year_records'], 16729)
                    self.assertFalse(receipt['images_labels_and_splits_changed'])
                    self.assertTrue((root / 'destination_cache_receipt.json').is_file())
            self.assertEqual(list(root.glob('classification-staging-*')), [])

    def test_valid_archive_stages_atomically(self): self.run_fixture()
    def test_changed_archive_rejected(self): self.run_fixture(changed=True)
    def test_existing_destination_preserved(self): self.run_fixture(existing=True)
    def test_traversal_rejected(self): self.run_fixture(unsafe='Classification/../../escape')
    def test_symlink_rejected(self): self.run_fixture(unsafe='Classification/images/link', symlink=True)

if __name__ == '__main__':
    unittest.main()
