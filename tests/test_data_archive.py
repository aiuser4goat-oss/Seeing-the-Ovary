import tempfile,unittest,zipfile
from pathlib import Path
from continuation.anatomy import DataArchive

class ArchiveLayoutTests(unittest.TestCase):
    def test_zip_and_expanded_layouts_match(self):
        name='Data/BiCycleStudy/image_metadata.csv'
        for layout in ['zip','expanded','kaggle_nested']:
            with self.subTest(layout=layout),tempfile.TemporaryDirectory() as d:
                root=Path(d)
                if layout=='zip':
                    with zipfile.ZipFile(root/'Data.zip','w') as z:z.writestr(name,b'PatientID,fold\n')
                else:
                    p=(root/'Data' if layout=='kaggle_nested' else root)/name
                    p.parent.mkdir(parents=True);p.write_bytes(b'PatientID,fold\n')
                with DataArchive(root) as z:
                    self.assertEqual(z.read(name),b'PatientID,fold\n')
                    self.assertIn(name,z.namelist())

    def test_ambiguous_roots_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for prefix in ['one','two']:
                p=root/prefix/'Data/BiCycleStudy/image_metadata.csv';p.parent.mkdir(parents=True);p.write_text('')
            with self.assertRaisesRegex(ValueError,'exactly one'):
                with DataArchive(root):pass

if __name__=='__main__':unittest.main()
