import contextlib
import csv
import io
import json
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from ablation.manifest import create, load, CLASSES
from ablation.package import make_jobs
from ablation.metrics import calculate
from ablation.collect import collect

ROOT = Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / 'tests')
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def fixture(self):
        data = self.root / 'data'
        for split in ['Train', 'Val']:
            for label in CLASSES:
                d = data / split / label
                d.mkdir(parents=True)
                # No actual images needed: manifest generator must never decode them.
                (d / 'sample.JPG').write_text('synthetic placeholder')
        return data

    def test_manifest_preserves_split_and_rejects_changes(self):
        out = self.root / 'manifest'
        create(self.fixture(), out)
        rows, protocol = load(out / 'split_manifest.csv', out / 'protocol.json')
        self.assertEqual(len(rows), 8)
        self.assertEqual(protocol['class_weights'], {str(i): 1 for i in range(4)})
        self.assertFalse(protocol['image_audit_performed'])
        with (out / 'split_manifest.csv').open('a') as f:
            f.write('extra,Normal,train\n')
        with self.assertRaises(ValueError):
            load(out / 'split_manifest.csv', out / 'protocol.json')

    def test_streaming_decoder_on_synthetic_files(self):
        import tensorflow as tf
        from ablation.data import make_dataset
        data = self.fixture()
        for path in data.rglob('*.JPG'):
            value = CLASSES.index(path.parent.name) * 50
            tf.io.write_file(str(path), tf.io.encode_jpeg(tf.fill([17, 23, 3], tf.cast(value, tf.uint8))))
        out = self.root / 'manifest'
        create(data, out)
        rows, _ = load(out / 'split_manifest.csv', out / 'protocol.json')
        val = make_dataset([r for r in rows if r['split'] == 'val'], data, False, 42)
        x, y = next(iter(val))
        self.assertEqual(tuple(x.shape), (4, 224, 224, 3))
        np.testing.assert_array_equal(y.numpy().argmax(axis=1), np.arange(4))
        self.assertGreater(float(tf.reduce_max(x)), 100.)

    def test_kernel_runs_bundled_code_with_correct_environment(self):
        jobs = self.root / 'jobs'
        with contextlib.redirect_stdout(io.StringIO()):
            make_jobs('testuser', 'testuser/frozen-data', jobs)
        for path in jobs.glob('*/run.py'):
            compile(path.read_text(), str(path), 'exec')
            meta = json.loads((path.parent / 'kernel-metadata.json').read_text())
            self.assertTrue(meta['is_private'])
            self.assertTrue(meta['enable_gpu'])
        script = (jobs / 'smoke/run.py').read_text().replace('/kaggle/working', str(self.root / 'working'))
        with patch('subprocess.run') as run:
            exec(compile(script, '<generated-kaggle-job>', 'exec'), {})
        args, kwargs = run.call_args
        self.assertEqual(kwargs['env']['PYTHONHASHSEED'], '42')
        self.assertIn('--require-gpu', args[0])
        self.assertTrue((kwargs['cwd'] / 'model/fuzzy_layers.py').is_file())

    def test_launch_sequence_and_resume(self):
        from ablation.launch import main
        jobs, results = self.root / 'jobs', self.root / 'downloaded'
        with contextlib.redirect_stdout(io.StringIO()):
            make_jobs('testuser', 'testuser/frozen-data', jobs)
        calls = []
        def fake_command(*args):
            calls.append(args)
            if args[1] == 'status':
                return 'Kernel has status "complete"'
            if args[1] == 'output':
                dest = Path(args[args.index('-p') + 1])
                dest.mkdir(parents=True, exist_ok=True)
                provenance = json.loads((jobs / dest.name / 'job-provenance.json').read_text())
                provenance.update(variant=dest.name, smoke_test=False)
                (dest / 'result.json').write_text(json.dumps(provenance))
            return ''
        argv = ['launch', '--jobs', str(jobs), '--results', str(results),
                '--variants', 'V0_baseline', 'V1_fuzzy_only']
        with patch('sys.argv', argv), patch('ablation.launch.shutil.which', return_value='kaggle'), patch('ablation.launch.command', side_effect=fake_command):
            main()
            count = len(calls)
            main()
        self.assertEqual([c[1] for c in calls], ['push', 'status', 'output', 'push', 'status', 'output'])
        self.assertEqual(len(calls), count)

    def test_launcher_status_parsing(self):
        from ablation.launch import status_of
        self.assertEqual(status_of('KernelWorkerStatus.COMPLETE'), 'complete')
        self.assertEqual(status_of('Kernel has status "running"'), 'running')
        with self.assertRaises(RuntimeError):
            status_of('Unauthorized')

    def write_result(self, name, pred):
        from ablation.train import source_hash
        y = np.tile(np.arange(4), 2)
        probs = np.full((8, 4), .05)
        probs[np.arange(8), pred] = .85
        d = self.root / 'results' / name
        d.mkdir(parents=True)
        result = calculate(y, probs)
        result.update(variant=name, smoke_test=False, seed=42, manifest_sha256='fixture',
                      source_sha256=source_hash(), versions={}, class_names=CLASSES,
                      class_weights={str(i): 1 for i in range(4)}, head='Dense(ReLU)->BatchNorm')
        (d / 'result.json').write_text(json.dumps(result))
        (d / 'validation_ids.json').write_text(json.dumps([str(i) for i in range(8)]))
        np.save(d / 'y_true.npy', y)
        np.save(d / f'predictions_{name}.npy', pred)
        np.save(d / f'probabilities_{name}.npy', probs)
        return d

    def test_metrics_and_exact_mcnemar(self):
        y = np.tile(np.arange(4), 2)
        self.write_result('V4_full_hybrid', y)
        other = y.copy()
        other[:4] = (other[:4] + 1) % 4
        self.write_result('V0_baseline', other)
        comparisons = collect(self.root / 'results', self.root / 'summary')
        self.assertEqual(comparisons[0]['p_exact'], .125)
        self.assertEqual(comparisons[0]['accuracy_delta'], .5)
        self.assertEqual(comparisons[0]['full_only_correct'], 4)

    def test_rejects_misaligned_predictions(self):
        y = np.tile(np.arange(4), 2)
        self.write_result('V4_full_hybrid', y)
        d = self.write_result('V0_baseline', y)
        (d / 'validation_ids.json').write_text(json.dumps([str(i) for i in reversed(range(8))]))
        with self.assertRaisesRegex(ValueError, 'Validation order'):
            collect(self.root / 'results', self.root / 'summary')

    def test_smoke_not_in_publication_table(self):
        d = self.write_result('V0_baseline', np.tile(np.arange(4), 2))
        r = json.loads((d / 'result.json').read_text())
        r['smoke_test'] = True
        (d / 'result.json').write_text(json.dumps(r))
        with self.assertRaisesRegex(ValueError, 'No completed'):
            collect(self.root / 'results', self.root / 'summary')


if __name__ == '__main__':
    unittest.main()
