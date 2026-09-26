import ast
import copy
import csv
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
from mi_pool_support import create_nested_balanced_subsets, missing_mi_grid, sizes_from_rates


class Labels:
    def __init__(self, classes=100, each=250):
        self.targets = np.repeat(np.arange(classes), each)

    def __len__(self):
        return len(self.targets)


class PoolTests(unittest.TestCase):
    def test_legacy_extension_and_replay(self):
        dataset = Labels()
        group = np.arange(len(dataset))
        old_sizes = [1000, 5000, 10000, 15000, 20000, 25000]
        # Reproduce the old implementation independently.
        rng = np.random.default_rng(42)
        per_class = [np.flatnonzero(dataset.targets == c) for c in range(100)]
        for values in per_class:
            rng.shuffle(values)
        old = {}
        for size in old_sizes:
            indices = np.concatenate([v[:size // 100] for v in per_class])
            np.random.default_rng(42 + size).shuffle(indices)
            old[size] = indices
        new_sizes = sizes_from_rates(25000, [.05, .1, .2, .5, .75, 1.0])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'nested_subsets_seed42.npz'
            np.savez(path, **{f'size_{s}': a for s, a in old.items()})
            result = create_nested_balanced_subsets(dataset, group, directory, new_sizes, 100)
            for size, indices in old.items():
                np.testing.assert_array_equal(result[size], indices)
            previous = set()
            for size in sorted(result):
                indices = result[size]
                self.assertEqual(len(set(indices)), size)
                self.assertTrue(previous <= set(indices))
                counts = np.bincount(dataset.targets[indices], minlength=100)
                self.assertLessEqual(counts.max() - counts.min(), 1)
                previous = set(indices)
            before = path.read_bytes()
            replay = create_nested_balanced_subsets(dataset, group, directory, new_sizes[::-1], 100)
            self.assertEqual(before, path.read_bytes())
            for size in result:
                np.testing.assert_array_equal(result[size], replay[size])
            # Changing the group cannot silently reuse stale cache.
            with self.assertRaises(ValueError):
                create_nested_balanced_subsets(dataset, group[:-1], directory, new_sizes, 100)
            self.assertEqual(before, path.read_bytes())
        with tempfile.TemporaryDirectory() as directory:
            fresh = create_nested_balanced_subsets(dataset, group, directory, new_sizes, 100)
            for size in new_sizes:
                np.testing.assert_array_equal(fresh[size], result[size])

    def test_incremental_grid_and_interrupted_pipeline(self):
        # Execute the real Pool-0 function with only model/data inference stubbed.
        # Avoid importing MI_check's GPU and dataset dependencies in unit tests.
        module = ast.parse(Path('MI_check.py').read_text(encoding='utf-8'))
        functions = [n for n in module.body if isinstance(n, ast.FunctionDef)
                     and n.name in {'main_nega_pool0', 'ensure_master_csv', 'append_master_row'}]
        columns = ['Scenario', 'seed', 'rate', 'model_name', 'epoch', 'bins', 'in_size',
                   'I(X;T)-In', 'I(T;Y)-In', 'out_size', 'I(X;T)-Out', 'I(T;Y)-Out', 'timestamp']
        data = Labels(classes=2, each=10)
        class Dataset:
            in_sample_set = data
            test_set = list(range(6))
            def subset(self, *args, **kwargs):
                return args[1]
        class Net:
            def to(self, *args): return self
            def load_state_dict(self, *args): pass
            def eval(self): pass
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'mi.csv'
            calls = []
            env = dict(Path=Path, np=np, csv=csv, copy=copy, datetime=datetime,
                       POOL0_MASTER_CSV=str(path), POOL0_VERBOSE_DIR=directory,
                       POOL0_RATE=0.0, POOL0_MODEL_DIR=directory, POOL0_DEIT_MODEL_DIR=directory,
                       POOL0_IN_SIZE_RATES=[.25, .5, 1.0], POOL0_BINS=[5, 10], device='cpu',
                       MASTER_CSV_COLUMNS=columns, sizes_from_rates=sizes_from_rates,
                       missing_mi_grid=missing_mi_grid,
                       process_yaml_file=lambda _: {'Scenario_Name':'toy', 'Dataset':{'name':'toy','group_size':20}},
                       load_best_checkpoint=lambda _: ('fake', None),
                       process_experiment_setup=lambda _: {'Model':Net(), 'Dataset':Dataset(), 'GroupSize':20,'NumClasses':2},
                       torch=SimpleNamespace(load=lambda *a, **k: {}, cuda=SimpleNamespace(is_available=lambda:False)),
                       create_or_load_group_A=lambda **k: np.arange(20),
                       create_nested_balanced_subsets=lambda **k: create_nested_balanced_subsets(
                           k['dataset'],k['group_A'],directory,k['subset_sizes'],k['num_classes']),
                       DataLoader=lambda data, **k: data,
                       collect_logits=lambda net, loader, device: (calls.append(len(loader)) or np.zeros((len(loader),2)), None),
                       mi_from_logits=lambda x,y,**k: (1.0, 2.0))
            exec(compile(ast.Module(body=functions, type_ignores=[]), 'MI_check.py', 'exec'), env)
            env['ensure_master_csv'](path)
            original = dict(zip(columns, ['toy',42,0.0,'toy_42_0.0',99,5,10,1,2,6,1,2,'old']))
            env['append_master_row'](path, original)
            old_bytes = path.read_bytes()
            append = env['append_master_row']
            def interrupted(path, row):
                append(path, row)
                raise RuntimeError('simulated interruption after completed row')
            env['append_master_row'] = interrupted
            with self.assertRaisesRegex(RuntimeError, 'simulated'):
                env['main_nega_pool0'](42,'unused')
            partial = path.read_bytes()
            env['append_master_row'] = append
            self.assertTrue(env['main_nega_pool0'](42,'unused'))
            self.assertTrue(path.read_bytes().startswith(partial))
            self.assertTrue(path.read_bytes().startswith(old_bytes))
            with path.open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows),6)
            self.assertEqual(len({(r['in_size'],r['bins']) for r in rows}),6)
            missing, _ = missing_mi_grid(path,'toy_42_0.0','toy',42,0.0,[5,10,20],[5,10])
            self.assertEqual(missing,[])
            calls.clear()
            self.assertFalse(env['main_nega_pool0'](42,'unused'))
            self.assertEqual(calls,[])
            # A new size reuses both saved out-bin values: no out inference.
            self.assertTrue(env['main_nega_pool0'](42,'unused',in_sizes=[7]))
            self.assertEqual(calls,[7])
            append(path, original)
            with self.assertRaisesRegex(ValueError,'Duplicate'):
                env['main_nega_pool0'](42,'unused')


if __name__ == '__main__':
    unittest.main()
