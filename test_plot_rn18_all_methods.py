"""Validate and export the expanded RN18 overlay without rewriting source CSVs."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.markers import MarkerStyle
import numpy as np
import pandas as pd

from test_plot_fixed_split_h0 import notebook_namespace
from test_plot_fixed_split_grid_data import execute_cell

ROOT = Path(__file__).resolve().parent


class AllMethodsTests(unittest.TestCase):
    def test_all_methods(self):
        sources = [ROOT / p for p in (
            'saved_logs/ft_final/MI_master_table_ft_multiple.csv',
            'saved_logs/pruning_final/MI_master_table_prune_multiple.csv',
            'saved_logs/extraction_final/MI_master_table_extraction_multiple.csv',
            'saved_logs/vanilla/MI_master_table_neg_pool0.csv')]
        hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in sources]
        ns, book = notebook_namespace()
        original = json.loads((ROOT/'tmp/distribution_check.before_overlay_family_sync.ipynb').read_text(encoding='utf-8'))
        for before, after in zip(original['cells'], book['cells']):
            if after.get('id') not in {'fixed-h0-overlay-config', 'fixed-h0-overlay-render',
                                      'fixed-h0-family-config', 'fixed-h0-family-render'}:
                self.assertEqual(before, after)
        for identity in ('b6bf5527-38a7-43f3-b5c0-ab17d70cb169', 'fixed-h0-grid-config',
                         'fixed-h0-grid-render', 'fixed-h0-single-config', 'fixed-h0-overlay-config'):
            execute_cell(book, identity, ns)
        before_panels = deepcopy(ns['PLOT_PANELS'])
        execute_cell(book, 'fixed-h0-overlay-render', ns)
        self.assertEqual(before_panels, ns['PLOT_PANELS'])
        fig, ax = ns['overlay_figures'][0], ns['overlay_axes'][0]
        counts = [len(c.get_offsets()) for c in ax.collections]
        self.assertEqual(counts, [50, 30] + [50]*7 + [1])
        self.assertEqual(len(ns['overlay_point_results'].query("architecture == 'RN18' and role == 'positive'")), 350)
        reference_ax = ns['axes_grid'].flat[0]
        for i in (0, 1, -1):
            np.testing.assert_array_equal(ax.collections[i].get_offsets(), reference_ax.collections[i].get_offsets())
        np.testing.assert_array_equal(ax.lines[0].get_xydata(), reference_ax.lines[0].get_xydata())
        self.assertEqual(ax.get_legend()._loc, 0)
        self.assertEqual(len(ax.get_legend().texts), 10)
        self.assertAlmostEqual(ax.bbox.width, ax.bbox.height)
        for collection in (ax.collections[:1] + ax.collections[2:]):
            self.assertTrue(np.all(collection.get_linewidths() == 0))
            self.assertEqual(len(collection.get_edgecolors()), 0)
        reference = ax.collections[1]
        self.assertEqual(len(reference.get_facecolors()), 0)
        np.testing.assert_allclose(reference.get_edgecolors()[0][:3],
                                   to_rgba(ns['OVERLAY_RN18_ROLE_COLORS']['reference_style'])[:3])
        marker = MarkerStyle(ns['OVERLAY_RN18_REFERENCE_MARKER'])
        np.testing.assert_allclose(reference.get_paths()[0].vertices,
                                   marker.get_path().transformed(marker.get_transform()).vertices)
        self.assertEqual(ax.get_legend().legend_handles[-1].get_marker(), ns['OVERLAY_RN18_REFERENCE_MARKER'])
        for index, role in ((0, 'negative_style'), (-1, 'victim_style')):
            np.testing.assert_allclose(ax.collections[index].get_facecolors()[0][:3],
                                       to_rgba(ns['OVERLAY_RN18_ROLE_COLORS'][role])[:3])
        # Exact per-method source coverage, without averaging or shifting any MI.
        for group, collection in zip(ns['overlay_panels'][0]['positive_groups'], ax.collections[2:-1]):
            points, xy = ns['fixed_h0'].selected_points(group['spec'], ns, ns['BINS'], ns['IN_SIZE'])
            np.testing.assert_array_equal(collection.get_offsets(), xy)
            np.testing.assert_allclose(collection.get_facecolors()[0][:3],
                                       to_rgba(ns['OVERLAY_RN18_METHOD_COLORS'][group['spec'].label])[:3])
            self.assertEqual(sorted(p['seed'] for p in points), list(range(50)))
        # The untouched DeiT overlay still reuses the original three positive groups.
        self.assertEqual([len(c.get_offsets()) for c in ns['overlay_axes'][1].collections], [50,30,50,50,50,1])
        self.assertEqual(hashes, [hashlib.sha256(p.read_bytes()).hexdigest() for p in sources])
        out = ROOT/ns['OVERLAY_OUTPUT_DIR']
        out.mkdir(parents=True, exist_ok=True)
        for suffix, dpi in [('pdf', ns['OVERLAY_DPI']), ('png', 200)]:
            ns['fixed_h0'].save_panel_figures([fig], [out/f'{ns["OVERLAY_RN18_OUTPUT_NAME"]}.{suffix}'],
                dpi=dpi, pad_inches=ns['OVERLAY_PAD_INCHES'], shared_crop=False)
        print('PASS: 350 positive points across 7 groups; same 50 negatives, 30 references, victim and H0.')
        print('Source CSV hashes and unrelated notebook cells unchanged.')
        print(ns['overlay_point_results'].query("architecture == 'RN18' and role == 'positive'").groupby('group').size())
        plt.close('all')


if __name__ == '__main__':
    unittest.main()
