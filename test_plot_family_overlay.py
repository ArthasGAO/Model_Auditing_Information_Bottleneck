"""Family split preserves source coordinates, anchors, H0 and existing figures."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.markers import MarkerStyle
from matplotlib.text import Text
import numpy as np
import pandas as pd

from test_plot_fixed_split_h0 import notebook_namespace
from test_plot_fixed_split_grid_data import execute_cell

ROOT = Path(__file__).resolve().parent


class FamilyOverlayTests(unittest.TestCase):
    def test_family_figures(self):
        protected = list((ROOT/'saved_plots/Fig1_SameArch_method_overlay').glob('*'))
        protected += [ROOT/p for p in (
            'saved_logs/ft_final/MI_master_table_ft_multiple.csv',
            'saved_logs/pruning_final/MI_master_table_prune_multiple.csv',
            'saved_logs/extraction_final/MI_master_table_extraction_multiple.csv',
            'saved_logs/vanilla/MI_master_table_neg_pool0.csv')]
        protected = [p for p in protected if p.is_file()]
        hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
        ns, book = notebook_namespace()
        before = json.loads((ROOT/'tmp/distribution_check.before_overlay_family_sync.ipynb').read_text(encoding='utf-8'))
        for original, current in zip(before['cells'], book['cells']):
            if current.get('id') not in {'fixed-h0-overlay-config', 'fixed-h0-overlay-render',
                                        'fixed-h0-family-config', 'fixed-h0-family-render'}:
                self.assertEqual(original, current)
        self.assertEqual(book['metadata'], before['metadata'])
        for identity in ('b6bf5527-38a7-43f3-b5c0-ab17d70cb169', 'fixed-h0-grid-config',
                         'fixed-h0-grid-render', 'fixed-h0-single-config', 'fixed-h0-overlay-config',
                         'fixed-h0-overlay-render', 'fixed-h0-family-config'):
            execute_cell(book, identity, ns)
        original_panels = deepcopy(ns['overlay_panels'])
        code = next(''.join(c['source']) for c in book['cells'] if c.get('id') == 'fixed-h0-family-render')
        with patch('matplotlib.pyplot.show'):
            exec(compile(code, 'family-render', 'exec'), ns)
        self.assertEqual(original_panels, ns['overlay_panels'])
        self.assertEqual(ns['family_names'], ['RN18_FT', 'RN18_Pruning', 'RN18_Extraction'])
        base_ax = ns['overlay_axes'][0]
        base_groups = ns['overlay_panels'][0]['positive_groups']
        lookup = {g['spec'].label: c for g, c in zip(base_groups, base_ax.collections[2:-1])}
        expected_names = [['FT-LL','FT-AL','RT-AL'], ['PR20% best','PR80% best'], ['Knockoff','DFMS']]
        for fig, ax, panel, names in zip(ns['family_figures'], ns['family_axes'], ns['family_panels'], expected_names):
            self.assertEqual([g['spec'].label for g in panel['positive_groups']], names)
            for i in (0, 1, -1):
                np.testing.assert_array_equal(ax.collections[i].get_offsets(), base_ax.collections[i].get_offsets())
            np.testing.assert_array_equal(ax.lines[0].get_xydata(), base_ax.lines[0].get_xydata())
            for name, collection in zip(names, ax.collections[2:-1]):
                np.testing.assert_array_equal(collection.get_offsets(), lookup[name].get_offsets())
                np.testing.assert_allclose(collection.get_facecolors()[0][:3],
                                           to_rgba(ns['FAMILY_METHOD_COLORS'][name])[:3], atol=1e-12)
            for index, role in ((0, 'negative_style'), (-1, 'victim_style')):
                np.testing.assert_allclose(ax.collections[index].get_facecolors()[0][:3],
                                           to_rgba(ns['FAMILY_ROLE_COLORS'][role])[:3], atol=1e-12)
            reference = ax.collections[1]
            self.assertEqual(len(reference.get_facecolors()), 0)
            np.testing.assert_allclose(reference.get_edgecolors()[0][:3],
                                       to_rgba(ns['FAMILY_ROLE_COLORS']['reference_style'])[:3], atol=1e-12)
            self.assertAlmostEqual(reference.get_linewidths()[0], ns['FAMILY_REFERENCE_RING_WIDTH'])
            self.assertAlmostEqual(reference.get_sizes()[0], ns['FAMILY_REFERENCE_RING_AREA'])
            self.assertGreater(reference.get_zorder(), ax.collections[0].get_zorder())
            marker = MarkerStyle(ns['FAMILY_REFERENCE_MARKER'])
            np.testing.assert_allclose(reference.get_paths()[0].vertices,
                                       marker.get_path().transformed(marker.get_transform()).vertices)
            reference_handle = ax.get_legend().legend_handles[-1]
            self.assertEqual(reference_handle.get_markerfacecolor(), 'none')
            self.assertEqual(reference_handle.get_marker(), ns['FAMILY_REFERENCE_MARKER'])
            np.testing.assert_allclose(to_rgba(reference_handle.get_markeredgecolor())[:3],
                                       to_rgba(ns['FAMILY_ROLE_COLORS']['reference_style'])[:3])
            for collection in (ax.collections[:1] + ax.collections[2:]):
                self.assertTrue(np.all(collection.get_linewidths() == 0))
                self.assertEqual(len(collection.get_edgecolors()), 0)
            self.assertEqual([t.get_text() for t in ax.get_legend().texts],
                             ['Victim model'] + names + ['negative suspects','reference models'])
            self.assertEqual(ax.get_legend()._loc, 0)
            self.assertAlmostEqual(ax.bbox.width, ax.bbox.height)
            self.assertFalse(ax.get_title())
            self.assertFalse(fig.texts)
            for text in fig.findobj(Text):
                if text.get_text():
                    self.assertEqual(text.get_fontproperties().get_name(), 'Times New Roman')
        actual = ns['family_point_results'].query("role == 'positive'")
        expected = ns['overlay_point_results'].query("architecture == 'RN18' and role == 'positive'")
        columns = [c for c in expected.columns if c != 'architecture']
        pd.testing.assert_frame_equal(actual[columns].sort_values(['group','model_name']).reset_index(drop=True),
                                      expected[columns].sort_values(['group','model_name']).reset_index(drop=True))
        self.assertEqual(actual.groupby('method_family').size().to_dict(), {'ft':150,'prune':100,'extraction':100})
        for pdf in ns['family_pdf_paths']:
            self.assertTrue(pdf.is_file() and pdf.stat().st_size > 1000)
            self.assertTrue(pdf.with_suffix('.png').is_file())
        self.assertEqual(hashes, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in protected})
        print('PASS: three family figures; 350 positives partitioned exactly; CSVs, prior plots, anchors and H0 unchanged.')
        plt.close('all')


if __name__ == '__main__':
    unittest.main()
