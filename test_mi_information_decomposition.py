"""Analytic and independent count checks; no model/checkpoint inference."""
from collections import Counter
import unittest

import numpy as np

from mi_information_decomposition import decomposition_from_states


def evaluate(v, s, y):
    ids = np.arange(len(y))
    return decomposition_from_states(v, s, y, victim_sample_ids=ids, suspect_sample_ids=ids)


def direct_cmi(a, b, c):
    """Independent probability-ratio definition, not an entropy difference."""
    abc, ac, bc, cc = Counter(zip(a, b, c)), Counter(zip(a, c)), Counter(zip(b, c)), Counter(c)
    n = len(a)
    return sum(count / n * np.log2(count * cc[z] / (ac[x, z] * bc[y, z]))
               for (x, y, z), count in abc.items())


class DecompositionTests(unittest.TestCase):
    def test_identical_models(self):
        r = evaluate([0, 0, 1, 2], [0, 0, 1, 2], [0, 1, 1, 0])
        for key in ("G_X", "L_X", "G_Y", "L_Y", "delta_X", "delta_Y"):
            self.assertAlmostEqual(r[key], 0)

    def test_directional_example_with_known_bits(self):
        # X=(Y,A,B), independent fair bits; V=Y, S=(A,B).
        rows = np.array([(y, a, b) for y in range(2) for a in range(2) for b in range(2)])
        y, a, b = rows.T
        r = evaluate(y, 2 * a + b, y)
        for key, value in {"G_X": 2, "L_X": 1, "G_Y": 0, "L_Y": 1,
                           "delta_X": 1, "delta_Y": -1}.items():
            self.assertAlmostEqual(r[key], value)

    def test_label_improvement_is_possible(self):
        y, a = np.array([(y, a) for y in range(2) for a in range(2)]).T
        r = evaluate(a, 2 * y + a, y)
        self.assertAlmostEqual(r["G_Y"], 1)
        self.assertAlmostEqual(r["L_Y"], 0)
        self.assertAlmostEqual(r["delta_Y"], 1)

    def test_joint_singletons_and_synergy(self):
        v, s = np.array([(v, s) for v in range(2) for s in range(2)]).T
        r = evaluate(v, s, v ^ s)
        for key in ("G_X", "L_X", "G_Y", "L_Y"):
            self.assertAlmostEqual(r[key], 1)
        self.assertAlmostEqual(r["I_V_Y"], 0)
        self.assertAlmostEqual(r["I_S_Y"], 0)
        self.assertEqual(r["occupancy_pair"]["singleton_sample_fraction"], 1)

    def test_both_marginals_singletonize(self):
        r = evaluate(np.arange(8), np.arange(8)[::-1], np.arange(8) % 2)
        self.assertAlmostEqual(r["I_X_V"], 3)
        self.assertAlmostEqual(r["I_V_Y"], 1)
        for key in ("G_X", "L_X", "G_Y", "L_Y"):
            self.assertAlmostEqual(r[key], 0)

    def test_independent_probability_ratio_and_bounds(self):
        rng = np.random.default_rng(917)
        v, s, y = rng.integers(0, 4, 137), rng.integers(0, 5, 137), rng.integers(0, 3, 137)
        r = evaluate(v, s, y)
        self.assertAlmostEqual(r["G_Y"], direct_cmi(s, y, v), places=12)
        self.assertAlmostEqual(r["L_Y"], direct_cmi(v, y, s), places=12)
        self.assertAlmostEqual(r["G_X"], direct_cmi(np.arange(len(y)), s, v), places=12)
        self.assertAlmostEqual(r["L_X"], direct_cmi(np.arange(len(y)), v, s), places=12)
        self.assertAlmostEqual(r["eq4_residual"], 0, places=12)
        self.assertAlmostEqual(r["eq5_residual"], 0, places=12)
        self.assertLessEqual(r["G_Y"], r["H_Y_given_V"] + 1e-12)
        self.assertLessEqual(r["L_Y"], r["H_Y_given_S"] + 1e-12)

    def test_pairing_and_invalid_input_rejected(self):
        with self.assertRaisesRegex(ValueError, "match in order"):
            decomposition_from_states([0, 1], [0, 1], [0, 0],
                                      victim_sample_ids=[10, 20], suspect_sample_ids=[20, 10])
        with self.assertRaisesRegex(ValueError, "unique sample"):
            decomposition_from_states([0, 1], [0, 1], [0, 0],
                                      victim_sample_ids=[10, 10], suspect_sample_ids=[10, 10])
        with self.assertRaises(ValueError):
            evaluate([0.0, 1.0], [0, 1], [0, 1])
        with self.assertRaises(ValueError):
            evaluate([0], [0, 1], [0, 1])


if __name__ == "__main__":
    unittest.main()
