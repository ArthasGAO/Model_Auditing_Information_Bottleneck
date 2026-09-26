"""Row replacement must preserve unrelated results and reject identity drift."""
import csv
from pathlib import Path
import tempfile
import unittest

import repair_cifar10_vgg16_hypothesis as repair


class RepairTests(unittest.TestCase):
    def test_preserves_other_case_bytes_and_order(self):
        with tempfile.TemporaryDirectory() as directory:
            old, new = Path(directory)/"old.csv", Path(directory)/"new.csv"
            header = b"scenario,in_size,bins,mi_kind,k_ref,value\r\n"
            first = b"other,25000,50,In,30,0.000000\r\n"
            last = b"another,25000,50,In,30,1e-08\r\n"
            old.write_bytes(header + first +
                            f"{repair.CASE},25000,50,In,30,0.5\r\n".encode() + last)
            rows = [{"scenario": repair.CASE, "in_size": "25000", "bins": "50",
                     "mi_kind": "In", "k_ref": "30", "value": "0.1"}]
            stats = repair.merge_case(old, rows, new, 1)
            self.assertEqual(new.read_bytes(), header + first +
                             f"{repair.CASE},25000,50,In,30,0.1\r\n".encode() + last)
            self.assertEqual(stats["preserved_rows"], 2)
            self.assertEqual(stats["replaced_rows"], 1)

    def test_wrong_identity_and_duplicate_replacements_rejected(self):
        for duplicate in (False, True):
            with self.subTest(duplicate=duplicate), tempfile.TemporaryDirectory() as directory:
                old, new = Path(directory)/"old.csv", Path(directory)/"new.csv"
                old.write_text(f"scenario,k_ref,value\n{repair.CASE},30,0.5\n")
                rows = [{"scenario": repair.CASE, "k_ref": "25", "value": "0.1"}]
                if duplicate:
                    rows *= 2
                with self.assertRaises(AssertionError):
                    repair.merge_case(old, rows, new, len(rows))


if __name__ == "__main__":
    unittest.main()
