import os
import tempfile
import unittest

import numpy as np

from potmill.structuregen.model import CNManager
from potmill.structuregen.optimizer import _append_descriptor_record, _read_descriptor_records


def _manager_from(descs, n_desc):
    m = CNManager(n_desc, energy_mode=False)
    for d in descs:
        m.update(d)
    return m


class TestDescriptorContainer(unittest.TestCase):
    def test_roundtrip_matches_per_array_feed(self):
        """The per-worker .bin container must accumulate the EXACT same information matrix as feeding
        the original descriptor arrays one by one. The old path was np.save/np.load (bit-exact for
        float64), so direct-feed of the arrays is the bitwise reference."""
        rng = np.random.default_rng(0)
        n_desc = 14
        # variable n_atoms per record (2..25), exercising the length-prefixed framing
        descs = [rng.standard_normal((int(n), n_desc)) for n in rng.integers(2, 26, size=37)]

        ref = _manager_from(descs, n_desc)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "desc_0.bin")
            for d in descs:
                _append_descriptor_record(path, d)
            records, offset = _read_descriptor_records(path, 0, n_desc)
            self.assertEqual(offset, os.path.getsize(path))

        self.assertEqual(len(records), len(descs))
        for got, want in zip(records, descs, strict=True):
            np.testing.assert_array_equal(got, want)  # values survive the bytes round-trip exactly

        got_mgr = _manager_from(records, n_desc)
        np.testing.assert_array_equal(got_mgr.cross, ref.cross)
        np.testing.assert_array_equal(got_mgr.sum, ref.sum)
        self.assertEqual(got_mgr.count, ref.count)

    def test_incremental_offset_reads_only_new_records(self):
        """Mimics the sync poll: read what exists, a peer appends more, the next read (from the
        returned offset) picks up only the new records."""
        rng = np.random.default_rng(1)
        n_desc = 8
        first = [rng.standard_normal((int(n), n_desc)) for n in (3, 5)]
        more = [rng.standard_normal((int(n), n_desc)) for n in (2, 4, 6)]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "desc_1.bin")
            for d in first:
                _append_descriptor_record(path, d)
            recs1, off1 = _read_descriptor_records(path, 0, n_desc)
            self.assertEqual(len(recs1), 2)
            for d in more:
                _append_descriptor_record(path, d)
            recs2, off2 = _read_descriptor_records(path, off1, n_desc)
            self.assertEqual(len(recs2), 3)
            self.assertEqual(off2, os.path.getsize(path))

    def test_torn_tail_is_not_consumed(self):
        """A partial trailing record (a peer mid-append) must be skipped, never fed truncated to the
        manager, and become readable once the peer finishes writing it."""
        rng = np.random.default_rng(2)
        n_desc = 6
        good = [rng.standard_normal((3, n_desc)), rng.standard_normal((4, n_desc))]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "desc_2.bin")
            for d in good:
                _append_descriptor_record(path, d)
            full = os.path.getsize(path)
            # torn record: prefix claims 5 rows but only 3 floats follow (far short of 5*n_desc)
            with open(path, "ab") as f:
                f.write(np.array(5, dtype="<u8").tobytes())
                f.write(np.zeros(3, dtype="<f8").tobytes())
            recs, off = _read_descriptor_records(path, 0, n_desc)
            self.assertEqual(len(recs), 2)  # only the two complete records
            self.assertEqual(off, full)  # offset stopped at the last complete record
            # peer finishes the record -> it becomes readable from the saved offset
            with open(path, "ab") as f:
                f.write(np.zeros(5 * n_desc - 3, dtype="<f8").tobytes())
            recs2, off2 = _read_descriptor_records(path, off, n_desc)
            self.assertEqual(len(recs2), 1)
            self.assertEqual(off2, os.path.getsize(path))


if __name__ == "__main__":
    unittest.main()
