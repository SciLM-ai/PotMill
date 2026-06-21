import os
import tempfile
import unittest

from potmill.config import ConfigManager
from potmill.resources import worker_layout


def _cfg(text=""):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "config.ini")
        with open(path, "w") as f:
            f.write(text)
        return ConfigManager(path)


class TestWorkerLayout(unittest.TestCase):
    def test_cuda_default_split(self):
        # 2 nodes, 128 cores, 8 GPUs -> 4 GPUs/node; defaults: device=cuda,
        # labeling_jobs_per_node=1, fit_jobs_per_node=2, 1 entropy/featurize per node.
        res = worker_layout(_cfg(), nnodes=2, ncores=128, ngpus=8)
        self.assertEqual(res.device, "cuda")
        self.assertEqual(res.gpus_per_node, 4)
        self.assertEqual(res.n_fit_workers, 4)  # 2/node * 2 nodes
        self.assertEqual(res.n_label_workers, 2)  # 1/node * 2 nodes
        self.assertEqual(res.n_entropy_workers, 2)  # 1/node * 2 nodes
        self.assertEqual(res.entropy_cores_per_job, 1)
        self.assertEqual(res.n_featurize_workers, 2)
        self.assertEqual(res.featurize_cores_per_job, 4)

    def test_cuda_entropy_and_featurize_knobs(self):
        cfg = _cfg(
            "[ourFeaturization]\nfeaturize_jobs_per_node = 5\n"
            "[ourStructureGen]\nentropy_jobs_per_node = 32\n"
            "[ourLabeling]\nlabeling_jobs_per_node = 1\n"
        )
        res = worker_layout(cfg, nnodes=4, ncores=128, ngpus=16)  # 4 GPUs/node
        self.assertEqual(res.n_entropy_workers, 128)  # 32 * 4
        self.assertEqual(res.n_featurize_workers, 20)  # 5 * 4

    def test_cuda_too_many_gpu_jobs_raises(self):
        # default labeling_jobs_per_node=1 + fit_jobs_per_node=4 = 5 > 4 GPUs/node
        cfg = _cfg("[ourFit]\nfit_jobs_per_node = 4\n")
        with self.assertRaises(AssertionError):
            worker_layout(cfg, nnodes=1, ncores=64, ngpus=4)

    def test_cpu_layout(self):
        cfg = _cfg(
            "[Main]\ndevice = cpu\n"
            "[ourStructureGen]\nentropy_jobs_per_node = 1\nentropy_cores_per_job = 8\n"
            "[ourLabeling]\nlabeling_jobs_per_node = 4\nlabeling_cores_per_job = 24\n"
            "[ourFeaturization]\nfeaturize_jobs_per_node = 2\nfeaturize_cores_per_job = 4\n"
            "[ourFit]\nfit_jobs_per_node = 2\nfit_cores_per_job = 4\n"
        )
        # 8 + 96 + 8 + 8 = 120 cores/node <= 128 - 2 reserve
        res = worker_layout(cfg, nnodes=4, ncores=512, ngpus=0)
        self.assertEqual(res.device, "cpu")
        self.assertEqual(res.n_label_workers, 16)  # 4/node * 4 nodes
        self.assertEqual(res.label_cores_per_job, 24)
        self.assertEqual(res.n_fit_workers, 8)
        self.assertEqual(res.fit_cores_per_job, 4)
        self.assertEqual(res.n_entropy_workers, 4)
        self.assertEqual(res.entropy_cores_per_job, 8)
        self.assertEqual(res.n_featurize_workers, 8)

    def test_strict_entropy_forces_single_worker(self):
        # strict_entropy_decrease=1 caps entropy to one serial worker regardless of jobs/nodes
        cfg = _cfg(
            "[Main]\ndevice = cpu\n"
            "[ourStructureGen]\nentropy_jobs_per_node = 4\nstrict_entropy_decrease = 1\n"
            "[ourLabeling]\nlabeling_jobs_per_node = 4\nlabeling_cores_per_job = 24\n"
            "[ourFeaturization]\nfeaturize_jobs_per_node = 2\nfeaturize_cores_per_job = 4\n"
            "[ourFit]\nfit_jobs_per_node = 2\nfit_cores_per_job = 4\n"
        )
        res = worker_layout(cfg, nnodes=4, ncores=512, ngpus=0)
        self.assertEqual(res.n_entropy_workers, 1)

    def test_cpu_budget_exceeded_raises(self):
        cfg = _cfg(
            "[Main]\ndevice = cpu\n"
            "[ourLabeling]\nlabeling_jobs_per_node = 4\nlabeling_cores_per_job = 30\n"
            "[ourFeaturization]\nfeaturize_jobs_per_node = 2\nfeaturize_cores_per_job = 4\n"
            "[ourFit]\nfit_jobs_per_node = 2\nfit_cores_per_job = 4\n"
        )
        # 1 + 120 + 8 + 8 = 137 cores/node > 128 - 2 reserve
        with self.assertRaises(AssertionError):
            worker_layout(cfg, nnodes=1, ncores=128, ngpus=0)


if __name__ == "__main__":
    unittest.main()
