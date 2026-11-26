import unittest

from scripts.run_promotion_live import validate_gpu_provenance


class GpuProvenanceTests(unittest.TestCase):
    def test_matching_gpu_family_needs_no_override(self):
        self.assertFalse(
            validate_gpu_provenance(
                "NVIDIA L4, 23034 MiB", "NVIDIA L4", False
            )
        )

    def test_mismatch_requires_explicit_override(self):
        with self.assertRaisesRegex(RuntimeError, "--allow-gpu-mismatch"):
            validate_gpu_provenance(
                "NVIDIA L4, 23034 MiB", "NVIDIA A100", False
            )

    def test_explicit_mismatch_is_reported(self):
        self.assertTrue(
            validate_gpu_provenance(
                "NVIDIA L4, 23034 MiB", "NVIDIA A100", True
            )
        )

    def test_unknown_source_gpu_is_never_allowed(self):
        with self.assertRaisesRegex(RuntimeError, "supported L4/A100"):
            validate_gpu_provenance("unknown GPU", "NVIDIA A100", True)


if __name__ == "__main__":
    unittest.main()
