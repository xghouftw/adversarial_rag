import tempfile
import unittest
from pathlib import Path

from reproduction.robustrag_setup import (
    SOURCE_HASHES,
    build_runner,
    verify_vendored_source,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_DIR / "third_party/robustrag_upstream"


class RobustRagSetupTests(unittest.TestCase):
    def test_vendored_source_matches_pinned_hashes(self):
        verify_vendored_source(SOURCE_DIR)

    def test_runner_is_complete_and_only_models_copy_is_patched(self):
        with tempfile.TemporaryDirectory() as temporary:
            runner_dir = Path(temporary) / "runner"
            build_runner(SOURCE_DIR, runner_dir)

            for relative_name in SOURCE_HASHES:
                source = (SOURCE_DIR / relative_name).read_text()
                runner = (runner_dir / relative_name).read_text()
                if relative_name == "src/models.py":
                    self.assertNotEqual(runner, source)
                    self.assertNotIn("import litellm\nfrom litellm", runner)
                    self.assertIn("        from openai import OpenAI", runner)
                    self.assertIn("        from litellm import batch_completion", runner)
                else:
                    self.assertEqual(runner, source)


if __name__ == "__main__":
    unittest.main()
