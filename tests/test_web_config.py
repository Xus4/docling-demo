import os
import unittest
from unittest.mock import patch

from config import AppConfig


class TestAppConfig(unittest.TestCase):
    def test_from_env_parses_size_and_types(self) -> None:
        env = {
            "MAX_FILE_SIZE": "20MB",
            "ALLOWED_TYPES": "pdf,docx,pptx",
            "DATA_DIR": "./data-test",
            "DEBUG": "true",
            "AUTO_CLEANUP": "yes",
            "CLEANUP_MAX_AGE_HOURS": "12",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = AppConfig.from_env()

        self.assertEqual(cfg.max_file_size_bytes, 20 * 1024 * 1024)
        self.assertEqual(cfg.allowed_types, {"pdf", "docx", "pptx"})
        self.assertTrue(cfg.debug)
        self.assertTrue(cfg.auto_cleanup)
        self.assertEqual(cfg.cleanup_max_age_hours, 12)
        self.assertTrue(cfg.llm_enable_thinking)


if __name__ == "__main__":
    unittest.main()
