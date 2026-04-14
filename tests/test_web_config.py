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
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = AppConfig.from_env()

        self.assertEqual(cfg.max_file_size_bytes, 20 * 1024 * 1024)
        self.assertEqual(cfg.allowed_types, {"pdf", "docx", "pptx"})
        self.assertTrue(cfg.debug)
        self.assertTrue(cfg.llm_enable_thinking)
        self.assertEqual(cfg.worker_max_parallel_jobs, 1)

    def test_worker_parallel_jobs_from_env(self) -> None:
        env = {
            "WORKER_MAX_PARALLEL_JOBS": "3",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = AppConfig.from_env()

        self.assertEqual(cfg.worker_max_parallel_jobs, 3)


if __name__ == "__main__":
    unittest.main()
