import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from job_worker import JobQueueWorker


class _FakeProc:
    def __init__(self, *, alive: bool, exitcode: int | None = 0) -> None:
        self._alive = alive
        self.exitcode = exitcode
        self.terminated = False
        self.join_calls = 0

    def is_alive(self) -> bool:
        return self._alive

    def terminate(self) -> None:
        self.terminated = True
        self._alive = False

    def join(self, timeout: float | None = None) -> None:
        _ = timeout
        self.join_calls += 1


class TestJobQueueWorkerParallel(unittest.TestCase):
    @patch("job_worker.threading.Thread")
    def test_max_parallel_jobs_minimum_one(self, _thread_cls: MagicMock) -> None:
        auth = MagicMock()
        service = MagicMock()
        worker = JobQueueWorker(auth, service, max_parallel_jobs=0)
        self.assertEqual(worker.max_parallel_jobs, 1)

    @patch("job_worker.threading.Thread")
    def test_cancel_targets_specific_job(self, _thread_cls: MagicMock) -> None:
        auth = MagicMock()
        service = MagicMock()
        worker = JobQueueWorker(auth, service, max_parallel_jobs=2)
        proc = _FakeProc(alive=True, exitcode=None)
        worker._active_procs["job-a"] = proc

        self.assertTrue(worker.cancel("job-a"))
        self.assertTrue(proc.terminated)
        self.assertFalse(worker.cancel("job-b"))

    @patch("job_worker.threading.Thread")
    def test_reap_marks_failed_for_abnormal_exit(self, _thread_cls: MagicMock) -> None:
        auth = MagicMock()
        service = MagicMock()
        worker = JobQueueWorker(auth, service, max_parallel_jobs=2)
        proc = _FakeProc(alive=False, exitcode=1)
        worker._active_procs["job-a"] = proc
        auth.get_job.return_value = SimpleNamespace(status="running")

        worker._reap_finished_processes()

        auth.mark_job_failed.assert_called_once_with("job-a", "转换进程异常退出")
        self.assertNotIn("job-a", worker._active_procs)

    @patch("job_worker.threading.Thread")
    def test_reap_does_not_mark_failed_when_not_running(self, _thread_cls: MagicMock) -> None:
        auth = MagicMock()
        service = MagicMock()
        worker = JobQueueWorker(auth, service, max_parallel_jobs=2)
        proc = _FakeProc(alive=False, exitcode=1)
        worker._active_procs["job-a"] = proc
        auth.get_job.return_value = SimpleNamespace(status="cancelled")

        worker._reap_finished_processes()

        auth.mark_job_failed.assert_not_called()
        self.assertNotIn("job-a", worker._active_procs)


if __name__ == "__main__":
    unittest.main()
