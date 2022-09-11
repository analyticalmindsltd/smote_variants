"""
This module tests the parallelization ecosystem.
"""

import time
import threading
import multiprocessing
from queue import Empty

import pytest

from smote_variants.evaluation import (TimeoutJobBase,
                                        ThreadTimeoutProcessPool,
                                        wait_for_lock,
                                        queue_get_default)

sleeps = [1, 2, 6, 7]

sleeps_no_timeout = [1, 2]

class SleepJob(TimeoutJobBase):
    """
    Sleep job for testing
    """
    def __init__(self, sleep):
        """
        The constructor of the sleep job
        """
        self.sleep = sleep

    def execute(self):
        """
        Execute the sleep job

        Returns:
            dict: the result of the sleep job
        """
        time.sleep(self.sleep)
        return {'slept': self.sleep}

    def timeout(self):
        """
        Carry out the timeout process for the sleep job

        Returns:
            dict: the return of the timeout post-processing
        """
        return {'slept': None}

def sleep_job(sleep):
    """
    Sleep job function

    Args:
        sleep (float): the time to sleep

    Returns:
        dict: the result of the sleeping
    """
    time.sleep(sleep)
    return {'slept': sleep}

def test_jobs_objects_timeout():
    """
    Testing the job objects with timeout.
    """

    ttpp = ThreadTimeoutProcessPool(n_jobs=2, timeout=4)

    results = ttpp.execute([SleepJob(sleep) for sleep in sleeps])

    assert len(results) == len(sleeps)

    assert all(isinstance(result, dict) for result in results)

def test_jobs_functions_timeout():
    """
    Testing the functions with timeout.
    """

    ttpp = ThreadTimeoutProcessPool(n_jobs=2, timeout=4)

    results = ttpp.apply(sleep_job, sleeps)

    assert len(results) == len(sleeps)

    dict_count = sum(isinstance(result, dict) for result in results)

    assert 0 < dict_count < len(sleeps)

def test_jobs_objects_no_timeout():
    """
    Testing the job objects without timeout.
    """

    ttpp = ThreadTimeoutProcessPool(n_jobs=2, timeout=-1)

    results = ttpp.execute([SleepJob(sleep) for sleep in sleeps_no_timeout])

    assert len(results) == len(sleeps_no_timeout)

    assert all(isinstance(result, dict) for result in results)

def test_jobs_functions_no_timeout():
    """
    Testing the job functions without timeout.
    """

    ttpp = ThreadTimeoutProcessPool(n_jobs=2, timeout=-1)

    results = ttpp.apply(sleep_job, sleeps_no_timeout)

    assert len(results) == len(sleeps_no_timeout)

    dict_count = sum(isinstance(result, dict) for result in results)

    assert dict_count == len(sleeps_no_timeout)

def test_exceptions():
    """
    Testing the exception in the base class
    """

    toj = TimeoutJobBase()

    with pytest.raises(RuntimeError) as _:
        toj.execute()

    with pytest.raises(RuntimeError) as _:
        toj.timeout()

def test_lock():
    """
    Testing the wait for lock functionality.
    """
    lock = multiprocessing.Lock()
    lock.acquire()

    thread = threading.Thread(target=wait_for_lock, args=(lock,))
    thread.start()

    time.sleep(3)

    lock.release()

    thread.join(1)

    time.sleep(1)

    assert not thread.is_alive()

class MockQueue: # pylint: disable=too-few-public-methods
    """
    Class mocking the Queue object
    """
    def get(self, block):
        """
        Mocking the get function of the Queue

        Args:
            block (bool): whether to block

        Returns:
            obj: the content of the queue
        """
        _ = block
        raise Empty

class MockJob: # pylint: disable=too-few-public-methods
    """
    Class mocking a TimeoutJob object
    """
    def timeout(self):
        """
        The timeout function

        Returns:
            int: 1
        """
        return 1

def test_queue_get_default():
    """
    Testing the queue_get_default functionality.
    """
    assert queue_get_default(MockQueue(), MockJob()) == 1
