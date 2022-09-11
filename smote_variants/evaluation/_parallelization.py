"""
This module implements the process based parallelization with timeout.
"""

import threading
import multiprocessing
from queue import Empty

__all__ = ['TimeoutJobBase',
           'ThreadTimeoutProcessPool',
           'wait_for_lock',
           'queue_get_default']

class TimeoutJobBase:
    """
    Preferred base class of a TimeoutJob
    """
    def execute(self):
        """
        Execute the job

        Returns:
            obj: the result of the job
        """
        raise RuntimeError(f"{self.__class__.__name__}.execute method not implemented")

    def timeout(self):
        """
        Executes the timeout clean-up

        Returns:
            obj: the result of the timeout clean-up
        """
        raise RuntimeError(f"{self.__class__.__name__}.timeout method not implemented")

class FunctionWrapperJob(TimeoutJobBase):
    """
    A class wrapping the function-task interface of the pool
    """
    def __init__(self, function, task):
        """
        Constructor of the wrapper

        Args:
            function (callable): the function to be called
            task (obj): the argument of the function
        """
        self.function = function
        self.task = task

    def execute(self):
        """
        Executes the job

        Returns:
            obj: the return value of the task
        """
        return self.function(self.task)

    def timeout(self):
        """
        Executes the timeout clean-up

        Returns:
            None: timed-out tasks indicated by None
        """
        return None

def execute_job_object(job, queue, queue_lock):
    """
    Execute a job object and put the result into a queue

    Args:
        job: the job object
        queue: the communication queue
    """
    result = job.execute()
    queue_lock.acquire()
    queue.put(result)
    queue_lock.release()

def queue_get_default(queue, job):
    """
    Return the content of the queue if available, otherwise return
    the timeout fallback

    Args:
        queue (multiprocessing.Queue): a queue
        job (TimeoutJob): a timeout job object

    Returns:
        obj: the content of the queue or the default timeout object
    """
    try:
        return queue.get(block=False)
    except Empty:
        return job.timeout()

def process_manager(job,
                    idx,
                    results,
                    process_start_lock,
                    timeout=None):
    """
    The timed-out process manager function, executes a process
    with timeout and writes the result into the results list/arra

    Args:
        job (obj): the job object
        idx (int): the index of the job in the the queue (and results)
        results (list): a list collecting the results
        timeout (float/None): the timeout time in seconds, None/negative
                                means no timeout
    """
    # creating the queue used to communicate with the process
    queue = multiprocessing.Queue()

    queue_lock = multiprocessing.Lock()

    # creating and executing the process with timeout
    process = multiprocessing.Process(target=execute_job_object,
                                      kwargs={'job': job,
                                                'queue': queue,
                                                'queue_lock': queue_lock})

    process.start()

    process_start_lock.release()

    if timeout is not None and timeout > 0:
        process.join(timeout)
    else:
        process.join()

    # if the process finished normally, the result is written into
    # the results output argument
    if not process.is_alive():
        results[idx] = queue_get_default(queue, job)

        return

    # remove anything from the queue to prevent deadlock
    try:
        queue_lock.acquire()
        results[idx] = queue.get(block=False)
    except Empty:
        pass

    # if the process is still running, the process is killed and
    # the timeout function is executed for potential clean-up
    process.kill()
    results[idx] = job.timeout()

def wait_for_lock(lock):
    """
    Wait for a multiprocessing lock

    Args:
        lock (multiprocessing.Lock): a lock to wait for
    """
    lock.acquire()

def pooling_thread(pool_object):
    """
    A function spinning up processes with timeout until all jobs in
    the pool are processed.

    Args:
        pool_object (ThreadTimeoutProcessPool): the pool object
    """
    while True:
        # acquiring the pool lock to determine the next job to be
        # executed
        pool_object.lock.acquire()

        # releasing the lock and exiting if all jobs has been done
        if pool_object.idx >= len(pool_object.jobs):
            pool_object.lock.release()
            break

        process_start_lock = threading.Lock()
        process_start_lock.acquire() # pylint: disable=consider-using-with

        # creating a new process manager thread to execute the job
        # with timeout
        thread = threading.Thread(target=process_manager,
                                    kwargs={'job': pool_object.jobs[pool_object.idx],
                                            'idx': pool_object.idx,
                                            'results': pool_object.results,
                                            'process_start_lock': process_start_lock,
                                            'timeout': pool_object.timeout},
                                    daemon=True)
        pool_object.idx = pool_object.idx + 1

        # the index indicating the next job to be done is updated, so the
        # lock can be released now

        thread.start()

        process_start_lock.acquire() # pylint: disable=consider-using-with
        process_start_lock.release()

        pool_object.lock.release()

        thread.join()

class ThreadTimeoutProcessPool:
    """
    A pool object being able to run n_jobs workers each subject
    to time-out to process a list of tasks
    """
    def __init__(self, n_jobs=1, timeout=-1):
        """
        The constructor of the object

        Args:
            n_jobs (int): the number of parallel jobs
            timeout (float/None): the timeout time in seconds,
                                    None/negative means no time-out
        """
        self.n_jobs = n_jobs
        self.timeout = timeout
        self.lock = multiprocessing.Lock()

        # all members need to be initialized in the constructor, even those
        # added and set by member functions
        self.jobs = None
        self.idx = -1
        self.results = None

    def execute(self, jobs):
        """
        Execute the jobs parallel with time-out.

        Args:
            jobs (list): the list of jobs to be executed

        Returns:
            list: the list of results
        """
        # setting the members to be accessed by the pooling threads
        self.jobs = jobs
        self.idx = 0
        self.results = [None] * len(self.jobs)

        # creating the pool of workers
        pool = [threading.Thread(target=pooling_thread,
                                 kwargs={'pool_object': self},
                                 daemon=True)
                                            for _ in range(self.n_jobs)]

        for thread in pool:
            thread.start()

        for thread in pool:
            thread.join()

        return self.results

    def apply(self, function, jobs):
        """
        apply-style interface of the pool object: applying a function to a list of jobs

        Args:
            function (callable): the function to be executed
            jobs (list): the list of arguments the function is executed with

        Returns:
            list: the results of the job execution
        """
        jobs = [FunctionWrapperJob(function, job) for job in jobs]
        return self.execute(jobs)
